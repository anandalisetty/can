import os
import re
import yaml
import joblib
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CAN-Autoencoder")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Config
# -------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# -------------------------
# Parse CAN Log
# -------------------------
def parse_can_log(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r'Timestamp:\s+([\d\.]+)\s+ID:\s+([0-9a-fA-F]+).*?DLC:\s+(\d+)\s+(.*)', line)
            if match:
                timestamp = float(match.group(1))
                can_id = match.group(2).strip()
                dlc = int(match.group(3))
                data = match.group(4).strip().split()
                data = [int(b, 16) for b in data]
                while len(data) < 8:
                    data.append(0)
                records.append((timestamp, can_id, *data))
    df = pd.DataFrame(records, columns=["Timestamp", "ID"] + [f"B{i}" for i in range(8)])
    return df

# -------------------------
# Dataset
# -------------------------
class CANDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

# -------------------------
# Model
# -------------------------
class MultiHeadAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.heads = nn.ModuleDict({
            'head1': nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim)),
            'head2': nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
        })

    def forward(self, x):
        encoded = self.shared_encoder(x)
        return {k: head(encoded) for k, head in self.heads.items()}

# -------------------------
# Preprocessing
# -------------------------
def preprocess_by_id(df, save_dir="quantile_models"):
    os.makedirs(save_dir, exist_ok=True)
    data_dict = {}
    for can_id, group in df.groupby("ID"):
        if len(group) < 100:
            continue
        try:
            qt_time = QuantileTransformer(output_distribution='normal')
            qt_data = QuantileTransformer(output_distribution='normal')
            group_features = group[["Timestamp"] + [f"B{i}" for i in range(8)]].copy()
            group_features.iloc[:, 0] = qt_time.fit_transform(group_features[["Timestamp"]]).squeeze()
            group_features.iloc[:, 1:] = qt_data.fit_transform(group_features.iloc[:, 1:])
            joblib.dump(qt_time, os.path.join(save_dir, f"{can_id}_time.pkl"))
            joblib.dump(qt_data, os.path.join(save_dir, f"{can_id}_data.pkl"))
            data_dict[can_id] = group_features.values
        except Exception as e:
            logger.warning(f"Skipping ID {can_id}: {e}")
    return data_dict

def load_quantile_models(can_ids, load_dir="quantile_models"):
    qt_dict = {}
    for can_id in can_ids:
        try:
            qt_time = joblib.load(os.path.join(load_dir, f"{can_id}_time.pkl"))
            qt_data = joblib.load(os.path.join(load_dir, f"{can_id}_data.pkl"))
            qt_dict[can_id] = (qt_time, qt_data)
        except Exception as e:
            logger.warning(f"Could not load quantile models for {can_id}: {e}")
    return qt_dict

def transform_test_data(df, qt_dict):
    data_dict = {}
    for can_id, group in df.groupby("ID"):
        if can_id not in qt_dict:
            continue
        qt_time, qt_data = qt_dict[can_id]
        try:
            group_features = group[["Timestamp"] + [f"B{i}" for i in range(8)]].copy()
            group_features.iloc[:, 0] = qt_time.transform(group_features[["Timestamp"]]).squeeze()
            group_features.iloc[:, 1:] = qt_data.transform(group_features.iloc[:, 1:])
            data_dict[can_id] = group_features.values
        except Exception as e:
            logger.warning(f"Skipping {can_id} during test transform: {e}")
    return data_dict

# -------------------------
# Train
# -------------------------
def train_autoencoder(data_dict, model_path, num_epochs=100, batch_size=256):
    all_data = np.concatenate(list(data_dict.values()))
    train_data, _ = train_test_split(all_data, test_size=0.2, random_state=42)
    loader = DataLoader(CANDataset(train_data), batch_size=batch_size, shuffle=True)

    model = MultiHeadAutoencoder(train_data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience, patience_counter = 10, 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = x.to(device)
            outputs = model(x)
            loss = sum(criterion(output, x) for output in outputs.values()) / len(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break

# -------------------------
# Inference
# -------------------------
def load_model(model_path, input_size):
    model = MultiHeadAutoencoder(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def compute_reconstruction_error(model, data, batch_size=256):
    loader = DataLoader(CANDataset(data), batch_size=batch_size)
    all_errors = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            outputs = model(x)
            error = sum(((x - out) ** 2).mean(dim=1) for out in outputs.values()) / len(outputs)
            all_errors.append(error.cpu().numpy())
    return np.concatenate(all_errors)

def detect_anomalies(errors, method='quantile', threshold=0.99):
    if method == 'quantile':
        thresh_val = np.quantile(errors, threshold)
    else:
        mean, std = np.mean(errors), np.std(errors)
        thresh_val = mean + 3 * std
    anomalies = errors > thresh_val
    return anomalies, thresh_val

def run_inference(test_data, model_path, batch_size, threshold_method, threshold_value):
    input_size = test_data.shape[1]
    model = load_model(model_path, input_size)
    errors = compute_reconstruction_error(model, test_data, batch_size)
    anomalies, threshold = detect_anomalies(errors, threshold_method, threshold_value)
    logger.info(f"Threshold: {threshold:.4f}, Anomalies: {np.sum(anomalies)}")
    return errors, anomalies

# -------------------------
# Main
# -------------------------
def main():
    config = load_config("config.yaml")
    df = parse_can_log(config["can_log"])

    if config["mode"] == "train":
        data_dict = preprocess_by_id(df, save_dir="quantile_models")
        if not data_dict:
            raise ValueError("No valid CAN IDs for training.")
        train_autoencoder(data_dict, model_path=config["model_path"],
                          num_epochs=config["num_epochs"], batch_size=config["batch_size"])

    elif config["mode"] == "test":
        qt_dict = load_quantile_models(df["ID"].unique(), load_dir="quantile_models")
        test_dict = transform_test_data(df, qt_dict)
        all_test_data = np.concatenate(list(test_dict.values()))
        run_inference(
            test_data=all_test_data,
            model_path=config["model_path"],
            batch_size=config["batch_size"],
            threshold_method=config["threshold_method"],
            threshold_value=config["threshold_value"]
        )

if __name__ == "__main__":
    main()
