# CAN Bus Anomaly Detection using Autoencoders

This project demonstrates an anomaly detection system for Controller Area Network (CAN bus) data using autoencoders.

## Overview

The system uses a multi-head autoencoder to learn the normal patterns in CAN bus data and identify anomalies based on reconstruction errors.

**Workflow:**

1. **Data Preprocessing:**
    - The raw CAN bus data is parsed and grouped by CAN ID.
    - Quantile transformation is applied to normalize the data.

2. **Model Training:**
    - A multi-head autoencoder is trained on the preprocessed data.
    - Early stopping is used to prevent overfitting.

3. **Anomaly Detection:**
    - The trained model is used to reconstruct the test data.
    - Anomalies are identified based on high reconstruction errors, using either a quantile-based or standard deviation-based threshold.

## Usage

**1. Installation:**

- Make sure you have Python 3.7+ installed.
- Install the required libraries:
    - Python 3.7+
    - PyTorch
    - Scikit-learn
    - Pandas
    - NumPy
    - PyYAML

**2. Configuration:**

- Create a `config.yaml` file with the following parameters:
yaml can_log: "path/to/your/can_log.txt" can_log_test: "path/to/your/can_log_test.txt" mode: "train" # or "test" quantile_models_path: "quantile_models" model_path: "model.pth" num_epochs: 100 batch_size: 256 threshold_method: "quantile" # or "std" threshold_value: 0.99 # or 3 (for std)

**3. Training:**

- Set `mode` to "train" in the `config.yaml` file.
- Run the script:
    python main.py

**4. Testing:**

    - Set `mode` to "test" in the `config.yaml` file.
    - Run the script:

**5. Results:**

- The detected anomalies will be saved to a file.

## Requirements

- Python 3.7+
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- PyYAML


Advantages of Multi-head Autoencoder over One-Class SVM
1. Multi-head Autoencoder

Handles high-dimensional data well: Autoencoders are effective in handling high-dimensional data like CAN bus data, where multiple signals are recorded simultaneously. They can learn complex relationships between these signals and identify subtle anomalies that may be missed by One-Class SVMs.
More robust to noisy data: Autoencoders can be more robust to noisy data due to their inherent ability to learn a compressed representation of the data, filtering out noise during the encoding and decoding process. This makes them potentially more suitable for real-world CAN bus data, which can be noisy.
Can capture temporal dependencies: Multi-head Autoencoders, with appropriate architecture modifications (like recurrent connections), can be adapted to capture temporal dependencies in CAN bus data. This is important because anomalies often manifest as deviations from expected sequences of events in the CAN bus communication.
Provides reconstruction error for anomaly scoring: Autoencoders provide a continuous reconstruction error as an anomaly score, which allows for more nuanced anomaly ranking and thresholding. One-Class SVM typically provides a binary classification (normal or anomaly) and may not offer the same level of flexibility in identifying the severity of anomalies.
Flexibility in handling multi-dimensional data: Multi-head Autoencoders can simultaneously reconstruct different aspects of the input data by utilizing separate heads. This allows for a more comprehensive analysis of anomalies across multiple dimensions of the CAN bus data.
2. One-Class SVM

Simpler to implement and train: One-Class SVM is generally simpler to implement and train compared to Multi-head Autoencoder. Autoencoders require careful hyperparameter tuning and architectural design.
Effective for smaller datasets: One-Class SVM can be effective for smaller datasets, while Autoencoders generally require larger datasets for effective training.
Well-suited for novelty detection: One-Class SVM is primarily designed for novelty detection, where the goal is to identify new, unseen data points that deviate from the training data. This can be relevant in specific CAN bus anomaly detection scenarios.
Overall

For CAN bus anomaly detection, Multi-head Autoencoders offer potential advantages due to their ability to handle high-dimensional, noisy data, capture temporal dependencies, and provide a continuous anomaly score. However, One-Class SVMs can be a viable alternative for smaller datasets or novelty detection scenarios. Ultimately, the best choice depends on the specific requirements and characteristics of the datase
