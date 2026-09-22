# Smart Home Energy Monitoring System using LSTM

## 📌 Overview

Smart Home Energy Monitoring System is a Python-based machine learning application designed to monitor and predict household appliance power consumption using Long Short-Term Memory (LSTM) neural networks.

The system analyzes historical appliance energy consumption data, performs data preprocessing and feature engineering, and uses time-series forecasting to predict power usage. It also compares actual and predicted consumption to identify abnormal usage patterns and generates an appliance health score.

A Tkinter-based graphical user interface (GUI) allows users to select an appliance and monitor its energy consumption, predictions, health status, and alerts.

---

## 🎯 Objectives

- Predict future appliance power consumption using LSTM.
- Analyze historical energy consumption patterns.
- Perform preprocessing and feature engineering on time-series data.
- Monitor appliance energy usage.
- Detect abnormal energy consumption.
- Generate an appliance health score.
- Visualize actual and predicted power consumption.
- Provide a simple and user-friendly monitoring interface.

---

## 💡 Key Features

### 1. Energy Consumption Prediction
Uses an LSTM deep learning model to learn patterns from historical power consumption data and predict future power usage.

### 2. Data Preprocessing
The system:
- Converts timestamps into datetime format.
- Resamples the data into fixed 25-minute intervals.
- Handles missing values.
- Applies smoothing to reduce fluctuations.
- Normalizes features using MinMaxScaler.

### 3. Feature Engineering

The following features are generated:

- `power` – appliance power consumption
- `hour` – hour of the day
- `day` – day information
- `lag1` – previous power reading
- `lag2` – power reading from the previous time step

These features help the model understand temporal and historical energy consumption patterns.

### 4. Time-Series Sequence Generation

The model uses a sequence length of 30 time steps.

A sliding-window approach is used to create sequences where:

**Previous 30 observations → Next power consumption prediction**

### 5. LSTM-Based Prediction

The model uses a stacked LSTM architecture:

- LSTM layer – 64 units
- Dropout – 20%
- LSTM layer – 32 units
- Dropout – 20%
- Dense layer – 16 units
- Output layer – 1 unit

The model is trained using the Adam optimizer and Mean Absolute Error (MAE) loss.

### 6. Anomaly Detection

The system calculates the difference between actual and predicted power consumption.

If the deviation exceeds a predefined threshold, the system generates an abnormal usage alert.

### 7. Appliance Health Score

A health score is calculated based on the prediction deviation.

The system categorizes the appliance condition as:

- Excellent
- Good
- Warning
- Critical

This provides an easy-to-understand representation of appliance behavior.

### 8. Performance Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Maximum Error
- Prediction Accuracy
- Average Health Score

### 9. Visualization

Matplotlib is used to visualize:

- Actual power consumption
- Predicted power consumption

This helps analyze how closely the model follows the actual energy consumption pattern.

### 10. Graphical User Interface

A Tkinter-based GUI allows users to:

- Select an appliance
- Start monitoring
- View predictions
- View deviation and health status
- Receive abnormal usage alerts

---

## 🏗️ System Workflow

```text
                 Appliance Dataset
                        ↓
                Data Preprocessing
                        ↓
              Time-Series Resampling
                        ↓
               Data Smoothing
                        ↓
                Feature Engineering
                        ↓
                  Normalization
                        ↓
             Sequence Generation
                        ↓
                  LSTM Model
                        ↓
               Power Prediction
                        ↓
          ┌─────────────┴─────────────┐
          ↓                           ↓
   Actual vs Predicted          Error Calculation
          ↓                           ↓
    Visualization              Health Score
                                      ↓
                              Anomaly Detection
                                      ↓
                                  GUI Alert