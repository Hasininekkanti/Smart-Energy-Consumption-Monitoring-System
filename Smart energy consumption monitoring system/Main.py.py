import pandas as pd
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

sequence_length = 20

def log(message):
    output_box.insert(tk.END, message + "\n")
    output_box.see(tk.END)
    root.update()

def smooth(data, window_size=3):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def train_and_monitor(file, appliance):

    log("\n==============================")
    log("Training model for: " + appliance)
    log("==============================")

    # Load only part of dataset for faster training
    data = pd.read_csv(file).head(5000)

    # Add time feature
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour

    features = data[["power","hour"]].values

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X = []
    y = []

    for i in range(len(features_scaled) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        y.append(features_scaled[i+sequence_length][0])

    X = np.array(X)
    y = np.array(y)

    # -------------------------------
    # LSTM MODEL (FAST VERSION)
    # -------------------------------

    model = Sequential()

    model.add(LSTM(32, return_sequences=True, input_shape=(sequence_length,2)))
    model.add(Dropout(0.2))

    model.add(LSTM(16))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    log("Training model...")

    model.fit(
        X,
        y,
        epochs=15,
        batch_size=32,
        verbose=0
    )

    log("Model trained successfully for " + appliance)

    stream_buffer = []
    errors = []
    health_scores = []
    total_alerts = []

    actual_values = []
    predicted_values = []

    alert_cooldown = 0

    # -------------------------------
    # IoT STREAM SIMULATION
    # -------------------------------

    for index, row in data.head(120).iterrows():

        timestamp = row["timestamp"]
        power = float(row["power"])
        hour = row["hour"]

        log("\nAppliance: " + appliance)
        log("Time: " + str(timestamp))
        log("Actual Power: " + str(power))

        stream_buffer.append([power, hour])

        if len(stream_buffer) >= sequence_length:

            seq = np.array(stream_buffer[-sequence_length:])
            seq_scaled = scaler.transform(seq)
            seq_scaled = seq_scaled.reshape(1, sequence_length,2)

            predicted_scaled = model.predict(seq_scaled, verbose=0)[0][0]

            dummy = np.array([[predicted_scaled, seq_scaled[0][-1][1]]])
            predicted = scaler.inverse_transform(dummy)[0][0]

            log("Predicted Power: " + str(round(predicted,2)))

            actual_values.append(power)
            predicted_values.append(predicted)

            error = abs(predicted - power)
            errors.append(error)

            max_deviation = 50
            health_score = max(0, 100 - (error / max_deviation * 100))
            health_scores.append(health_score)

            log("Deviation: " + str(round(error,2)))
            log("Health Score: " + str(round(health_score,2)) + " / 100")

            if error > 20 and alert_cooldown == 0:
                log("⚠ ALERT: " + appliance + " abnormal usage detected!")
                total_alerts.append(1)
                alert_cooldown = 5
            else:
                log("Normal usage")

            if alert_cooldown > 0:
                alert_cooldown -= 1

        log("-------------------")
        root.update()
        time.sleep(0.1)

    # -------------------------------
    # FINAL METRICS
    # -------------------------------

    if len(errors) > 0:

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.square(errors)))
        avg_health = np.mean(health_scores)

        log("\n===== SYSTEM METRICS =====")
        log("Mean Absolute Error (MAE): " + str(round(mae,2)))
        log("Root Mean Squared Error (RMSE): " + str(round(rmse,2)))
        log("Average Appliance Health: " + str(round(avg_health,2)))
        log("Total Alerts Generated: " + str(len(total_alerts)))
        log("==========================")

        actual_smooth = smooth(actual_values)
        predicted_smooth = smooth(predicted_values)

        plt.figure(figsize=(10,5))
        plt.plot(actual_smooth, label="Actual Power", linewidth=2)
        plt.plot(predicted_smooth, label="Predicted Power", linestyle="--", linewidth=2)

        plt.xlabel("Time Steps")
        plt.ylabel("Power Consumption")
        plt.title(appliance + " Power Consumption (Actual vs Predicted)")
        plt.legend()
        plt.grid(True)
        plt.show()


def start_monitoring():

    appliance = appliance_box.get()

    files = {
        "Fan": "fan_small_25000.csv",
        "Coffee": "coffee_small_25000.csv",
        "Fridge": "fridge_small_25000.csv",
        "TV": "tv_small_25000.csv",
        "WashingMachine": "washingmachine_small_25000.csv"
    }

    file = files[appliance]

    output_box.delete(1.0, tk.END)

    train_and_monitor(file, appliance)


# -------------------------------
# GUI
# -------------------------------

root = tk.Tk()
root.title("Smart Home Energy Monitoring System")
root.geometry("650x520")

title = tk.Label(root, text="Smart Home Energy Monitoring System", font=("Arial",16))
title.pack(pady=10)

label = tk.Label(root, text="Select Appliance")
label.pack()

appliance_box = ttk.Combobox(root)
appliance_box["values"] = ["Fan","Coffee","Fridge","TV","WashingMachine"]
appliance_box.current(0)
appliance_box.pack(pady=5)

start_button = tk.Button(root, text="Start Monitoring", command=start_monitoring)
start_button.pack(pady=10)

output_box = scrolledtext.ScrolledText(root, width=75, height=20)
output_box.pack(pady=10)

root.mainloop()