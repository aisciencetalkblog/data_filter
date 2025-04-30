import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

st.title("Multi-Signal Filtering and Summary Demo")

# --- Upload CSV or use demo ---
uploaded_file = st.file_uploader("Upload a CSV file with a 'Time' column and at least 3 signal columns", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Time"])
    st.success("File uploaded successfully.")
else:
    st.info("No file uploaded. Using demo dataset.")
    np.random.seed(42)
    time = pd.date_range(start="2025-01-01", periods=100, freq="H")
    sensor_1 = 22 + np.random.normal(0, 0.4, size=100).cumsum()
    sensor_2 = 23 + np.random.normal(0, 0.3, size=100).cumsum()
    sensor_3 = 21.5 + np.random.normal(0, 0.5, size=100).cumsum()
    df = pd.DataFrame({
        "Time": time,
        "Sensor_1": sensor_1,
        "Sensor_2": sensor_2,
        "Sensor_3": sensor_3
    })

# --- Signal selection ---
df["Time"] = pd.to_datetime(df["Time"])
signal_names = df.columns.drop("Time")
selected_signals = st.multiselect("Select at least 3 signals:", signal_names.tolist(), default=signal_names.tolist())
if len(selected_signals) < 3:
    st.warning("Please select at least 3 signals.")
    st.stop()

# --- Filter selection ---
filter_type = st.selectbox("Choose a summary filter type (across signals and time):",
                           ["Moving Average", "Gaussian", "Median"])

# --- Filter parameters ---
if filter_type in ["Moving Average", "Median"]:
    window = st.slider("Window size (across time)", min_value=3, max_value=25, value=5, step=2)
elif filter_type == "Gaussian":
    sigma = st.slider("Gaussian sigma (across time)", min_value=1.0, max_value=10.0, value=2.0)

# --- Compute across-signals summary ---
# Step 1: Aggregate across selected signals at each time step
if filter_type == "Median":
    summary_signal = df[selected_signals].median(axis=1)
elif filter_type == "Moving Average":
    summary_signal = df[selected_signals].mean(axis=1).rolling(window, center=True).mean()
elif filter_type == "Gaussian":
    avg_signal = df[selected_signals].mean(axis=1)
    summary_signal = gaussian_filter1d(avg_signal, sigma=sigma)

df["Filtered_Summary"] = summary_signal

# --- Combined Plot ---
st.subheader("Summary Plot: All Signals + Filtered")
fig, ax = plt.subplots()
for sig in selected_signals:
    ax.plot(df["Time"], df[sig], alpha=0.5, label=sig)
ax.plot(df["Time"], df["Filtered_Summary"], color="black", label="Filtered Summary", linewidth=2)
ax.set_title("Summary Plot with All Signals")
ax.legend()
plt.xticks(rotation=20)
st.pyplot(fig)

# --- Individual Plots ---
st.subheader("Individual Signals vs Filtered Summary")
for sig in selected_signals:
    fig, ax = plt.subplots()
    ax.plot(df["Time"], df[sig], label=sig)
    ax.plot(df["Time"], df["Filtered_Summary"], label="Filtered Summary", linestyle="--", linewidth=2)
    ax.set_title(f"{sig} with {filter_type} Summary Filter")
    ax.legend()
    plt.xticks(rotation=20)
    st.pyplot(fig)

# --- Download data ---
st.subheader("Download Combined Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download CSV with Filtered Summary",
    data=csv,
    file_name="filtered_summary.csv",
    mime="text/csv"
)
