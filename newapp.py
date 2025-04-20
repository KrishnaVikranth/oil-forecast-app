# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    model = load_model(
        'lstm_oilprodus_model.h5',
        compile=False,
        custom_objects={"mse": MeanSquaredError()}
    )
    return model

# Function to create sequences
def create_sequences(data, seq_length=100):
    X = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
    return np.array(X)

# Function to do recursive future prediction
def forecast_future(initial_sequence, model, scaler, future_steps=30):
    future_predictions = []
    current_sequence = initial_sequence.copy()

    for _ in range(future_steps):
        next_pred_scaled = model.predict(current_sequence, verbose=0)
        next_pred_unscaled = scaler.inverse_transform(next_pred_scaled)
        future_predictions.append(next_pred_unscaled.flatten()[0])

        next_pred_scaled = next_pred_scaled.reshape(1, 1, 1)
        current_sequence = np.concatenate((current_sequence[:, 1:, :], next_pred_scaled), axis=1)

    return future_predictions

# Streamlit App
st.set_page_config(page_title="Oil Production Forecast", layout="centered")

st.title("🛢️ Oil Production Forecasting App")

# 📄 Auto-load economic_data.csv
try:
    data = pd.read_csv('economic_data.csv')



    # Parse Date column
    data['Date'] = pd.to_datetime(data['Date'])
    last_date = data['Date'].max()

    # Extract the OILPRODUS column
    values = data['OILPRODUS'].values.reshape(-1, 1)

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    # Create sequences for in-sample prediction
    sequence_length = 100
    if len(scaled_data) <= sequence_length:
        st.error("❗ Not enough data to create sequences. Need more than 100 rows.")
    else:
        model = load_trained_model()

        X_test = create_sequences(scaled_data, sequence_length)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict in-sample
        predictions = model.predict(X_test)
        predictions_rescaled = scaler.inverse_transform(predictions)

        # Prepare actual data for comparison
        actual = values[sequence_length:]
        historical_dates = data['Date'].iloc[sequence_length:]

        # User Input for Future Forecasting
        st.subheader("🔮 Forecast Future Oil Production")
        future_steps = st.number_input(
            "Enter number of future steps to forecast:",
            min_value=1,
            max_value=500,
            value=30,
            step=1
        )

        if st.button("🚀 Forecast Future"):
            # Prepare last sequence safely
            last_sequence = scaled_data[-sequence_length:]
            last_sequence = np.nan_to_num(last_sequence)  # Replace NaN safely
            last_sequence = last_sequence.reshape(1, sequence_length, 1)

            # Forecast future
            future_forecast = forecast_future(last_sequence, model, scaler, future_steps)
            future_forecast = np.array(future_forecast)

            # Generate future dates
            #future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_steps, freq='MS')
            future_dates = future_dates.strftime('%Y-%m-%d')  # Format to remove timestamp


            # Prepare forecast DataFrame
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Forecasted_Oil_Production": np.round(future_forecast, 2)
            })

            # 📈 Interactive Plotly Plot
            st.subheader("📈 Prediction Results")
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=historical_dates, y=actual.flatten(), mode='lines', name='Actual Production (Historical)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=historical_dates, y=predictions_rescaled.flatten(), mode='lines', name='Model Predictions (Historical)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=future_dates, y=future_forecast, mode='lines', name='Future Forecast', line=dict(color='red', dash='dash')))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Oil Production",
                legend_title="Legend",
                template="plotly_white",
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

            # 📋 Forecasted Values Table
            st.subheader("📋 Forecasted Values")
            st.dataframe(forecast_df)

            # ⬇️ Download Button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Forecasted Data as CSV",
                data=csv,
                file_name='future_forecast.csv',
                mime='text/csv'
            )

except FileNotFoundError:
    st.error("❗ The file 'economic_data.csv' was not found. Please ensure it is placed in the same folder as app.py.")
except KeyError:
    st.error("❗ Column 'Date' or 'OILPRODUS' not found in economic_data.csv. Please check your file.")
