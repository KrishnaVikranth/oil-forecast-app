# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# Load the pre-trained ARIMA/SARIMAX model
@st.cache_resource
def load_trained_model():
    model = joblib.load('arima_model.pkl')
    return model

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
    historical_dates = data['Date']

    # Load the ARIMA/SARIMAX model
    model = load_trained_model()

    # User Inputs: Two separate inputs
    st.subheader("🔮 Forecast Future Oil Production")
    col1, col2 = st.columns(2)

    with col1:
        short_term_steps = st.number_input(
            "Short-Term Forecast (months)",
            min_value=1,
            max_value=60,
            value=12,
            step=1
        )
        if st.button("🚀 Forecast Short-Term"):
            forecast_steps = short_term_steps
            forecast_label = "Short-Term Forecast"
    with col2:
        long_term_steps = st.number_input(
            "Long-Term Forecast (months)",
            min_value=1,
            max_value=120,
            value=24,
            step=1
        )
        if st.button("🚀 Forecast Long-Term"):
            forecast_steps = long_term_steps
            forecast_label = "Long-Term Forecast"

    # Only proceed if forecast_steps is defined
    if 'forecast_steps' in locals():
        # Forecast future
        future_forecast = model.forecast(steps=forecast_steps)
        future_forecast = np.array(future_forecast)

        # Generate future dates
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
        future_dates = future_dates.strftime('%Y-%m-%d')

        # Prepare forecast DataFrame
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecasted_Oil_Production": np.round(future_forecast, 2)
        })

        # 📈 Interactive Plotly Plot
        st.subheader("📈 Prediction Results")

        # Filter historical data to only show from 2000 onwards
        mask_2000 = historical_dates >= pd.to_datetime('2000-01-01')
        filtered_dates = historical_dates[mask_2000]
        filtered_values = values.flatten()[mask_2000]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_values, mode='lines', name='Actual Production (Historical)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=future_forecast, mode='lines', name=f'{forecast_label}', line=dict(color='red', dash='dash')))

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
