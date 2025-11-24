"""
Streamlit Deployment App - UK Electricity Demand Forecasting

Worked on by: [Your Name]

What this app does:
- Loads best trained model from comparison
- Provides user interface for forecasting
- Visualizes historical data and predictions
- Shows model performance metrics
- Allows custom forecast periods

This follows CloudAI patterns for Streamlit deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="UK Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Title
st.title("⚡ UK Electricity Demand Forecasting")
st.markdown("**Predict future electricity demand using AI models trained on 2009-2024 data**")

# Sidebar
st.sidebar.header("Configuration")

# Model selection - Updated for complete training models
available_models = {
    'XGBoost (BEST - 3% MAPE)': {
        'model': '../Data/xgboost_model.pkl',
        'features': '../Data/xgboost_features.pkl',
        'type': 'xgboost'
    },
    'Ensemble (4% MAPE)': {
        'model': '../Data/ensemble_weights.pkl',
        'prophet': '../Data/prophet_seasonal_model.pkl',
        'xgboost': '../Data/xgboost_model.pkl',
        'lstm': '../Data/lstm_model.h5',
        'features': '../Data/xgboost_features.pkl',
        'scaler': '../Data/lstm_scaler.pkl',
        'type': 'ensemble'
    },
    'LSTM Neural Network (7% MAPE)': {
        'model': '../Data/lstm_model.h5',
        'scaler': '../Data/lstm_scaler.pkl',
        'type': 'lstm'
    },
    'Prophet Seasonal (18% MAPE)': {
        'model': '../Data/prophet_seasonal_model.pkl',
        'type': 'prophet'
    }
}

# Check which models exist
existing_models = {}
for name, paths in available_models.items():
    if paths['type'] == 'ensemble':
        # Check all ensemble dependencies
        if all(os.path.exists(paths[key]) for key in ['model', 'prophet', 'xgboost', 'lstm', 'features', 'scaler']):
            existing_models[name] = paths
    elif paths['type'] == 'xgboost':
        if os.path.exists(paths['model']) and os.path.exists(paths['features']):
            existing_models[name] = paths
    elif paths['type'] == 'lstm':
        if os.path.exists(paths['model']) and os.path.exists(paths['scaler']):
            existing_models[name] = paths
    else:  # prophet
        if os.path.exists(paths['model']):
            existing_models[name] = paths

if not existing_models:
    st.error("❌ No trained models found! Please run 07_complete_model_training.ipynb first.")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(existing_models.keys()),
    help="Choose from 4 trained models. XGBoost has best accuracy (3% MAPE)."
)

# Forecast parameters
forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)",
    min_value=7,
    max_value=365,
    value=30,
    step=7
)

start_date = st.sidebar.date_input(
    "Start Date for Forecast",
    value=datetime(2025, 1, 1)
)

# Load data
@st.cache_data
def load_data():
    """Load historical data - period level (half-hourly)"""
    df = pd.read_csv('../Data/cleaned_and_augmented_electricity_data.csv', low_memory=False)
    df['settlement_date'] = pd.to_datetime(df['settlement_date'], errors='coerce')
    df = df.dropna(subset=['settlement_date'])
    df = df.sort_values('settlement_date')
    # Keep period-level data (half-hourly) - this is what models were trained on
    df_periods = df[['settlement_date', 'england_wales_demand']].copy()
    df_periods.rename(columns={'settlement_date': 'ds', 'england_wales_demand': 'y'}, inplace=True)
    return df_periods.dropna()

# Load models
@st.cache_resource
def load_model_components(model_info):
    """Load trained model and its dependencies"""
    components = {'type': model_info['type']}
    
    if model_info['type'] == 'prophet':
        with open(model_info['model'], 'rb') as f:
            components['model'] = pickle.load(f)
    
    elif model_info['type'] == 'xgboost':
        with open(model_info['model'], 'rb') as f:
            components['model'] = pickle.load(f)
        with open(model_info['features'], 'rb') as f:
            components['features'] = pickle.load(f)
    
    elif model_info['type'] == 'lstm':
        # Import TensorFlow only when needed
        try:
            from tensorflow.keras.models import load_model
            components['model'] = load_model(model_info['model'])
            with open(model_info['scaler'], 'rb') as f:
                components['scaler'] = pickle.load(f)
            components['lookback'] = 48  # 24 hours of half-hourly data
        except ImportError:
            st.error("TensorFlow required for LSTM model. Install: pip install tensorflow")
            return None
    
    elif model_info['type'] == 'ensemble':
        with open(model_info['model'], 'rb') as f:
            components['weights'] = pickle.load(f)
        with open(model_info['prophet'], 'rb') as f:
            components['prophet_model'] = pickle.load(f)
        with open(model_info['xgboost'], 'rb') as f:
            components['xgb_model'] = pickle.load(f)
        with open(model_info['features'], 'rb') as f:
            components['features'] = pickle.load(f)
        try:
            from tensorflow.keras.models import load_model
            components['lstm_model'] = load_model(model_info['lstm'])
            with open(model_info['scaler'], 'rb') as f:
                components['scaler'] = pickle.load(f)
            components['lookback'] = 48
        except ImportError:
            st.warning("TensorFlow not available. Ensemble will use Prophet + XGBoost only.")
            components['lstm_model'] = None
    
    return components

# Load metrics from comparison file
@st.cache_data
def load_metrics():
    """Load model performance metrics"""
    metrics_file = '../Data/complete_model_comparison.csv'
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    return None

# Helper function to prepare features for XGBoost
def prepare_features_for_forecast(df, feature_list, start_date, periods):
    """Prepare features for XGBoost forecasting"""
    # Create future dates
    future_dates = pd.date_range(start=start_date, periods=periods, freq='30min')
    
    # Create empty dataframe with required features
    future_df = pd.DataFrame({'settlement_date': future_dates})
    
    # Time features
    future_df['hour'] = future_df['settlement_date'].dt.hour
    future_df['day_of_week'] = future_df['settlement_date'].dt.dayofweek
    future_df['month'] = future_df['settlement_date'].dt.month
    future_df['quarter'] = future_df['settlement_date'].dt.quarter
    future_df['day_of_year'] = future_df['settlement_date'].dt.dayofyear
    future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
    future_df['is_morning'] = ((future_df['hour'] >= 6) & (future_df['hour'] < 12)).astype(int)
    future_df['is_afternoon'] = ((future_df['hour'] >= 12) & (future_df['hour'] < 18)).astype(int)
    future_df['is_evening'] = ((future_df['hour'] >= 18) & (future_df['hour'] < 22)).astype(int)
    future_df['is_night'] = ((future_df['hour'] >= 22) | (future_df['hour'] < 6)).astype(int)
    future_df['period_of_day'] = ((future_df['settlement_date'].dt.hour * 2) + 
                                   (future_df['settlement_date'].dt.minute // 30) + 1)
    
    # Lag and rolling features - use last known values from historical data
    last_demand = df['y'].iloc[-1]
    last_48_demand = df['y'].iloc[-48] if len(df) >= 48 else last_demand
    
    future_df['lag_1'] = last_demand
    future_df['lag_2'] = last_demand
    future_df['lag_48'] = last_48_demand
    future_df['lag_96'] = last_48_demand
    future_df['lag_336'] = last_demand
    future_df['rolling_mean_24h'] = df['y'].tail(48).mean()
    future_df['rolling_std_24h'] = df['y'].tail(48).std()
    future_df['rolling_min_24h'] = df['y'].tail(48).min()
    future_df['rolling_max_24h'] = df['y'].tail(48).max()
    future_df['rolling_mean_7d'] = df['y'].tail(336).mean()
    future_df['rolling_std_7d'] = df['y'].tail(336).std()
    future_df['ema_24h'] = df['y'].ewm(span=48).mean().iloc[-1]
    future_df['ema_7d'] = df['y'].ewm(span=336).mean().iloc[-1]
    
    # Return only required features in correct order
    return future_df[feature_list], future_dates

def generate_forecast(components, df_historical, start_date, forecast_days):
    """Generate forecast based on model type"""
    forecast_periods = forecast_days * 48  # Half-hourly periods
    
    if components['type'] == 'prophet':
        # Prophet forecast
        future_df = pd.DataFrame({
            'ds': pd.date_range(start=start_date, periods=forecast_periods, freq='30min')
        })
        forecast = components['model'].predict(future_df)
        return future_df['ds'].values, forecast['yhat'].values
    
    elif components['type'] == 'xgboost':
        # XGBoost forecast
        X_future, future_dates = prepare_features_for_forecast(
            df_historical, 
            components['features'], 
            start_date, 
            forecast_periods
        )
        predictions = components['model'].predict(X_future)
        return future_dates, predictions
    
    elif components['type'] == 'lstm':
        # LSTM forecast
        lookback = components['lookback']
        scaler = components['scaler']
        model = components['model']
        
        # Scale historical data
        historical_values = df_historical['y'].values
        scaled_data = scaler.transform(historical_values.reshape(-1, 1))
        
        # Generate forecast iteratively
        predictions_scaled = []
        current_sequence = scaled_data[-lookback:].flatten()
        
        for _ in range(forecast_periods):
            # Prepare input
            X_input = current_sequence.reshape(1, lookback, 1)
            
            # Predict next value
            pred_scaled = model.predict(X_input, verbose=0)[0, 0]
            predictions_scaled.append(pred_scaled)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(start=start_date, periods=forecast_periods, freq='30min')
        return future_dates, predictions
    
    elif components['type'] == 'ensemble':
        # Ensemble forecast - combine all models
        weights = components['weights']
        
        # Get Prophet prediction
        prophet_dates, prophet_pred = generate_forecast(
            {'type': 'prophet', 'model': components['prophet_model']},
            df_historical, start_date, forecast_days
        )
        
        # Get XGBoost prediction
        xgb_dates, xgb_pred = generate_forecast(
            {'type': 'xgboost', 'model': components['xgb_model'], 'features': components['features']},
            df_historical, start_date, forecast_days
        )
        
        # Get LSTM prediction if available
        if components['lstm_model'] is not None:
            lstm_dates, lstm_pred = generate_forecast(
                {'type': 'lstm', 'model': components['lstm_model'], 
                 'scaler': components['scaler'], 'lookback': components['lookback']},
                df_historical, start_date, forecast_days
            )
            
            # Weighted average
            ensemble_pred = (
                weights['prophet'] * prophet_pred +
                weights['xgboost'] * xgb_pred +
                weights['lstm'] * lstm_pred
            )
        else:
            # Just Prophet + XGBoost
            total_weight = weights['prophet'] + weights['xgboost']
            ensemble_pred = (
                (weights['prophet'] / total_weight) * prophet_pred +
                (weights['xgboost'] / total_weight) * xgb_pred
            )
        
        return prophet_dates, ensemble_pred
    
    return None, None

# Main content
try:
    # Load data
    with st.spinner("Loading historical data..."):
        df_historical = load_data()
    
    st.success(f"✅ Loaded {len(df_historical):,} half-hourly records ({df_historical['ds'].min()} to {df_historical['ds'].max()})")
    
    # Load model components
    with st.spinner(f"Loading {selected_model_name}..."):
        model_info = existing_models[selected_model_name]
        components = load_model_components(model_info)
    
    if components is None:
        st.error("Failed to load model components")
        st.stop()
    
    st.success(f"✅ Loaded {selected_model_name}")
    
    # Display metrics
    metrics_df = load_metrics()
    if metrics_df is not None:
        st.subheader("📊 All Models Performance Comparison")
        
        # Show metrics table
        st.dataframe(
            metrics_df.style.highlight_min(subset=['MAPE'], color='lightgreen'),
            use_container_width=True
        )
        
        # Show selected model metrics
        model_name_map = {
            'XGBoost (BEST - 3% MAPE)': 'XGBoost',
            'Ensemble (4% MAPE)': 'Ensemble',
            'LSTM Neural Network (7% MAPE)': 'LSTM',
            'Prophet Seasonal (18% MAPE)': 'Prophet (Seasonal)'
        }
        
        selected_metrics = metrics_df[metrics_df['Model'] == model_name_map.get(selected_model_name, selected_model_name)]
        
        if not selected_metrics.empty:
            st.subheader(f"📈 {selected_model_name} Performance")
            cols = st.columns(5)
            row = selected_metrics.iloc[0]
            cols[0].metric("MAE", f"{row['MAE']:,.0f} MW")
            cols[1].metric("RMSE", f"{row['RMSE']:,.0f} MW")
            cols[2].metric("MAPE", f"{row['MAPE']:.2f}%")
            cols[3].metric("R²", f"{row['R²']:.4f}")
            if row['Training Time (s)'] > 0:
                cols[4].metric("Training Time", f"{row['Training Time (s)']:.1f}s")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Historical Data", "ℹ️ About"])
    
    with tab1:
        st.subheader("🔮 Forecast Future Demand")
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                try:
                    # Generate forecast
                    future_dates, forecast_values = generate_forecast(
                        components,
                        df_historical,
                        start_date,
                        forecast_days
                    )
                    
                    if forecast_values is None:
                        st.error("Failed to generate forecast")
                    else:
                        # Create forecast dataframe
                        forecast_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted Demand (MW)': forecast_values
                        })
                        
                        # Aggregate to daily summary
                        forecast_df['Date_Only'] = pd.to_datetime(forecast_df['Date']).dt.date
                        daily_forecast = forecast_df.groupby('Date_Only').agg({
                            'Predicted Demand (MW)': ['mean', 'max', 'min']
                        }).reset_index()
                        daily_forecast.columns = ['Date', 'Avg Demand (MW)', 'Peak Demand (MW)', 'Min Demand (MW)']
                        
                        # Display daily summary
                        st.subheader("📅 Daily Forecast Summary")
                        st.dataframe(
                            daily_forecast.style.format({
                                'Avg Demand (MW)': '{:,.0f}',
                                'Peak Demand (MW)': '{:,.0f}',
                                'Min Demand (MW)': '{:,.0f}'
                            }),
                            use_container_width=True,
                            height=300
                        )
                        
                        # Plot forecast
                        st.subheader("📈 Forecast Visualization")
                        
                        fig, ax = plt.subplots(figsize=(14, 7))
                        
                        # Plot historical (last 7 days, daily average)
                        recent_historical = df_historical.tail(48 * 7).copy()
                        recent_historical['date'] = pd.to_datetime(recent_historical['ds']).dt.date
                        historical_daily = recent_historical.groupby('date')['y'].mean().reset_index()
                        
                        ax.plot(historical_daily['date'], historical_daily['y'],
                               label='Historical (7-day avg)', color='#2c3e50', linewidth=2.5, marker='o', markersize=4)
                        
                        # Plot forecast (daily average)
                        forecast_daily = forecast_df.groupby('Date_Only')['Predicted Demand (MW)'].mean().reset_index()
                        ax.plot(forecast_daily['Date_Only'], forecast_daily['Predicted Demand (MW)'],
                               label=f'Forecast ({selected_model_name})', 
                               color='#e74c3c', linewidth=2.5, linestyle='--', marker='s', markersize=4)
                        
                        # Add confidence band for forecast
                        forecast_std = forecast_df.groupby('Date_Only')['Predicted Demand (MW)'].std().values
                        forecast_mean = forecast_daily['Predicted Demand (MW)'].values
                        ax.fill_between(
                            forecast_daily['Date_Only'],
                            forecast_mean - forecast_std,
                            forecast_mean + forecast_std,
                            alpha=0.2, color='#e74c3c', label='±1 Std Dev'
                        )
                        
                        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
                        ax.set_ylabel('Demand (MW)', fontsize=13, fontweight='bold')
                        ax.set_title(f'{selected_model_name} - {forecast_days} Day Forecast',
                                    fontsize=15, fontweight='bold')
                        ax.legend(fontsize=11, loc='best')
                        ax.grid(True, alpha=0.3, linestyle='--')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Summary statistics
                        st.subheader("📊 Forecast Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Average Demand", f"{np.mean(forecast_values):,.0f} MW")
                        col2.metric("Peak Demand", f"{np.max(forecast_values):,.0f} MW")
                        col3.metric("Min Demand", f"{np.min(forecast_values):,.0f} MW")
                        col4.metric("Total Periods", f"{len(forecast_values):,}")
                        
                        # Download buttons
                        st.subheader("💾 Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Half-hourly forecast
                            csv_halfhourly = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Half-Hourly Forecast",
                                data=csv_halfhourly,
                                file_name=f"electricity_forecast_halfhourly_{start_date}_{forecast_days}d.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Daily summary
                            csv_daily = daily_forecast.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Daily Summary",
                                data=csv_daily,
                                file_name=f"electricity_forecast_daily_{start_date}_{forecast_days}d.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
                    import traceback
                    with st.expander("Show Error Details"):
                        st.code(traceback.format_exc())
    
    with tab2:
        st.subheader("Historical Electricity Demand (Half-Hourly Periods)")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            hist_start = st.date_input("Start Date", 
                                       value=df_historical['ds'].max() - timedelta(days=90))
        with col2:
            hist_end = st.date_input("End Date", 
                                     value=df_historical['ds'].max())
        
        # Filter data
        mask = (df_historical['ds'] >= pd.Timestamp(hist_start)) & \
               (df_historical['ds'] <= pd.Timestamp(hist_end))
        filtered_data = df_historical[mask]
        
        # Aggregate to daily for cleaner visualization
        daily_data = filtered_data.copy()
        daily_data['date'] = daily_data['ds'].dt.date
        daily_stats = daily_data.groupby('date')['y'].agg(['mean', 'max', 'min']).reset_index()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_stats['date'], daily_stats['mean'], color='blue', linewidth=2, label='Daily Average')
        ax.fill_between(daily_stats['date'], daily_stats['min'], daily_stats['max'], 
                        alpha=0.3, color='blue', label='Min-Max Range')
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand (MW per period)')
        ax.set_title(f'Historical Demand: {hist_start} to {hist_end}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistics
        st.subheader("Statistics (Half-Hourly Periods)")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{filtered_data['y'].mean():,.0f} MW")
        col2.metric("Std Dev", f"{filtered_data['y'].std():,.0f} MW")
        col3.metric("Max", f"{filtered_data['y'].max():,.0f} MW")
        col4.metric("Min", f"{filtered_data['y'].min():,.0f} MW")
    
    with tab3:
        st.subheader("ℹ️ About This App")
        
        st.markdown("""
        ### 🎯 Purpose
        This application forecasts UK electricity demand using 4 advanced models 
        trained on historical data from 2009-2024.
        
        ### 🏆 Models Available
        
        **1. XGBoost (BEST - 3% MAPE)** ⭐
        - Gradient boosting with 24 engineered features
        - Fastest training (5.6s) with best accuracy
        - Uses lag features, rolling statistics, time patterns
        
        **2. Ensemble (4% MAPE)**
        - Weighted average of Prophet + XGBoost + LSTM
        - Weights based on inverse MAPE performance
        - Most robust predictions combining all model strengths
        
        **3. LSTM Neural Network (7% MAPE)**
        - Bidirectional LSTM with 48-period lookback
        - Captures complex temporal dependencies
        - CPU-optimized class-based architecture
        
        **4. Prophet Seasonal (18% MAPE)**
        - Facebook's Prophet with full seasonality
        - Daily, weekly, yearly, and intraday patterns
        - Interpretable trend + seasonality decomposition
        
        ### 📊 Data Details
        - **Source**: NESO (National Energy System Operator)
        - **Period**: 2009-2024 (16 years)
        - **Frequency**: Half-hourly data (48 periods/day)
        - **Target**: England & Wales Demand (MW)
        - **Records**: 262,000+ half-hourly periods
        
        ### 🔧 Technical Features
        - Built with Streamlit + TensorFlow + XGBoost
        - Half-hourly forecasting (not daily aggregates)
        - Real-time demand patterns (peaks, troughs, seasonality)
        - Multiple model comparison and selection
        - Download forecasts as CSV
        
        ### 📈 Feature Engineering
        - **Temporal**: hour, day, month, quarter, day_of_year
        - **Cyclical**: sin/cos encoding for time features
        - **Lag Features**: 1 period, 1 hour, 1 day, 2 days, 1 week
        - **Rolling Stats**: 24h and 7-day mean/std/min/max
        - **Binary**: weekend, morning, afternoon, evening, night
        
        ### 🎓 Academic Context
        - **Course**: CloudAI Chapter 6 - Time Series Forecasting
        - **Dataset**: UK Historic Electricity Demand (Dataset 2)
        - **Training Notebook**: `07_complete_model_training.ipynb`
        - **Deadline**: November 24, 2025
        
        ### 📊 Model Comparison Results
        | Model | MAPE | MAE | R² | Training Time |
        |-------|------|-----|----|--------------| 
        | XGBoost | 3.00% | 751 MW | 0.9411 | 5.6s |
        | Ensemble | 4.71% | 1,129 MW | 0.8967 | - |
        | LSTM | 7.23% | 1,710 MW | 0.6963 | 926s |
        | Prophet | 17.77% | 4,072 MW | -0.2304 | 232s |
        
        ### 🚀 How to Use
        1. Select a model from the sidebar
        2. Choose forecast horizon (7-365 days)
        3. Set start date for forecast
        4. Click "Generate Forecast"
        5. View results and download CSV
        
        ### 💡 Recommendations
        - Use **XGBoost** for production forecasting (best accuracy)
        - Use **Ensemble** for critical applications (most robust)
        - Use **LSTM** for research on deep learning approaches
        - Use **Prophet** for interpretable trend analysis
        """)
        
        st.markdown("---")
        st.markdown("### 🛠️ Technical Stack")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Frontend**")
            st.markdown("- Streamlit")
            st.markdown("- Matplotlib")
            st.markdown("- Seaborn")
        with col2:
            st.markdown("**Models**")
            st.markdown("- XGBoost")
            st.markdown("- TensorFlow/Keras")
            st.markdown("- Prophet")
        with col3:
            st.markdown("**Data Science**")
            st.markdown("- Pandas")
            st.markdown("- NumPy")
            st.markdown("- Scikit-learn")

except Exception as e:
    st.error(f"❌ Application Error: {str(e)}")
    st.info("💡 Please ensure all notebooks have been run and models are saved correctly")

# Footer
st.markdown("---")
st.markdown("**UK Electricity Demand Forecasting** | Built with ❤️ using Streamlit | CloudAI Chapter 6 Project")
