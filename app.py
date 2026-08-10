import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io

# Import data processing and machine learning logic from model.py
from model import (
    download_data,
    load_and_clean_data,
    aggregate_daily_sales,
    engineer_features,
    train_and_evaluate_models,
    forecast_future,
    LOCAL_PATH
)

# Set page config to wide mode with custom title and icon
st.set_page_config(
    page_title="Sales Forecasting & Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism design, clean typography, vibrant gradients)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* General styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title styling */
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .dashboard-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* KPI Card styling */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
        width: 100%;
    }
    .kpi-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: transform 0.3s ease-in-out, border-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }
    .kpi-subtitle {
        font-size: 0.75rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to format metric values
def format_metric(value, prefix="", suffix=""):
    if value >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.2f}M{suffix}"
    elif value >= 1_000:
        return f"{prefix}{value / 1_000:.1f}K{suffix}"
    else:
        return f"{prefix}{value:.2f}{suffix}"

# Cache data loading so it doesn't reload on every run
@st.cache_data
def get_cleaned_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            # Read from uploaded file buffer
            return load_and_clean_data(uploaded_file=uploaded_file)
        except Exception as e:
            st.error(f"Error loading custom file: {e}")
            return None
    else:
        # Download and load default Superstore dataset
        if not os.path.exists(LOCAL_PATH):
            download_data()
        return load_and_clean_data(LOCAL_PATH)

# Sidebar layout
st.sidebar.markdown("## 📊 SalesForecaster")
st.sidebar.markdown("Configuration and settings for data and forecasting models.")

# 1. Data Selection
st.sidebar.subheader("1. Data Source")
use_custom = st.sidebar.checkbox("Upload Custom CSV Dataset", value=False)
uploaded_file = None
if use_custom:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Sales CSV",
        type=["csv"],
        help="Upload a CSV with at least Order Date, Sales, Quantity, Product, and Category columns."
    )

# Load current dataset
with st.spinner("Processing sales dataset..."):
    df_raw = get_cleaned_data(uploaded_file if use_custom else None)

if df_raw is None:
    st.warning("Please upload a valid sales dataset or uncheck the custom upload option to use the default dataset.")
    st.stop()

# Aggregate and Engineer Features
df_daily = aggregate_daily_sales(df_raw)
df_features = engineer_features(df_daily)

# (Forecasting parameters are now located directly in the Future Forecast section below)

# Header Section
st.markdown('<div class="dashboard-title">Sales Forecasting & Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Time-series forecasting and sales intelligence on historical retail transactions</div>', unsafe_allow_html=True)

# Compute Main Metrics for KPIs
total_sales = df_raw['Sales'].sum()
total_qty = df_raw['Quantity'].sum()
avg_daily_sales = df_daily['Sales'].mean()

# Render custom KPI Cards using streamlit columns and CSS
kpi_cols = st.columns(3)
with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Total Revenue</div>
        <div class="kpi-value">{format_metric(total_sales, prefix="$")}</div>
        <div class="kpi-subtitle">Sum of all transaction revenue</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Products Sold</div>
        <div class="kpi-value">{format_metric(total_qty)}</div>
        <div class="kpi-subtitle">Total quantity of products sold</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Avg Daily Sales</div>
        <div class="kpi-value">{format_metric(avg_daily_sales, prefix="$")}</div>
        <div class="kpi-subtitle">Average daily sales revenue</div>
    </div>
    """, unsafe_allow_html=True)

# Main Dashboard Navigation Sections (Sequential Layout)
st.write("---")

# ----------------- SECTION 1: EDA -----------------
st.header("📈 Sales Trends & EDA")
st.subheader("Historical Sales Performance")

# 1. Historical Sales Trend
st.markdown("#### Daily Revenue Trend")
# Add simple rolling average line for better visual interpretation of trend
df_trend_plot = df_daily.copy()
df_trend_plot['7-Day Moving Avg'] = df_trend_plot['Sales'].rolling(7).mean()

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=df_trend_plot['Date'],
    y=df_trend_plot['Sales'],
    name='Daily Sales',
    line=dict(color='rgba(99, 102, 241, 0.4)', width=1)
))
fig_trend.add_trace(go.Scatter(
    x=df_trend_plot['Date'],
    y=df_trend_plot['7-Day Moving Avg'],
    name='7-Day Moving Avg',
    line=dict(color='#818CF8', width=2)
))
fig_trend.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Date"),
    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Sales ($)"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_trend, use_container_width=True)

# Grid for category and product breakdowns
col_break_1, col_break_2 = st.columns(2)

with col_break_1:
    st.markdown("#### Category Sales Breakdown")
    df_cat = df_raw.groupby('Category')['Sales'].sum().reset_index()
    fig_cat = px.pie(
        df_cat,
        values='Sales',
        names='Category',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_cat.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", y=-0.1)
    )
    st.plotly_chart(fig_cat, use_container_width=True)
    
with col_break_2:
    st.markdown("#### Top 10 Products by Sales")
    df_prod = df_raw.groupby('Product')['Sales'].sum().reset_index()
    df_prod = df_prod.sort_values('Sales', ascending=True).tail(10)
    fig_prod = px.bar(
        df_prod,
        x='Sales',
        y='Product',
        orientation='h',
        color_discrete_sequence=['#A855F7']
    )
    fig_prod.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)'),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_prod, use_container_width=True)
    
# Grid for regional and seasonality breakdowns
col_break_3, col_break_4 = st.columns(2)

with col_break_3:
    st.markdown("#### Regional Sales Performance")
    df_region = df_raw.groupby('Store/Region')['Sales'].sum().reset_index()
    fig_region = px.bar(
        df_region,
        x='Store/Region',
        y='Sales',
        color='Sales',
        color_continuous_scale='Purples'
    )
    fig_region.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Region"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Sales ($)")
    )
    st.plotly_chart(fig_region, use_container_width=True)
    
with col_break_4:
    st.markdown("#### Seasonality: Monthly Average Sales")
    df_monthly_avg = df_daily.copy()
    df_monthly_avg['Month'] = df_monthly_avg['Date'].dt.strftime('%B')
    df_monthly_avg['MonthNum'] = df_monthly_avg['Date'].dt.month
    df_monthly_avg = df_monthly_avg.groupby(['MonthNum', 'Month'])['Sales'].mean().reset_index()
    df_monthly_avg = df_monthly_avg.sort_values('MonthNum')
    
    fig_month = px.line(
        df_monthly_avg,
        x='Month',
        y='Sales',
        markers=True,
        line_shape='spline',
        color_discrete_sequence=['#EC4899']
    )
    fig_month.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Month"),
        yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Average Sales ($)")
    )
    st.plotly_chart(fig_month, use_container_width=True)

# ----------------- SECTION 2: MODELS -----------------
st.write("---")
st.header("⚔️ Model Evaluation and Comparison")
st.markdown("""
    We compare three models using chronological train-test splits (80% training data, 20% test set).
    Evaluating time-series forecasting models requires metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
""")

# Train and evaluate models
with st.spinner("Training forecasting models..."):
    try:
        models, metrics_df, predictions_df = train_and_evaluate_models(df_features)
        
        # Display evaluation table
        st.dataframe(
            metrics_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='rgba(99, 102, 241, 0.2)'),
            use_container_width=True
        )
        
        # Determine best model
        best_model_idx = metrics_df['MAE'].idxmin()
        best_model_name = metrics_df.loc[best_model_idx, 'Model']
        st.success(f"🏆 **Best Performing Model**: **{best_model_name}** has the lowest MAE on the test dataset.")
        
        # Show test predictions chart
        st.markdown("#### Test Set Forecast vs Actual Sales")
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=predictions_df['Date'],
            y=predictions_df['Actual'],
            name='Actual Sales',
            line=dict(color='rgba(255, 255, 255, 0.6)', width=1.5)
        ))
        for model_name in ['Linear Regression', 'Random Forest', 'XGBoost']:
            fig_eval.add_trace(go.Scatter(
                x=predictions_df['Date'],
                y=predictions_df[model_name],
                name=model_name,
                line=dict(width=1.5, dash='dash' if model_name != best_model_name else 'solid')
            ))
        fig_eval.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Test Period Dates"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Sales ($)"),
            hovermode="x unified"
        )
        st.plotly_chart(fig_eval, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error training models: {e}")

# ----------------- SECTION 3: FUTURE FORECAST -----------------
st.write("---")
st.header("🔮 Future Demand Forecasting")
st.markdown("""
    Select your forecast horizon and model to generate future sales predictions.
    This uses a **recursive forecasting pipeline** that recomputes lag and rolling averages day-by-day.
""")

# Input controls inside the section
col_input_1, col_input_2 = st.columns(2)
with col_input_1:
    horizon = st.selectbox("Forecast Horizon (Days)", options=[7, 30, 90], index=1, key="fc_horizon")
with col_input_2:
    selected_model_name = st.selectbox(
        "Forecasting Model",
        options=["XGBoost", "Random Forest", "Linear Regression"],
        index=0,
        key="fc_model"
    )

if st.button("🔮 Run Demand Forecast", type="primary", key="fc_btn"):
    with st.spinner("Generating recursive future predictions..."):
        try:
            # 1. Fit models on full features
            models, metrics_df, predictions_df = train_and_evaluate_models(df_features)
            
            # 2. Run recursive forecasting
            forecast_df = forecast_future(
                models=models,
                historical_daily_df=df_daily,
                horizon=horizon,
                selected_model_name=selected_model_name
            )
            
            # 3. Create historical + predicted plot (last 120 days history for context)
            df_hist_tail = df_daily.tail(120).copy()
            
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=df_hist_tail['Date'],
                y=df_hist_tail['Sales'],
                name='Historical Sales',
                line=dict(color='#6366F1', width=2)
            ))
            fig_fc.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Predicted Sales'],
                name=f'Predicted ({selected_model_name})',
                line=dict(color='#EC4899', width=2.5, dash='dash')
            ))
            fig_fc.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Timeline"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.05)', title="Sales ($)"),
                hovermode="x unified"
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            
            # Display and export predictions table
            col_tab_1, col_tab_2 = st.columns([2, 1])
            with col_tab_1:
                st.markdown("#### Future Predictions Table")
                st.dataframe(forecast_df, use_container_width=True)
            with col_tab_2:
                st.markdown("#### Export Forecast")
                csv_buffer = io.StringIO()
                forecast_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"sales_forecast_{horizon}days.csv",
                    mime="text/csv",
                    key="fc_dl_btn"
                )
                
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
else:
    st.info("Click the **Run Demand Forecast** button above to generate and visualize future sales predictions.")
