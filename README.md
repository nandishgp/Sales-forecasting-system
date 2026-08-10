# Sales Forecasting & Analytics Dashboard

An interactive Sales Forecasting and Analytics Dashboard that analyzes historical sales and predicts future demand. This is a 3rd-year Data Science project utilizing Python, Streamlit, and Machine Learning algorithms (Linear Regression, Random Forest, and XGBoost) to perform time-series analysis and demand planning.

## 📊 Project Overview
This application provides businesses with a central portal to view historical transaction analytics and generate forecast demand horizons (7, 30, or 90 days). It supports:
- **Historical Sales Analysis (EDA)**: Interactive breakdown of revenue by Category, Region, Top 10 Products, and Monthly Seasonality.
- **Model Comparison**: Evaluation of three separate time-series regression models (Linear Regression, Random Forest, and XGBoost) utilizing chronological train-test validation.
- **Recursive Demand Forecasting**: Dynamic, step-by-step prediction of future sales, recomputing rolling averages and lags programmatically.
- **Custom Data Upload**: Option for users to drag-and-drop their own transactional CSV datasets, which are automatically cleaned, mapped, and parsed for forecasting.

---

## 💾 Dataset
The system defaults to downloading and loading the standard **Kaggle Sample Superstore Sales** dataset:
- **Source**: [Sample Superstore Sales on GitHub](https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv)
- **Original Attributes**: Order Date, Product Name, Category, Quantity, Sales (Revenue), Region, etc.
- **Derived Fields**:
  - `Price`: Computed as `Sales / Quantity` (revenue per unit).
  - Date attributes: Day, Week, Month, Quarter, Year, Day-of-week.
  - Lag features: Previous day's sales, Previous week's sales.
  - Rolling features: 7-day moving average, 30-day moving average.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/Sales-Forecasting.git
   cd Sales-Forecasting
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install required packages using:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### 1. Standalone Pipeline Run
To download the dataset, perform cleaning, engineering, and print test-set model evaluations, run `model.py` directly:
```bash
python model.py
```

### 2. Launching the Interactive Dashboard
To launch the Streamlit app locally:
```bash
streamlit run app.py
```
This will start the development server and automatically open the application in your browser (usually at `http://localhost:8501`).

### 3. Running the Jupyter Notebook
To explore the step-by-step EDA and cleaning logic:
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 🤖 Models & Key Findings

### Evaluation Metrics
We validate the models using a **chronological 80-20 train-test split** to mimic real-world forecasting. Performance is measured using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

### Model Comparison Summary
- **Linear Regression**: Best for capturing long-term growth and linear trend lines. However, it struggles with non-linear seasonal cycles.
- **Random Forest**: Good at capturing localized seasonal spikes and non-linear patterns, but does not extrapolate trend curves as well.
- **XGBoost**: Typically performs the best on the test set, displaying the lowest MAE and MAPE. It successfully balances short-term lag seasonality with the overall long-term trend.

---

## 📷 Screenshots
*Add your dashboard screenshots here!*

- **Dashboard Overview**:
  ![Overview](screenshots/dashboard_overview.png)
- **Forecasting Tab**:
  ![Forecasting](screenshots/dashboard_forecasting.png)
- **Model Evaluation**:
  ![Model Comparison](screenshots/dashboard_models.png)
