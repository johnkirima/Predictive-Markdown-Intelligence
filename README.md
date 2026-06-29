🛒 Predictive Markdown Intelligence
End-to-End Machine Learning for Retail Demand Forecasting & Markdown Optimization
Predicting SKU-level daily demand using synthetic retail data, feature engineering, and gradient boosted tree models to support smarter pricing and inventory decisions.






📖 Overview
Retail markdown decisions are difficult because lowering prices increases demand while reducing margins.

This project simulates a realistic fashion retail environment and develops an end-to-end machine learning pipeline that predicts daily SKU demand under different pricing conditions.

The final objective is to support better:

📦 Inventory planning
💰 Markdown timing
📈 Demand forecasting
🛍 Retail pricing decisions
🚀 Project Highlights
✅ Synthetic retail data generation

✅ Leakage-free feature engineering

✅ Multiple baseline models

✅ XGBoost Tweedie optimization

✅ Error analysis

✅ Deployment-ready prediction pipeline

✅ Professional GitHub documentation

🏗 Repository Structure
Predictive-Markdown-Intelligence/

│
├── data/
│
├── models/
│
├── results/
│
├── 01_baseline_models.ipynb
├── 02_Linear_Regression.ipynb
├── 03_Random_Forest.ipynb
├── 04_XGBoost.ipynb
├── 05_Predictions and Error Analysis.ipynb
├── 06_EXECUTIVE_SUMMARY.ipynb
├── 07_Deployment_Prep.ipynb
│
├── build_modeling_table.py
├── config.py
├── generate_calendar.py
├── generate_daily_sales.py
├── generate_price_schedule.py
├── generate_sku_master.py
├── visualize_data.py
│
└── README.md
📊 Dataset
The synthetic retail dataset contains approximately

260,000+ SKU-Day observations
180 Fashion SKUs
2 Years of Daily Sales
Multiple Product Categories
The data was intentionally designed to reproduce real retail characteristics including

Zero-inflated demand
Long-tail sales
Price elasticity
Markdown fatigue
Seasonality
Inventory constraints
Product life cycle
⚙ Feature Engineering
Important features include

Rolling 7-day demand
Lag features
Discount percentage
Markdown stage
Inventory ratio
Days until season end
Days since launch
Day of week
Holiday indicators
Future information leakage was removed before model training.

🤖 Models
The project compares several machine learning models.

Model	Purpose
Linear Regression	Baseline
Random Forest	Non-linear benchmark
XGBoost	Gradient boosting
XGBoost Poisson	Count prediction
✅ XGBoost Tweedie	Final production model
📈 Final Model
The selected production model is

✅ XGBoost Tweedie
Chosen because it naturally models

Sparse demand
Long-tail purchases
Zero inflation
Stable expected demand
📉 Evaluation
Evaluation includes

RMSE
MAE
RMSLE
Residual analysis
SKU-level diagnostics
Metric Scale
The validation data is approximately 99.99% zero demand.

Therefore:

average demand is extremely small
RMSE values appear numerically small
metrics are reported on the original units sold scale
This behavior is expected for highly sparse retail demand forecasting.

🚀 Deployment
Deployment artifacts include

trained Tweedie model
feature ordering
preprocessing pipeline
The project also contains notebook-based deployment preparation for future API integration.

💡 Business Value
A retailer using this system could

reduce stockouts
reduce overstocks
improve markdown timing
improve inventory allocation
increase gross margin
reduce inventory waste
🔮 Future Improvements
SHAP Explainability
FastAPI Deployment
Docker Container
Markdown Optimization Engine
Reinforcement Learning Pricing
Real Retail Data Validation
🧑‍💻 Author
John Kirima
Business Analytics & Information Systems

Machine Learning • Data Science • AI Engineering