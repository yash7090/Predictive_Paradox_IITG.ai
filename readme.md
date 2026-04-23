Overview: "Short-Term Power Demand Forecasting"

An end-to-end machine learning pipeline designed to forecast national power demand using 10 years of historical grid telemetry, thermodynamic weather data, and macroeconomic indicators. 

The final model (XGBoost) achieved a 2.89% Mean Absolute Percentage Error (MAPE) on a hold-out test set, successfully mapping highly non-linear human energy consumption patterns.

->Methodology for Short-Term Power Demand Forecasting

Step 1: Data Ingestion and Initial Validation
I loaded three separate datasets: historical hourly grid power demand, thermodynamic weather data, and macroeconomic annual GDP indicators. I performed initial sanity checks to identify null values and physically impossible negative or zero demand entries.

Step 2: Timeline Continuity and Preprocessing
The raw grid logs contained irregular timestamp entries. I standardized the timeline by flooring all timestamps to the nearest hour and calculating the mean for any duplicate hours. I then generated a perfect, continuous hourly date range and forward-filled any missing gaps to ensure the data flow required for time-series forecasting.

Step 3: Dataset Integration
Because GDP is an annual metric and power demand is hourly, I extracted the specific yearly GDP values and broadcasted them across every hour of their respective years. I then merged the weather and economic features into the main power dataframe.

Step 4: Temporal Feature Engineering
To map human routines and cyclical consumption, I extracted discrete time-based features from the datetime index. This included the hour of the day, day of the week, month of the year, and a binary flag to indicate weekends.

Step 5: Self-Healing Outlier Correction
The real-world grid data contained massive human-entry typos. To fix this without deleting data, I built a Year-Month Cohort Interquartile Range smoothing function. It calculated dynamic statistical boundaries for each specific month of each specific year, clipping extreme typos to safe historical boundaries while preserving legitimate seasonal peaks.

Step 6: Thermodynamic and Momentum Feature Engineering
To capture the physical inertia of the power grid, I created lag features for demand representing the past 1 hour, 24 hours, and 168 hours, along with a 24-hour rolling mean. To account for structural heat accumulation in buildings, I also engineered a 24-hour temperature lag and a 24-hour rolling average for temperature.

Step 7: Strict Temporal Splitting and Feature Scaling
To prevent future data from leaking into the training phase, I enforced a chronological split. The models were trained exclusively on data up to 2023, and tested on data from 2024 onward. I then applied a Standard Scaler to normalize the massive mathematical variance between trillion-dollar GDP numbers and single-digit temporal features.

Step 8: Model Training and Evaluation
I trained a suite of machine learning models including Linear Regression, Ridge Regression, Random Forest, LightGBM, and XGBoost. I evaluated their performance on the unseen test set using Mean Absolute Percentage Error to determine percentage-based accuracy, alongside RMSE and MAE metrics.

Step 9: Manual Model Tuning & Bias-Variance Analysis
To optimize performance without overfitting, I bypassed automated grid searches entirely and performed strict manual tuning based on the physical realities of the grid data. By manually testing complexity constraints and analyzing the bias-variance tradeoff, I identified the exact mathematical sweet spot for XGBoost—allowing it to capture weather-driven demand spikes without memorizing sensor noise.

Step 10: Diagnostics and Feature Importance Extraction
Finally, I analyzed the feature importance using XGBoost's gain metrics to rank the primary drivers of grid demand. I also performed a diagnostic residual analysis on the baseline Linear Regression model to verify that non-linear, tree-based algorithms were necessary to handle the extreme spikes in power consumption during weather shocks.

->Non-linear ensemble models vastly outperformed linear baselines in capturing summer cooling spikes. 

| Model             | Test MAPE | RMSE   |
| XGBoost (Selected)| 2.89%     | 489.14 |
| LightGBM          | 2.95%     | 489.48 |
| Random Forest     | 3.19%     | 522.54 |
| Ridge Regression  | 5.08%     | 714.59 |
| Linear Regression | 5.08%     | 714.59 |

->Quick Start

Ensure you have Python 3.8+ installed.

Clone the repository
git clone [https://github.com/yash7090/Predictive_Paradox_IITG_ai.git](https://github.com/yash7090/Predictive_Paradox_IITG_ai.git)

pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn openpyxl time

jupyter notebook

Run pred_para_pipeline_yash.ipynb