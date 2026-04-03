import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def predict_salary(df):
    # Sample 20,000 rows to speed up training (adjust if needed)
    df_sample = df.sample(n=20000, random_state=42)

    # --- Step 1: Extract average salary from salary range ---
    def extract_avg_salary(s):
        try:
            s = s.replace('$', '').replace(',', '').strip().upper()
            parts = s.split('-')
            nums = []
            for p in parts:
                if 'K' in p:
                    nums.append(float(p.replace('K', '')) * 1000)
                else:
                    nums.append(float(p))
            return sum(nums) / len(nums)
        except:
            return None

    df_sample['avg_salary'] = df_sample['salary range'].apply(extract_avg_salary)

    # Drop rows with missing avg_salary
    df_sample = df_sample.dropna(subset=['avg_salary'])

    # --- Step 2: Additional feature engineering ---
    df_sample['num_skills'] = df_sample['skills'].str.split(',').apply(lambda x: len(x) if isinstance(x, list) else 0)

    # --- Step 3: Select features and target ---
    X = df_sample[['skills', 'job title', 'experience', 'num_skills']]
    y = df_sample['avg_salary']

    # --- Step 4: Preprocessing Pipeline ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('skills', TfidfVectorizer(max_features=200), 'skills'),
            ('job_title', OneHotEncoder(handle_unknown='ignore'), ['job title']),
            ('experience', OneHotEncoder(handle_unknown='ignore'), ['experience']),
            ('num_skills', 'passthrough', ['num_skills'])
        ])

    # --- Step 5: XGBoost Model Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('xgb', XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        ))
    ])

    # --- Step 6: Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Step 7: Train Model ---
    model_pipeline.fit(X_train, y_train)

    # --- Step 8: Evaluate Performance ---
    y_pred = model_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("✅ Model Training Completed")
    print(f"R² Score: {r2:.3f} ({round(r2 * 100)}%)")
    print(f"MAE: $ {mae:.2f}")
    print(f"RMSE: $ {rmse:.2f}")

    return model_pipeline, r2, mae, rmse
