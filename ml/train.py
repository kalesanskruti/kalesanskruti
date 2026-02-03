import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def train_model():
    input_file = 'data/processed_train_FD001.csv'
    model_dir = 'ml/models'
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"Loading preprocessed data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Define features and target
    # We use settings and sensors as features
    feature_cols = ['setting1', 'setting2', 'setting3'] + ['s' + str(i) for i in range(1, 22)]
    target_col = 'RUL'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train Random Forest Regressor
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Validation R2 Score: {r2:.2f}")
    
    # Save the model and scaler
    model_path = os.path.join(model_dir, 'rul_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train_model()
