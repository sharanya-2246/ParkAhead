import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ParkingPredictor:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.model = None
        self.feature_cols = None
    
    def load_model(self, facility_id=None):
        model_path = os.path.join(self.models_dir, f'parkahead_model_{facility_id or "all"}.pkl')
        features_path = os.path.join(self.models_dir, f'parkahead_features_{facility_id or "all"}.pkl')
        if os.path.exists(model_path) and os.path.exists(features_path):
            self.model = joblib.load(model_path)
            self.feature_cols = joblib.load(features_path)
            return True
        return False
    
    def predict_at_eta(self, historical_df, facility_id, eta_datetime, capacity=None):
        if self.model is None or self.feature_cols is None:
            raise ValueError("Model not loaded. Train the model first.")
        
        df_fac = historical_df[historical_df['facility_id'] == facility_id].copy()
        if df_fac.empty:
            raise ValueError(f"No historical data for facility {facility_id}")
        
        df_fac = df_fac.sort_values('timestamp')
        latest = df_fac.iloc[-1]
        
        if capacity is None and 'capacity' in latest:
            capacity = int(latest['capacity'])
        
        # Create features for prediction point (using latest as proxy)
        pred_features = {}
        pred_features['hour'] = eta_datetime.hour
        pred_features['day_of_week'] = eta_datetime.weekday()
        pred_features['month'] = eta_datetime.month
        pred_features['is_weekend'] = 1 if eta_datetime.weekday() in [5,6] else 0
        pred_features['is_morning'] = 1 if 6 <= eta_datetime.hour <= 11 else 0
        pred_features['is_afternoon'] = 1 if 12 <= eta_datetime.hour <= 17 else 0
        pred_features['is_evening'] = 1 if 18 <= eta_datetime.hour <= 22 else 0
        pred_features['is_night'] = 1 if eta_datetime.hour >= 23 or eta_datetime.hour <= 5 else 0
        
        # Use recent history
        pred_features['occupancy_lag1'] = latest['occupancy']
        pred_features['occupancy_lag2'] = df_fac.iloc[-2]['occupancy'] if len(df_fac) > 1 else latest['occupancy']
        pred_features['occupancy_lag3'] = df_fac.iloc[-3]['occupancy'] if len(df_fac) > 2 else latest['occupancy']
        pred_features['occupancy_rolling_mean_3'] = df_fac['occupancy'].tail(3).mean()
        pred_features['occupancy_rolling_mean_6'] = df_fac['occupancy'].tail(6).mean()
        pred_features['occupancy_change'] = df_fac['occupancy'].iloc[-1] - df_fac['occupancy'].iloc[-2] if len(df_fac) > 1 else 0
        
        X_pred = pd.DataFrame([pred_features])[self.feature_cols]
        predicted_occupancy = self.model.predict(X_pred)[0]
        
        # Apply constraints
        if capacity:
            predicted_occupancy = max(0, min(predicted_occupancy, capacity))
        else:
            predicted_occupancy = max(0, predicted_occupancy)
        
        predicted_available = (capacity - predicted_occupancy) if capacity else None
        if predicted_available is not None and predicted_available < 0:
            predicted_available = 0
        
        predicted_occ_rate = (predicted_occupancy / capacity * 100) if capacity and capacity > 0 else None
        
        return {
            'facility_id': facility_id,
            'predicted_occupancy': round(float(predicted_occupancy), 1),
            'predicted_available_spaces': int(round(float(predicted_available))) if predicted_available is not None else None,
            'predicted_occupancy_rate': round(float(predicted_occ_rate), 1) if predicted_occ_rate is not None else None,
            'capacity': capacity,
            'eta': eta_datetime,
            'model_used': 'RandomForest'
        }
