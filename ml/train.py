import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml.preprocessing import ParkingDataPreprocessor
from ml.feature_engineering import FeatureEngineer

class ParkingModelTrainer:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        self.preprocessor = ParkingDataPreprocessor()
        self.fe = FeatureEngineer()
        self.models = {}
        self.metrics = {}
        self.feature_cols = self.fe.get_feature_columns()
    
    def prepare_data(self, data_source='birmingham'):
        if data_source == 'birmingham':
            df = self.preprocessor.load_birmingham()
        elif data_source == 'klcc':
            df = self.preprocessor.load_klcc()
            # KLCC might not have capacity - need to handle
        else:
            raise ValueError("Unknown data source")
        
        df_clean = self.preprocessor.clean_data(df)
        df_feat = self.fe.create_features(df_clean)
        self.cleaning_report = self.preprocessor.get_cleaning_report()
        return df_feat
    
    def train_models(self, df, target='occupancy', facility_id=None):
        df_train = df.copy()
        if facility_id:
            df_train = df_train[df_train['facility_id'] == facility_id]
        
        # Remove rows with NaN in features
        df_train = df_train.dropna(subset=self.feature_cols + [target])
        
        X = df_train[self.feature_cols]
        y = df_train[target]
        
        # Chronological split (80/10/10)
        n = len(df_train)
        train_idx = int(n * 0.8)
        val_idx = int(n * 0.9)
        
        X_train = X.iloc[:train_idx]
        y_train = y.iloc[:train_idx]
        X_val = X.iloc[train_idx:val_idx]
        y_val = y.iloc[train_idx:val_idx]
        X_test = X.iloc[val_idx:]
        y_test = y.iloc[val_idx:]
        
        results = {}
        
        # Baseline - persistence
        y_pred_persist = X_test['occupancy_lag1'].values
        results['persistence'] = self._compute_metrics(y_test, y_pred_persist, 'Persistence')
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        results['linear_regression'] = self._compute_metrics(y_test, y_pred_lr, 'Linear Regression')
        self.models['linear_regression'] = lr
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        results['random_forest'] = self._compute_metrics(y_test, y_pred_rf, 'Random Forest')
        self.models['random_forest'] = rf
        
        # Try XGBoost if available
        try:
            import xgboost as xgb
            xgbr = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            xgbr.fit(X_train, y_train)
            y_pred_xgb = xgbr.predict(X_test)
            results['xgboost'] = self._compute_metrics(y_test, y_pred_xgb, 'XGBoost')
            self.models['xgboost'] = xgbr
        except:
            pass
        
        self.metrics = results
        self.X_test = X_test
        self.y_test = y_test
        self.df_test = df_train.iloc[val_idx:].copy()
        
        return results
    
    def _compute_metrics(self, y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {
            'name': name,
            'mae': round(float(mae), 3),
            'rmse': round(float(rmse), 3),
            'r2': round(float(r2), 3)
        }
    
    def save_best_model(self, facility_id=None):
        # Save Random Forest as default
        if 'random_forest' in self.models:
            model_path = os.path.join(self.models_dir, f'parkahead_model_{facility_id or "all"}.pkl')
            features_path = os.path.join(self.models_dir, f'parkahead_features_{facility_id or "all"}.pkl')
            joblib.dump(self.models['random_forest'], model_path)
            joblib.dump(self.feature_cols, features_path)
            return model_path
        return None
    
    def get_metrics_df(self):
        return pd.DataFrame(self.metrics).T
