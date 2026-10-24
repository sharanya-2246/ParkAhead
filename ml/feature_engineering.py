import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        pass
    
    def create_features(self, df, facility_id_col='facility_id', timestamp_col='timestamp', 
                       occupancy_col='occupancy'):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Time features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek  # 0=Monday
        df['day_name'] = df[timestamp_col].dt.day_name()
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_morning'] = df['hour'].between(6, 11).astype(int)
        df['is_afternoon'] = df['hour'].between(12, 17).astype(int)
        df['is_evening'] = df['hour'].between(18, 22).astype(int)
        df['is_night'] = df['hour'].between(23, 5).astype(int)
        
        # Lag features per facility
        for fid in df[facility_id_col].unique():
            mask = df[facility_id_col] == fid
            df.loc[mask, 'occupancy_lag1'] = df.loc[mask, occupancy_col].shift(1)
            df.loc[mask, 'occupancy_lag2'] = df.loc[mask, occupancy_col].shift(2)
            df.loc[mask, 'occupancy_lag3'] = df.loc[mask, occupancy_col].shift(3)
            df.loc[mask, 'occupancy_rolling_mean_3'] = df.loc[mask, occupancy_col].rolling(window=3, min_periods=1).mean()
            df.loc[mask, 'occupancy_rolling_mean_6'] = df.loc[mask, occupancy_col].rolling(window=6, min_periods=1).mean()
            df.loc[mask, 'occupancy_change'] = df.loc[mask, occupancy_col].diff()
        
        # Fill NaN from lags
        df['occupancy_lag1'] = df['occupancy_lag1'].fillna(df[occupancy_col])
        df['occupancy_lag2'] = df['occupancy_lag2'].fillna(df[occupancy_col])
        df['occupancy_lag3'] = df['occupancy_lag3'].fillna(df[occupancy_col])
        df['occupancy_rolling_mean_3'] = df['occupancy_rolling_mean_3'].fillna(df[occupancy_col])
        df['occupancy_rolling_mean_6'] = df['occupancy_rolling_mean_6'].fillna(df[occupancy_col])
        df['occupancy_change'] = df['occupancy_change'].fillna(0)
        
        return df
    
    def get_feature_columns(self):
        return ['hour', 'day_of_week', 'month', 'is_weekend', 'is_morning', 
                'is_afternoon', 'is_evening', 'is_night',
                'occupancy_lag1', 'occupancy_lag2', 'occupancy_lag3',
                'occupancy_rolling_mean_3', 'occupancy_rolling_mean_6', 
                'occupancy_change']
