import os
import pandas as pd
import numpy as np
from datetime import datetime

class ParkingDataPreprocessor:
    def __init__(self, raw_data_dir=None):
        self.raw_data_dir = raw_data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
        self.cleaning_report = {}
    
    def load_birmingham(self, filename='birmingham_parking.csv'):
        path = os.path.join(self.raw_data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Birmingham data not found: {path}")
        df = pd.read_csv(path)
        # Standardize columns
        df = df.rename(columns={
            'SystemCodeNumber': 'facility_id',
            'Capacity': 'capacity',
            'Occupancy': 'occupancy',
            'LastUpdated': 'timestamp'
        })
        return df
    
    def load_klcc(self, filename='klcc_parking.txt'):
        path = os.path.join(self.raw_data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"KLCC data not found: {path}")
        # Try semicolon separated
        df = pd.read_csv(path, sep=';', header=None, names=['facility_id', 'occupancy', 'timestamp'])
        return df
    
    def clean_data(self, df, facility_id_col='facility_id', capacity_col='capacity', 
                   occupancy_col='occupancy', timestamp_col='timestamp'):
        report = {}
        df = df.copy()
        report['initial_rows'] = len(df)
        
        # Parse timestamps
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        invalid_ts = df[timestamp_col].isna().sum()
        report['invalid_timestamps'] = int(invalid_ts)
        df = df.dropna(subset=[timestamp_col])
        
        # Ensure numeric
        if occupancy_col in df.columns:
            df[occupancy_col] = pd.to_numeric(df[occupancy_col], errors='coerce')
        if capacity_col in df.columns:
            df[capacity_col] = pd.to_numeric(df[capacity_col], errors='coerce')
        
        # Handle missing
        report['missing_occupancy'] = int(df[occupancy_col].isna().sum()) if occupancy_col in df.columns else 0
        report['missing_capacity'] = int(df[capacity_col].isna().sum()) if capacity_col in df.columns else 0
        
        # Drop rows with critical missing
        df = df.dropna(subset=[occupancy_col]) if occupancy_col in df.columns else df
        
        # Clean capacity
        if capacity_col in df.columns:
            # Fill missing capacity per facility with mode/median
            for fid in df[facility_id_col].unique():
                mask = (df[facility_id_col] == fid) & df[capacity_col].isna()
                if mask.sum() > 0:
                    cap_mode = df[(df[facility_id_col] == fid) & df[capacity_col].notna()][capacity_col].mode()
                    if not cap_mode.empty:
                        df.loc[mask, capacity_col] = cap_mode.iloc[0]
            df = df.dropna(subset=[capacity_col])
            # Remove zero/negative capacity
            invalid_cap = ((df[capacity_col] <= 0)).sum()
            report['invalid_capacity_removed'] = int(invalid_cap)
            df = df[df[capacity_col] > 0]
        
        # Clean occupancy
        if occupancy_col in df.columns:
            # Negative occupancy -> set to 0
            neg_occ = (df[occupancy_col] < 0).sum()
            report['negative_occupancy_corrected'] = int(neg_occ)
            df.loc[df[occupancy_col] < 0, occupancy_col] = 0
            
            if capacity_col in df.columns:
                # Occupancy > capacity - investigate but cap or flag
                over_cap = (df[occupancy_col] > df[capacity_col]).sum()
                report['occupancy_greater_than_capacity'] = int(over_cap)
                # Cap at capacity to maintain physical validity
                df.loc[df[occupancy_col] > df[capacity_col], occupancy_col] = df.loc[df[occupancy_col] > df[capacity_col], capacity_col]
        
        # Calculate derived fields
        if capacity_col in df.columns and occupancy_col in df.columns:
            df['available_spaces'] = df[capacity_col] - df[occupancy_col]
            df['available_spaces'] = df['available_spaces'].clip(lower=0)
            df['occupancy_rate'] = (df[occupancy_col] / df[capacity_col] * 100).round(2)
        
        # Remove duplicates
        before_dup = len(df)
        df = df.drop_duplicates(subset=[facility_id_col, timestamp_col])
        report['duplicates_removed'] = int(before_dup - len(df))
        
        # Sort
        df = df.sort_values([facility_id_col, timestamp_col]).reset_index(drop=True)
        
        report['final_rows'] = len(df)
        report['facilities'] = df[facility_id_col].nunique()
        self.cleaning_report = report
        return df
    
    def get_cleaning_report(self):
        return self.cleaning_report
