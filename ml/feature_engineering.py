
import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self):
        pass

    def create_features(
        self,
        df,
        facility_id_col='facility_id',
        timestamp_col='timestamp',
        occupancy_col='occupancy'
    ):
        df = df.copy()

        # Convert timestamps to datetime format
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], errors='coerce'
        )

        # Sort each facility's records chronologically
        df = df.sort_values(
            [facility_id_col, timestamp_col]
        ).reset_index(drop=True)

        # Time-based features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        # 0 = Monday, 6 = Sunday
        df['day_name'] = df[timestamp_col].dt.day_name()
        df['month'] = df[timestamp_col].dt.month

        df['is_weekend'] = (
            df['day_of_week'].isin([5, 6])
        ).astype(int)

        df['is_morning'] = (
            df['hour'].between(6, 11)
        ).astype(int)

        df['is_afternoon'] = (
            df['hour'].between(12, 17)
        ).astype(int)

        df['is_evening'] = (
            df['hour'].between(18, 22)
        ).astype(int)

        # Night includes 11 PM to 5 AM
        df['is_night'] = (
            (df['hour'] >= 23) | (df['hour'] <= 5)
        ).astype(int)

        # Historical occupancy features for each facility
        grouped = df.groupby(facility_id_col)[occupancy_col]

        # Previous occupancy values
        df['occupancy_lag1'] = grouped.shift(1)
        df['occupancy_lag2'] = grouped.shift(2)
        df['occupancy_lag3'] = grouped.shift(3)

        # Use only previous observations for rolling averages
        previous_occupancy = grouped.shift(1)

        df['occupancy_rolling_mean_3'] = (
            previous_occupancy
            .groupby(df[facility_id_col])
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df['occupancy_rolling_mean_6'] = (
            previous_occupancy
            .groupby(df[facility_id_col])
            .rolling(window=6, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Change in occupancy based only on earlier observations
        df['occupancy_change'] = (
            grouped.diff()
            .groupby(df[facility_id_col])
            .shift(1)
        )

        # Keep initial missing historical features as NaN.
        # The trainer will remove rows without sufficient history.
        return df

    def get_feature_columns(self):
        return [
            'hour',
            'day_of_week',
            'month',
            'is_weekend',
            'is_morning',
            'is_afternoon',
            'is_evening',
            'is_night',
            'occupancy_lag1',
            'occupancy_lag2',
            'occupancy_lag3',
            'occupancy_rolling_mean_3',
            'occupancy_rolling_mean_6',
            'occupancy_change'
        ]
