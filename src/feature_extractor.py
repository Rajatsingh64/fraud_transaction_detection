from src.exception import SrcException
from src.config import mongo_client ,TARGET_COLUMN , REALTIME_FEATURES
from pymongo import DESCENDING 
import pandas as pd
import numpy as np
import os ,sys


required_columns = REALTIME_FEATURES + ([TARGET_COLUMN] if isinstance(TARGET_COLUMN, str) else TARGET_COLUMN)

########################
# Feature Engineering
########################

def create_is_night_tx(df):
    """Flag transactions that occur at night (0–6 AM)."""
    df = df.copy()
    df['IS_NIGHT_TX'] = df['TX_HOUR'].apply(lambda x: 1 if x < 6 else 0)
    return df

def create_is_weekend_tx(df):
    """Flag if the transaction occurred on a weekend (Saturday=5, Sunday=6)."""
    df = df.copy()
    df['TX_IS_WEEKEND'] = df['TX_WEEK_DAY'].apply(lambda x: 1 if x in [5, 6] else 0)
    return df

def create_is_tx_amount_high(df, threshold=180):
    """Flag if the amount is higher than a given threshold."""
    df = df.copy()
    df['IS_TX_AMOUNT_HIGH'] = df['TX_AMOUNT'].apply(lambda x: 1 if x > threshold else 0)
    return df

def create_rolling_features(df, window_days=7):
    """Create rolling counts of transactions over last N days per customer and terminal."""
    df = df.copy()
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df = df.sort_values('TX_DATETIME')

    # Make TX_DATETIME unique for rolling window (handle duplicate timestamps)
    df['TX_DATETIME_Unique'] = df.groupby('TX_DATETIME').cumcount()
    df['TX_DATETIME_UNIQUE_IDX'] = df['TX_DATETIME'] + pd.to_timedelta(df['TX_DATETIME_Unique'], unit='ms')

    df.set_index('TX_DATETIME_UNIQUE_IDX', inplace=True)

    df['CUSTOMER_TX_COUNT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].rolling(f'{window_days}D').count().reset_index(level=0, drop=True)
    df['TERMINAL_TX_COUNT_7D'] = df.groupby('TERMINAL_ID')['TX_AMOUNT'].rolling(f'{window_days}D').count().reset_index(level=0, drop=True)

    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['TX_DATETIME_Unique'], inplace=True)

    return df

def create_amount_stats(df, window=7):
    """Create average and max amount per customer over last N days and overall average."""
    df = df.copy()
    df = df.sort_values(by='TX_DATETIME')

    df['CUSTOMER_AVG_AMOUNT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )

    df['CUSTOMER_MAX_AMOUNT_7D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(
        lambda x: x.rolling(window=window, min_periods=1).max()
    )

    df['AVG_AMOUNT_CUSTOMER'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('mean')

    return df

def create_is_5x_avg(df):
    """Flag if transaction amount > 5x average of customer."""
    df = df.copy()
    df['IS_TX_5X_AVG'] = (df['TX_AMOUNT'] > (5 * df['AVG_AMOUNT_CUSTOMER'])).astype(int)
    return df

def create_tx_month(df):
    """Extract month from TX_DATETIME."""
    df = df.copy()
    df['TX_MONTH'] = pd.to_datetime(df['TX_DATETIME']).dt.month
    return df

def create_monthly_tx_counts(df):
    """Add monthly transaction counts for each customer and terminal."""
    df = df.copy()

    if 'TX_MONTH' not in df.columns:
        df = create_tx_month(df)

    df['CUSTOMER_TX_COUNT_MONTH'] = df.groupby(['CUSTOMER_ID', 'TX_MONTH'])['TX_DATETIME'].transform('count')
    df['TERMINAL_TX_COUNT_MONTH'] = df.groupby(['TERMINAL_ID', 'TX_MONTH'])['TX_DATETIME'].transform('count')

    return df

def create_ratio_features(df):
    """Create ratio features comparing TX_AMOUNT with averages and max."""
    df = df.copy()
    df['TX_OVER_CUSTOMER_AVG'] = df['TX_AMOUNT'] / (df['AVG_AMOUNT_CUSTOMER'] + 1e-5)
    df['TX_OVER_MAX_LAST_7D'] = df['TX_AMOUNT'] / (df['CUSTOMER_MAX_AMOUNT_7D'] + 1e-5)
    return df


def create_rolling_tx_count_1d(df):
    """
    Adds rolling 1-day transaction count per customer.
    Requires TX_DATETIME to be a datetime column.
    """
    df = df.copy()
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])  # ensure it's datetime
    df = df.sort_values(by=['CUSTOMER_ID', 'TX_DATETIME'])
    df['TX_DATETIME_Unique'] = df.groupby('TX_DATETIME').cumcount()
    df['TX_DATETIME_UNIQUE_IDX'] = df['TX_DATETIME'] + pd.to_timedelta(df['TX_DATETIME_Unique'], unit='ms')


    df.set_index('TX_DATETIME_UNIQUE_IDX', inplace=True)

    df['ROLLING_TX_COUNT_1D'] = (
        df.groupby('CUSTOMER_ID')['TX_AMOUNT']
        .rolling('1D')
        .count()
        .reset_index(level=0, drop=True)
    )

    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['TX_DATETIME_Unique'], inplace=True)

    return df

def create_terminal_risk(df: pd.DataFrame, past_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Assigns a risk score to each TERMINAL_ID based on historical fraud rate.

    - In training mode: calculates from df itself.
    - In prediction mode: calculates from past_df to avoid leakage.

    Parameters:
        df (pd.DataFrame): The current or combined dataframe to apply terminal risk.
        past_df (pd.DataFrame): Past transactions used for terminal fraud rate (in prediction).

    Returns:
        pd.DataFrame: DataFrame with TERMINAL_RISK column added.
    """
    df = df.copy()

    if past_df is not None and not past_df.empty and "TX_FRAUD" in past_df.columns:
        # Prediction mode: calculate risk from past_df only
        terminal_fraud_rate = past_df.groupby("TERMINAL_ID")["TX_FRAUD"].mean()
        df["TERMINAL_RISK"] = df["TERMINAL_ID"].map(terminal_fraud_rate).fillna(0)

    elif "TX_FRAUD" in df.columns:
        # Training mode: calculate risk from df itself
        terminal_fraud_rate = df.groupby("TERMINAL_ID")["TX_FRAUD"].mean()
        df["TERMINAL_RISK"] = df["TERMINAL_ID"].map(terminal_fraud_rate).fillna(0)

    else:
        # No fraud labels available
        df["TERMINAL_RISK"] = 0

    return df


def create_weekend_night_flag(df):
    """Flag transactions that occur on weekend nights."""
    df = df.copy()
    df['WEEKEND_NIGHT'] = ((df['TX_IS_WEEKEND'] == 1) & (df['IS_NIGHT_TX'] == 1)).astype(int)
    return df

def create_time_since_last_tx(df):
    """Calculate time in seconds since last transaction per customer."""
    df = df.copy()
    df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])
    df['TIME_SINCE_LAST_TX'] = df.groupby('CUSTOMER_ID')['TX_DATETIME'].diff().dt.total_seconds()
    df['TIME_SINCE_LAST_TX'] = df['TIME_SINCE_LAST_TX'].fillna(999999)  # Large number if no prior tx
    return df

def add_weekday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds weekday-related features to the DataFrame:
    - TX_HOUR: hour of the transaction
    - TX_WEEK_DAY: day of the week (0=Monday, 6=Sunday)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with at least a 'TX_DATETIME' column.

    Returns:
    --------
    pd.DataFrame
        DataFrame with new weekday features added.
    """
    df = df.copy()
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
    df["TX_WEEK_DAY"] = df["TX_DATETIME"].dt.weekday
    return df

#####################
# Feature Generator
#####################

def generate_features(current_df: pd.DataFrame, 
                      past_df: pd.DataFrame = None, 
                      required_columns= required_columns, 
                      mode: str = "training") -> pd.DataFrame:
    """
    Generates features for fraud detection based on mode.
    """
    if current_df is None or current_df.empty:
        raise ValueError("`current_df` must be a non-empty DataFrame")

    try:
        current_df = add_weekday_features(current_df)

        if mode == "prediction":
            
            if past_df is not None and not past_df.empty:
                if required_columns:
                    missing_cols_past = set(required_columns) - set(past_df.columns)
                    if missing_cols_past:
                        raise ValueError(f"Missing columns in past_df: {missing_cols_past}")
                    past_df = past_df[required_columns].copy()
                else:
                    past_df = past_df.copy()

                past_df = add_weekday_features(past_df)

                if TARGET_COLUMN not in current_df.columns:
                    current_df[TARGET_COLUMN]=0 #place holder for current prediction 

                # Filter to avoid leakage
                cutoff_time = pd.to_datetime(current_df["TX_DATETIME"]).min()
                past_df["TX_DATETIME"] = pd.to_datetime(past_df["TX_DATETIME"])
                past_df = past_df[past_df["TX_DATETIME"] < cutoff_time]

                combined_df = pd.concat([past_df, current_df]).sort_values("TX_DATETIME")
            else:
                combined_df = current_df.copy()

        elif mode == "training":
            past_df=None
            if required_columns:
                missing_cols = set(required_columns) - set(current_df.columns)
                if missing_cols:
                    raise ValueError(f"Missing columns in training data: {missing_cols}")
                combined_df = current_df[required_columns].copy()
            else:
                combined_df = current_df.copy()
        else:
            raise ValueError("`mode` must be either 'training' or 'prediction'")

        # Base features
        combined_df = create_is_night_tx(combined_df)
        combined_df = create_is_weekend_tx(combined_df)
        combined_df = create_is_tx_amount_high(combined_df)
        combined_df = create_tx_month(combined_df)
        combined_df = create_weekend_night_flag(combined_df)

        # Rolling & statistical features
        try:
            combined_df = create_amount_stats(combined_df)
            combined_df = create_is_5x_avg(combined_df)
            combined_df = create_rolling_features(combined_df)
            combined_df = create_monthly_tx_counts(combined_df)
            combined_df = create_ratio_features(combined_df)
            combined_df = create_rolling_tx_count_1d(combined_df)
            combined_df = create_terminal_risk(combined_df, past_df if mode == "prediction" else None)
            combined_df = create_time_since_last_tx(combined_df)
        except Exception as fe:
            raise fe

        # Final output matching current transactions only   
        current_df = combined_df.loc[
            combined_df["TRANSACTION_ID"].isin(current_df["TRANSACTION_ID"])
        ].copy()

        return current_df

    except Exception as e:
        raise SrcException(e,sys)