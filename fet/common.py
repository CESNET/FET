"""
    Common and helper functions.
"""

import pandas as pd

directional_columns = [
    ("dst_ip", "src_ip"),
    ("dst_mac", "src_mac"),
    ("dst_port", "src_port"),
    ("bytes", "bytes_rev"),
    ("packets", "packets_rev"),
    ("tcp_flags", "tcp_flags_rev"),
]

flow_key = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]


def convert_times(df, inplace=False):
    """Convert time strings and calculate duration.

    Args:
        df (pandas.DataFrame): DataFrame with time_first and time_last.
        inplace (bool, optional): Extract features within provided DataFrame
            or return new DataFrame. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame is returned only if inplace=False,
        otherwise returns None.
    """
    if not inplace:
        df = df.copy()

    df["time_first"] = pd.to_datetime(df["time_first"])
    df["time_last"] = pd.to_datetime(df["time_last"])
    df["duration"] = (df["time_last"] - df["time_first"]).dt.total_seconds()

    if not inplace:
        return df
