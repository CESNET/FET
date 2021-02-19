"""
    Helper functions.
"""

import pandas as pd
import numpy as np

directional_columns = [
    ("dst_ip", "src_ip"),
    ("dst_mac", "src_mac"),
    ("dst_port", "src_port"),
    ("bytes", "bytes_rev"),
    ("packets", "packets_rev"),
    ("tcp_flags", "tcp_flags_rev"),
]

flow_key = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol"]


def swap_directions(df, swap, inplace=False):
    """Swap directional columns.

    Args:
        df (pandas.DataFrame): DataFrame with directional columns.
        swap (pandas.Series): Bool series of affected rows.
        inplace (bool, optional): Extract features within provided DataFrame
            or return new DataFrame. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame is returned only if inplace=False,
            otherwise returns None.
    """
    if not inplace:
        df = df.copy()

    for a, b in directional_columns:
        df.loc[swap, [a, b]] = df.loc[swap, [b, a]].values

    df.loc[swap, "ppi_pkt_directions"] = df.loc[swap, "ppi_pkt_directions"].apply(
        lambda x: "["
        + "|".join([str(-int(y)) for y in x.strip("[]").split("|")])
        + "]"
    )

    if not inplace:
        return df


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


def concatenate_ppi(fields):
    """Concatenate per packet information lists.

    Args:
        fields (list): List of string representations from ppi_pkt_* field.

    Returns:
        string: Concatenated representation.
    """
    return "[" + "|".join([x.strip("[]") for x in fields]) + "]"


def aggregate_pstats(df, window="5min"):
    """Time aggregation of basic + pstats fields.

    Args:
        df (pandas.DataFrame): DataFrame with basic + pstats fields.
        window (str, optional): Aggregation time window. Defaults to "5min".
    """
    df = df.astype(
        {
            "tcp_flags": int,
            "tcp_flags_rev": int,
            "ppi_pkt_directions": str,
            "ppi_pkt_flags": str,
            "ppi_pkt_lengths": str,
            "ppi_pkt_times": str,
        }
    )

    df["time"] = df["time_first"].dt.ceil(window)

    group = df.groupby(["time"] + flow_key, as_index=False)[
        [
            "time_first",
            "time_last",
            "packets",
            "packets_rev",
            "bytes",
            "bytes_rev",
            "dir_bit_field",
            "dst_mac",
            "src_mac",
            "tcp_flags",
            "tcp_flags_rev",
            "ppi_pkt_directions",
            "ppi_pkt_flags",
            "ppi_pkt_lengths",
            "ppi_pkt_times",
        ]
    ].agg(
        {
            "time_first": np.min,
            "time_last": np.max,
            "packets": np.sum,
            "packets_rev": np.sum,
            "bytes": np.sum,
            "bytes_rev": np.sum,
            "dir_bit_field": lambda x: x.iloc[0],
            "dst_mac": lambda x: x.iloc[0],
            "src_mac": lambda x: x.iloc[0],
            "tcp_flags": np.bitwise_or.reduce,
            "tcp_flags_rev": np.bitwise_or.reduce,
            "ppi_pkt_directions": lambda x: concatenate_ppi(x.tolist()),
            "ppi_pkt_flags": lambda x: concatenate_ppi(x.tolist()),
            "ppi_pkt_lengths": lambda x: concatenate_ppi(x.tolist()),
            "ppi_pkt_times": lambda x: concatenate_ppi(x.tolist()),
        }
    )

    group["duration"] = (group["time_last"] - group["time_first"]).dt.total_seconds()

    return group
