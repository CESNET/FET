"""
    Per flow features extraction.
"""

import statistics
from datetime import datetime

import pandas as pd

loop_stats_fields = [
    "fin_count",
    "syn_count",
    "rst_count",
    "psh_count",
    "ack_count",
    "urg_count",
    "lengths_min",
    "lengths_max",
    "lengths_mean",
    "lengths_std",
    "fwd_lengths_min",
    "fwd_lengths_max",
    "fwd_lengths_mean",
    "fwd_lengths_std",
    "bwd_lengths_min",
    "bwd_lengths_max",
    "bwd_lengths_mean",
    "bwd_lengths_std",
    "pkt_iat_min",
    "pkt_iat_max",
    "pkt_iat_mean",
    "pkt_iat_std",
    "fwd_pkt_iat_min",
    "fwd_pkt_iat_max",
    "fwd_pkt_iat_mean",
    "fwd_pkt_iat_std",
    "bwd_pkt_iat_min",
    "bwd_pkt_iat_max",
    "bwd_pkt_iat_mean",
    "bwd_pkt_iat_std",
]

feature_cols = [
    "duration",
    "bytes_per_s",
    "bytes_rev_per_s",
    "packets_per_s",
    "packets_rev_per_s",
    "bytes_ratio",
    "bytes_mean",
    "packets_ratio",
] + loop_stats_fields


def convert_lengths(pkt_lengths):
    """Convert lengths from PPI_PKT_LENGHTS representation.

    Args:
        pkt_lengths (str): PPI_PKT_LENGTHS.

    Returns:
        dict: List of packet lengths.
    """

    if pkt_lengths == "[]":
        return []

    return [int(x) for x in pkt_lengths.strip("[]").split("|")]


def convert_directions(pkt_directions):
    """Convert directions from PPI_PKT_DIRECTIONS representation.

    Args:
        pkt_directions (str): PPI_PKT_DIRECTIONS.

    Returns:
        forward (list): Indexes of forward packets.
        backward (list): Indexes of backward packets.
    """
    if pkt_directions == "[]":
        return [], []

    forward = []
    backward = []

    for i, val in enumerate(pkt_directions.strip("[]").split("|")):
        if val == "1":
            forward.append(i)
        else:
            backward.append(i)

    return forward, backward


def flags_stats(row):
    """Calculate flags statistics.

    Args:
        row (dict): Row within a dataframe.

    Returns:
        dict: Dictionary with statistics.
    """
    stats = {
        "fin_count": 0,
        "syn_count": 0,
        "rst_count": 0,
        "psh_count": 0,
        "ack_count": 0,
        "urg_count": 0,
    }

    if row["ppi_pkt_flags"] == "[]":
        flags = []
    else:
        flags = [int(x) for x in row["ppi_pkt_flags"].strip("[]").split("|")]

    for f in flags:
        if f & 1 == 1:
            stats["fin_count"] += 1
        if f & 2 == 2:
            stats["syn_count"] += 1
        if f & 4 == 4:
            stats["rst_count"] += 1
        if f & 8 == 8:
            stats["psh_count"] += 1
        if f & 16 == 16:
            stats["ack_count"] += 1
        if f & 32 == 32:
            stats["urg_count"] += 1

    return stats


def lengths_stats(row):
    """Calculate packet lengths statistics.

    Args:
        row (dict): Row within a dataframe.

    Returns:
        dict: Dictionary with statistics.
    """
    stats = {
        "lengths_min": 0,
        "lengths_max": 0,
        "lengths_mean": 0,
        "lengths_std": 0,
        "fwd_lengths_min": 0,
        "fwd_lengths_max": 0,
        "fwd_lengths_mean": 0,
        "fwd_lengths_std": 0,
        "bwd_lengths_min": 0,
        "bwd_lengths_max": 0,
        "bwd_lengths_mean": 0,
        "bwd_lengths_std": 0,
    }

    lengths = row["ppi_pkt_lengths"]
    fwd_lengths = [lengths[i] for i in row["fwd"]]
    bwd_lengths = [lengths[i] for i in row["bwd"]]

    if lengths:
        stats["lengths_min"] = min(lengths)
        stats["lengths_max"] = max(lengths)
        stats["lengths_mean"] = statistics.mean(lengths)
        stats["lengths_std"] = statistics.pstdev(lengths)

    if fwd_lengths:
        stats["fwd_lengths_min"] = min(fwd_lengths)
        stats["fwd_lengths_max"] = max(fwd_lengths)
        stats["fwd_lengths_mean"] = statistics.mean(fwd_lengths)
        stats["fwd_lengths_std"] = statistics.pstdev(fwd_lengths)

    if bwd_lengths:
        stats["bwd_lengths_min"] = min(bwd_lengths)
        stats["bwd_lengths_max"] = max(bwd_lengths)
        stats["bwd_lengths_mean"] = statistics.mean(bwd_lengths)
        stats["bwd_lengths_std"] = statistics.pstdev(bwd_lengths)

    return stats


def iat_stats(row):
    """Calculate inter arrival times statistics.

    Args:
        row (dict): Row within a dataframe.

    Returns:
        dict: Dictionary with statistics.
    """
    stats = {
        "pkt_iat_min": 0,
        "pkt_iat_max": 0,
        "pkt_iat_mean": 0,
        "pkt_iat_std": 0,
        "fwd_pkt_iat_min": 0,
        "fwd_pkt_iat_max": 0,
        "fwd_pkt_iat_mean": 0,
        "fwd_pkt_iat_std": 0,
        "bwd_pkt_iat_min": 0,
        "bwd_pkt_iat_max": 0,
        "bwd_pkt_iat_mean": 0,
        "bwd_pkt_iat_std": 0,
    }

    if row["ppi_pkt_times"] == "[]":
        times = []
    else:
        times = row["ppi_pkt_times"].strip("[]").split("|")

    times = [datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f") for x in times]

    fwd_times = [times[i] for i in row["fwd"]]
    bwd_times = [times[i] for i in row["bwd"]]

    packets_iat = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    forward_iat = [(b - a).total_seconds() for a, b in zip(fwd_times, fwd_times[1:])]
    backward_iat = [(b - a).total_seconds() for a, b in zip(bwd_times, bwd_times[1:])]

    if packets_iat:
        stats["pkt_iat_min"] = min(packets_iat)
        stats["pkt_iat_max"] = max(packets_iat)
        stats["pkt_iat_mean"] = statistics.mean(packets_iat)
        stats["pkt_iat_std"] = statistics.pstdev(packets_iat)

    if forward_iat:
        stats["fwd_pkt_iat_min"] = min(forward_iat)
        stats["fwd_pkt_iat_max"] = max(forward_iat)
        stats["fwd_pkt_iat_mean"] = statistics.mean(forward_iat)
        stats["fwd_pkt_iat_std"] = statistics.pstdev(forward_iat)

    if backward_iat:
        stats["bwd_pkt_iat_min"] = min(backward_iat)
        stats["bwd_pkt_iat_max"] = max(backward_iat)
        stats["bwd_pkt_iat_mean"] = statistics.mean(backward_iat)
        stats["bwd_pkt_iat_std"] = statistics.pstdev(backward_iat)

    return stats


def loop_flow_stats(row):
    """Calculate flow statistics of a single row - appliable over datafram.

    Args:
        row (dict): Row within a dataframe.

    Returns:
        dict: Dictionary with statistics.
    """
    stats = {}

    stats.update(flags_stats(row))
    stats.update(lengths_stats(row))
    stats.update(iat_stats(row))

    return stats


def extract_per_flow_stats(df, inplace=False, min_packets=2):
    """Extracts per flow statistics.

    Args:
        df (pandas.DataFrame): Dataframe with basic and pstats values.
        inplace (bool, optional): Extract features within provided DataFrame
            or return new DataFrame. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame is returned only if inplace=False - otherwise
            returns None.
    """
    if not inplace:
        df = df.copy()

    df.drop(df[df["packets"] < min_packets].index, inplace=True)

    df["time_first"] = pd.to_datetime(df["time_first"])
    df["time_last"] = pd.to_datetime(df["time_last"])
    df["duration"] = (df["time_last"] - df["time_first"]).dt.total_seconds()

    df["bytes_per_s"] = df["bytes"] / df["duration"]
    df["bytes_rev_per_s"] = df["bytes_rev"] / df["duration"]
    df["packets_per_s"] = df["packets"] / df["duration"]
    df["packets_rev_per_s"] = df["packets_rev"] / df["duration"]

    df["bytes_ratio"] = df["bytes_rev"] / df["bytes"]
    df["bytes_mean"] = df["bytes"] / df["packets"]
    df["packets_ratio"] = df["packets_rev"] / df["packets"]

    df["fwd"], df["bwd"] = zip(*df["ppi_pkt_directions"].apply(convert_directions))
    df["ppi_pkt_lengths"] = df["ppi_pkt_lengths"].map(convert_lengths)

    df[loop_stats_fields] = df.apply(loop_flow_stats, axis=1, result_type="expand")

    if not inplace:
        return df
