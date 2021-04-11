"""
    Per flow features extraction.
"""

import statistics
from datetime import datetime

import numpy as np
import pandas as pd

from fet.common import flow_key, directional_columns

basic_fields = [
    "dst_ip",
    "src_ip",
    "bytes",
    "bytes_rev",
    "link_bit_field",
    "time_first",
    "time_last",
    "dst_mac",
    "src_mac",
    "packets",
    "packets_rev",
    "dst_port",
    "src_port",
    "dir_bit_field",
    "protocol",
    "tcp_flags",
    "tcp_flags_rev",
]

loop_stats_fields = [
    "fin_count",
    "syn_count",
    "rst_count",
    "psh_count",
    "ack_count",
    "urg_count",
    "fin_ratio",
    "syn_ratio",
    "rst_ratio",
    "psh_ratio",
    "ack_ratio",
    "urg_ratio",
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
    "norm_pkt_iat_mean",
    "norm_pkt_iat_std",
    "norm_fwd_pkt_iat_mean",
    "norm_fwd_pkt_iat_std",
    "norm_bwd_pkt_iat_mean",
    "norm_bwd_pkt_iat_std",
]

feature_cols = [
    "bytes_rate",
    "bytes_rev_rate",
    "bytes_total_rate",
    "packets_rate",
    "packets_rev_rate",
    "packets_total_rate",
] + loop_stats_fields


def convert_lengths(pkt_lengths):
    """Convert lengths from PPI_PKT_LENGHTS representation.

    Args:
        pkt_lengths (str): PPI_PKT_LENGTHS.

    Returns:
        list: List of packet lengths.
    """

    if pkt_lengths == "[]":
        return []

    return [int(x) for x in pkt_lengths.strip("[]").split("|")]


def convert_directions(pkt_directions):
    """Convert directions from PPI_PKT_DIRECTIONS representation.

    Args:
        pkt_directions (str): PPI_PKT_DIRECTIONS.

    Returns:
        tuple: Tuple containing:

        - directions (list): Converted list of directions: 1, -1 values.
        - forward (list): Indexes of forward packets.
        - backward (list): Indexes of backward packets.
    """
    if pkt_directions == "[]":
        return [], [], []

    directions = []
    forward = []
    backward = []

    for i, val in enumerate(pkt_directions.strip("[]").split("|")):
        if val == "1":
            directions.append(1)
            forward.append(i)
        else:
            directions.append(-1)
            backward.append(i)

    return directions, forward, backward


def convert_merged_lengths(lengths, directions):
    """Convert lengths to merged representation.

    Merged representation sums lengths for consecutive packets which
    are in the same direction.

    Args:
        lengths (list): List of all packet lengths.
        directions (list): List of directions: 1, -1 values.

    Returns:
        list: List with merged packet lengths.
    """
    merged = []
    tmp_sum = 0
    direction = 1

    for i, l in enumerate(lengths):
        if directions[i] != direction:
            merged.append(tmp_sum)
            tmp_sum = 0

        tmp_sum += l
        direction = directions[i]

    if tmp_sum != 0:
        merged.append(tmp_sum)

    return merged


def convert_flags(pkt_flags):
    """Convert flags from PPI_PKT_FLAGS representation.

    Args:
        pkt_flags (str): PPI_PKT_FLAGS.

    Returns:
        list: List of packet flags (as integers).
    """

    if pkt_flags == "[]":
        return []
    else:
        return [int(x) for x in pkt_flags.strip("[]").split("|")]


def contains_handshake(flags):
    """Determines if TCP handshake is present.

    Args:
        flags (list): List of packet flags (as integers).

    Returns:
        bool: True (contains handshake) or False (does not).
    """
    if len(flags) >= 3 and flags[0] == 2 and flags[1] == 18 and flags[2] == 16:
        return True

    return False


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
        "fin_ratio": 0,
        "syn_ratio": 0,
        "rst_ratio": 0,
        "psh_ratio": 0,
        "ack_ratio": 0,
        "urg_ratio": 0,
    }

    flags = row["ppi_pkt_flags"]

    fin_count = 0
    syn_count = 0
    rst_count = 0
    psh_count = 0
    ack_count = 0
    urg_count = 0

    for f in flags:
        if f & 1 == 1:
            fin_count += 1
        if f & 2 == 2:
            syn_count += 1
        if f & 4 == 4:
            rst_count += 1
        if f & 8 == 8:
            psh_count += 1
        if f & 16 == 16:
            ack_count += 1
        if f & 32 == 32:
            urg_count += 1

    total = len(flags)

    stats["fin_count"] = fin_count
    stats["syn_count"] = syn_count
    stats["rst_count"] = rst_count
    stats["psh_count"] = psh_count
    stats["ack_count"] = ack_count
    stats["urg_count"] = urg_count

    stats["fin_ratio"] = fin_count / total
    stats["syn_ratio"] = syn_count / total
    stats["rst_ratio"] = rst_count / total
    stats["psh_ratio"] = psh_count / total
    stats["ack_ratio"] = ack_count / total
    stats["urg_ratio"] = urg_count / total

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

    # skip handshake
    if contains_handshake(row["ppi_pkt_flags"]):
        lengths = lengths[3:]
        fwd_lengths = fwd_lengths[2:]
        bwd_lengths = bwd_lengths[1:]

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
        "norm_pkt_iat_mean": 0,
        "norm_pkt_iat_std": 0,
        "norm_fwd_pkt_iat_mean": 0,
        "norm_fwd_pkt_iat_std": 0,
        "norm_bwd_pkt_iat_mean": 0,
        "norm_bwd_pkt_iat_std": 0,
    }

    if row["ppi_pkt_times"] == "[]":
        times = []
    else:
        times = row["ppi_pkt_times"].strip("[]").split("|")

    times = [datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f") for x in times]

    fwd_times = [times[i] for i in row["fwd"]]
    bwd_times = [times[i] for i in row["bwd"]]

    # skip handshake
    if contains_handshake(row["ppi_pkt_flags"]):
        times = times[3:]
        fwd_times = fwd_times[2:]
        bwd_times = bwd_times[1:]

    packets_iat = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    forward_iat = [(b - a).total_seconds() for a, b in zip(fwd_times, fwd_times[1:])]
    backward_iat = [(b - a).total_seconds() for a, b in zip(bwd_times, bwd_times[1:])]

    # normalized inter arrival times (0 = short, 1 = long)
    norm_packets_iat = [1 if x > 5.0 else 0 for x in packets_iat]
    norm_forward_iat = [1 if x > 5.0 else 0 for x in forward_iat]
    norm_backward_iat = [1 if x > 5.0 else 0 for x in backward_iat]

    if packets_iat:
        stats["pkt_iat_min"] = min(packets_iat)
        stats["pkt_iat_max"] = max(packets_iat)
        stats["pkt_iat_mean"] = statistics.mean(packets_iat)
        stats["pkt_iat_std"] = statistics.pstdev(packets_iat)
        stats["norm_pkt_iat_mean"] = statistics.mean(norm_packets_iat)
        stats["norm_pkt_iat_std"] = statistics.pstdev(norm_packets_iat)

    if forward_iat:
        stats["fwd_pkt_iat_min"] = min(forward_iat)
        stats["fwd_pkt_iat_max"] = max(forward_iat)
        stats["fwd_pkt_iat_mean"] = statistics.mean(forward_iat)
        stats["fwd_pkt_iat_std"] = statistics.pstdev(forward_iat)
        stats["norm_fwd_pkt_iat_mean"] = statistics.mean(norm_forward_iat)
        stats["norm_fwd_pkt_iat_std"] = statistics.pstdev(norm_forward_iat)

    if backward_iat:
        stats["bwd_pkt_iat_min"] = min(backward_iat)
        stats["bwd_pkt_iat_max"] = max(backward_iat)
        stats["bwd_pkt_iat_mean"] = statistics.mean(backward_iat)
        stats["bwd_pkt_iat_std"] = statistics.pstdev(backward_iat)
        stats["norm_bwd_pkt_iat_mean"] = statistics.mean(norm_backward_iat)
        stats["norm_bwd_pkt_iat_std"] = statistics.pstdev(norm_backward_iat)

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


def prep_convert(df):
    """Applies conversions of default pstats columns.

    Args:
        df (pandas.DataFrame): Dataframe with basic and pstats values.
    """
    df["time_first"] = pd.to_datetime(df["time_first"])
    df["time_last"] = pd.to_datetime(df["time_last"])
    df["duration"] = (df["time_last"] - df["time_first"]).dt.total_seconds()

    df["ppi_pkt_lengths"] = df["ppi_pkt_lengths"].map(convert_lengths)

    df["ppi_pkt_flags"] = df["ppi_pkt_flags"].map(convert_flags)

    df[["ppi_pkt_directions", "fwd", "bwd"]] = pd.DataFrame(
        df["ppi_pkt_directions"].apply(convert_directions).tolist(),
        index=df.index,
        columns=["ppi_pkt_directions", "fwd", "bwd"],
    )


def extract_features(df, inplace=False, min_packets=2):
    """Extracts per flow statistics.

    Args:
        df (pandas.DataFrame): Dataframe with basic and pstats values.
        inplace (bool, optional): Extract features within provided DataFrame
            or return new DataFrame. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame is returned only if inplace=False,
        otherwise returns None.
    """
    if not inplace:
        df = df.copy()

    df.columns = df.columns.str.lower()

    df.drop(df[df["packets"] + df["packets_rev"] < min_packets].index, inplace=True)

    prep_convert(df)

    df["bytes_rate"] = df["bytes"] / df["duration"]
    df["bytes_rev_rate"] = df["bytes_rev"] / df["duration"]
    df["bytes_total_rate"] = (df["bytes"] + df["bytes_rev"]) / df["duration"]

    df["packets_rate"] = df["packets"] / df["duration"]
    df["packets_rev_rate"] = df["packets_rev"] / df["duration"]
    df["packets_total_rate"] = (df["packets"] + df["packets_rev"]) / df["duration"]

    flow_stats = df.apply(loop_flow_stats, axis=1, result_type="expand")
    df[flow_stats.columns] = flow_stats

    if not inplace:
        return df


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


def concatenate_ppi(fields):
    """Concatenate per packet information lists.

    Args:
        fields (list): List of string representations from ppi_pkt_* field.

    Returns:
        string: Concatenated representation.
    """
    return "[" + "|".join([x.strip("[]") for x in fields if x != "[]"]) + "]"


def aggregate(df, window="5min"):
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
