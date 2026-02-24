"""Oracle utilities for TFBind8 workflow."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


def build_oracle_one_hot_df(
    test_df: pd.DataFrame,
    seq_cols: Iterable[str] = ("0", "1", "2", "3", "4", "5", "6", "7"),
    score_col: str = "binding_scores",
    flatten: bool = True,
) -> pd.DataFrame:
    """Create oracle DataFrame with one-hot encoded sequences.

    Args:
        test_df: Input dataframe containing sequence columns and score column.
        seq_cols: Column names for sequence positions.
        score_col: Column name for binding scores.
        flatten: If True, store flattened one-hot vectors.

    Returns:
        DataFrame with columns: score_col, one_hot_seq
    """
    df_oracle = test_df[list(seq_cols) + [score_col]].copy()
    x_oracle = df_oracle.iloc[:, :-1].values

    if flatten:
        df_oracle["one_hot_seq"] = [
            tf.one_hot(seq, depth=4).numpy().flatten() for seq in x_oracle
        ]
    else:
        df_oracle["one_hot_seq"] = [tf.one_hot(seq, depth=4).numpy() for seq in x_oracle]

    df_oracle.drop(columns=df_oracle.columns[:-2], inplace=True)
    return df_oracle

def oracle_lookup_one_hot(df_oracle: pd.DataFrame, one_hot_sequence: np.ndarray) -> float:
    """Looks up the real binding score for a given one-hot encoded sequence."""
    flattened_seq = np.asarray(one_hot_sequence).flatten()
    match = df_oracle[df_oracle["one_hot_seq"].apply(lambda x: np.array_equal(x, flattened_seq))]
    if not match.empty:
        return float(match["binding_scores"].values[0])
    return 0.0


def oracle_exists_one_hot(df_oracle: pd.DataFrame, one_hot_sequence: np.ndarray) -> bool:
    """Checks if a given one-hot encoded sequence exists in the oracle dataframe."""
    flattened_seq = np.asarray(one_hot_sequence).flatten()
    match = df_oracle[df_oracle["one_hot_seq"].apply(lambda x: np.array_equal(x, flattened_seq))]
    return not match.empty



########################Oracle with encoded sequences (not one-hot)########################

def build_oracle_df(test_df: pd.DataFrame,
                    seq_cols: Iterable[str] = ("0", "1", "2", "3", "4", "5", "6", "7"),
                    score_col: str = "binding_scores") -> pd.DataFrame:
    """Create oracle DataFrame with number encoded sequences.
       Put therefore the numbers from the columns into a list and stor it in new colum encoded_sequence
    Args:
        test_df: Input dataframe containing sequence columns and score column.
        seq_cols: Column names for sequence positions.
        score_col: Column name for binding scores."""
    df_oracle = test_df[list(seq_cols) + [score_col]].copy()
    x_oracle = df_oracle.iloc[:, :-1].values
    df_oracle["encoded_sequence"] = [list(seq) for seq in x_oracle]
    df_oracle.drop(columns=df_oracle.columns[:-2], inplace=True)
    return df_oracle

def oracle_lookup(df_oracle: pd.DataFrame, sequence) -> float:
    """Looks up the real binding score for a given sequence."""
    # Convert numpy array to list for comparison
    if isinstance(sequence, np.ndarray):
        sequence = sequence.flatten().tolist()
    elif not isinstance(sequence, list):
        sequence = list(sequence)
    
    match = df_oracle[df_oracle["encoded_sequence"].apply(lambda x: list(x) == sequence)]
    if not match.empty:
        return float(match["binding_scores"].values[0])
    return 0.0

def oracle_exists(df_oracle: pd.DataFrame, sequence) -> bool:
    """Checks if a given sequence exists in the oracle dataframe."""
    # Convert numpy array to list for comparison
    if isinstance(sequence, np.ndarray):
        sequence = sequence.flatten().tolist()
    elif not isinstance(sequence, list):
        sequence = list(sequence)
    
    match = df_oracle[df_oracle["encoded_sequence"].apply(lambda x: list(x) == sequence)]
    return not match.empty



def oracle_score_range(df_oracle: pd.DataFrame, score_col: str = "binding_scores") -> Tuple[float, float]:
    """Return min/max score for the oracle."""
    return float(df_oracle[score_col].min()), float(df_oracle[score_col].max())
