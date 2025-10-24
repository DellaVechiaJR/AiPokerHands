
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@dataclass
class EncodedDataset:

    features: np.ndarray
    labels: Optional[np.ndarray]
    feature_columns: Iterable[str]
    encoders: Dict[str, LabelEncoder]
    label_encoder: Optional[LabelEncoder]

def _encode_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:

    encoders: Dict[str, LabelEncoder] = {}
    encoded = frame.copy()
    for column in encoded.columns:
        if encoded[column].dtype == object:
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))
            encoders[column] = encoder
    return encoded, encoders

def load_poker_dataset(
    csv_path: str | Path,
    *,
    label_column: Optional[str] = None,
    drop_columns: Optional[Iterable[str]] = None,
) -> EncodedDataset:

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing the dataset.
    label_column:
        Optional name of the column that represents the poker hand class. When
        provided, the column is removed from the feature matrix and returned
        separately as ``labels``.
    drop_columns:
        Optional iterable with columns that should be removed prior to
        preprocessing.

    Returns
    -------
    EncodedDataset
        Dataclass containing the encoded feature matrix, optional labels,
        encoders, and metadata about the transformation.

    csv_path = Path(csv_path)
    data_frame = pd.read_csv(csv_path)

    if drop_columns:
        data_frame = data_frame.drop(columns=list(drop_columns), errors="ignore")

    label_encoder: Optional[LabelEncoder] = None
    labels: Optional[np.ndarray] = None

    if label_column and label_column in data_frame.columns:
        labels_series = data_frame.pop(label_column)
        if labels_series.dtype == object:
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels_series.astype(str))
        else:
            labels = labels_series.to_numpy()
    elif label_column:
        raise ValueError(f"Column '{label_column}' not found in dataset {csv_path}.")

    encoded_features, encoders = _encode_features(data_frame)

    return EncodedDataset(
        features=encoded_features.to_numpy(dtype=float),
        labels=labels,
        feature_columns=tuple(encoded_features.columns),
        encoders=encoders,
        label_encoder=label_encoder,
    )

def split_dataset(
    csv_path: str | Path,
    *,
    train_size: float = 0.7,
    random_state: int = 42,
    stratify_column: Optional[str] = None,
    output_train_path: Optional[str | Path] = None,
    output_test_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    Parameters
    ----------
    csv_path:
        Path to the CSV file containing the dataset.
    train_size:
        Proportion of data that should belong to the training split. The
        remainder of the dataset is used as test data.
    random_state:
        Seed used for the random shuffling performed during the split.
    stratify_column:
        Optional name of a column that should be used to stratify the split.
        This is useful for preserving class balance when a label column is
        available.
    output_train_path / output_test_path:
        Optional paths where the resulting CSV files should be stored.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        DataFrames containing the training and test partitions respectively.

    csv_path = Path(csv_path)
    data_frame = pd.read_csv(csv_path)

    stratify_data = None
    if stratify_column:
        if stratify_column not in data_frame.columns:
            raise ValueError(
                f"Column '{stratify_column}' not found in dataset {csv_path}."
            )
        stratify_data = data_frame[stratify_column]

    train_df, test_df = train_test_split(
        data_frame,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_data,
    )

    if output_train_path:
        Path(output_train_path).parent.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_train_path, index=False)
    if output_test_path:
        Path(output_test_path).parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(output_test_path, index=False)

    return train_df, test_df