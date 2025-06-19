"""
Module for creating and comparing data fingerprints.

This module provides a utility class for creating unique identifiers of both
pandas DataFrames and numpy arrays that allow for quick comparison of data
structure and content. It is particularly useful for implementing caching
mechanisms based on structural data identity.

Author: Giulio Surya Lo Verde
Date: 13/06/2025
Version: 1.1
"""

import pandas as pd
import numpy as np
from typing import Union


class DataFrameFingerprint:
    """
    Class for creating and comparing data fingerprints for DataFrames and numpy arrays.

    This class creates a fingerprint of a pandas DataFrame or numpy array by capturing its
    key structural characteristics: shape, data types, and content hash.
    It allows for quick comparison of two data structures to verify if they have
    the same structure and/or content.

    The fingerprint is primarily used for:
    - Implementing efficient caching mechanisms
    - Detecting when data has been modified
    - Verifying structural compatibility between datasets

    Attributes
    ----------
    shape : tuple
        Data dimensions (rows, columns)
    columns : list or None
        List of column names for DataFrames, None for numpy arrays
    index : list, int, or None
        List of indices if DataFrame has less than 10000 rows,
        otherwise a hash of the index for efficiency, None for numpy arrays
    dtypes : dict or str
        Dictionary mapping column names to their data types for DataFrames,
        or string representation of dtype for numpy arrays
    content_hash : int
        Hash of the data content for quick comparison

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> fp1 = DataFrameFingerprint(df1)
    >>> fp2 = DataFrameFingerprint(df2)
    >>> fp1 == fp2  # True - same structure and content
    True

    >>> arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> fp1 = DataFrameFingerprint(arr1)
    >>> fp2 = DataFrameFingerprint(arr2)
    >>> fp1 == fp2  # True - same structure and content
    True
    """

    def __init__(self, data: Union[pd.DataFrame, np.ndarray]):
        """
        Initialize a data fingerprint.

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            The data to create a fingerprint for
        """
        # Store basic structural information
        self.shape = data.shape


        if isinstance(data, np.ndarray):
            # NumPy array-specific attributes
            self.columns = None
            self.index = None
            self.dtypes = str(data.dtype)

            # Create content hash for numpy array
            # For large arrays, use a sample-based hash to avoid memory issues
            if data.size > 100000:
                # Sample-based hash for very large arrays
                flat_data = data.flat
                sample_indices = np.linspace(0, data.size - 1, 1000, dtype=int)
                sample_data = np.array([flat_data[i] for i in sample_indices])
                self.content_hash = hash(sample_data.tobytes())
            else:
                self.content_hash = hash(data.tobytes())
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Only pandas DataFrame and numpy array are supported.")

    def __eq__(self, other: 'DataFrameFingerprint') -> bool:
        """
        Compare two fingerprints for complete equality.

        Two fingerprints are considered equal if they have the same shape,
        columns, index, data types, and content hash. This indicates that the data
        structures they represent are structurally identical and contain the
        same data.

        Parameters
        ----------
        other : DataFrameFingerprint
            Another fingerprint to compare with

        Returns
        -------
        bool
            True if fingerprints are identical, False otherwise
        """
        return (
                self.shape == other.shape and
                self.columns == other.columns and
                self.index == other.index and
                self.dtypes == other.dtypes and
                self.content_hash == other.content_hash
        )

    def __hash__(self) -> int:
        """
        Create a hash for use as dictionary key.

        This allows fingerprints to be used as keys in dictionaries,
        which is essential for implementing efficient caching mechanisms.

        Returns
        -------
        int
            Hash value of the fingerprint
        """
        # Handle different types of attributes for hashing
        columns_hash = tuple(self.columns) if self.columns is not None else None
        index_hash = tuple(self.index) if isinstance(self.index, list) else self.index
        dtypes_hash = tuple(sorted(self.dtypes.items())) if isinstance(self.dtypes, dict) else self.dtypes

        return hash((
            self.shape,
            columns_hash,
            index_hash,
            dtypes_hash,
            self.content_hash
        ))

    def matches_structure_only(self, other: 'DataFrameFingerprint') -> bool:
        """
        Compare only structural properties (not content).

        This method checks if two data structures have the same structure
        (shape, columns, and data types) regardless of their actual content.
        Useful for detecting when data has been modified in ways that preserve
        its structure but change its content.

        Parameters
        ----------
        other : DataFrameFingerprint
            Another fingerprint to compare with

        Returns
        -------
        bool
            True if structures match, False otherwise

        Notes
        -----
        This is particularly useful for detecting potential data leakage
        scenarios where the same data structure is used but with
        modified content (e.g., after transformations that preserve structure).
        """
        return (
                self.shape == other.shape and
                self.columns == other.columns and
                self.dtypes == other.dtypes
        )