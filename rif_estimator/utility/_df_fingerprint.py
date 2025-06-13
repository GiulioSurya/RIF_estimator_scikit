"""
Module for creating and comparing pandas DataFrame fingerprints.

This module provides a utility class for creating unique identifiers of DataFrames
that allow for quick comparison of DataFrame structure and content.
It is particularly useful for implementing caching mechanisms based on
structural data identity.

Author: Giulio Surya Lo Verde
Date: 13/06/2025
Version: 1.0
"""

import pandas as pd


class DataFrameFingerprint:
    """
    Class for creating and comparing DataFrame fingerprints.

    This class creates a fingerprint of a pandas DataFrame by capturing its
    key structural characteristics: shape, column names, index, and data types.
    It allows for quick comparison of two DataFrames to verify if they have
    the same structure and/or content.

    The fingerprint is primarily used for:
    - Implementing efficient caching mechanisms
    - Detecting when a DataFrame has been modified
    - Verifying structural compatibility between DataFrames

    Attributes
    ----------
    shape : tuple
        DataFrame dimensions (rows, columns)
    columns : list
        List of column names
    index : list or int
        List of indices if DataFrame has less than 10000 rows,
        otherwise a hash of the index for efficiency
    dtypes : dict
        Dictionary mapping column names to their data types

    Examples
    --------
    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df2 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> fp1 = DataFrameFingerprint(df1)
    >>> fp2 = DataFrameFingerprint(df2)
    >>> fp1 == fp2  # True - same structure and content
    True

    >>> df3 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
    >>> fp3 = DataFrameFingerprint(df3)
    >>> fp1.matches_structure_only(fp3)  # True - same structure, different content
    True
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize a DataFrame fingerprint.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to create a fingerprint for
        """
        # Store basic structural information
        self.shape = df.shape
        self.columns = df.columns.tolist()

        # For large DataFrames, store hash instead of full index to save memory
        # Threshold of 10000 rows is a reasonable compromise between accuracy and efficiency
        self.index = df.index.tolist() if len(df.index) < 10000 else hash(tuple(df.index.tolist()))

        # Store data types for each column
        self.dtypes = df.dtypes.to_dict()

    def __eq__(self, other: 'DataFrameFingerprint') -> bool:
        """
        Compare two fingerprints for complete equality.

        Two fingerprints are considered equal if they have the same shape,
        columns, index, and data types. This indicates that the DataFrames
        they represent are structurally identical and likely contain the
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
                self.dtypes == other.dtypes
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
        return hash((
            self.shape,
            tuple(self.columns),
            tuple(self.index) if isinstance(self.index, list) else self.index,
            tuple(sorted(self.dtypes.items()))
        ))

    def matches_structure_only(self, other: 'DataFrameFingerprint') -> bool:
        """
        Compare only structural properties (not index/content).

        This method checks if two DataFrames have the same structure
        (shape, columns, and data types) regardless of their actual content
        or index values. Useful for detecting when a DataFrame has been
        modified in ways that preserve its structure but change its content.

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
        scenarios where the same DataFrame structure is used but with
        modified content (e.g., after reset_index() or other transformations).
        """
        return (
                self.shape == other.shape and
                self.columns == other.columns and
                self.dtypes == other.dtypes
        )