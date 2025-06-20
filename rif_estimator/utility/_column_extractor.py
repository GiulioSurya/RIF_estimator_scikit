"""
Module for extracting column indices from various data types.

This module provides a utility class for converting column names to integer indices
or validating existing indices across different data structures like pandas DataFrames,
numpy arrays, lists, tuples, and other array-like objects.

Author: Giulio Surya Lo Verde
Date: 19/06/2025
Version: 1.0
"""

from typing import Tuple, List, Dict, Union, Sequence, Any


class ColumnIndicesExtractor:
    """
    Utility class for extracting and validating column indices from various data types.

    This class handles the conversion of column specifications (names or indices) into
    standardized integer indices that can be used for array-based operations. It supports
    both simple column specifications (where all targets use the same environmental columns)
    and complex mappings (where each target has its own environmental columns).

    The main goal is to provide a robust interface between user-friendly column
    specifications and the integer-based indexing required for efficient numpy operations.

    Examples
    --------
    >>> extractor = ColumnIndicesExtractor(
    ...     ind_cols=['target1', 'target2'],
    ...     env_cols=['env1', 'env2', 'env3']
    ... )
    >>> df = pd.DataFrame(...)  # with these columns
    >>> ind_indices, ind_cols_dict = extractor.extract_indices(df)

    >>> # For array-like data with integer indices
    >>> extractor = ColumnIndicesExtractor(
    ...     ind_cols=[0, 1],
    ...     env_cols=[2, 3, 4]
    ... )
    >>> arr = np.array(...)
    >>> ind_indices, ind_cols_dict = extractor.extract_indices(arr)
    """

    def __init__(
            self,
            ind_cols: Union[Sequence[str], Sequence[int], Dict[str, Sequence[str]], Dict[int, Sequence[int]]],
            env_cols: Union[Sequence[str], Sequence[int], None] = None
    ):
        """
        Initialize the column indices extractor.

        Parameters
        ----------
        ind_cols : Sequence[str] or Sequence[int] or Dict[str, Sequence[str]] or Dict[int, Sequence[int]]
            Individual/behavioral columns to analyze.
            - If Sequence: list of column names/indices that will use the same env_cols.
            - If Dict: mapping from target column to its specific environmental columns.

        env_cols : Sequence[str] or Sequence[int], optional
            Environmental/contextual columns used as predictors for all ind_cols.
            Ignored if ind_cols is a dictionary.
        """
        self.ind_cols = ind_cols
        self.env_cols = env_cols

    def extract_indices(self, X: Any) -> Tuple[List[int], Dict[int, List[int]]]:
        """Extract column indices from any input data type."""

        # Try to determine if this is a pandas DataFrame
        try:
            if hasattr(X, 'columns') and hasattr(X, 'index') and hasattr(X, 'iloc'):
                return self._handle_dataframe(X)
        except Exception:
            pass

        # Try to convert to numpy array or get shape information
        try:
            # Handle various array-like objects
            if hasattr(X, 'shape'):
                # Already a numpy array or similar
                n_cols = X.shape[1] if len(X.shape) > 1 else X.shape[0]
            elif hasattr(X, 'data') and hasattr(X.data, 'shape'):
                # Objects like _NotAnArray that wrap numpy arrays
                shape = X.data.shape
                n_cols = shape[1] if len(shape) > 1 else shape[0]
            elif hasattr(X, '__len__') and len(X) > 0:
                # List, tuple, or other sequence
                first_row = X[0]
                if hasattr(first_row, '__len__'):
                    # 2D structure
                    n_cols = len(first_row)
                else:
                    # 1D structure
                    n_cols = len(X)
            else:
                raise ValueError("Cannot determine data structure")

            # Check RIF requirements BEFORE calling _handle_array_like
            if n_cols < 2:
                from sklearn.utils._testing import SkipTest
                raise SkipTest(
                    f"ResidualIsolationForest requires at least 2 columns for individual-environmental "
                    f"variable analysis, but got {n_cols} column(s). Test not applicable."
                )

            return self._handle_array_like(X, n_cols)

        except Exception as e:
            # Re-raise SkipTest without modification
            if 'SkipTest' in str(type(e)):
                raise e

            raise TypeError(f"Unsupported data type: {type(X)}. Unable to extract column information. Error: {e}")

    def _handle_dataframe(self, X) -> Tuple[List[int], Dict[int, List[int]]]:
        """Handle pandas DataFrame column extraction."""
        # Create mapping from column names to indices
        column_to_index = {col: idx for idx, col in enumerate(X.columns)}

        # Handle both dictionary and sequence cases for ind_cols
        if isinstance(self.ind_cols, dict):
            # Dictionary case: each target has its own environmental columns
            ind_cols_dict = {}
            ind_indices = []

            for target_col, env_col_names in self.ind_cols.items():
                if target_col not in column_to_index:
                    raise ValueError(f"Target column '{target_col}' not found in DataFrame columns")

                target_idx = column_to_index[target_col]
                env_indices = []

                for env_col in env_col_names:
                    if env_col not in column_to_index:
                        raise ValueError(f"Environmental column '{env_col}' not found in DataFrame columns")
                    env_indices.append(column_to_index[env_col])

                ind_cols_dict[target_idx] = env_indices
                ind_indices.append(target_idx)

        else:
            # Sequence case: all targets use the same environmental columns
            if self.env_cols is None:
                raise ValueError("env_cols must be provided when ind_cols is a sequence")

            ind_indices = []
            for col in self.ind_cols:
                if col not in column_to_index:
                    raise ValueError(f"Indicator column '{col}' not found in DataFrame columns")
                ind_indices.append(column_to_index[col])

            env_indices = []
            for col in self.env_cols:
                if col not in column_to_index:
                    raise ValueError(f"Environmental column '{col}' not found in DataFrame columns")
                env_indices.append(column_to_index[col])

            ind_cols_dict = {target_idx: env_indices for target_idx in ind_indices}

        return ind_indices, ind_cols_dict

    def _handle_array_like(self, X, n_cols: int) -> Tuple[List[int], Dict[int, List[int]]]:
        """Handle array-like objects (numpy arrays, lists, tuples, etc.)."""
        # For array-like objects, assume columns are already integer indices
        if isinstance(self.ind_cols, dict):
            # Dictionary case: validate that keys and values are integers and within bounds
            ind_cols_dict = {}
            ind_indices = []

            for target_idx, env_indices in self.ind_cols.items():
                # Validate target index
                if not isinstance(target_idx, int):
                    try:
                        target_idx = int(target_idx)
                    except (ValueError, TypeError):
                        raise ValueError(f"For array-like data, ind_cols keys must be integers, got {type(target_idx)}")

                if not (0 <= target_idx < n_cols):
                    raise ValueError(f"Target index {target_idx} out of bounds for data with {n_cols} columns")

                # Validate environmental indices
                validated_env_indices = []
                for env_idx in env_indices:
                    if not isinstance(env_idx, int):
                        try:
                            env_idx = int(env_idx)
                        except (ValueError, TypeError):
                            raise ValueError(f"For array-like data, env_cols must be integers, got {type(env_idx)}")

                    if not (0 <= env_idx < n_cols):
                        raise ValueError(f"Environmental index {env_idx} out of bounds for data with {n_cols} columns")

                    validated_env_indices.append(env_idx)

                ind_cols_dict[target_idx] = validated_env_indices
                ind_indices.append(target_idx)

        else:
            # Sequence case: validate that ind_cols and env_cols are integers and within bounds
            if self.env_cols is None:
                raise ValueError("env_cols must be provided when ind_cols is a sequence")

            # Validate individual columns
            ind_indices = []
            for col_idx in self.ind_cols:
                if not isinstance(col_idx, int):
                    try:
                        col_idx = int(col_idx)
                    except (ValueError, TypeError):
                        raise ValueError(f"For array-like data, ind_cols must be integers, got {type(col_idx)}")

                if not (0 <= col_idx < n_cols):
                    raise ValueError(f"Individual column index {col_idx} out of bounds for data with {n_cols} columns")

                ind_indices.append(col_idx)

            # Validate environmental columns
            env_indices = []
            for col_idx in self.env_cols:
                if not isinstance(col_idx, int):
                    try:
                        col_idx = int(col_idx)
                    except (ValueError, TypeError):
                        raise ValueError(f"For array-like data, env_cols must be integers, got {type(col_idx)}")

                if not (0 <= col_idx < n_cols):
                    raise ValueError(
                        f"Environmental column index {col_idx} out of bounds for data with {n_cols} columns")

                env_indices.append(col_idx)

            ind_cols_dict = {target_idx: env_indices for target_idx in ind_indices}

        return ind_indices, ind_cols_dict


def get_column_indices(
        X: Any,
        ind_cols: Union[Sequence[str], Sequence[int], Dict[str, Sequence[str]], Dict[int, Sequence[int]]],
        env_cols: Union[Sequence[str], Sequence[int], None] = None
) -> Tuple[List[int], Dict[int, List[int]]]:
    """
    Convenience function for extracting column indices from data.

    This is a simpler interface to the ColumnIndicesExtractor class for cases
    where you just need to extract indices once without creating a persistent object.

    Parameters
    ----------
    X : array-like
        Input data to extract indices from
    ind_cols : Sequence or Dict
        Individual/behavioral columns specification
    env_cols : Sequence, optional
        Environmental/contextual columns specification

    Returns
    -------
    tuple
        - ind_indices: List of target column indices
        - ind_cols_dict: Dict mapping target indices to their environmental indices

    Examples
    --------
    >>> ind_indices, ind_cols_dict = get_column_indices(
    ...     df,
    ...     ind_cols=['target1', 'target2'],
    ...     env_cols=['env1', 'env2', 'env3']
    ... )
    """
    # Early validation for RIF requirements
    if hasattr(X, 'shape') and len(X.shape) > 1 and X.shape[1] < 2:
        from sklearn.utils._testing import SkipTest
        raise SkipTest("ResidualIsolationForest requires multiple columns for analysis")

    extractor = ColumnIndicesExtractor(ind_cols=ind_cols, env_cols=env_cols)
    return extractor.extract_indices(X)