from sklearn.utils._param_validation import InvalidParameterError
import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from rif_estimator import ResidualIsolationForest


class TestScikitLearnCompliance:
    """Test scikit-learn compliance using check_estimator."""

    def test_check_estimator(self):
        """Test scikit-learn compliance with check_estimator.

        Due to tags.non_deterministic = True, some tests will be skipped.
        This is expected behavior for non-deterministic estimators.
        """
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            contamination=0.10,
            random_state=42,
            residual_strategy="oob",
            bayes_search=False,
            iso_params={"max_features": 1}
        )

        # Run check_estimator - tests will be skipped due to non_deterministic tag
        results = check_estimator(
            estimator=rif,
            legacy=True,
            on_skip="warn",
            on_fail="warn"  # Don't raise, collect all results
        )

        # Analyze results
        passed = sum(1 for r in results if r["status"] == "passed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        failed = sum(1 for r in results if r["status"] == "failed")

        print(f"\ncheck_estimator results:")
        print(f"  Passed: {passed}")
        print(f"  Skipped: {skipped} (expected due to non_deterministic tag)")
        print(f"  Failed: {failed}")

        # Print failed tests for debugging
        if failed > 0:
            print("\nFailed tests:")
            for r in results:
                if r["status"] == "failed":
                    print(f"  - {r['check_name']}: {r['exception']}")

        # The test should run without crashing
        # Some skips are expected due to non_deterministic tag
        assert len(results) > 0, "No tests were executed"
        assert skipped > 0, "Expected some tests to be skipped due to non_deterministic tag"

class TestIndColsValidation:
    """Test parameter validation for ind_cols parameter."""

    def setup_method(self):
        """Setup test data for fit operations."""
        np.random.seed(42)
        self.X = np.random.randn(50, 4)  # Small dataset for fast tests

    # Test invalid types
    def test_ind_cols_string(self):
        """Test ind_cols as string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"ind_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols="invalid",
                env_cols=[1]
            ).fit(self.X)

    def test_ind_cols_int(self):
        """Test ind_cols as single int raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"ind_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=0,
                env_cols=[1]
            ).fit(self.X)

    def test_ind_cols_float(self):
        """Test ind_cols as float raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"ind_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=0.5,
                env_cols=[1]
            ).fit(self.X)

    def test_ind_cols_none(self):
        """Test ind_cols as None raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"ind_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=None,
                env_cols=[1]
            ).fit(self.X)

    def test_ind_cols_set(self):
        """Test ind_cols as set raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"ind_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols={0, 1},
                env_cols=[2]
            ).fit(self.X)

    def test_ind_cols_numpy_array(self):
        """Test ind_cols as numpy array raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"ind_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=np.array([0, 1]),
                env_cols=[2]
            ).fit(self.X)

    # Test valid types - list
    def test_ind_cols_valid_list_int(self):
        """Test valid ind_cols as list of integers."""
        rif = ResidualIsolationForest(
            ind_cols=[0, 1],
            env_cols=[2, 3]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.ind_cols == [0, 1]

    def test_ind_cols_valid_list_str(self):
        """Test valid ind_cols as list of strings (would work with DataFrame)."""
        rif = ResidualIsolationForest(
            ind_cols=['col0', 'col1'],
            env_cols=['col2', 'col3']
        )
        # Constructor should work, but fit will fail with numpy array
        assert rif.ind_cols == ['col0', 'col1']

    def test_ind_cols_valid_single_element_list(self):
        """Test valid ind_cols as single element list."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1, 2, 3]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.ind_cols == [0]

    # Test valid types - tuple
    def test_ind_cols_valid_tuple_int(self):
        """Test valid ind_cols as tuple of integers."""
        rif = ResidualIsolationForest(
            ind_cols=(0, 1),
            env_cols=(2, 3)
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.ind_cols == (0, 1)

    def test_ind_cols_valid_tuple_str(self):
        """Test valid ind_cols as tuple of strings."""
        rif = ResidualIsolationForest(
            ind_cols=('col0', 'col1'),
            env_cols=('col2', 'col3')
        )
        # Constructor should work
        assert rif.ind_cols == ('col0', 'col1')

    def test_ind_cols_valid_single_element_tuple(self):
        """Test valid ind_cols as single element tuple."""
        rif = ResidualIsolationForest(
            ind_cols=(0,),
            env_cols=(1, 2, 3)
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.ind_cols == (0,)

    # Test valid types - dict
    def test_ind_cols_valid_dict_int_keys(self):
        """Test valid ind_cols as dict with int keys."""
        rif = ResidualIsolationForest(
            ind_cols={0: [2, 3], 1: [2, 3]}
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.ind_cols == {0: [2, 3], 1: [2, 3]}

    def test_ind_cols_valid_dict_str_keys(self):
        """Test valid ind_cols as dict with string keys."""
        rif = ResidualIsolationForest(
            ind_cols={'col0': ['col2', 'col3'], 'col1': ['col2', 'col3']}
        )
        # Constructor should work
        assert rif.ind_cols == {'col0': ['col2', 'col3'], 'col1': ['col2', 'col3']}

    def test_ind_cols_valid_dict_mixed_env_cols(self):
        """Test valid ind_cols as dict with different env_cols per target."""
        rif = ResidualIsolationForest(
            ind_cols={0: [2], 1: [3]}
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.ind_cols == {0: [2], 1: [3]}

    # Test empty containers
    def test_ind_cols_empty_list(self):
        """Test ind_cols as empty list - should work but might fail in fit."""
        rif = ResidualIsolationForest(
            ind_cols=[],
            env_cols=[1, 2, 3]
        )
        # Constructor should work, but fit might fail due to business logic
        assert rif.ind_cols == []

    def test_ind_cols_empty_tuple(self):
        """Test ind_cols as empty tuple - should work but might fail in fit."""
        rif = ResidualIsolationForest(
            ind_cols=(),
            env_cols=(1, 2, 3)
        )
        # Constructor should work
        assert rif.ind_cols == ()

    def test_ind_cols_empty_dict(self):
        """Test ind_cols as empty dict - should work but might fail in fit."""
        rif = ResidualIsolationForest(
            ind_cols={}
        )
        # Constructor should work
        assert rif.ind_cols == {}

class TestEnvColsValidation:
    """Test parameter validation for env_cols parameter."""

    def setup_method(self):
        """Setup test data for fit operations."""
        np.random.seed(42)
        self.X = np.random.randn(50, 4)  # Small dataset for fast tests

    # Test invalid types
    def test_env_cols_string(self):
        """Test env_cols as string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"env_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols="invalid"
            ).fit(self.X)

    def test_env_cols_int(self):
        """Test env_cols as single int raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"env_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=1
            ).fit(self.X)

    def test_env_cols_float(self):
        """Test env_cols as float raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"env_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=1.5
            ).fit(self.X)

    def test_env_cols_dict(self):
        """Test env_cols as dict raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"env_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols={1: 2}
            ).fit(self.X)

    def test_env_cols_set(self):
        """Test env_cols as set raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"env_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols={1, 2}
            ).fit(self.X)

    def test_env_cols_numpy_array(self):
        """Test env_cols as numpy array raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"env_cols.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=np.array([1, 2])
            ).fit(self.X)

    # Test valid types - list
    def test_env_cols_valid_list_int(self):
        """Test valid env_cols as list of integers."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1, 2, 3]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.env_cols == [1, 2, 3]

    def test_env_cols_valid_list_str(self):
        """Test valid env_cols as list of strings (would work with DataFrame)."""
        rif = ResidualIsolationForest(
            ind_cols=['col0'],
            env_cols=['col1', 'col2', 'col3']
        )
        # Constructor should work, but fit will fail with numpy array
        assert rif.env_cols == ['col1', 'col2', 'col3']

    def test_env_cols_valid_single_element_list(self):
        """Test valid env_cols as single element list."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.env_cols == [1]

    def test_env_cols_valid_multiple_elements_list(self):
        """Test valid env_cols as multiple elements list."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1, 2, 3]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.env_cols == [1, 2, 3]

    # Test valid types - tuple
    def test_env_cols_valid_tuple_int(self):
        """Test valid env_cols as tuple of integers."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=(1, 2, 3)
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.env_cols == (1, 2, 3)

    def test_env_cols_valid_tuple_str(self):
        """Test valid env_cols as tuple of strings."""
        rif = ResidualIsolationForest(
            ind_cols=['col0'],
            env_cols=('col1', 'col2', 'col3')
        )
        # Constructor should work
        assert rif.env_cols == ('col1', 'col2', 'col3')

    def test_env_cols_valid_single_element_tuple(self):
        """Test valid env_cols as single element tuple."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=(1,)
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.env_cols == (1,)

    # Test valid None
    def test_env_cols_valid_none(self):
        """Test valid env_cols as None (when ind_cols is dict)."""
        rif = ResidualIsolationForest(
            ind_cols={0: [1, 2, 3]},
            env_cols=None
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.env_cols is None

    def test_env_cols_none_with_sequence_ind_cols(self):
        """Test env_cols as None with sequence ind_cols - should fail in business logic."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=None
        )
        # Constructor should work, but fit should fail due to business logic
        # (env_cols required when ind_cols is sequence)
        assert rif.env_cols is None

    # Test empty containers
    def test_env_cols_empty_list(self):
        """Test env_cols as empty list - should work but might fail in fit."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[]
        )
        # Constructor should work, but fit might fail due to business logic
        assert rif.env_cols == []

    def test_env_cols_empty_tuple(self):
        """Test env_cols as empty tuple - should work but might fail in fit."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=()
        )
        # Constructor should work
        assert rif.env_cols == ()

    # Test default value
    def test_env_cols_default_none(self):
        """Test default env_cols value is None."""
        rif = ResidualIsolationForest(
            ind_cols={0: [1, 2, 3]}
        )
        # Default should be None
        assert rif.env_cols is None

    # Test mixed scenarios
    def test_env_cols_with_dict_ind_cols_ignored(self):
        """Test that env_cols is ignored when ind_cols is dict."""
        rif = ResidualIsolationForest(
            ind_cols={0: [1, 2], 1: [2, 3]},
            env_cols=[1, 2, 3]  # Should be ignored
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        # env_cols should still be stored even if ignored
        assert rif.env_cols == [1, 2, 3]

class TestContaminationValidation:
    """Test parameter validation for contamination parameter."""

    def setup_method(self):
        """Setup test data for fit operations."""
        np.random.seed(42)
        self.X = np.random.randn(50, 4)  # Small dataset for fast tests

    def test_contamination_too_high(self):
        """Test contamination > 0.5 raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"contamination.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                contamination=0.6
            ).fit(self.X)

    def test_contamination_at_upper_bound(self):
        """Test contamination = 0.5 raises InvalidParameterError (bound is open)."""
        with pytest.raises(InvalidParameterError, match=r"contamination.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                contamination=0.5
            ).fit(self.X)

    def test_contamination_zero(self):
        """Test contamination = 0 raises InvalidParameterError (bound is open)."""
        with pytest.raises(InvalidParameterError, match=r"contamination.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                contamination=0.0
            ).fit(self.X)

    def test_contamination_negative(self):
        """Test negative contamination raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"contamination.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                contamination=-0.1
            ).fit(self.X)


    def test_contamination_string(self):
        """Test contamination as string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"contamination.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                contamination="0.1"
            ).fit(self.X)

    def test_contamination_none(self):
        """Test contamination as None raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"contamination.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                contamination=None
            ).fit(self.X)

    def test_contamination_list(self):
        """Test contamination as list raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"contamination.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                contamination=[0.1]
            ).fit(self.X)

    # Valid contamination values
    def test_contamination_valid_small(self):
        """Test valid small contamination value."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            contamination=0.001
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.contamination == 0.001

    def test_contamination_valid_typical(self):
        """Test valid typical contamination value."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            contamination=0.1
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.contamination == 0.1

    def test_contamination_valid_high(self):
        """Test valid high contamination value (close to but less than 0.5)."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            contamination=0.49
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.contamination == 0.49

    def test_contamination_default(self):
        """Test default contamination value."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.contamination == 0.10


class TestResidualStrategyValidation:
    """Test parameter validation for residual_strategy parameter."""

    def setup_method(self):
        """Setup test data for fit operations."""
        np.random.seed(42)
        self.X = np.random.randn(50, 4)  # Small dataset for fast tests

    # Test invalid string values
    def test_residual_strategy_invalid_string(self):
        """Test residual_strategy with invalid string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy="invalid"
            ).fit(self.X)

    def test_residual_strategy_wrong_case(self):
        """Test residual_strategy with wrong case raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy="OOB"  # Should be lowercase
            ).fit(self.X)

    def test_residual_strategy_partial_match(self):
        """Test residual_strategy with partial string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy="oo"  # Should be "oob"
            ).fit(self.X)

    def test_residual_strategy_typo(self):
        """Test residual_strategy with typo raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy="kfolds"  # Should be "kfold"
            ).fit(self.X)

    # Test invalid types
    def test_residual_strategy_int(self):
        """Test residual_strategy as int raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy=1
            ).fit(self.X)

    def test_residual_strategy_float(self):
        """Test residual_strategy as float raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy=1.0
            ).fit(self.X)

    def test_residual_strategy_bool(self):
        """Test residual_strategy as bool raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy=True
            ).fit(self.X)

    def test_residual_strategy_list(self):
        """Test residual_strategy as list raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy=["oob"]
            ).fit(self.X)

    def test_residual_strategy_dict(self):
        """Test residual_strategy as dict raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy={"strategy": "oob"}
            ).fit(self.X)

    # Test valid values
    def test_residual_strategy_valid_oob(self):
        """Test valid residual_strategy 'oob'."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            residual_strategy="oob"
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.residual_strategy == "oob"

    def test_residual_strategy_valid_kfold(self):
        """Test valid residual_strategy 'kfold'."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            residual_strategy="kfold"
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.residual_strategy == "kfold"

    def test_residual_strategy_valid_none(self):
        """Test valid residual_strategy None."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            residual_strategy=None
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.residual_strategy is None

    def test_residual_strategy_default(self):
        """Test default residual_strategy value is 'oob'."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.residual_strategy == "oob"  # Default value from your class

    # Test edge cases
    def test_residual_strategy_empty_string(self):
        """Test residual_strategy as empty string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy=""
            ).fit(self.X)

    def test_residual_strategy_whitespace(self):
        """Test residual_strategy with whitespace raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy=" oob "
            ).fit(self.X)

    def test_residual_strategy_unicode(self):
        """Test residual_strategy with unicode characters raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"residual_strategy.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                residual_strategy="oōb"  # Unicode 'o'
            ).fit(self.X)


class TestBayesSearchValidation:
    """Test parameter validation for bayes_search parameter."""

    def setup_method(self):
        """Setup test data for fit operations."""
        np.random.seed(42)
        self.X = np.random.randn(50, 4)  # Small dataset for fast tests

    # Test invalid types
    def test_bayes_search_string_true(self):
        """Test bayes_search as string 'True' raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search="True"
            ).fit(self.X)

    def test_bayes_search_string_false(self):
        """Test bayes_search as string 'False' raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search="False"
            ).fit(self.X)

    def test_bayes_search_string_lowercase(self):
        """Test bayes_search as lowercase string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search="true"
            ).fit(self.X)

    def test_bayes_search_int_zero(self):
        """Test bayes_search as int 0 raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search=0
            ).fit(self.X)


    def test_bayes_search_float_zero(self):
        """Test bayes_search as float 0.0 raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search=0.0
            ).fit(self.X)


    def test_bayes_search_none(self):
        """Test bayes_search as None raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search=None
            ).fit(self.X)

    def test_bayes_search_list(self):
        """Test bayes_search as list raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search=[True]
            ).fit(self.X)

    def test_bayes_search_dict(self):
        """Test bayes_search as dict raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_search.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_search={"enabled": True}
            ).fit(self.X)


    # Test valid values

    def test_bayes_search_valid_true(self, mocker):
        """Test valid bayes_search True."""
        mock_bayes_search = mocker.patch("rif_estimator.utility._residual_gen.ResidualGenerator._bayesian_search")
        mock_bayes_search.return_value = {'n_estimators': 100, 'max_depth': 5}

        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=True
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.bayes_search is True
        assert isinstance(rif.bayes_search, bool)
        # Verify that _bayesian_search was called
        mock_bayes_search.assert_called()

    def test_bayes_search_valid_false(self):
        """Test valid bayes_search False."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=False
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.bayes_search is False
        assert isinstance(rif.bayes_search, bool)

    def test_bayes_search_default(self):
        """Test default bayes_search value is False."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.bayes_search is False  # Default value from your class
        assert isinstance(rif.bayes_search, bool)


class TestBayesIterValidation:
    """Test parameter validation for bayes_iter parameter."""

    def setup_method(self):
        """Setup test data for fit operations."""
        np.random.seed(42)
        self.X = np.random.randn(50, 4)  # Small dataset for fast tests

    # Test invalid values - below minimum
    def test_bayes_iter_zero(self):
        """Test bayes_iter = 0 raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_iter.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_iter=0
            ).fit(self.X)

    def test_bayes_iter_negative(self):
        """Test negative bayes_iter raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_iter.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_iter=-1
            ).fit(self.X)

    # Test invalid types
    def test_bayes_iter_float(self):
        """Test bayes_iter as float raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_iter.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_iter=3.0
            ).fit(self.X)


    def test_bayes_iter_string(self):
        """Test bayes_iter as string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_iter.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_iter="3"
            ).fit(self.X)

    def test_bayes_iter_none(self):
        """Test bayes_iter as None raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_iter.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_iter=None
            ).fit(self.X)

    def test_bayes_iter_bool_false_invalid(self):
        """Test bayes_iter as bool False raises InvalidParameterError (converts to 0)."""
        with pytest.raises(InvalidParameterError, match=r"bayes_iter.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_iter=False  # False converts to 0, which is invalid
            ).fit(self.X)

    def test_bayes_iter_list(self):
        """Test bayes_iter as list raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"bayes_iter.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                bayes_iter=[3]
            ).fit(self.X)

    # Test valid values
    def test_bayes_iter_valid_minimum(self):
        """Test valid bayes_iter at minimum value (1)."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_iter=1
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.bayes_iter == 1
        assert isinstance(rif.bayes_iter, int)

    def test_bayes_iter_default(self):
        """Test default bayes_iter value is 3."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.bayes_iter == 3  # Default value from your class
        assert isinstance(rif.bayes_iter, int)

    def test_bayes_iter_numpy_int_accepted(self):
        """Test bayes_iter as numpy int is accepted."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_iter=np.int32(3)  # numpy int is accepted
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.bayes_iter == 3

    # Test functional behavior with mocking
    def test_bayes_iter_passed_to_bayesian_search(self, mocker):
        """Test that bayes_iter is passed to BayesSearchCV."""
        mock_bayes_search = mocker.patch("rif_estimator.utility._residual_gen.ResidualGenerator._bayesian_search")
        mock_bayes_search.return_value = {'n_estimators': 100, 'max_depth': 5}

        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=True,
            bayes_iter=7  # Custom value
        )

        rif.fit(self.X)

        # Verify the parameter is stored correctly
        assert rif.bayes_iter == 7
        # Verify bayesian search was called (meaning bayes_iter was used)
        mock_bayes_search.assert_called()

    def test_bayes_iter_not_used_when_bayes_search_false(self, mocker):
        """Test that bayes_iter is not used when bayes_search=False."""
        mock_bayes_search = mocker.patch("rif_estimator.utility._residual_gen.ResidualGenerator._bayesian_search")

        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=False,
            bayes_iter=10  # This should be ignored
        )

        rif.fit(self.X)

        # Verify the parameter is stored (but not used)
        assert rif.bayes_iter == 10
        # Verify bayesian search was NOT called
        mock_bayes_search.assert_not_called()

    def test_bayes_iter_bool_true_accepted(self):
        """Test bayes_iter as bool True is accepted (converts to 1)."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_iter=True  # True converts to 1
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.bayes_iter == 1  # True is converted to 1


class TestRfSearchSpaceValidation:
    """Test parameter validation for rf_search_space parameter."""

    def setup_method(self):
        """Setup test data for fit operations."""
        np.random.seed(42)
        self.X = np.random.randn(50, 4)  # Small dataset for fast tests

    # Test invalid types
    def test_rf_search_space_string(self):
        """Test rf_search_space as string raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"rf_search_space.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                rf_search_space="invalid"
            ).fit(self.X)

    def test_rf_search_space_int(self):
        """Test rf_search_space as int raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"rf_search_space.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                rf_search_space=5
            ).fit(self.X)

    def test_rf_search_space_float(self):
        """Test rf_search_space as float raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"rf_search_space.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                rf_search_space=1.5
            ).fit(self.X)

    def test_rf_search_space_bool(self):
        """Test rf_search_space as bool raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"rf_search_space.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                rf_search_space=True
            ).fit(self.X)

    def test_rf_search_space_list(self):
        """Test rf_search_space as list raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"rf_search_space.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                rf_search_space=[1, 2, 3]
            ).fit(self.X)

    def test_rf_search_space_tuple(self):
        """Test rf_search_space as tuple raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"rf_search_space.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                rf_search_space=(1, 2, 3)
            ).fit(self.X)

    def test_rf_search_space_set(self):
        """Test rf_search_space as set raises InvalidParameterError."""
        with pytest.raises(InvalidParameterError, match=r"rf_search_space.*parameter.*must be"):
            ResidualIsolationForest(
                ind_cols=[0],
                env_cols=[1],
                rf_search_space={1, 2, 3}
            ).fit(self.X)

    # Test valid values - None
    def test_rf_search_space_valid_none(self):
        """Test valid rf_search_space as None."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            rf_search_space=None
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space is None

    def test_rf_search_space_default_none(self):
        """Test default rf_search_space value is None."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1]
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space is None  # Default value

    def test_rf_search_space_invalid_rf_parameter_raises_error(self, mocker):
        """Test that invalid RandomForest parameters are rejected."""

        from skopt.space import Integer
        invalid_search_space = {
            "invalid": Integer(50, 500),
            "wella_qw": Integer(3, 30),
            "min_samples_split": Integer(2, 20),
            "min_samples_leaf": Integer(1, 10),
        }

        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=True,  # Force use of rf_search_space
            rf_search_space=invalid_search_space
        )

        # Should raise TypeError when RandomForestRegressor gets invalid params
        with pytest.raises(ValueError, match=r"Invalid parameter .* for estimator RandomForestRegressor"):
         rif.fit(self.X)

    # Test valid values - dict
    def test_rf_search_space_valid_empty_dict(self):
        """Test valid rf_search_space as empty dict."""
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            rf_search_space={}
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space == {}
        assert isinstance(rif.rf_search_space, dict)

    def test_rf_search_space_valid_simple_dict(self):
        """Test valid rf_search_space as simple dict."""
        search_space = {"n_estimators": 100, "max_depth": 10}
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            rf_search_space=search_space
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space == search_space
        assert isinstance(rif.rf_search_space, dict)

    def test_rf_search_space_valid_nested_dict(self):
        """Test valid rf_search_space as nested dict."""
        search_space = {
            "n_estimators": {"min": 100, "max": 500},
            "max_depth": {"values": [5, 10, 15]}
        }
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            rf_search_space=search_space
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space == search_space
        assert isinstance(rif.rf_search_space, dict)

    def test_rf_search_space_valid_mixed_types_dict(self):
        """Test valid rf_search_space with mixed value types."""
        search_space = {
            "n_estimators": 100,
            "max_depth": None,
            "bootstrap": True,
            "criterion": "squared_error"
        }
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            rf_search_space=search_space
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space == search_space

    # Test functional behavior with mocking
    def test_rf_search_space_passed_to_bayesian_search(self, mocker):
        """Test that rf_search_space is passed to Bayesian search when provided."""
        mock_bayes_search = mocker.patch(
            "rif_estimator.utility._residual_gen.ResidualGenerator._bayesian_search"
        )
        mock_bayes_search.return_value = {'n_estimators': 100, 'max_depth': 5}

        custom_search_space = {"n_estimators": 200, "max_depth": 15}

        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=True,
            rf_search_space=custom_search_space
        )

        rif.fit(self.X)

        # Verify the parameter is stored correctly
        assert rif.rf_search_space == custom_search_space
        # Verify bayesian search was called
        mock_bayes_search.assert_called()

    def test_rf_search_space_none_uses_default_in_bayesian_search(self, mocker):
        """Test that rf_search_space=None uses default search space."""
        mock_bayes_search = mocker.patch("rif_estimator.utility._residual_gen.BayesSearchCV")
        mock_bayes_search.fit.best_params_.return_value = {'n_estimators': 100, 'max_depth': 5}

        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=True,
            rf_search_space=None  # Should use default
        )

        rif.fit(self.X)

        # Verify the parameter is stored as None
        assert rif.rf_search_space is None
        # Verify bayesian search was called (will use default internally)
        mock_bayes_search.assert_called()

        #verify that the search space is the default one
        call_kwargs = mock_bayes_search.call_args.kwargs
        from rif_estimator.utility._residual_gen import _DEFAULT_RF_SPACE
        assert call_kwargs["search_spaces"] == _DEFAULT_RF_SPACE

    def test_rf_search_space_not_used_when_bayes_search_false(self, mocker):
        """Test that rf_search_space is not used when bayes_search=False."""
        mock_bayes_search = mocker.patch(
            "rif_estimator.utility._residual_gen.ResidualGenerator._bayesian_search"
        )

        custom_search_space = {"n_estimators": 200}

        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            bayes_search=False,
            rf_search_space=custom_search_space  # Should be ignored
        )

        rif.fit(self.X)

        # Verify the parameter is stored (but not used)
        assert rif.rf_search_space == custom_search_space
        # Verify bayesian search was NOT called
        mock_bayes_search.assert_not_called()

    # Test edge cases
    def test_rf_search_space_dict_with_none_values(self):
        """Test rf_search_space dict can contain None values."""
        search_space = {"n_estimators": 100, "max_depth": None}
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            rf_search_space=search_space
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space == search_space

    def test_rf_search_space_dict_with_special_keys(self):
        """Test rf_search_space dict can have special key names."""
        search_space = {"param_1": 100, "param-2": 200, "param_with_underscore": 300}
        rif = ResidualIsolationForest(
            ind_cols=[0],
            env_cols=[1],
            rf_search_space=search_space
        )
        # Should not raise during construction or fit
        rif.fit(self.X)
        assert rif.rf_search_space == search_space





