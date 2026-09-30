import pandas as pd
import pytest

from spring_engine.data.transform import compose_examples

@pytest.fixture
def sample_labels() -> pd.DataFrame:
    return pd.DataFrame({
        'cell_id': [1, 2],
        'onset_date': pd.to_datetime(['2025-01-28', '2025-06-05']),
        'sin_doy': [0.448229, 0.455907],
        'cos_doy': [0.893919, -0.890028]
    })

@pytest.fixture
def sample_features() -> pd.DataFrame:
    return pd.DataFrame({
        'cell_id': [1, 1, 2, 2],
        'year': [2024, 2025, 2025, 2025],
        'month': [12, 1, 5, 6],
        'ppt_sum': [91.2, 32.3, 155.7, 172.2],
        'tmax_mean': [11.5, 4.2, 22.6, 27.4],
        'tmin_mean': [-3.1, -5.8, 4.5, 6.5],
        'chill_sum': [14.9, 21.7, 4.2, 0.7],
        'gdd_sum': [23.2, 0.3, 194.6, 349.2]
    })

def test_label_retention(sample_labels: pd.DataFrame, sample_features: pd.DataFrame):
    examples = compose_examples(sample_labels, sample_features, max_lag=1)

    pd.testing.assert_frame_equal(
        examples[sample_labels.columns],
        sample_labels
    )

def test_feature_lag(sample_labels: pd.DataFrame, sample_features: pd.DataFrame):
    examples = compose_examples(sample_labels, sample_features, max_lag=1)

    expected = pd.DataFrame({
        'ppt_sum_lag1': [91.2, 155.7],
        'tmax_mean_lag1': [11.5, 22.6],
        'tmin_mean_lag1': [-3.1, 4.5],
        'chill_sum_lag1': [14.9, 4.2],
        'gdd_sum_lag1': [23.2, 194.6]
    })

    pd.testing.assert_frame_equal(
        examples[expected.columns],
        expected
    )