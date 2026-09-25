import math
import pandas as pd

from spring_engine.data.clean import PHENOLOGY_SCHEMA
from spring_engine.data.transform import compose_labels

def _make_obs(test_case: list[dict]) -> pd.DataFrame:
    template = {
        "site_id": 1,
        "latitude": 39.889560,
        "longitude": -87.199478,
        "state": "IN",
        "individual_id": 1,
        "observation_date": "2025-07-18",
        "phenophase_status": 0
    }

    obs = [
        {**template, **override, 'observation_id': i}
        for i, override in enumerate(test_case, start=1)
    ]

    return pd.DataFrame(obs).astype(PHENOLOGY_SCHEMA)

def _make_labels(test_case: list[dict]) -> pd.DataFrame:
    df = _make_obs(test_case)
    labels, _ = compose_labels(df, trans_gap=14, cycle_gap=182)

    return labels

def test_cycle_gap():
    test_case = [
        {'observation_date': '2024-07-17', 'phenophase_status': 0},
        {'observation_date': '2024-07-18', 'phenophase_status': 1},
        {'observation_date': '2025-07-17', 'phenophase_status': 0},
        {'observation_date': '2025-07-18', 'phenophase_status': 1},
        {'observation_date': '2025-07-19', 'phenophase_status': 1}
    ]

    labels = _make_labels(test_case)
    assert labels['observation_date'].tolist() == [pd.Timestamp('2024-07-18'), pd.Timestamp('2025-07-18')]

def test_onset_date():
    test_case = [
        {'observation_date': '2025-07-12', 'phenophase_status': 0},
        {'observation_date': '2025-07-16', 'phenophase_status': 0},
        {'observation_date': '2025-07-18', 'phenophase_status': 1}
    ]

    labels = _make_labels(test_case)
    assert labels['onset_date'].tolist() == [pd.Timestamp('2025-07-17')]

def test_trans_gap():
    test_case = [
        {'site_id': 1, 'observation_date': '2025-07-04', 'phenophase_status': 0},
        {'site_id': 1, 'observation_date': '2025-07-18', 'phenophase_status': 1},
        {'site_id': 2, 'observation_date': '2025-07-04', 'phenophase_status': 0},
        {'site_id': 2, 'observation_date': '2025-07-19', 'phenophase_status': 1}
    ]

    labels = _make_labels(test_case)
    assert labels['observation_date'].tolist() == [pd.Timestamp('2025-07-18')]

def test_trig_doy():
    test_case = [
        {'site_id': 1, 'observation_date': '2024-12-30', 'phenophase_status': 0},
        {'site_id': 1, 'observation_date': '2025-01-01', 'phenophase_status': 1},
        {'site_id': 2, 'observation_date': '2024-12-31', 'phenophase_status': 0},
        {'site_id': 2, 'observation_date': '2025-01-02', 'phenophase_status': 1}
    ]

    labels = _make_labels(test_case)

    sin_dif = labels.iloc[0]['sin_doy'] - labels.iloc[1]['sin_doy']
    cos_dif = labels.iloc[0]['cos_doy'] - labels.iloc[1]['cos_doy']
    k = math.sqrt(sin_dif**2 + cos_dif**2)

    assert k < 0.02