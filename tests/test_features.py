import pandas as pd
import pytest
import xarray as xr

from spring_engine.data.transform import engineer_thermal_features

def _make_cells(test_case: list[dict]) -> xr.Dataset:
    template = {
        "time": "2025-07-18",
        "ppt": 0.0,
        "tmax": 29.4,
        "tmin": 21.7
    }

    cells = [
        {**template, **override, 'cell_id': i}
        for i, override in enumerate(test_case, start=1)
    ]

    df = pd.DataFrame(cells)
    df['time'] = pd.to_datetime(df['time'])

    return df.set_index(['time', 'cell_id']).to_xarray()

def _make_features(test_case: list[dict]) -> xr.Dataset:
    ds = _make_cells(test_case)

    return engineer_thermal_features(ds, chill_bounds=(0.0, 7.0), gdd_bounds=(10.0, 30.0))

def test_chill():
    test_case = [
        {'tmax': 7.0, 'tmin': 0.0},
        {'tmax': 3.5, 'tmin': -3.5},
        {'tmax': 10.5, 'tmin': 3.5},
        {'tmax': 14.0, 'tmin': 7.0}
    ]

    features = _make_features(test_case)
    assert features.chill.isel(time=0).values == pytest.approx([1.0, 0.5, 0.5, 0.0])

def test_gdd():
    test_case = [
        {'tmax': 30.0, 'tmin': 10.0},
        {'tmax': 20.0, 'tmin': 0.0},
        {'tmax': 40.0, 'tmin': 20.0},
        {'tmax': 10.0, 'tmin': -10.0}
    ]

    features = _make_features(test_case)
    assert features.gdd.isel(time=0).values == pytest.approx([10.0, 5.0, 15.0, 0.0])