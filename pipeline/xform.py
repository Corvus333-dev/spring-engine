import calendar
import xarray as xr

def engineer_monthly_features(ds):
    """
    Aggregates daily weather data into monthly features by computing per-month means of temperature extrema and sums of
    precipitation, chill accumulation, and growing degree-days (GDD).

    Args:
        ds (xr.Dataset): Weather dataset with dimensions [time, lat, lon] and data variables
            ['ppt', 'tmax', 'tmin', 'chill', 'gdd'].

    Returns:
        xr.Dataset: Feature dataset reduced to dimensions [lat, lon], with monthly-encoded data variable names
            (e.g., oct_chill_sum, apr_tmax_mean, etc.).
    """
    features = {}

    monthly_bins = {
        'ppt_sum': ds.ppt.groupby('time.month').sum('time'),
        'tmax_mean': ds.tmax.groupby('time.month').mean('time'),
        'tmin_mean': ds.tmin.groupby('time.month').mean('time'),
        'chill_sum': ds.chill.groupby('time.month').sum('time'),
        'gdd_sum': ds.gdd.groupby('time.month').sum('time'),
    }

    for k, v in monthly_bins.items():
        for month in v.month.values:
            month_abbr = calendar.month_abbr[int(month)].lower()
            name = f"{month_abbr}_{k}"

            features[name] = (
                v.sel(month=month)
                .drop_vars('month') # Drop scalar coordinate for reconstruction
            )

    return xr.Dataset(features)

def engineer_thermal_features(ds, chill_bounds, gdd_bounds):
    """
    Derives winter chill and growing degree-day (GDD) features from daily temperature ranges. Approximates relative
    chill accumulation via the fractional overlap between `chill_bounds` and [tmin, tmax]. Calculates daily heat
    accumulation using a modified GDD formula that clips temperatures to `gdd_bounds` before averaging.

    Args:
        ds (xr.Dataset): Weather dataset with data variables [..., tmax, tmin].
        chill_bounds (tuple of float): Winter chill bounds (min, max).
        gdd_bounds (tuple of float): Growing degree-day bounds (min, max).

    Returns:
        xr.Dataset: Weather dataset with added data variables [chill, gdd].
    """
    tmax, tmin = ds.tmax, ds.tmin
    trange = tmax - tmin
    nz_trange = xr.where(trange == 0, 1e-6, trange)

    tmax_chill = tmax.clip(max=chill_bounds[1])
    tmin_chill = tmin.clip(min=chill_bounds[0])

    chill_isect = (tmax_chill - tmin_chill).clip(min=0)
    chill = chill_isect / nz_trange

    is_static_chill = (trange == 0) & (tmax >= chill_bounds[0]) & (tmax <= chill_bounds[1])
    chill = xr.where(is_static_chill, 1.0, chill)

    tmax_gdd = tmax.clip(min=gdd_bounds[0], max=gdd_bounds[1])
    tmin_gdd = tmin.clip(min=gdd_bounds[0], max=gdd_bounds[1])

    gdd = ((tmax_gdd + tmin_gdd) / 2) - gdd_bounds[0]

    ds['chill'] = chill
    ds['gdd'] = gdd

    return ds