import calendar
import pandas as pd
import xarray as xr

def compose_labels(df, trans_gap, cycle_gap):
    """
    Extracts phenophase event onset labels and curates according to the following rules:

        - For a specific plant individual, an event onset is assigned as the midpoint between a positive (1)
          phenophase observation and the nearest preceding negative (0) phenophase observation occurring within
          `trans_gap` days.
        - An event onset is accepted only if no prior event onsets from any plant individual at the corresponding site
          exist within the preceding `cycle_gap` days. This effectively retains the earliest detected phenophase at each
          site for every phenological cycle.

    Calculates event onset day of year via 365-day year standardization to account for leap years. Extracts spatial
    coordinates for each unique site represented in the curated phenophase onset labels.

    Args:
        df (pd.DataFrame): Phenology DataFrame with fixed column schema (enforced in pipeline.clean).
        trans_gap (int): Maximum observer cadence window (e.g., 14 days between observations) for 0 -> 1 transitions.
        cycle_gap (int): Minimum days between phenophase onsets (typically quasi-annual cycles) at a given site.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - labels: Curated phenophase onset labels with columns [onset_date, onset_doy] added to original schema.
            - label_sites: Unique label site coordinates with columns [latitude, longitude].
    """
    df = df.sort_values(['site_id', 'individual_id', 'observation_date'])

    df['prev_status'] = df.groupby(['site_id', 'individual_id'])['phenophase_status'].shift(1)
    df['prev_date'] = df.groupby(['site_id', 'individual_id'])['observation_date'].shift(1)

    is_trans = (df['prev_status'] == 0) & (df['phenophase_status'] == 1)
    valid_gap = (df['observation_date'] - df['prev_date']) <= pd.Timedelta(days=trans_gap)

    pool = df[is_trans & valid_gap].copy()
    pool['onset_date'] = pool['prev_date'] + (pool['observation_date'] - pool['prev_date']) / 2

    labels = (
        pool.groupby('site_id', group_keys=False)
        .apply(lambda x: _filter_consecutive_onsets(x, cycle_gap))
        .drop(columns=['prev_status', 'prev_date'])
    )

    is_leap_year = labels['onset_date'].dt.is_leap_year
    post_feb = labels['onset_date'].dt.month > 2

    labels['onset_doy'] = labels['onset_date'].dt.dayofyear
    labels.loc[is_leap_year & post_feb, 'onset_doy'] -= 1

    label_sites = labels[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)

    return labels, label_sites

def _filter_consecutive_onsets(site_group, cycle_gap):
    """
    Filters event onsets for a site, retaining records separated by more than `cycle_gap` days.

    Args:
        site_group (pd.DataFrame): Site-specific subset of plant individuals with a valid phenophase transition.
        cycle_gap (int): Minimum days between phenophase onsets (typically quasi-annual cycles) at a given site.

    Returns:
        pd.DataFrame: Filtered event onset records.
    """
    valid_rows = []
    last_onset = pd.NaT

    site_group = site_group.sort_values('onset_date')

    for _, row in site_group.iterrows():
        if pd.isna(last_onset) or (row['onset_date'] - last_onset) > pd.Timedelta(days=cycle_gap):
            valid_rows.append(row)
            last_onset = row['onset_date']

    return pd.DataFrame(valid_rows)

def select_weather_subset(ds, label_sites):
    """
    Subsets the weather dataset by selecting the nearest grid cells for all unique label sites. This reduces the spatial
    domain prior to feature engineering so that features are computed only for grid cells associated with label sites.

    Args:
        ds (xr.Dataset): Weather dataset with dimensions [time, lat, lon].
        label_sites (pd.DataFrame): Unique label sites DataFrame with columns [latitude, longitude].

    Returns:
        xr.Dataset: Weather subset with dimensions [time, point] and coordinates [time, lat(point), lon(point)].
    """
    lat = xr.DataArray(label_sites['latitude'], dims='point')
    lon = xr.DataArray(label_sites['longitude'], dims='point')

    ds = ds.sel(lat=lat, lon=lon, method='nearest')

    return ds

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