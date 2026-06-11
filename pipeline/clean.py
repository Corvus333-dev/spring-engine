import pandas as pd
from pathlib import Path
import json
import warnings
import xarray as xr

PHENOLOGY_SCHEMA = {
        'observation_id': 'int32',
        'site_id': 'int32',
        'latitude': 'float32',
        'longitude': 'float32',
        'individual_id': 'int32',
        'observation_date': 'datetime64[ns]',
        'day_of_year': 'int16',
        'phenophase_status': 'int8'
}

class WeatherLoader:
    """
    Assembles an annual series of daily PRISM NetCDF weather grids into a block-distributed xarray Dataset.

    Args:
        input_dir (pathlib.Path): Contains grid files.
        chunk_size (int): Chunk size for spatial dimensions.
    """
    def __init__(self, input_dir, chunk_size=100):
        self.input_dir = input_dir
        self.chunk_size = chunk_size
        self.file_index = self._build_file_index()

    def load_weather_data(self, phenophase_year):
        """
        Loads local weather data into a Dask-backed xarray Dataset. Uses a file index to retrieve grids for a given
        phenophase year, which are concatenated along a temporal axis. The dataset is chunked to optimize memory usage.

        Args:
            phenophase_year (int): Spring phenophase year. Jun-Dec grids map to the following year.

        Returns:
            xr.Dataset: Weather dataset with dimensions [time, lat, lon].

        Raises:
            ValueError: If no data exists for `phenophase_year`.
        """
        df = self.file_index[self.file_index['phenophase_year'] == phenophase_year]

        if df.empty:
            raise ValueError(f"No data for phenophase year '{phenophase_year}'")

        grid_files = df['path'].tolist()

        ds = xr.open_mfdataset(
            grid_files,
            chunks={'lat': self.chunk_size, 'lon': self.chunk_size},  # Chunk spatially per daily grid
            compat='override',
            preprocess=self._preprocess,
            engine='netcdf4',
            data_vars='minimal',
            coords='minimal',
            combine='by_coords',
            parallel=True,
        )

        return ds.chunk({'time': -1})  # Rechunk temporally across all grids

    def _build_file_index(self):
        """
        Scans NetCDF files under 'input_dir' and builds a per-file index containing path and phenophase year. The latter
        is calculated via a +1 year offset for records from the summer solstice onward.

        Returns:
            pd.DataFrame: File index with columns ['path', 'phenophase_year'].

        Raises:
            FileNotFoundError: If no NetCDF files exist under `input_dir`.
        """
        if not (grid_files := list(self.input_dir.glob('*.nc'))):
            raise FileNotFoundError(f"No data for resolution '{self.input_dir.name}'")

        records = []

        for f in grid_files:
            date, _ = self._parse_tokens(f)
            records.append({'path': f, 'date': date})

        df = pd.DataFrame(records)

        m, d = df['date'].dt.month, df['date'].dt.day
        after_spring = (m > 6) | ((m == 6) & (d >= 21))
        df['phenophase_year'] = df['date'].dt.year + after_spring

        return df.drop(columns=['date'])

    @classmethod
    def _preprocess(cls, ds: xr.Dataset) -> xr.Dataset:
        """Creates temporal axis, renames generic grid variable, and cleans up CRS artifacts"""
        source = Path(ds.encoding['source'])
        date, var = cls._parse_tokens(source)

        ds = ds.rename({'Band1': var})
        ds = ds.drop_vars('crs', errors='ignore')
        ds[var].attrs.pop('grid_mapping', None)

        return ds.expand_dims(time=[date])

    @staticmethod
    def _parse_tokens(grid_file: Path) -> tuple[pd.Timestamp, str]:
        """Parses date and variable tokens from a PRISM NetCDF filename"""
        tokens = grid_file.stem.split('_')
        try:
            date = pd.to_datetime(tokens[-1], format='%Y%m%d')
            var = tokens[-4]
        except (IndexError, ValueError) as e:
            e.add_note(f"Unexpected filename: {grid_file.name}")
            raise

        return date, var

def load_phenology_data(species_id, phenophase_id, input_dir):
    """
     Loads local phenology data for a given species and phenophase into a DataFrame. Reads all JSON files under
     `input_dir`, extracting only values required for model training.

    Args:
        species_id (int): Unique species identifier.
        phenophase_id (int): Unique phenophase identifier.
        input_dir (pathlib.Path): Contains observation files.

    Returns:
        pd.DataFrame: DataFrame of matching observation entries, with a fixed column schema.

    Raises:
        FileNotFoundError: If no data exists for `species_id`.
        json.JSONDecodeError: If the file cannot be decoded as JSON.
        OSError: If the file cannot be accessed.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
        ValueError: If the file does not contain a non-empty list.

    Warns:
        UserWarning: If no records are found matching `phenophase_id` for the species.

    Notes:
        - Species search tool (includes phenophases): https://naturesnotebook.usanpn.org/npnapps/species.
        - Refer to `ingest.download_phenology_metadata()` output if needed.
    """
    if not (obs_files := list(input_dir.glob('*.json'))):
        raise FileNotFoundError(f"No data for species '{species_id}'")

    rows = []

    for obs_file in obs_files:
        try:
            with obs_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            e.add_note(f"Error loading '{obs_file.name}'")
            raise

        if not isinstance(data, list) or not data:
            raise ValueError(f"Expected a non-empty list in '{obs_file.name}'")

        for obs_entry in data:
            if obs_entry.get('phenophase_id') == phenophase_id:
                rows.append({k: obs_entry.get(k) for k in PHENOLOGY_SCHEMA})

    if not rows:
        warnings.warn(
            f"No observations for species '{species_id}' and phenophase '{phenophase_id}'. "
            "This may be due to an invalid phenophase or sparse data coverage",
            category=UserWarning
        )

    return pd.DataFrame.from_records(rows, columns=PHENOLOGY_SCHEMA.keys())

def clean_phenology_data(df, lat_bounds, lon_bounds):
    """
    Cleans phenology data by deduplicating on `observation_id` and filtering invalid rows. Validates non-negative
    integer IDs, spatial (lat/lon) and temporal (DOY) bounds, date format (YYYY-MM-DD), date-DOY consistency, and
    phenophase status ∈ {-1, 0, 1}. Logs counts of failed checks to console.

    Args:
        df (pd.DataFrame): DataFrame of observation entries, with a fixed column schema.
        lat_bounds (tuple of float): Latitudinal bounds (min, max).
        lon_bounds (tuple of float): Longitudinal bounds (min, max).

    Returns:
        pd.DataFrame: Clean DataFrame copy with enforced dtypes.
    """
    print("Cleaning phenology data...")
    masks = {}

    df = df.drop_duplicates(subset='observation_id', keep='last', ignore_index=True)
    unique_len = len(df)

    id_cols = ['observation_id', 'site_id', 'individual_id']

    for col in id_cols:
        s = pd.to_numeric(df[col], errors='coerce')
        masks[f"invalid '{col}'"] = (s.isna() | (s % 1 != 0) | (s < 0))
        df[col] = s

    st_bbox = {'latitude': lat_bounds, 'longitude': lon_bounds, 'day_of_year': (1, 366)}

    for col, (min_val, max_val) in st_bbox.items():
        s = pd.to_numeric(df[col], errors='coerce')
        masks[f"invalid {col}"] = s.isna() | (s < min_val) | (s > max_val)
        df[col] = s

    parsed_dates = pd.to_datetime(df['observation_date'], format='%Y-%m-%d', errors='coerce')
    masks["invalid 'observation_date'"] = parsed_dates.isna()
    df['observation_date'] = parsed_dates

    valid_dates = ~masks["invalid 'observation_date'"]
    expected_doy = parsed_dates.dt.dayofyear
    actual_doy = df['day_of_year']
    masks["'day_of_year' mismatch"] = valid_dates & (actual_doy != expected_doy)

    uny = pd.to_numeric(df['phenophase_status'], errors='coerce')
    masks["invalid 'phenophase_status'"] = ~uny.isin((-1, 0, 1))
    df['phenophase_status'] = uny

    invalid = pd.Series(False, index=df.index)
    for issue, mask in masks.items():
        n = mask.sum()
        if n:
            print(f"{issue}: {n}")
        invalid |= mask

    cleaned = df.loc[~invalid].copy()
    cleaned = cleaned.astype(PHENOLOGY_SCHEMA)

    print(f"Kept {len(cleaned)}/{unique_len} unique observations")

    return cleaned.reset_index(drop=True)

def clean_weather_data(ds, variables, lat_bounds, lon_bounds):
    """
    Ensures that overall structure and coordinate values of a PRISM-sourced xarray Dataset are clean.

    Args:
        ds (xr.Dataset): Weather dataset with dimensions [time, lat, lon].
        variables (tuple of str): Weather data variables (e.g., ppt, tmax, tmin).
        lat_bounds (tuple of float): Latitudinal bounds (min, max).
        lon_bounds (tuple of float): Longitudinal bounds (min, max).

    Returns:
        xr.Dataset: Clean weather dataset with enforced spatial coordinate dtypes.

    Raises:
        ValueError: If dataset structure or coordinate values are invalid.
        TypeError: If time values are not 'datetime64'.

    Warns:
        UserWarning: If dataset contains extra dimensions or variables.
    """
    expected_dims = {'time', 'lat', 'lon'}
    expected_vars = set(variables)

    expected = expected_dims | expected_vars
    actual = set(ds.variables) # Capture dimension coordinates and data variables

    if missing := expected - actual:
        raise ValueError(f"Expected dimensions and/or variables are missing: {missing}")

    if extra := actual - expected:
        warnings.warn(
            f"Unexpected dimensions and/or variables are present: {extra}. "
            "This may result in suboptimal Dask operations",
            category=UserWarning
        )

    # Downcast for consistency with phenology schema
    ds = ds.assign_coords(lat=ds.lat.astype('float32'), lon=ds.lon.astype('float32'))

    if ds.indexes['time'].dtype.kind != 'M':
        raise TypeError("Dimension coordinate 'time' values are not 'datetime64'")

    for dim in expected_dims:
        idx = ds.indexes[dim]

        if idx.hasnans:
            raise ValueError(f"Dimension coordinate '{dim}' contains NaN values")

        if idx.has_duplicates:
            raise ValueError(f"Dimension coordinate '{dim}' contains duplicate values")

        if not idx.is_monotonic_increasing:
            raise ValueError(f"Dimension coordinate '{dim}' values are not in ascending order")

    if ds.lat[0] < lat_bounds[0] or ds.lat[-1] > lat_bounds[-1]:
        raise ValueError("Latitude coordinates are out of bounds")

    if ds.lon[0] < lon_bounds[0] or ds.lon[-1] > lon_bounds[-1]:
        raise ValueError("Longitude coordinates are out of bounds")

    return ds