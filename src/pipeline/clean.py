import pandas as pd
from pathlib import Path
import json
from tqdm.auto import tqdm
import warnings
import xarray as xr
import zipfile

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

    Warns:
        UserWarning: If no records are found matching `phenophase_id` for the species.

    Raises:
        FileNotFoundError: If no data exists for `species_id`.
        json.JSONDecodeError: If the file cannot be decoded as JSON.
        OSError: If the file cannot be accessed.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
        ValueError: If the file does not contain a non-empty list.

    Notes:
        - Species search tool (includes phenophases): https://naturesnotebook.usanpn.org/npnapps/species.
        - Refer to `ingest.download_phenology_metadata()` output if needed.
    """
    if not (obs_files := list(input_dir.glob('*.json'))):
        raise FileNotFoundError(f"No data for species '{species_id}'. Run 'download_phenology_data()'")

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
        pd.DataFrame: Cleaned DataFrame copy with enforced dtypes.
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

def extract_weather_data(io_dir):
    """
    Extracts a NetCDF file from each ZIP archive in `input_dir`. Deletes the archive only after verifying a successful
    extraction. Prints a failure count (if any).

    Args:
        io_dir (pathlib.Path): Contains grid archives and receives extracted grid files.

    Notes:
        Assumes one NetCDF file per archive and that the extracted file does not already exist.
    """
    failed = 0

    grid_archives = list(io_dir.glob('*.zip'))
    pbar = tqdm(grid_archives, desc="Extracting weather data")

    for grid_archive in pbar:
        try:
            with zipfile.ZipFile(grid_archive) as z:
                grid_file = next(n for n in z.namelist() if n.endswith('.nc'))
                z.extract(grid_file, io_dir)
        except (zipfile.BadZipFile, StopIteration):
            failed += 1
            continue

        if not (io_dir / grid_file).exists():
            failed += 1
            continue

        grid_archive.unlink()

    if failed > 0:
        print(f"Failed to extract {failed} NetCDF files")

def _parse_tokens(grid_file: Path):
    """Parses date and variable tokens from a PRISM NetCDF filename"""
    tokens = grid_file.stem.split('_')
    try:
        date = pd.to_datetime(tokens[-1], format='%Y%m%d')
        var = tokens[-4]
    except (IndexError, ValueError) as e:
        e.add_note(f"Unexpected filename: {grid_file.name}")
        raise

    return date, var

def build_weather_index(input_dir):
    """
    Scans NetCDF files under 'input_dir' and builds a per-file index containing path, date, variable, and phenophase
    year. The latter is calculated via a +1 year offset for records from June onward.

    Args:
        input_dir (pathlib.Path): Contains grid files.

    Raises:
        FileNotFoundError: If no NetCDF files exist under `input_dir`.

    Returns:
        pd.DataFrame: Weather index with columns ['path', 'date', 'var', 'py'].
    """
    if not (grid_files := list(input_dir.glob('*.nc'))):
        raise FileNotFoundError(f"No data for resolution '{input_dir.name}'. Run 'download_weather_data()'")

    records = []

    for f in grid_files:
        date, var = _parse_tokens(f)
        records.append({'path': f, 'date': date, 'var': var})

    df = pd.DataFrame(records)
    df['py'] = df['date'].dt.year + (df['date'].dt.month >= 6)

    return df

def _preprocess(ds: xr.Dataset):
    """Creates temporal axis, renames generic grid variable, and cleans up CRS artifacts"""
    source = Path(ds.encoding['source'])
    date, var = _parse_tokens(source)

    ds = ds.rename({'Band1': var})
    ds = ds.drop_vars('crs', errors='ignore')
    ds[var].attrs.pop('grid_mapping', None)

    return ds.expand_dims(time=[date])

def load_weather_data(idx_df, py, s=100):
    """
    Loads local weather data into a Dask-backed xarray Dataset. Uses an index to retrieve grids for a given phenophase
    year, which are concatenated along a temporal axis. The dataset is chunked to optimize memory usage.

    Args:
        idx_df (pd.DataFrame): Weather index with corresponding paths and phenophase years.
        py (int): Spring phenophase year. Jun-Dec grids map to the following year.
        s (int): Chunk size for spatial dimensions.

    Raises:
        ValueError: If no data exists for 'py'.

    Returns:
        xr.Dataset: Weather dataset with dimensions [time, lat, lon].

    Notes:
        Uses the module-level `_preprocess` helper for temporal axis creation.
    """
    df = idx_df[idx_df['py'] == py]

    if df.empty:
        raise ValueError(f"No data for phenophase year '{py}'. Check offset range")

    grid_files = df['path'].tolist()

    ds = xr.open_mfdataset(
        grid_files,
        chunks={'lat': s, 'lon': s}, # Chunk spatially per daily grid
        compat='override',
        preprocess=_preprocess,
        engine='netcdf4',
        data_vars='minimal',
        coords='minimal',
        combine='by_coords',
        parallel=True,
    )

    ds = ds.chunk({'time': -1}) # Rechunk temporally across all grids

    return ds