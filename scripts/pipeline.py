from datetime import datetime, timedelta
import json
import pandas as pd
from pathlib import Path
import random
import requests
import time
from tqdm.auto import tqdm
import warnings

# Create project root path relative to this module
ROOT = Path(__file__).resolve().parent.parent

NPN_URL = "https://services.usanpn.org/npn_portal"
PRISM_URL = "https://services.nacse.org/prism/data/get"

def fetch_with_retry(session, url, context, params=None, alpha=2, attempts=3, timeout=60):
    """
    Fetches data from an API endpoint using an existing session, with base-2 exponential backoff and ±10% jitter.
    Sleep duration is capped at 5 minutes.

    Args:
        session (requests.Session): Active HTTP session.
        url (str): Endpoint URL.
        context (str): Identifier used in error output.
        params (dict, optional): Query parameters passed to request.
        alpha (int): Backoff coefficient in seconds. Defaults to 2.
        attempts (int): Total number of attempts. Defaults to 3.
        timeout (int): Request timeout in seconds. Defaults to 60.

    Returns:
        requests.Response: HTTP response on success.

    Raises:
        requests.exceptions.RequestException: If all attempts fail.
    """
    MAX_SLEEP = 300

    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Failed for {context} ({attempt + 1}/{attempts}): {e}")

            if attempt < attempts - 1:
                time.sleep(min((random.uniform(0.9, 1.1) * alpha * 2**attempt), MAX_SLEEP))
            else:
                raise

    raise AssertionError("I like turtles") # Unreachable

def download_phenology_metadata(sleep=2):
    """
    Downloads phenophase and species metadata from the National Phenology Network API using the `fetch_with_retry`
    helper, and saves the responses as formatted JSON files under `data/phenology/metadata/`.

    Args:
        sleep (int | float): Base sleep duration in seconds after a successful download. A ±10% jitter is applied to
            this value. Defaults to 2.

    Raises:
        requests.exceptions.RequestException: If all attempts fail.
        json.JSONDecodeError: If the response body cannot be decoded as JSON.
    """
    data_dir = ROOT / 'data' / 'phenology' / 'metadata'
    data_dir.mkdir(parents=True, exist_ok=True)

    endpoints = {
        'phenophases': f"{NPN_URL}/phenophases/getPhenophases.json",
        'species': f"{NPN_URL}/species/getSpecies.json"
    }

    pbar = tqdm(endpoints.items(), desc="Downloading phenology metadata")

    with requests.Session() as session:

        for name, url in pbar:
            data_file = data_dir / f"{name}.json"

            try:
                response = fetch_with_retry(session=session, url=url, context=name, alpha=sleep)
                data = response.json()
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                e.add_note("Download aborted")
                pbar.close()
                raise

            with data_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            time.sleep(random.uniform(0.9, 1.1) * sleep)

def download_phenology_data(species_id, start_year, end_year, sleep=2):
    """
    Downloads yearly observation data for a species from the National Phenology Network API using the `fetch_with_retry`
    helper, and saves the responses as formatted JSON files under `data/phenology/observations/{species_id}/`. Existing
    files are skipped to avoid redundant downloads.

    Args:
        species_id (int): Unique species identifier.
        start_year (int): First year of data to download.
        end_year (int): Last year of data to download.
        sleep (int | float): Base sleep duration in seconds after a successful download. A ±10% jitter is applied to
            this value. Defaults to 2.

    Raises:
        FileNotFoundError: If the species metadata file is missing. Run `download_phenology_metadata()` if needed.
        ValueError: If `species_id` does not match a species in the metadata.
        requests.exceptions.RequestException: If all attempts fail.
        json.JSONDecodeError: If the response body cannot be decoded as JSON.

    Notes:
        The helper `utils.lookup_species` can be used to retrieve a valid `species_id`.
    """
    species_file = ROOT / 'data' / 'phenology' / 'metadata' / 'species.json'

    try:
        with open(species_file, 'r', encoding='utf-8') as f:
            species_metadata = json.load(f)
    except FileNotFoundError as e:
        e.add_note("Species metadata missing. Run 'download_phenology_metadata()'")
        raise

    species_entry = next((s for s in species_metadata if s['species_id'] == species_id), None)

    if species_entry is None:
        raise ValueError(f"Invalid species_id: {species_id}")

    data_dir = ROOT / 'data' / 'phenology' / 'observations' / str(species_id)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_params = {'species_id': species_id, 'request_src': 'SpringEngine'}
    url = f"{NPN_URL}/observations/getObservations.json"

    years = range(start_year, end_year + 1)

    pbar = tqdm(years, desc="Loading phenology data")

    with requests.Session() as session:
        for year in pbar:
            data_file = data_dir / f"{year}.json"

            if data_file.exists():
                pbar.set_postfix(status='local')
                continue
            else:
                pbar.set_postfix(status='remote')

            params = {**base_params, 'start_date': f"{year}-01-01", 'end_date': f"{year}-12-31"}

            try:
                response = fetch_with_retry(session=session, url=url, context=f"{year}", params=params, alpha=sleep)
                data = response.json()
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                e.add_note("Download aborted")
                pbar.close()
                raise

            with data_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            time.sleep(random.uniform(0.9, 1.1) * sleep)

def download_weather_data(start_year, end_year, resolution='4km', variables=('ppt', 'tmax', 'tmin'), sleep=4):
    """
    Downloads daily weather data from the PRISM Group API using the `fetch_with_retry` helper, and saves the responses
    as NetCDF files under `data/weather/grids/{resolution}/`. Existing files are skipped to avoid redundant downloads.
    Grid cells are retrieved for each element in `variables`, within the range [start_year, end_year].

    Args:
        start_year (int): First year of data to download.
        end_year (int): Last year of data to download.
        resolution (str): Grid cell resolution. Defaults to '4km'.
        variables (tuple of str): Weather variables to download. Defaults to ('ppt', 'tmax', 'tmin').
        sleep (int | float): Base sleep duration in seconds after a successful download. A ±10% jitter is applied to
            this value. Defaults to 4.

    Raises:
        requests.exceptions.RequestException: If all attempts fail.

    Notes:
        PRISM monitors download activity and may restrict access for excessive requests. The default `sleep` delay
        conservatively throttles request frequency to maintain a stable session; avoid reducing it. Large year ranges
        may take many hours to download.
    """
    data_dir = ROOT / 'data' / 'weather' / 'grids' / resolution
    data_dir.mkdir(parents=True, exist_ok=True)

    headers = {'User-Agent': 'SpringEngine (phenology research)'}

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    current_date = start_date
    total_days = (end_date - start_date).days + 1

    pbar = tqdm(total=total_days, desc="Loading weather data")

    with requests.Session() as session:
        session.headers.update(headers)

        while current_date <= end_date:
            date = current_date.strftime("%Y%m%d")

            for var in variables:
                data_file = data_dir / f"{date}_{var}.nc"

                if data_file.exists():
                    pbar.set_postfix(status='local')
                    continue
                else:
                    pbar.set_postfix(status='remote')

                url = f"{PRISM_URL}/us/{resolution}/{var}/{date}"

                try:
                    response = fetch_with_retry(session=session, url=url, context=date, alpha=sleep, timeout=120)
                except requests.exceptions.RequestException as e:
                    e.add_note("Download aborted")
                    pbar.close()
                    raise

                with data_file.open('wb') as f:
                    f.write(response.content)

                time.sleep(random.uniform(0.9, 1.1) * sleep)

            current_date += timedelta(days=1)
            pbar.update(1)

def load_phenology_data(species_id, phenophase_id):
    """
     Loads local phenology data for a given species and phenophase into a DataFrame. Reads all JSON files under
    `data/phenology/observations/{species_id}/`, extracting only values required for model training.

    Args:
        species_id (int): Unique species identifier.
        phenophase_id (int): Unique phenophase identifier.

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
        - List of all phenophases: `data/phenology/metadata/phenophases.json`
        - Species search tool (includes phenophases): https://naturesnotebook.usanpn.org/npnapps/species
    """
    data_dir = ROOT / 'data' / 'phenology' / 'observations' / str(species_id)

    if not data_dir.is_dir() or not any(data_dir.glob("*.json")):
        raise FileNotFoundError(f"No data for species '{species_id}'. Run 'download_phenology_data()'")

    cols = [
        'observation_id',
        'site_id',
        'latitude',
        'longitude',
        'elevation_in_meters',
        'individual_id',
        'observation_date',
        'day_of_year',
        'phenophase_status'
    ]

    rows = []

    for data_file in data_dir.glob("*.json"):
        try:
            with data_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            e.add_note(f"Error loading '{data_file.name}'")
            raise

        if not isinstance(data, list) or not data:
            raise ValueError(f"Expected a non-empty list in '{data_file.name}'")

        for obs_entry in data:
            if obs_entry.get('phenophase_id') == phenophase_id:
                rows.append({k: obs_entry.get(k) for k in cols})

    if not rows:
        warnings.warn(f"No observations for species '{species_id}' "
                      f"and phenophase '{phenophase_id}'. "
                      f"This may be due to an invalid phenophase or sparse data coverage",
                      category=UserWarning
        )

    return pd.DataFrame.from_records(rows, columns=cols)