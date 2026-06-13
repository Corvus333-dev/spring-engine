from datetime import datetime, timedelta
import json
import random
import requests
import time
from tqdm.auto import tqdm
import zipfile

NPN_URL = "https://services.usanpn.org/npn_portal"
PRISM_URL = "https://services.nacse.org/prism/data/get"

def _fetch_with_retry(session, url, context, params=None, alpha=2, attempts=3, timeout=60):
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

def download_phenology_metadata(output_dir, sleep=2):
    """
    Downloads phenophase and species metadata from the National Phenology Network API and saves the responses as
    formatted JSON files. Existing files are skipped to avoid redundant downloads.

    Args:
        output_dir (pathlib.Path): Receives metadata files.
        sleep (int | float): Base sleep duration in seconds after a successful download. A ±10% jitter is applied to
            this value. Defaults to 2.

    Raises:
        requests.exceptions.RequestException: If all attempts fail.
        json.JSONDecodeError: If the response body cannot be decoded as JSON.

    Notes:
        Uses the module-level `_fetch_with_retry` helper for network requests.
    """
    endpoints = {
        'phenophases': f"{NPN_URL}/phenophases/getPhenophases.json",
        'species': f"{NPN_URL}/species/getSpecies.json"
    }

    pbar = tqdm(endpoints.items(), desc="Downloading phenology metadata")

    with requests.Session() as session:

        for name, url in pbar:
            meta_file = output_dir / f"{name}.json"

            if meta_file.exists():
                pbar.set_postfix(status='local')
                continue
            else:
                pbar.set_postfix(status='remote')

            try:
                response = _fetch_with_retry(session=session, url=url, context=name, alpha=sleep)
                data = response.json()
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                e.add_note("Download aborted")
                pbar.close()
                raise

            with meta_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            time.sleep(random.uniform(0.9, 1.1) * sleep)

def download_phenology_data(species_id, start_year, end_year, input_dir, output_dir, sleep=2):
    """
    Downloads yearly observation data for a species from the National Phenology Network API and saves the responses as
    formatted JSON files. Existing files are skipped to avoid redundant downloads.

    Args:
        species_id (int): Unique species identifier.
        start_year (int): First year of data to download.
        end_year (int): Last year of data to download.
        input_dir (pathlib.Path): Contains species metadata file.
        output_dir (pathlib.Path): Receives observation files.
        sleep (int | float): Base sleep duration in seconds after a successful download. A ±10% jitter is applied to
            this value. Defaults to 2.

    Raises:
        FileNotFoundError: If the species metadata file is missing. Run `download_phenology_metadata()` if needed.
        ValueError: If `species_id` does not match a species in the metadata.
        requests.exceptions.RequestException: If all attempts fail.
        json.JSONDecodeError: If the response body cannot be decoded as JSON.

    Notes:
        - Uses the module-level `_fetch_with_retry` helper for network requests.
        - The `tools.lookup_species` tool can be used to retrieve a valid `species_id`.
    """
    species_meta_file = input_dir / 'species.json'

    try:
        with open(species_meta_file, 'r', encoding='utf-8') as f:
            species_meta = json.load(f)
    except FileNotFoundError as e:
        e.add_note("No species metadata file")
        raise

    species_entry = next((s for s in species_meta if s['species_id'] == species_id), None)

    if species_entry is None:
        raise ValueError(f"Invalid species_id: {species_id}")

    base_params = {'species_id': species_id, 'request_src': 'SpringEngine'}
    url = f"{NPN_URL}/observations/getObservations.json"

    years = range(start_year, end_year + 1)

    pbar = tqdm(years, desc="Downloading phenology data")

    with requests.Session() as session:
        for year in pbar:
            obs_file = output_dir / f"{year}.json"

            if obs_file.exists():
                pbar.set_postfix(status='local')
                continue
            else:
                pbar.set_postfix(status='remote')

            params = {**base_params, 'start_date': f"{year}-01-01", 'end_date': f"{year}-12-31"}

            try:
                response = _fetch_with_retry(session=session, url=url, context=f"{year}", params=params, alpha=sleep)
                data = response.json()
            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                e.add_note("Download aborted")
                pbar.close()
                raise

            with obs_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            time.sleep(random.uniform(0.9, 1.1) * sleep)

def download_weather_data(start_year, end_year, region, resolution, variables, output_dir, sleep=10):
    """
    Downloads daily weather data from the PRISM Group API and saves the responses as ZIP archives containing NetCDF grid
    packages. Existing files are skipped to avoid redundant downloads. Grid cells are retrieved for each element in
    `variables` using a summer solstice-aligned phenophase year, which requires data from the preceding calendar year.

    Args:
        start_year (int): First year of data to download.
        end_year (int): Last year of data to download.
        region (str): Geographic boundary of grid packages.
        resolution (str): Grid cell resolution.
        variables (tuple of str): Weather data variables (e.g., ppt, tmax, tmin).
        output_dir (pathlib.Path): Receives grid archives.
        sleep (int | float): Base sleep duration in seconds after a successful download. A ±10% jitter is applied to
            this value. Defaults to 4.

    Raises:
        requests.exceptions.RequestException: If all attempts fail.

    Notes:
        - Uses the module-level `_fetch_with_retry` helper for network requests.
        - PRISM monitors download activity and may restrict access for excessive requests.
    """
    headers = {'User-Agent': 'SpringEngine (phenology research)'}
    grid_code = {'4km': '25m', '800m': '30s'}[resolution]  # NetCDF filename token

    start_date = datetime(start_year - 1, 6, 21)
    end_date = datetime(end_year, 6, 20)
    current_date = start_date
    total_days = (end_date - start_date).days + 1

    pbar = tqdm(total=total_days, desc="Downloading weather data")

    with requests.Session() as session:
        session.headers.update(headers)

        while current_date <= end_date:
            date = current_date.strftime("%Y%m%d")

            for var in variables:
                grid_archive = output_dir / f"{date}_{var}.zip"
                grid_file = output_dir / f"prism_{var}_{region}_{grid_code}_{date}.nc"

                if grid_archive.exists() or grid_file.exists():
                    pbar.set_postfix(status='local')
                    continue
                else:
                    pbar.set_postfix(status='remote')

                url = f"{PRISM_URL}/{region}/{resolution}/{var}/{date}?format=nc"

                try:
                    response = _fetch_with_retry(session=session, url=url, context=date, alpha=sleep, timeout=120)
                except requests.exceptions.RequestException as e:
                    e.add_note("Download aborted")
                    pbar.close()
                    raise

                with grid_archive.open('wb') as f:
                    f.write(response.content)

                time.sleep(random.uniform(0.9, 1.1) * sleep)

            current_date += timedelta(days=1)
            pbar.update(1)

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

    if not (grid_archives := list(io_dir.glob('*.zip'))):
        return

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