import pandas as pd

def write_table(df, start_year, end_year, output_dir):
    """
    Saves a DataFrame representing a range [start_year, end_year] as a Parquet file.

    Args:
        df (pd.DataFrame): Data to write to the table.
        start_year (int): First year of data represented by the DataFrame.
        end_year (int): Last year of data represented by the DataFrame.
        output_dir (pathlib.Path): Receives the Parquet file.

    Raises:
        Exception: If export fails. Any incomplete Parquet file is removed.
    """
    table_file = output_dir / f"{start_year}_{end_year}.parquet"

    try:
        df.to_parquet(table_file, engine='pyarrow')
    except Exception as e:
        e.add_note("Failed to export table")
        table_file.unlink(missing_ok=True)
        raise

def write_partition(ds, year, output_dir):
    """
    Saves an xarray Dataset for a given year as a Hive-style partition of a Parquet dataset.

    Args:
        ds (xr.Dataset): Data to write to the partition.
        year (int): Calendar year represented by the data.
        output_dir (pathlib.Path): Parent directory that receives the `year=<year>` partition.

    Raises:
        Exception: If export fails. Any incomplete Parquet file is removed.
    """
    partition_dir = output_dir / f"year={year}" # Hive-style partition
    partition_dir.mkdir(parents=True, exist_ok=True)

    partition_file = partition_dir / 'data.parquet'

    df = ds.to_dataframe().reset_index()

    try:
        df.to_parquet(partition_file, engine='pyarrow')
    except Exception as e:
        e.add_note(f"Failed to export partition: year={year}")
        partition_file.unlink(missing_ok=True)
        raise

def read_table(start_year, end_year, input_dir):
    """
    Reads a DataFrame representing a range [start_year, end_year] from a Parquet file.

    Args:
        start_year (int): First year of data represented by the DataFrame.
        end_year (int): Last year of data represented by the DataFrame.
        input_dir (pathlib.Path): Contains the Parquet file.

    Returns:
        pd.DataFrame: Data read from the table.

    Raises:
        FileNotFoundError: If the Parquet file does not exist under `input_dir`.
    """
    table_file = input_dir / f"{start_year}_{end_year}.parquet"

    if not table_file.exists():
        raise FileNotFoundError(f"No table for range: [{start_year}, {end_year}]")

    return pd.read_parquet(table_file, engine='pyarrow')

def read_partitions(start_year, end_year, input_dir):
    """
    Reads a DataFrame for a range [start_year, end_year] from Hive-style partitions of a Parquet dataset.

    Args:
        start_year (int): First year of data represented by the dataset.
        end_year (int): Last year of data represented by the dataset.
        input_dir (pathlib.Path): Parent directory that contains the `year=<year>` partitions.

    Returns:
        pd.DataFrame: Data read from the partitions.

    Raises:
        FileNotFoundError: If any of the Parquet partitions do not exist under `input_dir`.
    """
    years =  list(range(start_year, end_year + 1))
    missing = []

    for year in years:
        partition_file = input_dir / f"year={year}/data.parquet"

        if not partition_file.exists():
            missing.append(f"year={year}")

    if missing:
        raise FileNotFoundError(f"No partitions for years: {missing}")

    return pd.read_parquet(input_dir, engine='pyarrow', filters=[('year', 'in', years)])