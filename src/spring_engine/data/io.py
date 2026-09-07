import pandas as pd

def write_labels_store(df, start_year, end_year, output_dir):
    """
    Saves phenology labels representing a range [start_year, end_year] as a Parquet dataset.

    Args:
        df (pd.DataFrame): Phenophase onset labels, including spatial coordinates and metadata.
        start_year (int): First year of data represented by the labels.
        end_year (int): Last year of data represented by the labels.
        output_dir (pathlib.Path): Receives the label file.

    Raises:
        Exception: If export fails. Any incomplete Parquet file is removed.
    """
    labels_file = output_dir / f"{start_year}_{end_year}.parquet"

    try:
        df.to_parquet(labels_file, engine='pyarrow')
    except Exception as e:
        e.add_note(f"Failed to export labels")
        labels_file.unlink(missing_ok=True)
        raise

def write_features_store(ds, year, output_dir):
    """
    Saves weather features for a given year as a Hive-style partition of a Parquet dataset.

    Args:
        ds (xr.Dataset): Features dataset with dimensions [lat, lon].
        year (int): Calendar year represented by the features.
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

def read_labels_store(start_year, end_year, input_dir):
    """
    Reads phenology labels representing a range [start_year, end_year] from a Parquet dataset.

    Args:
        start_year (int): First year of data represented by the labels.
        end_year (int): Last year of data represented by the labels.
        input_dir (pathlib.Path): Contains the label file.

    Returns:
        pd.DataFrame: Phenophase onset labels, including spatial coordinates and metadata.

    Raises:
        FileNotFoundError: If the Parquet label file does not exist under `input_dir`.
    """
    labels_file = input_dir / f"{start_year}_{end_year}.parquet"

    if not labels_file.exists():
        raise FileNotFoundError(f"No labels for range: [{start_year}, {end_year}]")

    return pd.read_parquet(labels_file, engine='pyarrow')

def read_features_store(start_year, end_year, input_dir):
    """
    Reads weather features for a range [start_year, end_year] from Hive-style partitions of a Parquet dataset.

    Args:
        start_year (int): First year of data represented by the features.
        end_year (int): Last year of data represented by the features.
        input_dir (pathlib.Path): Parent directory that contains the `year=<year>` partitions.

    Returns:
        pd.DataFrame: Weather features containing a full climate history within the specified range.

    Raises:
        FileNotFoundError: If any of the Parquet partitions do not exist under `input_dir`.
    """
    years =  list(range(start_year, end_year + 1))
    missing = []

    for year in years:
        features_file = input_dir / f"year={year}/data.parquet"

        if not features_file.exists():
            missing.append(f"year={year}")

    if missing:
        raise FileNotFoundError(f"No features for partitions: {missing}")

    return pd.read_parquet(input_dir, engine='pyarrow', filters=[('year', 'in', years)])