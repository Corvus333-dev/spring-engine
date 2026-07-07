def write_features_store(ds, year, output_dir):
    """
    Saves weather features for a given year as a Hive-style partition of a Parquet dataset.

    Args:
        ds (xr.Dataset): Features dataset with dimensions [lat, lon].
        year (int): Calendar year represented by the features.
        output_dir (pathlib.Path): Parent directory that receives the `year=<year>` partition.

    Raises:
        Exception: If materialization or export fails. Any incomplete Parquet file is removed.

    Notes:
        Triggers high-compute eager evaluation of the xarray dataset.
    """
    partition_dir = output_dir / f"year={year}" # Hive-style partition
    partition_dir.mkdir(parents=True, exist_ok=True)

    partition_file = partition_dir / 'data.parquet'

    try:
        df = ds.to_dataframe().reset_index() # IMPORTANT: materialize lazy dataset
        df.to_parquet(partition_file, engine='pyarrow')
    except Exception as e:
        e.add_note(f"Failed to export partition: year={year}")
        partition_file.unlink(missing_ok=True)
        raise

def write_labels_store(df, start_year, end_year, output_dir):
    """
    Saves phenology labels representing a range [start_year, end_year] as a Parquet dataset.

    Args:
        df (pd.DataFrame): Phenophase onset labels, including spatial coordinates and metadata.
        start_year (int): First year of data represented by the labels.
        end_year (int): Last year of data represented by the labels.
        output_dir (pathlib.Path): Receives label files.

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