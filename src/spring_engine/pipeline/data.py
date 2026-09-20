import pandas as pd
from typing import TYPE_CHECKING
from tqdm.auto import tqdm

from spring_engine.data import clean, ingest, io, transform

if TYPE_CHECKING:
    from spring_engine.config import DataConfig, DirConfig

class DataPipeline:
    def __init__(self, data_cfg: 'DataConfig', dir_cfg: 'DirConfig'):
        self.data_cfg = data_cfg
        self.dir_cfg = dir_cfg
        self.loader = clean.WeatherLoader(input_dir=dir_cfg.grids)

    def run(self):
        self._ingest_data()
        labels, label_sites = self._process_labels()
        crosswalk = self._process_features(label_sites)
        self._process_examples(labels, crosswalk)

    def _ingest_data(self):
        ingest.download_phenology_metadata(output_dir=self.dir_cfg.meta)

        ingest.download_phenology_data(
            species_id=self.data_cfg.species_id,
            start_year=self.data_cfg.start_year,
            end_year=self.data_cfg.end_year,
            input_dir=self.dir_cfg.meta,
            output_dir=self.dir_cfg.obs
        )

        ingest.download_weather_data(
            start_year=self.data_cfg.start_year - 1,
            end_year=self.data_cfg.end_year,
            region=self.data_cfg.region,
            resolution=self.data_cfg.resolution,
            variables=self.data_cfg.variables,
            output_dir=self.dir_cfg.grids
        )

        ingest.extract_weather_data(io_dir=self.dir_cfg.grids)

    def _process_labels(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = clean.load_phenology_data(
            species_id=self.data_cfg.species_id,
            phenophase_id=self.data_cfg.phenophase_id,
            input_dir=self.dir_cfg.obs
        )

        df = clean.clean_phenology_data(
            df=df,
            lat_bounds=self.data_cfg.lat_bounds,
            lon_bounds=self.data_cfg.lon_bounds
        )

        return transform.compose_labels(
            df=df,
            trans_gap=self.data_cfg.trans_gap,
            cycle_gap=self.data_cfg.cycle_gap
        )

    def _process_features(self, label_sites: pd.DataFrame) -> pd.DataFrame:
        years = range(self.data_cfg.start_year - 1, self.data_cfg.end_year + 1)
        pbar = tqdm(years, desc="Processing features")

        for i, year in enumerate(pbar):
            ds = self.loader.load_weather_data(year)

            ds = clean.clean_weather_data(
                ds=ds,
                variables=self.data_cfg.variables,
                lat_bounds=self.data_cfg.lat_bounds,
                lon_bounds=self.data_cfg.lon_bounds
            )

            ds, year_crosswalk = transform.subset_weather(ds, label_sites)

            ds = transform.engineer_thermal_features(
                ds=ds,
                chill_bounds=self.data_cfg.chill_bounds,
                gdd_bounds=self.data_cfg.gdd_bounds
            )

            features = transform.engineer_monthly_features(ds)

            # Establish reference crosswalk and verify grid consistency
            if i == 0:
               crosswalk = year_crosswalk
            elif not crosswalk.equals(year_crosswalk):
               raise ValueError(f"Variant crosswalk for year: {year}")

            io.write_partition(
                ds=features,
                year=year,
                output_dir=self.dir_cfg.features
            )

        return crosswalk

    def _process_examples(self, labels: pd.DataFrame, crosswalk: pd.DataFrame):
        labels = labels.merge(crosswalk, how='left', on='site_id', validate='many_to_one')

        features = io.read_partitions(
            start_year=self.data_cfg.start_year - 1,
            end_year=self.data_cfg.end_year,
            input_dir=self.dir_cfg.features
        )

        examples = transform.compose_examples(labels, features)

        io.write_table(
            df=examples,
            start_year=self.data_cfg.start_year,
            end_year=self.data_cfg.end_year,
            output_dir=self.dir_cfg.examples
        )