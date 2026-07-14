import pandas as pd
from typing import TYPE_CHECKING
import xarray as xr

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

        df = self._clean_phenology_data()
        label_sites = self._transform_phenology_data(df)

        for year in range(self.data_cfg.start_year - 1, self.data_cfg.end_year + 1):
            ds = self._clean_weather_data(year)
            self._transform_weather_data(ds, label_sites, year)

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

    def _clean_phenology_data(self) -> pd.DataFrame:
        df = clean.load_phenology_data(
            species_id=self.data_cfg.species_id,
            phenophase_id=self.data_cfg.phenophase_id,
            input_dir=self.dir_cfg.obs
        )

        return clean.clean_phenology_data(
            df=df,
            lat_bounds=self.data_cfg.lat_bounds,
            lon_bounds=self.data_cfg.lon_bounds
        )

    def _clean_weather_data(self, year: int) -> xr.Dataset:
        ds = self.loader.load_weather_data(year)

        return clean.clean_weather_data(
            ds=ds,
            variables=self.data_cfg.variables,
            lat_bounds=self.data_cfg.lat_bounds,
            lon_bounds=self.data_cfg.lon_bounds
        )

    def _transform_phenology_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df, label_sites = transform.compose_labels(
            df=df,
            trans_gap=self.data_cfg.trans_gap,
            cycle_gap=self.data_cfg.cycle_gap
        )

        io.write_labels_store(
            df=df,
            start_year=self.data_cfg.start_year,
            end_year=self.data_cfg.end_year,
            output_dir=self.dir_cfg.labels
        )

        return label_sites

    def _transform_weather_data(self, ds: xr.Dataset, label_sites: pd.DataFrame, year: int):
        ds = transform.select_weather_subset(ds, label_sites)

        ds = transform.engineer_thermal_features(
            ds=ds,
            chill_bounds=self.data_cfg.chill_bounds,
            gdd_bounds=self.data_cfg.gdd_bounds
        )

        ds = transform.engineer_monthly_features(ds)

        io.write_features_store(
            ds=ds,
            year=year,
            output_dir=self.dir_cfg.features
        )