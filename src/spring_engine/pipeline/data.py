import pandas as pd
from typing import TYPE_CHECKING
from tqdm.auto import tqdm
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
        labels, label_sites = self._transform_phenology_data(df)

        years = range(self.data_cfg.start_year - 1, self.data_cfg.end_year + 1)
        pbar = tqdm(years, desc="Processing weather data")

        for i, year in enumerate(pbar):
            ds = self._clean_weather_data(year)
            features, year_crosswalk = self._transform_weather_data(ds, label_sites)
            self._write_data(features, year)

            # Establish reference crosswalk and verify grid consistency
            if i == 0:
               crosswalk = year_crosswalk
            elif not crosswalk.equals(year_crosswalk):
               raise ValueError(f"Variant crosswalk for year: {year}")

        labels = labels.merge(crosswalk, how='left', on='site_id', validate='many_to_one')
        self._write_data(labels)

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

    def _transform_phenology_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return transform.compose_labels(
            df=df,
            trans_gap=self.data_cfg.trans_gap,
            cycle_gap=self.data_cfg.cycle_gap
        )

    def _transform_weather_data(self, ds: xr.Dataset, label_sites: pd.DataFrame) -> tuple[xr.Dataset, pd.DataFrame]:
        ds, crosswalk = transform.subset_weather(ds, label_sites)

        ds = transform.engineer_thermal_features(
            ds=ds,
            chill_bounds=self.data_cfg.chill_bounds,
            gdd_bounds=self.data_cfg.gdd_bounds
        )

        ds = transform.engineer_monthly_features(ds)

        return ds, crosswalk

    def _write_data(self, data: pd.DataFrame | xr.Dataset, year: int | None = None):
        if isinstance(data, pd.DataFrame):
            io.write_labels_store(
                df=data,
                start_year=self.data_cfg.start_year,
                end_year=self.data_cfg.end_year,
                output_dir=self.dir_cfg.labels
            )
        elif isinstance(data, xr.Dataset):
            io.write_features_store(
                ds=data,
                year=year,
                output_dir=self.dir_cfg.features
            )
        else:
            raise TypeError(f"Invalid data type: {type(data)}")