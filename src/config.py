from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

@dataclass(frozen=True)
class DataConfig:
    start_year: int = 2009
    end_year: int = 2025
    species_id: int = 36
    phenophase_id: int = 373

    # Grid specs
    resolution: str = '4km'
    variables: Tuple[str, ...] = ('ppt', 'tmax', 'tmin')

    # Data extrema
    latitude: Tuple[float, float] = (24.396308, 49.384358)
    longitude: Tuple[float, float] = (-124.848974, -66.885444)
    precipitation: Tuple[int, int] = (0, 1092)
    temperature: Tuple[float, float] = (-56.7, 56.7)
    day_of_year: Tuple[int, int] = (1, 366)

class DirConfig:
    def __init__(self, cfg: DataConfig):
        root = Path(__file__).resolve().parents[1]
        data = root / 'data'

        self.meta = data / 'phenology' / 'metadata'
        self.obs = data / 'phenology' / 'observations' / str(cfg.species_id)
        self.grids = data / 'weather' / 'grids' / cfg.resolution

        self._create_dirs()

    def _create_dirs(self):
        for path in (self.meta, self.obs, self.grids):
            path.mkdir(parents=True, exist_ok=True)

# Static initialization
data_cfg = DataConfig()
dir_cfg = DirConfig(data_cfg)