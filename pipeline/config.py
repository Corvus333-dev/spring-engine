from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataConfig:
    start_year: int = 2009
    end_year: int = 2025
    species_id: int = 36
    phenophase_id: int = 373

    region: str = 'us'
    resolution: str = '4km'
    variables: tuple[str, ...] = ('ppt', 'tmax', 'tmin')

    lat_bounds: tuple[float, float] = (24.0, 50.0)
    lon_bounds: tuple[float, float] = (-125.0, -66.0)

    chill_bounds = (0.0, 7.0)
    gdd_bounds = (10.0, 30.0)

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