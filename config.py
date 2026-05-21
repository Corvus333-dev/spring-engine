from pathlib import Path

class Golem:
    """
    A self-destructing lazy initializer that creates a target directory on first access and then displaces itself on the
    owner class with the resulting Path object (i.e., subsequent access yields the Path object directly).

    Args:
        subpath (str): Directory path relative to owner's 'ROOT' attribute.
    """

    def __init__(self, subpath):
        self.subpath = subpath

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        path = owner.ROOT / self.subpath
        path.mkdir(parents=True, exist_ok=True)

        # Crumble: overwrite descriptor
        setattr(owner, self.name, path)

        # Only reached on first access
        return path

class Fortress:
    """Central configuration and lazy-initialization registry for project paths"""
    ROOT = Path(__file__).resolve().parent

    metadata = Golem('data/phenology/metadata')
    observations = Golem('data/phenology/observations')
    grids = Golem('data/weather/grids')