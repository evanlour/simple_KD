import importlib.metadata

try:
    __version__ = importlib.metadata.version("simple_kd")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
