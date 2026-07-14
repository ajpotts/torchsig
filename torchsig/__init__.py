# pylint: disable=missing-module-docstring
__version__ = "2.2.0"

# Importing HierarchicalMetadataObject here helps avoid circular imports
from torchsig.utils.abstractions import HierarchicalMetadataObject

from . import datasets, signals, transforms, utils

__all__ = ["__version__", "datasets", "signals", "transforms", "utils"]
