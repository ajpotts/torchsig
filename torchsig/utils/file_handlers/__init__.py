"""TorchSig File Handlers"""

from . import base_handler, batched_hdf5, hdf5, hdf5_homogeneous, npy
from .base_handler import BaseFileHandler, FileReader, FileWriter
from .batched_hdf5 import BatchedHDF5Reader, BatchedHDF5Writer
from .hdf5 import HDF5Reader, HDF5Writer
from .hdf5_homogeneous import (
    HomogeneousHDF5Reader,
    HomogeneousHDF5Writer,
)
from .metadata_reader import MetadataIndexError, MetadataReader
from .npy import NPYReader
from .ogg import OGGReader
from .wav import WAVReader

__all__ = [
    "BaseFileHandler",
    "BatchedHDF5Reader",
    "BatchedHDF5Writer",
    "FileReader",
    "FileWriter",
    "HDF5FileHandler",
    "HDF5Reader",
    "HDF5Writer",
    "HomogeneousHDF5Reader",
    "HomogeneousHDF5Writer",
    "MetadataIndexError",
    "MetadataReader",
    "NPYReader",
    "OGGReader",
    "WAVReader",
    "base_handler",
    "batched_hdf5",
    "hdf5",
    "hdf5_homogeneous",
    "npy",
]
