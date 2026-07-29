"""TorchSig File Handlers"""

from . import base_handler, hdf5, hdf5_batched, npy
from .base_handler import BaseFileHandler, FileReader, FileWriter
from .hdf5 import HDF5Reader, HDF5Writer
from .hdf5_batched import BatchedHDF5Reader, BatchedHDF5Writer
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
    "MetadataIndexError",
    "MetadataReader",
    "NPYReader",
    "OGGReader",
    "WAVReader",
    "base_handler",
    "hdf5",
    "hdf5_batched",
    "npy",
]
