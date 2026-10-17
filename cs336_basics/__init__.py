import importlib.metadata
from .tokenizer import BPE_tokenizer
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
