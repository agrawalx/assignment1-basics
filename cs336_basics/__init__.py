import importlib.metadata
from .train_tokenizer import BPE_tokenizer
from .tokenizer import Tokenizer
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
