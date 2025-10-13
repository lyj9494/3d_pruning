"""
This module contains type annotations for the project, using
1. Python type hints (https://docs.python.org/3/library/typing.html) for Python objects
2. jaxtyping (https://github.com/google/jaxtyping/blob/main/API.md) for PyTorch tensors

Two types of typing checking can be used:
1. Static type checking with mypy (install with pip and enabled as the default linter in VSCode)
2. Runtime type checking with typeguard (install with pip and triggered at runtime, mainly for tensor dtype and shape checking)
"""

# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)
from collections import defaultdict

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt

# Config type
from argparse import Namespace
from collections import defaultdict
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

# PyTorch Tensor type
from torch import Tensor
from torch.nn import Parameter, Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.data_loader import DataLoaderShard

# Runtime type checking decorator
from typeguard import typechecked as typechecker

from typing import *



# Custom types
class FuncArgs(TypedDict):
    """Type for instantiating a function with keyword arguments"""

    name: str
    kwargs: Dict[str, Any]

    @staticmethod
    def validate(variable):
        necessary_keys = ["name", "kwargs"]
        for key in necessary_keys:
            assert key in variable, f"Key {key} is missing in {variable}"
        if not isinstance(variable["name"], str):
            raise TypeError(
                f"Key 'name' should be a string, not {type(variable['name'])}"
            )
        if not isinstance(variable["kwargs"], dict):
            raise TypeError(
                f"Key 'kwargs' should be a dictionary, not {type(variable['kwargs'])}"
            )
        return variable
