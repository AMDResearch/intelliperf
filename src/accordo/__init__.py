"""Accordo: Automated side-by-side correctness validation for GPU kernels.

Public API:
    - ValidationConfig: Configuration for kernel validation
    - KernelArg: Structured kernel argument representation
    - ValidationResult: Result of validation with detailed metrics
    - ArrayMismatch: Information about array validation failures
    - AccordoValidator: Main validator class (TODO: implement)
    - Exceptions: AccordoError, AccordoBuildError, AccordoTimeoutError, etc.

Example:
    >>> from accordo import ValidationConfig, KernelArg
    >>> config = ValidationConfig(
    ...     kernel_name="my_kernel",
    ...     kernel_args=[
    ...         KernelArg(name="result", type="double*", direction="out"),
    ...         KernelArg(name="input", type="const double*", direction="in"),
    ...     ],
    ...     tolerance=1e-6
    ... )
"""

# Public API exports
from .config import KernelArg, ValidationConfig
from .exceptions import (
	AccordoBuildError,
	AccordoError,
	AccordoProcessError,
	AccordoTimeoutError,
	AccordoValidationError,
)
from .result import ArrayMismatch, ValidationResult

# Version
__version__ = "0.2.0-dev"

# Public API
__all__ = [
	# Config
	"ValidationConfig",
	"KernelArg",
	# Results
	"ValidationResult",
	"ArrayMismatch",
	# Exceptions
	"AccordoError",
	"AccordoBuildError",
	"AccordoTimeoutError",
	"AccordoProcessError",
	"AccordoValidationError",
	# TODO: Add AccordoValidator when implemented
	# "AccordoValidator",
]
