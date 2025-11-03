"""Accordo: Automated side-by-side correctness validation for GPU kernels.

Public API:
    - ValidationConfig: Configuration for kernel validation
    - KernelArg: Structured kernel argument representation
    - Snapshot: Captured kernel argument data from binary execution
    - ValidationResult: Result of validation with detailed metrics
    - ArrayMismatch: Information about array validation failures
    - AccordoValidator: Main validator class for kernel validation
    - Exceptions: AccordoError, AccordoBuildError, AccordoTimeoutError, etc.

Quick Example (one-off validation):
    >>> from accordo import ValidationConfig, KernelArg, AccordoValidator
    >>> config = ValidationConfig(
    ...     kernel_name="my_kernel",
    ...     kernel_args=[
    ...         KernelArg(name="result", type="double*"),
    ...         KernelArg(name="input", type="const double*"),
    ...     ],
    ...     tolerance=1e-6
    ... )
    >>> validator = AccordoValidator(config)
    >>> result = validator.validate(
    ...     reference_binary=["./app_ref"],
    ...     optimized_binary=["./app_opt"],
    ...     working_directory=".",
    ...     baseline_time_ms=10.0
    ... )

Efficient Example (multiple optimizations vs same reference):
    >>> # Capture reference once (returns Snapshot object)
    >>> ref_snapshot = validator.capture_snapshot(
    ...     binary=["./app_ref"],
    ...     working_directory=".",
    ...     timeout_seconds=30
    ... )
    >>> print(ref_snapshot)  # Snapshot(binary='./app_ref', arrays=3, execution_time_ms=12.50)
    >>>
    >>> # Compare multiple optimizations
    >>> for opt_binary in optimized_binaries:
    ...     opt_snapshot = validator.capture_snapshot(
    ...         binary=opt_binary,
    ...         working_directory=".",
    ...         timeout_seconds=60
    ...     )
    ...     result = validator.compare_snapshots(ref_snapshot, opt_snapshot)
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
from .snapshot import Snapshot
from .validator import AccordoValidator

# Version
__version__ = "0.2.0"

# Public API
__all__ = [
	# Config
	"ValidationConfig",
	"KernelArg",
	# Validator
	"AccordoValidator",
	# Snapshot
	"Snapshot",
	# Results
	"ValidationResult",
	"ArrayMismatch",
	# Exceptions
	"AccordoError",
	"AccordoBuildError",
	"AccordoTimeoutError",
	"AccordoProcessError",
	"AccordoValidationError",
]
