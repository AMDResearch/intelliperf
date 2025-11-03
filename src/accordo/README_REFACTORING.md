# Accordo API Refactoring Progress

## ✅ Completed

### 1. Package Structure
- Created `_internal/` directory for private implementation
- Created `_internal/ipc/` subdirectory for IPC modules
- Added proper `__init__.py` files

### 2. Public API Classes

#### `config.py` - Configuration
- ✅ `KernelArg`: Structured kernel argument with name, type, and direction
  - Supports auto-inference of direction from type
  - Backward compatible with plain strings and dicts
- ✅ `ValidationConfig`: Configuration for validation
  - Supports `kernel_args` as KernelArg, str, or dict
  - Includes `additional_includes` for custom types
  - Configurable tolerance and timeout

#### `result.py` - Results
- ✅ `ArrayMismatch`: Detailed mismatch information
- ✅ `ValidationResult`: Validation result with metrics
  - Properties: `num_arrays_validated`, `success_rate`
  - Method: `summary()` for human-readable output

#### `exceptions.py` - Exceptions
- ✅ `AccordoError`: Base exception
- ✅ `AccordoBuildError`: Build failures
- ✅ `AccordoTimeoutError`: Timeout errors (with timeout_seconds attribute)
- ✅ `AccordoProcessError`: Process crashes (with exit_code attribute)
- ✅ `AccordoValidationError`: Validation failures

### 3. Internal Modules

#### `_internal/codegen.py`
- ✅ `generate_kernel_header()`: Creates KernelArguments.hpp
  - Supports `additional_includes` parameter
  - Generates flat argument structures

#### `_internal/hip.py`
- ✅ `hip_try()`: HIP error checking
- ✅ `open_ipc_handle()`: Open IPC memory handles
- ✅ `memcpy_d2h()`: Device-to-host memory copy

#### `_internal/ipc/communication.py`
- ✅ `read_ipc_handles()`: Read IPC handles from file
- ✅ `send_response()`: Send completion signal via pipe
- ✅ `get_kern_arg_data()`: Get kernel argument data via IPC
  - Supports dynamic timeout based on baseline
  - Type mapping for common GPU types (float*, double*, __half*, etc.)

### 4. Public API Export
- ✅ Updated `__init__.py` to export public API classes
- ✅ Clean imports: `from accordo import ValidationConfig, KernelArg, ValidationResult`

## 🚧 TODO

### 1. AccordoValidator Implementation
The main validator class needs to be implemented. This will:
- Wrap the existing functionality from `formula_base.py`
- Manage Accordo C++ library building (via CMake)
- Handle process management and IPC communication
- Provide the `validate(reference_app, optimized_app)` method

**Key Design Decisions:**
- Lazy build on first use
- Auto-detect Accordo path
- Dynamic timeout calculation
- Clean error handling with custom exceptions

### 2. Builder Module (`_internal/builder.py`)
Manage CMake builds of Accordo C++ library:
- Check if build needed
- Run CMake with proper flags
- Handle build errors
- Cache build results

### 3. Runtime Module (`_internal/runtime.py`)
Manage process execution and IPC:
- Launch instrumented processes
- Monitor process health
- Handle timeouts
- Clean up resources

### 4. Integration with IntelliPerf
Update `formula_base.py` to use the new API:
```python
from accordo import AccordoValidator, ValidationConfig, KernelArg

config = ValidationConfig(
    kernel_name=kernel,
    kernel_args=[KernelArg(name=f"arg{i}", type=t) for i, t in enumerate(kernel_args)],
    tolerance=tolerance
)

validator = AccordoValidator(config)
result = validator.validate(
    reference_app=self._reference_app,
    optimized_app=self._application,
    baseline_time_ms=getattr(self, "baseline_time_ms", None)
)

return Result(
    success=result.is_valid,
    error_report=result.error_message if not result.is_valid else ""
)
```

### 5. Tests
- Unit tests for each module
- Integration tests with real kernels
- Test backward compatibility

### 6. Documentation
- API reference docs
- Usage examples
- Migration guide

## 📝 Design Notes

### Flat Arguments Only (For Now)
Current implementation supports only flat argument lists:
```python
# ✅ Supported
kernel_args=[
    KernelArg("output", "float*"),
    KernelArg("input", "const float*"),
    KernelArg("size", "int")
]

# ❌ Not yet supported (nested structs)
kernel_args=[
    KernelArg("matrix", "Matrix*")  # Matrix has nested pointers
]
```

Future enhancement: Add visitor/accessor pattern for nested types.

### Backward Compatibility
The API accepts multiple input formats:
- `KernelArg` objects (new, preferred)
- Plain strings (backward compatible)
- Dicts (for JSON/LLM generation)

All formats are automatically normalized to `KernelArg` instances.

### Type Mapping
Supported GPU types in `_internal/ipc/communication.py`:
- `double*`, `float*`
- `int*`, `std::size_t*`
- `__half*` (FP16)
- `__hip_bfloat16*` (BFloat16)

Additional types can be added to the `type_map` dictionary.

## 🎯 Next Steps

1. **Implement AccordoValidator** - This is the main missing piece
2. **Test with existing formulas** - Ensure backward compatibility
3. **Update formula_base.py** - Integrate new API
4. **Add comprehensive tests** - Cover edge cases
5. **Document migration path** - Help users adopt new API

## 📚 Reference

See `accordo.todo` for the complete design document.

