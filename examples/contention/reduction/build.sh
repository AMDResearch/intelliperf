script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd $script_dir
hipcc initial_code.hip -o reduction.unoptimized
hipcc optimized_code.hip -o reduction.optimized
popd