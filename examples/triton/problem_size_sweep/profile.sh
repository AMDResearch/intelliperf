#!/bin/bash

# Function to handle errors without exiting the shell
handle_error() {
  echo "Error: $1"
}

# Check for input argument
if [ $# -lt 1 ]; then
  handle_error "Usage: $0 <python_file> --version <swizzle|unswizzle> [problem_size_args...]"
  exit 1
fi

# Extract filename without extension
FILE="$1"
shift  # Remove the script name from arguments

# Parse arguments
VERSION=""
PROBLEM_ARGS=""
ITERATION=""
while (( "$#" )); do
  case "$1" in
    --version)
      VERSION="$2"
      shift 2
      ;;
    --iteration)
      ITERATION="$2"
      shift 2
      ;;
    *)
      PROBLEM_ARGS="$PROBLEM_ARGS $1"
      shift
      ;;
  esac
done

if [ -z "$VERSION" ]; then
    echo "Error: --version argument is required."
    exit 1
fi
if [ -z "$ITERATION" ]; then
    echo "Error: --iteration argument is required."
    exit 1
fi

# Create a unique workload name
BASENAME=$(basename "$FILE" .py)
WORKLOAD_NAME="${BASENAME}_${VERSION}_${ITERATION}"
WORKLOAD_PATH="workloads/${WORKLOAD_NAME}"

DEVICE=0
LOG_FILE="${BASENAME}_analysis.log"
PROFILER="rocprof-compute"
UNIT="ms"

EXTENSION="${FILE##*.}"
if [ "$EXTENSION" == "hip" ]; then
  EXECUTABLE="${BASENAME}.out"
else
  EXECUTABLE="$FILE"
fi

if ! command -v "$PROFILER" &> /dev/null; then
  echo "Error: $PROFILER is not installed or not in PATH."
  exit 1
fi

# Compile the HIP file
if [ "$EXTENSION" == "hip" ]; then
  echo "Compiling $FILE..."
  hipcc "$FILE" -o "$EXECUTABLE"
  if [ $? -ne 0 ]; then
    handle_error "Compilation failed for $FILE."
    exit 1
  fi
  echo "Compilation successful. Executable: $EXECUTABLE"
fi

# Profile the executable using rocprof-compute with a workload name
echo "Profiling $EXECUTABLE with rocprof-compute..."
rm -rf "$WORKLOAD_PATH"

if [ "$EXTENSION" == "py" ]; then
  $PROFILER profile --name "$WORKLOAD_NAME" --device $DEVICE -- python "$EXECUTABLE" --version "$VERSION" $PROBLEM_ARGS
else
  $PROFILER profile --name "$WORKLOAD_NAME" --device $DEVICE -- ./"$EXECUTABLE" --version "$VERSION" $PROBLEM_ARGS
fi

if [ $? -ne 0 ]; then
  handle_error "Profiling failed for $EXECUTABLE."
  exit 1
fi

echo "Profiling completed successfully."

# Analyze the profiling results and log output
echo "Analyzing the profiling data..."

GPU_WORKLOAD_PATH=$(find "$WORKLOAD_PATH" -mindepth 1 -maxdepth 1 -type d | head -n 1)

$PROFILER analyze -p "$GPU_WORKLOAD_PATH" --time-unit $UNIT --save-dfs "$GPU_WORKLOAD_PATH/analyze" | tee "$LOG_FILE"
if [ $? -ne 0 ]; then
  handle_error "Analysis failed for workload at $WORKLOAD_PATH."
  exit 1
fi

echo "Analysis completed successfully. Results saved to $LOG_FILE."