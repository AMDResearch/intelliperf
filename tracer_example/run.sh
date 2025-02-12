exec="dummy"

if [ ! -f "$exec" ]; then
    echo "Compiling $exec.hip"
    if hipcc -g -O0 -o "$exec" "$exec.hip"; then
        echo "Compilation successful."
    else
        echo "Compilation failed."
        exit 1
    fi
fi

LOG_MAESTRO_TRACER=1 LD_PRELOAD=../src/tracer/tracer.so ./"$exec"
