set(EXAMPLES
    "matrix_transpose"
    "reduce"
    "synthetic"
    "transpose_scale_add"
    "transpose_scale_add_templated"
)

foreach(example IN LISTS EXAMPLES)
    add_example(${example} ${example}/${example}.hip)
endforeach()
