set(EXAMPLES
    "reduction"
    "reduction_optimized"
)

foreach(example IN LISTS EXAMPLES)
    add_example(${example} ${example}/${example}.hip)
endforeach() 
