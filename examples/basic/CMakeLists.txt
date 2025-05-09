set(EXAMPLES
    "vector_add"
)

foreach(example IN LISTS EXAMPLES)
    add_example(${example} ${example}/${example}.hip)
endforeach() 
