cmake_minimum_required(VERSION 3.10)
project(Examples LANGUAGES HIP)

add_library(maestro_examples INTERFACE)

target_compile_options(maestro_examples INTERFACE -Wno-unused-result -O3 -g)
option(INSTRUMENT "Instrument the code with Omniprobe" OFF)

function(add_example name source_file)
    add_executable(${name} ${source_file})
    target_link_libraries(${name} PRIVATE maestro_examples)
    if(INSTRUMENT)
        target_link_libraries(${name} PRIVATE -L${CMAKE_SOURCE_DIR}/../external/logduration/build -lMemAnalysis64)
    endif()
endfunction()

add_subdirectory(bank_conflict)
add_subdirectory(basic)
add_subdirectory(contention)
