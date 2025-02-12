include_guard()

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/RyzenCompilerOptions.cmake)

CPMAddPackage("gh:catchorg/Catch2@3.4.0")
message("catch2_SOURCE_DIR: ${Catch2_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

include(CTest)
include(Catch)

function(ryzen_add_test TARGET)
    add_executable(${TARGET} ${ARGN})
    target_link_libraries(${TARGET} PRIVATE ryzendf::ryzendf
                                    PRIVATE Catch2::Catch2WithMain)
    ryzen_compiler_warnings(${TARGET})
    ryzen_compiler_options(${TARGET})
    catch_discover_tests(${TARGET}
                        DISCOVERY_MODE PRE_TEST)
endfunction()
