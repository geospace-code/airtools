cmake_minimum_required(VERSION 3.15)

project(airtools LANGUAGES Fortran)

enable_testing()

add_library(logmart fortran/logmart.f90)

add_executable(test_logmart fortran/test_logmart.f90 fortran/random_utils.f90)
target_link_libraries(test_logmart PRIVATE logmart)

add_test(NAME Logmart COMMAND test_logmart)
