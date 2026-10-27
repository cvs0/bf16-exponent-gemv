cmake_minimum_required(VERSION 3.16)
project(expcode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(EXPCODE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Clang selects the newest /usr/lib/gcc tree. Some images install a newer
# libgcc without the matching libstdc++ headers. Point Clang at a GCC
# install that has C++ headers when that happens.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_expcode_gcc_install "")
    set(_expcode_gcc_best 0)
    file(GLOB _expcode_cxx_roots LIST_DIRECTORIES true "/usr/include/c++/*")
    foreach(_root IN LISTS _expcode_cxx_roots)
        get_filename_component(_ver "${_root}" NAME)
        if(_ver MATCHES "^[0-9]+$" AND EXISTS "${_root}/cstdio"
           AND EXISTS "/usr/lib/gcc/x86_64-linux-gnu/${_ver}")
            if(_ver GREATER _expcode_gcc_best)
                set(_expcode_gcc_best "${_ver}")
                set(_expcode_gcc_install "/usr/lib/gcc/x86_64-linux-gnu/${_ver}")
            endif()
        endif()
    endforeach()
    if(_expcode_gcc_install)
        add_compile_options("--gcc-install-dir=${_expcode_gcc_install}")
        add_link_options("--gcc-install-dir=${_expcode_gcc_install}")
    endif()
endif()

find_package(OpenMP REQUIRED)

set(EXPCODE_WARNINGS -Wall -Wextra -Wpedantic -Werror)
# Keep the scalar stripe order intact. Explicit SIMD intrinsics are unaffected.
set(EXPCODE_FP -mfma -ffp-contract=off -fno-math-errno -fno-strict-aliasing)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND EXPCODE_FP -fno-vectorize -fno-slp-vectorize)
else()
    list(APPEND EXPCODE_FP -fno-tree-vectorize -fno-tree-slp-vectorize)
endif()

add_library(expcode_kernels STATIC
    src/gemv_avx2.cpp
    src/gemv_avx512.cpp
    src/gemv_dispatch.cpp
)
target_include_directories(expcode_kernels
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(expcode_kernels PRIVATE ${EXPCODE_WARNINGS} ${EXPCODE_FP})
target_link_libraries(expcode_kernels PUBLIC OpenMP::OpenMP_CXX)
set_source_files_properties(src/gemv_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(src/gemv_avx512.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma")

add_executable(expcode_tests tests/test_main.cpp)
target_link_libraries(expcode_tests PRIVATE expcode_kernels)
target_compile_options(expcode_tests PRIVATE ${EXPCODE_WARNINGS} ${EXPCODE_FP})

add_executable(expcode_bench src/bench_main.cpp src/model.cpp)
target_include_directories(expcode_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(expcode_bench PRIVATE expcode_kernels)
target_compile_options(expcode_bench PRIVATE ${EXPCODE_WARNINGS} ${EXPCODE_FP})

if(EXPCODE_SANITIZERS)
    foreach(tgt expcode_kernels expcode_tests expcode_bench)
        target_compile_options(${tgt} PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
        target_link_options(${tgt} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()

enable_testing()
add_test(NAME expcode_tests COMMAND expcode_tests)

install(DIRECTORY include/expcode DESTINATION include)
