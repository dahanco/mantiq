# Mantiq's CPU build: mntq-cpu runs GLM-5.3-Flash (glm5-next) and Qwen3.5/3.6-MoE GGUFs through the pinned llama.cpp's
# ggml with its CPU backend alone - no CUDA, no HIP, no Metal.  The engine source is the Metal build's (src/metal),
# which already carries a full CPU path (GlmOptions::use_gpu = false, the oracle the Metal path is checked against), so
# this build is that engine with the Metal backend left out and use_gpu defaulting to false.  It needs none of the
# CUDA/HIP engine and none of the x86 CPU kernels (AVX-512/AVX2, cpuid), so this build is its own short list of targets;
# the top-level CMakeLists.txt returns right after including it.
#
#   -DMNTQ_ENABLE_CPU=ON              this build
#   -DMNTQ_GGML_DIR=<llama.cpp>       a checkout at the pinned commit (default: fetched)
#   -DMNTQ_CPU_PORTABLE=ON            no -mcpu=native (a build for other CPUs)
enable_language(C)

option(MNTQ_CPU_PORTABLE "Build mntq-cpu for any CPU" OFF)
set(MNTQ_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(MNTQ_CPU_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()

if(MNTQ_GGML_DIR)
  set(_mntq_llamacpp_dir ${MNTQ_GGML_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(mntq_llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
    GIT_SHALLOW FALSE)
  FetchContent_GetProperties(mntq_llamacpp)
  if(NOT mntq_llamacpp_POPULATED)
    FetchContent_Populate(mntq_llamacpp)
  endif()
  set(_mntq_llamacpp_dir ${mntq_llamacpp_SOURCE_DIR})
endif()
# third_party/ggml's two patches patch the Metal backend and its allocator, and this build has neither: plain ggml
add_subdirectory(${_mntq_llamacpp_dir}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

# the engine reports Mantiq's release (VERSION) as its version
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _mntq_version LIMIT_COUNT 1)
add_library(mntq_cpu STATIC src/metal/glm_metal.cpp src/metal/glm_metal_load.cpp src/metal/expert_pool.cpp
                            src/metal/glm_metal_dflash.cpp src/metal/spec.cpp src/metal/suffix_drafter.cpp
                            src/metal/sampler.cpp)
target_include_directories(mntq_cpu PUBLIC src/metal)
target_compile_definitions(mntq_cpu PUBLIC MNTQ_VERSION="${_mntq_version}" MNTQ_CPU_ONLY=1 MNTQ_EXE_NAME="mntq-cpu")
target_link_libraries(mntq_cpu PUBLIC ggml ggml-base ggml-cpu)

add_executable(mntq-cpu src/metal/main.cpp src/metal/serve.cpp)
target_link_libraries(mntq-cpu PRIVATE mntq_cpu)
