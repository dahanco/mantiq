# Mantiq's Metal build (Apple Silicon): mntq-metal runs GLM-5.3-Flash through the pinned llama.cpp's ggml with its Metal
# backend.  It needs none of the CUDA/HIP engine and none of the x86 CPU kernels (AVX-512/AVX2, cpuid), so this build
# is its own short list of targets; the top-level CMakeLists.txt returns right after including it.
#
#   -DMNTQ_ENABLE_METAL=ON            this build
#   -DMNTQ_GGML_DIR=<llama.cpp>       a checkout at the pinned commit (default: fetched)
#   -DMNTQ_METAL_PORTABLE=ON          no -mcpu=native (a build for other Macs)
if(NOT APPLE)
  message(FATAL_ERROR "MNTQ_ENABLE_METAL builds for macOS on Apple Silicon")
endif()
enable_language(C OBJC)

option(MNTQ_METAL_PORTABLE "Build mntq-metal for any Apple Silicon Mac" OFF)
set(MNTQ_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)   # the shaders live inside the binary: nothing to install
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(MNTQ_METAL_PORTABLE)
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
# ggml is built from a copy in the build tree with third_party/ggml/metal-sparse-mla.patch applied (ggml_sparse_mla:
# the DSA layers' sparse attention as one Metal kernel; the checkout stays untouched).  The copy is redone when the
# checkout, its ggml.h or the patch changes.  A checkout the patch does not apply to (or no git) builds as it is, and the
# engine keeps its gather + mul_mm attention.
set(_mntq_gg ${_mntq_llamacpp_dir}/ggml)
set(_mntq_mla_patch ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/metal-sparse-mla.patch)
set(_mntq_alloc_patch ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/metal-allocator-free-blocks.patch)
set(_mntq_mla_copy ${CMAKE_BINARY_DIR}/mntq-ggml)
file(SHA256 ${_mntq_mla_patch} _mntq_mla_patch_sha)
file(SHA256 ${_mntq_alloc_patch} _mntq_alloc_patch_sha)
file(SHA256 ${_mntq_gg}/include/ggml.h _mntq_mla_src_sha)
set(_mntq_mla_stamp "${_mntq_gg}|${_mntq_mla_src_sha}|${_mntq_mla_patch_sha}|${_mntq_alloc_patch_sha}")
set(_mntq_mla_done "")
if(EXISTS ${_mntq_mla_copy}/.mntq-stamp)
  file(READ ${_mntq_mla_copy}/.mntq-stamp _mntq_mla_done)
endif()
set(_mntq_mla ON)
if(NOT _mntq_mla_done STREQUAL _mntq_mla_stamp)
  find_package(Git)
  if(NOT GIT_FOUND)
    message(WARNING "git was not found to apply third_party/ggml/metal-sparse-mla.patch: ggml is built as it is, and the "
                    "DSA layers keep the gather + mul_mm attention")
    set(_mntq_mla OFF)
  else()
    file(REMOVE_RECURSE ${_mntq_mla_copy})
    file(COPY ${_mntq_gg}/ DESTINATION ${_mntq_mla_copy})
    # (GIT_CEILING_DIRECTORIES: git applies the patch as a plain patch tool, not inside the repository holding the
    # build tree)
    execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                            ${GIT_EXECUTABLE} apply --ignore-whitespace ${_mntq_mla_patch}
                    WORKING_DIRECTORY ${_mntq_mla_copy} RESULT_VARIABLE _mntq_mla_rc ERROR_VARIABLE _mntq_mla_err)
    if(_mntq_mla_rc EQUAL 0)
      # The pinned ggml allocator's 256 free ranges can fragment during long
      # Metal prefill and abort at ~92K input tokens. Patch ONLY the copied ggml
      # tree, never the vendored llama.cpp checkout. Fail closed on patch drift.
      execute_process(COMMAND ${CMAKE_COMMAND} -E env GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}
                              ${GIT_EXECUTABLE} apply --ignore-whitespace ${_mntq_alloc_patch}
                      WORKING_DIRECTORY ${_mntq_mla_copy} RESULT_VARIABLE _mntq_alloc_rc
                      ERROR_VARIABLE _mntq_alloc_err)
      if(NOT _mntq_alloc_rc EQUAL 0)
        message(FATAL_ERROR "Metal ggml allocator patch failed against the pinned ggml copy:\n${_mntq_alloc_err}")
      endif()
      file(WRITE ${_mntq_mla_copy}/.mntq-stamp "${_mntq_mla_stamp}")
    else()
      message(WARNING "third_party/ggml/metal-sparse-mla.patch (made for the pinned ggml) does not apply to ${_mntq_gg} - "
                      "it is built as it is, and the DSA layers keep the gather + mul_mm attention:\n${_mntq_mla_err}")
      file(REMOVE_RECURSE ${_mntq_mla_copy})
      set(_mntq_mla OFF)
    endif()
  endif()
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_mntq_mla_patch} ${_mntq_alloc_patch} ${_mntq_gg}/include/ggml.h)
if(_mntq_mla)
  add_subdirectory(${_mntq_mla_copy} ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
else()
  add_subdirectory(${_mntq_gg} ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
endif()

# the engine reports Mantiq's release (VERSION) as its version
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _mntq_version LIMIT_COUNT 1)
add_library(mntq_metal STATIC src/metal/glm_metal.cpp src/metal/glm_metal_load.cpp src/metal/expert_pool.cpp
                              src/metal/glm_metal_dflash.cpp src/metal/spec.cpp src/metal/suffix_drafter.cpp
                              src/metal/sampler.cpp)
target_include_directories(mntq_metal PUBLIC src/metal)
target_compile_definitions(mntq_metal PUBLIC MNTQ_VERSION="${_mntq_version}")
target_link_libraries(mntq_metal PUBLIC ggml ggml-base ggml-cpu ggml-metal)
if(_mntq_mla)
  target_compile_definitions(mntq_metal PRIVATE MNTQ_METAL_SPARSE_MLA=1)
endif()

add_executable(mntq-metal src/metal/main.cpp src/metal/serve.cpp)
target_link_libraries(mntq-metal PRIVATE mntq_metal)
