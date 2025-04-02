# ROCm Module
if(USE_ROCM)
  find_rocm(${USE_ROCM} REQUIRED)
else(USE_ROCM)
  return()
endif()

###### Borrowed from MSHADOW project

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

################################################################################################
# Config rocm compilation and append ROCm libraries to linker_libs
# Usage:
#  dgl_config_rocm(linker_libs)
macro(dgl_config_rocm linker_libs)
  if(NOT ROCM_FOUND)
    message(FATAL_ERROR "Cannot find ROCm.")
  endif()

  enable_language(HIP)

  add_definitions(-DDGL_USE_ROCM)
  # We need the newest stuff that isn't turned on by default yet
  add_definitions(-DHIP_ENABLE_WARP_SYNC_BUILTINS)
  # Used by third_party/HugeCTR/gpu_cache
  add_compile_definitions(LIBCUDACXX_VERSION)

  list(APPEND ${linker_libs} 
    hip::host
    roc::hipblas
    roc::hipsparse
    hip::hiprand)
endmacro()
