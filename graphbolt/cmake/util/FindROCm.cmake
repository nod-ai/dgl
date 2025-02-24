#######################################################
# Usage:
#   find_rocm(${USE_ROCM})
#
# - When USE_ROCM=ON, use auto search
#
# Please use the CMAKE variable ROCM_HOME to set ROCm directory
#
# Provide variables:
#
# - ROCM_FOUND
#

macro(find_rocm use_rocm)
  # ROCM prints out a nonsense version here from
  # ROCmCMakeBuildToolsConfigVersion.cmake that is just going to confuse people
  find_package(ROCM REQUIRED)
  find_package(HIP REQUIRED)
  find_package(hipBLAS REQUIRED)
  find_package(hipRAND REQUIRED)
  find_package(hipSPARSE REQUIRED)
endmacro(find_rocm)
