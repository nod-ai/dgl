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
  find_package_and_print_version(HIP REQUIRED)
  find_package_and_print_version(hipBLAS REQUIRED)
  find_package_and_print_version(hipRAND REQUIRED)
  find_package_and_print_version(hipSPARSE REQUIRED)
endmacro(find_rocm)
