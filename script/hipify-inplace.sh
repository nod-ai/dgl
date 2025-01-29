#!/bin/bash

# Hipifies the DGL codebase. That is, translates CUDA terminology to HIP, using
# the HIPIFY tooling (https://rocm.docs.amd.com/projects/HIPIFY/) plus some
# custom regex replacements. This performs the hipification inplace, saving the
# original files with a .prehip extension and then modifying the existing files
# without renaming them. This means HIP files will have cuda file extensions
# (e.g. .cu), so some tooling will need to be modified to work with this (e.g.
# set language mode in CMake). The advantage is that file path references like
# include statements don't need to change.

set -euo pipefail

cd "${DGL_HOME}"

HIPIFY_LOG="/tmp/hipify-inplace.log"

function find_code() {
    find $@ -name '*.cu' -o -name '*.CU'
    find $@ -name '*.cpp' -o -name '*.cxx' -o -name '*.c' -o -name '*.cc'
    find $@ -name '*.CPP' -o -name '*.CXX' -o -name '*.C' -o -name '*.CC'
    find $@ -name '*.cuh' -o -name '*.CUH'
    find $@ -name '*.h' -o -name '*.hpp' -o -name '*.inc' -o -name '*.inl' -o -name '*.hxx' -o -name '*.hdl'
    find $@ -name '*.H' -o -name '*.HPP' -o -name '*.INC' -o -name '*.INL' -o -name '*.HXX' -o -name '*.HDL'
}

# There isn't any direct cuda code in the tests, but we still want to run our
# own sed replacements and want a prehip file for what they were before.
declare -a srcs=(
    $(find_code src include tests third_party/HugeCTR/gpu_cache)
)

declare -a log_files=()
for src in ${srcs[@]}; do
    log_file="$(mktemp --tmpdir hipify_${src//\//_}.XXX.log)"
    log_files+=("${log_file}")
    ( set -x ; hipify-perl -print-stats -inplace $src &> "${log_file}" ) &
done

sleep 1 # Hack to make it more likely the echo below prints after the last job command line from above.
echo "Waiting for hipify jobs to complete"
wait

cat "${log_files[@]}" > "${HIPIFY_LOG}" && rm "${log_files[@]}"
echo "Logs written to ${HIPIFY_LOG}"

# Additional fixes for project-specific things and things hipify misses or gets
# wrong.
for src in ${srcs[@]}; do
    sed -i 's@#include <hipblas.h>@#include <hipblas/hipblas.h>@' $src
    sed -i 's@#include <hipsparse.h>@#include <hipsparse/hipsparse.h>@' $src
    sed -i 's@#include <cuda_fp8.h>@#include <hip/hip_fp8.h>@' $src
    sed -i 's@#include <cuda_bf16.h>@#include <hip/hip_bf16.h>@' $src
    sed -i 's@\bDGL_USE_CUDA\b@DGL_USE_ROCM@g' $src
    sed -i 's@\bCUB_VERSION\b@HIPCUB_VERSION@g' $src
    sed -i 's@\bCUDART_ZERO_BF16\b@HIPRT_ZERO_BF16@g' $src
    sed -i 's@\bCUDART_INF_BF16\b@HIPRT_INF_BF16@g' $src
    sed -i 's@\bthrust::cuda::par@thrust::hip::par@g' $src
    sed -i 's@\b__nv_fp8_e4m3\b@__hip_fp8_e4m3@g' $src
    sed -i 's@\b__nv_fp8_e5m2\b@__hip_fp8_e5m2@g' $src
    sed -i 's@\bCUBLAS_GEMM_DEFAULT_TENSOR_OP\b@HIPBLAS_GEMM_DEFAULT@g' $src
    sed -i 's@\bcurand4\b@hiprand4@g' $src
    # hipify uses the old one
    sed -i 's@\bhip_bfloat16\b@__hip_bfloat16@g' $src
    sed -i 's@\b__trap();@abort();@' $src

    # If no changes were made, delete the prehip file.
    if cmp -s "${src}" "${src}.prehip"; then
        echo "Deleting unchanged ${src}.prehip"
        rm "${src}.prehip"
    fi
done
