#!/bin/bash

# Reverses hipify.sh and hipify-tensoradapter.py by copying prehip files back to
# their original locations.

set -euo pipefail

cd "${DGL_HOME}"

function find_prehip() {
    find $@ -name '*.prehip'
}

declare -a prehip_srcs=(
    $(find_prehip src include tests third_party/HugeCTR/gpu_cache tensoradapter)
)

for prehip_src in ${prehip_srcs[@]}; do
    mv "${prehip_src}" "${prehip_src%%.prehip}"
done
