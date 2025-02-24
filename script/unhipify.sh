#!/bin/bash

# Reverses hipify.sh and hipify-tensoradapter.py by copying prehip files back to
# their original locations.

set -euo pipefail

cd "${DGL_HOME}"

function find_sed_prehip() {
    find $@ -name '*.sed.prehip' | sort
}

function find_prehip() {
    find $@ -name '*.prehip' | sort
}

declare -a sed_prehip_srcs=(
    $(find_sed_prehip  src include tests third_party/HugeCTR/gpu_cache tensoradapter graphbolt)
)

declare -a prehip_srcs=(
    $(find_prehip  src include tests third_party/HugeCTR/gpu_cache tensoradapter graphbolt)
)
prehip_srcs=(
    $(comm -23 <(printf '%s\n' "${prehip_srcs[@]}") <(printf '%s\n' "${sed_prehip_srcs[@]}"))
)

for prehip_src in ${sed_prehip_srcs[@]}; do
    mv "${prehip_src}" "${prehip_src%%.sed.prehip}"
done

for prehip_src in ${prehip_srcs[@]}; do
    mv "${prehip_src}" "${prehip_src%%.prehip}"
done
