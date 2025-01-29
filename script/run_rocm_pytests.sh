#!/bin/bash

# Runs all Python tests that should work with the HIP/ROCM port (including CPU
# tests). Note that this deliberately keeps going even if one of the pytest
# invocations fails, but will return non-zero if any of the invocations fail
# (the highest numbered exit code).


bash ./script/run_pytest.sh -g \
    tests/python/pytorch/ \
    --ignore=tests/python/pytorch/graphbolt \
    --ignore=tests/python/pytorch/distributed \
    --deselect=tests/python/pytorch/dataloading/test_dataloader.py::test_distributed_dataloaders

final_ret=$?

bash ./script/run_pytest.sh -g \
    tests/python/common/ \
    --ignore=tests/python/common/test_partition.py

ret=$?
if (( ret != 0 && ret > final_ret )); then
    final_ret=$ret
fi

# Python CPU tests
bash ./script/run_pytest.sh -c \
    tests/python/pytorch/ \
    --ignore=tests/python/pytorch/graphbolt \
    --ignore=tests/python/pytorch/distributed \
    --deselect=tests/python/pytorch/dataloading/test_dataloader.py::test_distributed_dataloaders

ret=$?
if (( ret != 0 && ret > final_ret )); then
    final_ret=$ret
fi

bash ./script/run_pytest.sh -c \
    tests/python/common/ \
    --ignore=tests/python/common/test_partition.py

ret=$?
if (( ret != 0 && ret > final_ret )); then
    final_ret=$ret
fi

if (( final_ret != 0 )); then
    echo "Failed with exit code $final_ret. Scroll up for details"
fi

exit $final_ret
