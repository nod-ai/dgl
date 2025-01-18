#!/usr/bin/env python3
"""Hipifies the DGL tensoradapter plugin for PyTorch

Instead of the generic HIPIFY tooling that we use for the rest of DGL, we use
the PyTorch Hipify Python tooling. This is customized for some PyTorch-specific
things and designed to work with PyTorch plugins. We wrap the tooling to make it
perform the hipification in place and save the original file with a .prehip
extension, mirroring the behavior of the hipify-inplace tool and our wrapper
script for the rest of DGL.

The PyTorch hipify is generally easier to work with (mostly because it's
Python), so we might consider using it for the rest of DGL as well rather than
having two different tools here, but it requires some modification to do that.
"""

import os
import pathlib
import shutil

from torch.utils.hipify import hipify_python

TENSOR_ADAPTER_DIR = pathlib.Path(os.environ["DGL_HOME"]) / "tensoradapter"
 
for prehip_file in TENSOR_ADAPTER_DIR.rglob("*.prehip"):
    orig_file = prehip_file.with_suffix("")
    shutil.copy2(prehip_file, orig_file)

hipify_result = hipify_python.hipify(
    project_directory=str(TENSOR_ADAPTER_DIR),
    output_directory=str(TENSOR_ADAPTER_DIR),
    is_pytorch_extension=True,
)

for orig_path, result in hipify_result.items():
    # For now, we follow the hipify-inplace convention of adding a .prehip
    # extension to original files and modifying the hipified files in place. 
    prehip_path = f"{orig_path}.prehip"
    shutil.copy2(orig_path, prehip_path)
    hipified_path = result.hipified_path or orig_path
    os.rename(hipified_path, orig_path)
    with open(orig_path) as f:
        content = f.read()
    # Do our own replacement. This is the only extra one we need.
    updated_content = content.replace("DGL_USE_CUDA", "DGL_USE_ROCM")
    if content == updated_content:
        # If we didn't change the contents and neither did torch hipify, delete
        # the .prehip file.
        if "skipped" in result.status:
            os.unlink(prehip_path)
    else:
        with open(orig_path, "w") as f:
            f.write(updated_content)
