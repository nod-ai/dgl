#!/usr/bin/env python3
"""Hipifies the DGL plugins for PyTorch

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

import argparse
import os
import pathlib
import shutil

from torch.utils.hipify import hipify_python

parser = argparse.ArgumentParser(
    prog="hipify-torch-extension",
    description="Hipify all files underneath the given directory."
)
parser.add_argument('dir_path', type=pathlib.Path) 
dir_path = parser.parse_args().dir_path
 
for prehip_file in dir_path.rglob("*.prehip"):
    orig_file = prehip_file.with_suffix("")
    shutil.copy2(prehip_file, orig_file)

hipify_result = hipify_python.hipify(
    project_directory=str(dir_path),
    output_directory=str(dir_path),
    is_pytorch_extension=True,
)

for orig_path, result in hipify_result.items():
    # For now, we follow the hipify-inplace convention of adding a .prehip
    # extension to original files and modifying the hipified files in place. 
    prehip_path = f"{orig_path}.prehip"
    shutil.copy2(orig_path, prehip_path)
    hipified_path = result.hipified_path or orig_path
    os.rename(hipified_path, orig_path)
