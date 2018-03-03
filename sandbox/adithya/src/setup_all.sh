#!/bin/bash

set -e

venv_dir=deeprl_hw*

if command -v python3 > /dev/null 2>&1; then
  python3 -m venv ${venv_dir}
else
  virtualenv ${venv_dir}
fi

source ${venv_dir}/bin/activate
pip install -U -r requirements.txt

# Note this does not install tensorflow.
# I was able to install it by running (after activating venv):
#   pip install --upgrade tensorflow
# If you have a GPU:
#   pip install --upgrade tensorflow-gpu

# It seems this TF installation is not optimized for my CPU.
# To fix, we need to install tensorflow from source: (or from distribution sources?)
#   https://www.tensorflow.org/install/install_sources
#   http://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
