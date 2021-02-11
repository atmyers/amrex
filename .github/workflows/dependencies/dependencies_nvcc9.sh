#!/usr/bin/env bash
#
# Copyright 2020 Axel Huebl
#
# License: BSD-3-Clause-LBNL

# search recursive inside a folder if a file contains tabs
#
# @result 0 if no files are found, else 1
#

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends\
    build-essential     \
    g++-5.5             \
    gfortran-5.5        \
    libopenmpi-dev      \
    openmpi-bin         \
    nvidia-cuda-dev     \
    nvidia-cuda-toolkit


