#!/bin/bash

python3 setup.py build_ext --inplace
cp loglikelihood.c loglikelihood.cpython-39-x86_64-linux-gnu.so ..

