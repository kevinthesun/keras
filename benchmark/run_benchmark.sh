#!/usr/bin/env bash
backend="tensorflow mxnet"

for gpu_num in 1 2 4 8; do
    export GPU_NUM=$gpu_num
    for back in $backend; do
        export KERAS_BACKEND=$back
        python benchmark.py
    done
done
