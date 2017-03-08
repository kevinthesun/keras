#!/usr/bin/env bash
backend="tensorflow mxnet"

for back in $backend; do
    export KERAS_BACKEND=$back
    python benchmark.py
done
