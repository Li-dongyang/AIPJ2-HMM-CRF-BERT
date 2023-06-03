#!/bin/sh
export CUDA_VISIBLE_DEVICES=3
python lstm-crf/main.py

# go build -ldflags="-s -w"