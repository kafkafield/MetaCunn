#!/bin/bash

# test flops counts for fft theano

nvprof --csv --metrics flops_sp python fft-conv1-benchmark.py >> flops_fft_con1.txt 2>&1
nvprof --csv --metrics flops_sp python fft-conv2-benchmark.py >> flops_fft_con2.txt 2>&1
nvprof --csv --metrics flops_sp python fft-conv3-benchmark.py >> flops_fft_con3.txt 2>&1
nvprof --csv --metrics flops_sp python fft-conv4-benchmark.py >> flops_fft_con4.txt 2>&1
nvprof --csv --metrics flops_sp python fft-conv5-benchmark.py >> flops_fft_con5.txt 2>&1

