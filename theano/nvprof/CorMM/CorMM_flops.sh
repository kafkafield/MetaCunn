#!/bin/bash

# test flops counts for Cormm theano

nvprof --csv --metrics flops_sp python Corr-conv1-benchmark.py >> flops_Cormm_con1.txt 2>&1
nvprof --csv --metrics flops_sp python Corr-conv2-benchmark.py >> flops_Cormm_con2.txt 2>&1
nvprof --csv --metrics flops_sp python Corr-conv3-benchmark.py >> flops_Cormm_con3.txt 2>&1
nvprof --csv --metrics flops_sp python Corr-conv4-benchmark.py >> flops_Cormm_con4.txt 2>&1
nvprof --csv --metrics flops_sp python Corr-conv5-benchmark.py >> flops_Cormm_con5.txt 2>&1

