#!/bin/bash

# test flops counts for Cormm theano

nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python Corr-conv1-benchmark.py >> metrics_Cormm_con1.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python Corr-conv2-benchmark.py >> metrics_Cormm_con2.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python Corr-conv3-benchmark.py >> metrics_Cormm_con3.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python Corr-conv4-benchmark.py >> metrics_Cormm_con4.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python Corr-conv5-benchmark.py >> metrics_Cormm_con5.txt 2>&1

