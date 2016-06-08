#!/bin/bash


nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python fft-conv1-benchmark.py >> metrics_fft_con1.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python fft-conv2-benchmark.py >> metrics_fft_con2.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python fft-conv3-benchmark.py >> metrics_fft_con3.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python fft-conv4-benchmark.py >> metrics_fft_con4.txt 2>&1
nvprof --csv --metrics achieved_occupancy,ipc,warp_execution_efficiency,gld_efficiency,gst_efficiency,shared_efficiency python fft-conv5-benchmark.py >> metrics_fft_con5.txt 2>&1


