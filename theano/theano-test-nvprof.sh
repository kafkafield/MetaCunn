#!/bin/bash
#nvprof --analysis-metrics -o ccn2.nvprof th benchmark-filter-ccn2.lua
#nvprof --analysis-metrics -o cunn.nvprof th benchmark-filter-cunn-cuDnn.lua
#nvprof --analysis-metrics -o fbcunn.nvprof th benchmark-filter-fbcunn.lua 


#nvprof --kernels "cgemm_sm35_ldg_nc_64x8x64x16x16" --metrics all --events all -o fbcunn-cgemm_sm35_ldg_nc_64x8x64x16x16.nvprof th benchmark-filter-fbcunn.lua
#nvprof --kernels "cgemm_sm35_ldg_nn_64x8x64x16x16" --metrics all --events all -o fbcunn-cgemm_sm35_ldg_nn_64x8x64x16x16.nvprof th benchmark-filter-fbcunn.lua
#nvprof --kernels "decimateInFrequency2DKernel128" --metrics all --events all -o fbcunn-decimateInFrequency2DKernel128.nvprof th benchmark-filter-fbcunn.lua
#nvprof --kernels "decimateInFrequency2DKernel" --metrics all --events all -o fbcunn-decimateInFrequency2DKernel.nvprof th benchmark-filter-fbcunn.lua


nvprof --kernels "sgemm_sm35_ldg_tn_32x16x64x8x16" --metrics all --events all -o caffe-sgemm_sm35_ldg_tn_32x16x64x8x16.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
nvprof --kernels "sgemm_sm35_ldg_nn_64x16x64x16x16" --metrics all --events all -o caffe-sgemm_sm35_ldg_nn_64x16x64x16x16.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
nvprof --kernels "im2col_gpu_kernel" --metrics all --events all -o caffe-im2col_gpu_kernel.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
