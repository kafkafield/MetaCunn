#!/bin/bash
#nvprof --analysis-metrics -o ccn2.nvprof th benchmark-filter-ccn2.lua
#nvprof --analysis-metrics -o cunn.nvprof th benchmark-filter-cunn-cuDnn.lua
#nvprof --analysis-metrics -o fbcunn.nvprof th benchmark-filter-fbcunn.lua 


#nvprof --kernels "cgemm_sm35_ldg_nc_64x8x64x16x16" --metrics all --events all -o fbcunn-cgemm_sm35_ldg_nc_64x8x64x16x16.nvprof th benchmark-filter-fbcunn.lua
#nvprof --kernels "cgemm_sm35_ldg_nn_64x8x64x16x16" --metrics all --events all -o fbcunn-cgemm_sm35_ldg_nn_64x8x64x16x16.nvprof th benchmark-filter-fbcunn.lua
#nvprof --kernels "decimateInFrequency2DKernel128" --metrics all --events all -o fbcunn-decimateInFrequency2DKernel128.nvprof th benchmark-filter-fbcunn.lua
#nvprof --kernels "decimateInFrequency2DKernel" --metrics all --events all -o fbcunn-decimateInFrequency2DKernel.nvprof th benchmark-filter-fbcunn.lua

# con1
#nvprof --kernels "wgrad_alg0_engine" --metrics all --events all -o cudnn-wgrad_alg0_engine.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
#nvprof --kernels "cudnn_dgrad_sm35_ldg_nt_64x16x64x16x16" --metrics all --events all -o cudnn-cudnn_dgrad_sm35_ldg_nt_64x16x64x16x16.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
#nvprof --kernels "cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x16x128x8x32" --metrics all --events all -o cudnn-cudnn_convolve_precomputed_sgemm_sm35_ldg_nn_64x16x128x8x32.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0

# conv5
nvprof --kernels "convolve_wgrad_engine" --metrics all --events all -o cudnn-convolve_wgrad_engine-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0
nvprof --kernels "cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32" --metrics all --events all -o cudnn-cudnn_dgrad_sm35_ldg_nt_64x16x128x8x32-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0
nvprof --kernels "cudnn_convolve_sm35_ldg_nn_64x16x128x8x32" --metrics all --events all -o cudnn-cudnn_convolve_sm35_ldg_nn_64x16x128x8x32-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0
