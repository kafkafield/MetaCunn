#!/bin/bash

#nvprof --kernels "sgemm_sm35_ldg_tn_32x16x64x8x16" --metrics all --events all -o theano-CorMM-sgemm_sm35_ldg_tn_32x16x64x8x16.nvprof python pylearn2_benchmark.py
#nvprof --kernels "sgemm_sm35_ldg_nn_64x16x64x16x16" --metrics all --events all -o theano-CorMM-sgemm_sm35_ldg_nn_64x16x64x16x16.nvprof python pylearn2_benchmark.py
#nvprof --kernels "im2col_kernel" --metrics all --events all -o theano-CorMM-im2col_kernel.nvprof python pylearn2_benchmark.py

#nvprof --kernels "sgemm_sm35_ldg_tn_32x16x64x8x16" --metrics all --events all -o theano-Meta-Optimizer-sgemm_sm35_ldg_tn_32x16x64x8x16.nvprof python pylearn2_benchmark_optimizer.py
#nvprof --kernels "sgemm_sm_heavy_nn_ldg" --metrics all --events all -o theano-Meta-Optimizer-sgemm_sm_heavy_nn_ldg.nvprof python pylearn2_benchmark_optimizer.py
#nvprof --kernels "sgemm_sm35_ldg_nt_128x16x64x16x16" --metrics all --events all -o theano-Meta-Optimizer-sgemm_sm35_ldg_nt_128x16x64x16x16.nvprof python pylearn2_benchmark_optimizer.py


nvprof --kernels "k_copy_4d" --metrics all --events all -o theano-cufft-k_copy_4d.nvprof python pylearn2_benchmark_cufft.py
nvprof --kernels "radixM_kernel" --metrics all --events all -o theano-radixM_kernel.nvprof python pylearn2_benchmark_cufft.py
