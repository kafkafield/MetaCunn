#!/bin/bash

# conv1
#nvprof --kernels "sgemm_sm35_ldg_tn_32x16x64x8x16" --metrics all --events all -o caffe-sgemm_sm35_ldg_tn_32x16x64x8x16.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
#nvprof --kernels "sgemm_sm35_ldg_nn_64x16x64x16x16" --metrics all --events all -o caffe-sgemm_sm35_ldg_nn_64x16x64x16x16.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
#nvprof --kernels "im2col_gpu_kernel" --metrics all --events all -o caffe-im2col_gpu_kernel.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0
# conv5
#nvprof --kernels "sgemm_sm35_ldg_nn_64x16x64x16x16" --metrics all --events all -o caffe-sgemm_sm35_ldg_nn_64x16x64x16x16-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0
#nvprof --kernels "sgemm_sm35_ldg_nt_64x16x64x16x16" --metrics all --events all -o caffe-sgemm_sm35_ldg_nt_64x16x64x16x16-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0
#nvprof --kernels "sgemm_sm35_ldg_tn_128x8x256x16x32" --metrics all --events all -o caffe-sgemm_sm35_ldg_tn_128x8x256x16x32-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0
#nvprof --kernels "sgemm_sm35_ldg_tn_64x16x128x8x32" --metrics all --events all -o caffe-sgemm_sm35_ldg_tn_64x16x128x8x32-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0
#nvprof --kernels "im2col_gpu_kernel" --metrics all --events all -o caffe-im2col_gpu_kernel-conv5.nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0

# Testing Caffe without cuDNN
cd /home/jedy/deeplearning/convnet-benchmarks-master/caffe/caffe
sed -i "5s/USE_CUDNN/#USE_CUDNN/" Makefile.config

make clean
make -j32

cd /home/jedy/deeplearning/convnet-benchmarks-master/caffe

# collect flop counts for caffe original
nvprof --csv --metrics flops_sp ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0 --logtostderr=1 >>./flops_caffe/flops_caffe_conv1.txt 2>&1
nvprof --csv --metrics flops_sp ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv2.prototxt --iterations=1 --gpu 0 --logtostderr=1 >>./flops_caffe/flops_caffe_conv2.txt 2>&1
nvprof --csv --metrics flops_sp ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv3.prototxt --iterations=1 --gpu 0 --logtostderr=1 >>./flops_caffe/flops_caffe_conv3.txt 2>&1
nvprof --csv --metrics flops_sp ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv4.prototxt --iterations=1 --gpu 0 --logtostderr=1 >>./flops_caffe/flops_caffe_conv4.txt 2>&1
nvprof --csv --metrics flops_sp ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0 --logtostderr=1 >>./flops_caffe/flops_caffe_conv5.txt 2>&1
