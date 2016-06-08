#!/bin/bash

cd /home/jedy/deeplearning/convnet-benchmarks-master/theano

sed -i "47s/'iw': .*/'iw': 128,/" pylearn2_benchmark_cufft.py
sed -i "48s/'ih': .*/'ih': 128,/" pylearn2_benchmark_cufft.py
sed -i "49s/'bs': .*/'bs': 64,/" pylearn2_benchmark_cufft.py
sed -i "43s/'ni': .*/'ni': 3,/" pylearn2_benchmark_cufft.py
sed -i "45s/'kw': .*/'kw': 11,/" pylearn2_benchmark_cufft.py
sed -i "46s/'kh': .*/'kh': 11,/" pylearn2_benchmark_cufft.py
sed -i "44s/'no': .*/'no': 64,/" pylearn2_benchmark_cufft.py
sed -i "50s/'dw': .*/'dw': 1,/" pylearn2_benchmark_cufft.py
sed -i "51s/'dh': .*/'dh': 1,/" pylearn2_benchmark_cufft.py
pwd

# Filter size

for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512
do
        sed -i "44s/'no': .*/'no': "$val",/" pylearn2_benchmark_cufft.py
        echo " filters: $val "

        SKIP=legacy python pylearn2_benchmark_cufft.py 


done


sed -i "47s/'iw': .*/'iw': 128,/" pylearn2_benchmark_cufft.py
sed -i "48s/'ih': .*/'ih': 128,/" pylearn2_benchmark_cufft.py
sed -i "49s/'bs': .*/'bs': 64,/" pylearn2_benchmark_cufft.py
sed -i "43s/'ni': .*/'ni': 3,/" pylearn2_benchmark_cufft.py
sed -i "45s/'kw': .*/'kw': 11,/" pylearn2_benchmark_cufft.py
sed -i "46s/'kh': .*/'kh': 11,/" pylearn2_benchmark_cufft.py
sed -i "44s/'no': .*/'no': 64,/" pylearn2_benchmark_cufft.py
sed -i "50s/'dw': .*/'dw': 1,/" pylearn2_benchmark_cufft.py
sed -i "51s/'dh': .*/'dh': 1,/" pylearn2_benchmark_cufft.py
