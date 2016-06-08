#!/bin/bash

cd /home/jedy/deeplearning/convnet-benchmarks-master/theano

sed -i "47s/'iw': .*/'iw': 128,/" pylearn2_benchmark_CorrMM.py
sed -i "48s/'ih': .*/'ih': 128,/" pylearn2_benchmark_CorrMM.py
sed -i "49s/'bs': .*/'bs': 64,/" pylearn2_benchmark_CorrMM.py
sed -i "43s/'ni': .*/'ni': 3,/" pylearn2_benchmark_CorrMM.py
sed -i "45s/'kw': .*/'kw': 11,/" pylearn2_benchmark_CorrMM.py
sed -i "46s/'kh': .*/'kh': 11,/" pylearn2_benchmark_CorrMM.py
sed -i "44s/'no': .*/'no': 64,/" pylearn2_benchmark_CorrMM.py
sed -i "50s/'dw': .*/'dw': 1,/" pylearn2_benchmark_CorrMM.py
sed -i "51s/'dh': .*/'dh': 1,/" pylearn2_benchmark_CorrMM.py
pwd

# Batch size
for val in 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512
do
        sed -i "49s/'bs': .*/'bs': "$val",/" pylearn2_benchmark_CorrMM.py

        echo " batch size: $val "

        SKIP=legacy python pylearn2_benchmark_CorrMM.py

done

sed -i "47s/'iw': .*/'iw': 128,/" pylearn2_benchmark_CorrMM.py
sed -i "48s/'ih': .*/'ih': 128,/" pylearn2_benchmark_CorrMM.py
sed -i "49s/'bs': .*/'bs': 64,/" pylearn2_benchmark_CorrMM.py
sed -i "43s/'ni': .*/'ni': 3,/" pylearn2_benchmark_CorrMM.py
sed -i "45s/'kw': .*/'kw': 11,/" pylearn2_benchmark_CorrMM.py
sed -i "46s/'kh': .*/'kh': 11,/" pylearn2_benchmark_CorrMM.py
sed -i "44s/'no': .*/'no': 64,/" pylearn2_benchmark_CorrMM.py
sed -i "50s/'dw': .*/'dw': 1,/" pylearn2_benchmark_CorrMM.py
sed -i "51s/'dh': .*/'dh': 1,/" pylearn2_benchmark_CorrMM.py
