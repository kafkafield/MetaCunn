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

for val in 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400

do
	sed -i "49s/'bs': .*/'bs': "$val",/" pylearn2_benchmark_cufft.py
	
	echo " batch size: $val "

	SKIP=legacy python pylearn2_benchmark_cufft.py >> ./test/batch-cufft-output.txt 2>&1
	
done
