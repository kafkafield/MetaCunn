#!/bin/bash


cd /home/jedy/deeplearning/convnet-benchmarks-master/theano

sed -i "49s/'bs': .*/'bs': "64",/" pylearn2_benchmark.py
pwd

for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
do
	sed -i "47s/'iw': .*/'iw': "$val",/" pylearn2_benchmark.py
	sed -i "48s/'ih': .*/'ih': "$val",/" pylearn2_benchmark.py
	echo " input: $val * $val"

	SKIP=legacy python pylearn2_benchmark.py >> ./test/input-output.txt 2>&1
	
done
