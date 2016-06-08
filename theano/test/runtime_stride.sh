#!/bin/bash

cd /home/jedy/deeplearning/convnet-benchmarks-master/theano

sed -i "47s/'iw': .*/'iw': 128,/" pylearn2_benchmark.py
sed -i "48s/'ih': .*/'ih': 128,/" pylearn2_benchmark.py
sed -i "49s/'bs': .*/'bs': 128,/" pylearn2_benchmark.py
sed -i "43s/'ni': .*/'ni': 3,/" pylearn2_benchmark.py
sed -i "45s/'kw': .*/'kw': 11,/" pylearn2_benchmark.py
sed -i "46s/'kh': .*/'kh': 11,/" pylearn2_benchmark.py
sed -i "44s/'no': .*/'no': 96,/" pylearn2_benchmark.py
sed -i "50s/'dw': .*/'dw': 1,/" pylearn2_benchmark.py
sed -i "51s/'dh': .*/'dh': 1,/" pylearn2_benchmark.py
pwd

# for val in 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
for val in 1 2 3 4 5 6 7 8 9 10 11
do
	sed -i "50s/'dw': .*/'dw': "$val",/" pylearn2_benchmark.py
	sed -i "51s/'dh': .*/'dh': "$val",/" pylearn2_benchmark.py
	echo " stride: $val "

	SKIP=legacy python pylearn2_benchmark.py >> ./test/stride-output.txt 2>&1
	
done
