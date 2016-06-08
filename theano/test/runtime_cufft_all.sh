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

# Batch size
for val in 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512

do
	sed -i "49s/'bs': .*/'bs': "$val",/" pylearn2_benchmark_cufft.py
	
	echo " batch size: $val "

	SKIP=legacy python pylearn2_benchmark_cufft.py >> ./test/batch-cufft-output.txt 2>&1
	
done

# Reset batch size to default value
sed -i "49s/'bs': .*/'bs': 64,/" pylearn2_benchmark_cufft.py

# Input size
for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
do
        sed -i "47s/'iw': .*/'iw': "$val",/" pylearn2_benchmark_cufft.py
        sed -i "48s/'ih': .*/'ih': "$val",/" pylearn2_benchmark_cufft.py
        echo " input: $val * $val"

        SKIP=legacy python pylearn2_benchmark_cufft.py >> ./test/input-cufft-output.txt 2>&1

done

# Reset input size to default value
sed -i "47s/'iw': .*/'iw': 128,/" pylearn2_benchmark_cufft.py
sed -i "48s/'ih': .*/'ih': 128,/" pylearn2_benchmark_cufft.py
# Filter size

for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512
do
        sed -i "44s/'no': .*/'no': "$val",/" pylearn2_benchmark_cufft.py
        echo " filters: $val "

        SKIP=legacy python pylearn2_benchmark_cufft.py >> ./test/filter-cufft-output.txt 2>&1

done

# Reset filter size to default value
sed -i "44s/'no': .*/'no': 64,/" pylearn2_benchmark_cufft.py
# Kernel size
for val in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
        sed -i "45s/'kw': .*/'kw': "$val",/" pylearn2_benchmark_cufft.py
        sed -i "46s/'kh': .*/'kh': "$val",/" pylearn2_benchmark_cufft.py
        echo " ksize: $val "

        SKIP=legacy python pylearn2_benchmark_cufft.py >> ./test/ksize-cufft-output.txt 2>&1

done

# Reset kernel size to defualt value
sed -i "45s/'kw': .*/'kw': 11,/" pylearn2_benchmark_cufft.py
sed -i "46s/'kh': .*/'kh': 11,/" pylearn2_benchmark_cufft.py

# Stride size
for val in 1
do
        sed -i "50s/'dw': .*/'dw': "$val",/" pylearn2_benchmark_cufft.py
        sed -i "51s/'dh': .*/'dh': "$val",/" pylearn2_benchmark_cufft.py
        echo " stride: $val "

        SKIP=legacy python pylearn2_benchmark_cufft.py >> ./test/stride-cufft-output.txt 2>&1

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
