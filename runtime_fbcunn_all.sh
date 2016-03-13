#!/bin/bash


# Batch size
for val in 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512
do
	sed -i '22s/bs = .*/bs = '$val',/' benchmark-batch-fbcunn.lua
	echo " batch size: $val "

	th benchmark-batch-fbcunn.lua >> fbfftlog.log #>> ./runtime_test/batch_fbcunn.out

done

val=64
sed -i '22s/bs = .*/bs = '$val',/' benchmark-batch-fbcunn.lua

# Input size

for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
do
        sed -i '20s/iw = .*/iw = '$val',/' benchmark-batch-fbcunn.lua
        sed -i '20s/ih = .*/ih = '$val',/' benchmark-batch-fbcunn.lua
        echo " filter: $val "

        th benchmark-batch-fbcunn.lua >> fbfftlog.log #>> ./runtime_test/filter_fbcunn.out

done

val=128
sed -i '20s/iw = .*/iw = '$val',/' benchmark-batch-fbcunn.lua
sed -i '20s/ih = .*/ih = '$val',/' benchmark-batch-fbcunn.lua

# Filter size

for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512
do
        sed -i '17s/no = .*/no = '$val',/' benchmark-batch-fbcunn.lua
        echo " filter: $val "

        th benchmark-batch-fbcunn.lua >> fbfftlog.log #>> ./runtime_test/filter_fbcunn.out

done

val=64
sed -i '17s/no = .*/no = '$val',/' benchmark-batch-fbcunn.lua

# Kernel size
for val in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
        sed -i '18s/kw = .*/kw = '$val',/' benchmark-batch-fbcunn.lua
        sed -i '19s/kh = .*/kh = '$val',/' benchmark-batch-fbcunn.lua
        echo " kernel size: $val*$val "

        th benchmark-batch-fbcunn.lua >> fbfftlog.log #>> ./runtime_test/ksize_fbcunn.out

done

val=11
sed -i '18s/kw = .*/kw = '$val',/' benchmark-batch-fbcunn.lua
sed -i '19s/kh = .*/kh = '$val',/' benchmark-batch-fbcunn.lua

# Stride size

for val in 1 #2 3 4 5 6 7 8 9 10 11 #12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
        sed -i '23s/dw = .*/dw = '$val',/' benchmark-batch-fbcunn.lua
        sed -i '24s/dh = .*/dh = '$val',/' benchmark-batch-fbcunn.lua
        echo " stride: $val "

        th benchmark-batch-fbcunn.lua >> fbfftlog.log #>> ./runtime_test/stride_fbcunn.out

done
