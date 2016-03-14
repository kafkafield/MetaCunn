#!/bin/bash

logfile=metacunnlog.log
programfile=benchmark-batch-metacunn.lua

# Batch size
for val in 32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512
do
	sed -i '22s/bs = .*/bs = '$val',/' $programfile
	echo " batch size: $val "

	th $programfile >> $logfile #>> ./runtime_test/batch_metacunn.out

done

val=64
sed -i '22s/bs = .*/bs = '$val',/' $programfile

# Input size

for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
do
        sed -i '20s/iw = .*/iw = '$val',/' $programfile
        sed -i '21s/ih = .*/ih = '$val',/' $programfile
        echo " filter: $val "

        th $programfile >> $logfile #>> ./runtime_test/filter_metacunn.out

done

val=128
sed -i '20s/iw = .*/iw = '$val',/' $programfile
sed -i '21s/ih = .*/ih = '$val',/' $programfile

# Filter size

for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512
do
        sed -i '17s/no = .*/no = '$val',/' $programfile
        echo " filter: $val "

        th $programfile >> $logfile #>> ./runtime_test/filter_metacunn.out

done

val=64
sed -i '17s/no = .*/no = '$val',/' $programfile

# Kernel size
for val in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
        sed -i '18s/kw = .*/kw = '$val',/' $programfile
        sed -i '19s/kh = .*/kh = '$val',/' $programfile
        echo " kernel size: $val*$val "

        th $programfile >> $logfile #>> ./runtime_test/ksize_metacunn.out

done

val=11
sed -i '18s/kw = .*/kw = '$val',/' $programfile
sed -i '19s/kh = .*/kh = '$val',/' $programfile

# Stride size

for val in 1 2 3 4 5 6 7 8 9 10 11 #12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50
do
        sed -i '23s/dw = .*/dw = '$val',/' $programfile
        sed -i '24s/dh = .*/dh = '$val',/' $programfile
        echo " stride: $val "

        th $programfile >> $logfile #>> ./runtime_test/stride_metacunn.out

done

val=1
sed -i '23s/dw = .*/dw = '$val',/' $programfile
sed -i '24s/dh = .*/dh = '$val',/' $programfile

#ni size

for val in 1 3 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384
do
        sed -i '16s/ni = .*/ni = '$val',/' $programfile
        echo " filter: $val "

        th $programfile >> $logfile #>> ./runtime_test/filter_fbcunn.out

done

val=3
sed -i '16s/ni = .*/ni = '$val',/' $programfile
