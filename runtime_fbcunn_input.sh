# Filter size

for val in 1 3 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384
do
        sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-fbcunn.lua
        echo " filter: $val "

        th benchmark-batch-fbcunn.lua >> fbfftlog.log #>> ./runtime_test/filter_fbcunn.out

done

val=3
sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-fbcunn.lua

# Filter size

for val in 1 3 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384
do
        sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-cunn.lua
        echo " filter: $val "

        th benchmark-batch-cunn.lua >> cunnlog.log #>> ./runtime_test/filter_fbcunn.out

done

val=3
sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-cunn.lua

# Filter size

for val in 1 3 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384
do
        sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-ccn2.lua
        echo " filter: $val "

        th benchmark-batch-ccn2.lua >> ccn2log.log #>> ./runtime_test/filter_fbcunn.out

done

val=3
sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-ccn2.lua

# Filter size

for val in 1 3 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304 320 336 352 368 384
do
        sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-cudnn.lua
        echo " filter: $val "

        th benchmark-batch-cudnn.lua >> cudnnlog.log #>> ./runtime_test/filter_fbcunn.out

done

val=3
sed -i '16s/ni = .*/ni = '$val',/' benchmark-batch-cudnn.lua