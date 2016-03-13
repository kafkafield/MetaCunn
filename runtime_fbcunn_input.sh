for val in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
do
        sed -i '20s/iw = .*/iw = '$val',/' benchmark-batch-fbcunn.lua
        sed -i '21s/ih = .*/ih = '$val',/' benchmark-batch-fbcunn.lua
        echo " filter: $val "

        th benchmark-batch-fbcunn.lua >> fbfftlog.log #>> ./runtime_test/filter_fbcunn.out

done

val=128
sed -i '20s/iw = .*/iw = '$val',/' benchmark-batch-fbcunn.lua
sed -i '21s/ih = .*/ih = '$val',/' benchmark-batch-fbcunn.lua