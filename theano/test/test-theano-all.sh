#!/bin/bash

rm *txt

./runtime_batch.sh
./runtime_filter.sh
./runtime_input.sh
./runtime_ksize.sh
./runtime_stride.sh


