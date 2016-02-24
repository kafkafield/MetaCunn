gcc metaHard.cpp -lstdc++ -std=c++11 -I /root/torch/install/include/ -fPIC -shared -o metahard.so
nvidia-smi --query-gpu=memory.used --format=csv -lms 1 -f ./heihei &
pgrep nvidia-smi | xargs kill -s 9