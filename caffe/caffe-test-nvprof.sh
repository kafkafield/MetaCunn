 cd /home/jedy/deeplearning/convnet-benchmarks-master/caffe/caffe
 sed -i "5s/USE_CUDNN/#USE_CUDNN/" Makefile.config

 make clean
 make -j32

 cd /home/jedy/deeplearning/convnet-benchmarks-master/caffe

# collect flop counts for caffe original
nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv1.prototxt --iterations=1 --gpu 0 --logtostderr=1
nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv2.prototxt --iterations=1 --gpu 0 --logtostderr=1
nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv3.prototxt --iterations=1 --gpu 0 --logtostderr=1
nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv4.prototxt --iterations=1 --gpu 0 --logtostderr=1
nvprof ./caffe/build/tools/caffe time --model=proto_forceGradInput/conv5.prototxt --iterations=1 --gpu 0 --logtostderr=1
