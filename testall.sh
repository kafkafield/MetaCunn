#test cunn
#sed -i '3s/logfile=.*/logfile=cunnlog.log/' runtime_fbcunn_all.sh
#sed -i '4s/programfile=.*/programfile=benchmark-batch-cunn.lua/' runtime_fbcunn_all.sh
#./runtime_fbcunn_all.sh
#test cudnn
#sed -i '3s/logfile=.*/logfile=cudnnlog.log/' runtime_fbcunn_all.sh
#sed -i '4s/programfile=.*/programfile=benchmark-batch-cudnn.lua/' runtime_fbcunn_all.sh
#./runtime_fbcunn_all.sh
#test ccn2
#sed -i '3s/logfile=.*/logfile=ccn2log.log/' runtime_fbcunn_all.sh
#sed -i '4s/programfile=.*/programfile=benchmark-batch-ccn2.lua/' runtime_fbcunn_all.sh
#./runtime_fbcunn_all.sh
#test fbfft
#sed -i '3s/logfile=.*/logfile=fbfftlog.log/' runtime_fbcunn_all.sh
#sed -i '4s/programfile=.*/programfile=benchmark-batch-fbcunn.lua/' runtime_fbcunn_all.sh
#./runtime_fbcunn_all.sh
#test metacunn
sed -i '3s/logfile=.*/logfile=metacunnlog.log/' runtime_fbcunn_all.sh
sed -i '4s/programfile=.*/programfile=benchmark-batch-metacunn.lua/' runtime_fbcunn_all.sh
./runtime_fbcunn_all.sh
