


# Digit Recognition via FCN

![](http://i.imgur.com/PLQqyzn.jpg)

![Map](http://i.imgur.com/1wBU1E5.png)


## To train:
- Git clone and put it into CAFFE_ROOT/examples/;
- Download the fcn-32s-pascalcontext.caffemodel [**here**](http://pan.baidu.com/s/1o80M1R0 "fcn32s-pascalcontext.caffemodel") and move it into CAFFE_ROOT/models/fcn-32s-pascalcontext.caffemodel;

- Download the **[dataset](http://pan.baidu.com/s/1mhMUYk4 "dataset")** and put it into CAFFE_ROOT/data/;
- Run CAFFE_ROOT/examples/digits2.0/covert.py for converting data into lmdb;
- Run solve.py to start training.


## To test:
- Download the pre-trained model or train your own model as mentioned above;
- Run CAFFE_ROOT/examples/digits2.0/test_fcn11_full.m(need Matlab and matcaffe);


Code base on **[caffe](http://caffe.berkeleyvision.org/ "Caffe")**

Contributors:
hduxiejun@126.com
