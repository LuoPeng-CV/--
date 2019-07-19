# 服务器环境搭建

本文档介绍的环境搭建主要包括

- Tensorflow
- caffe
- matlab
- pycharm





# 1.Tensorflow
在安装tensorflow之前首先安装pip：
```
首先切换到/workspace然后下载文件
cd /workspace/
wget https://bootstrap.pypa.io/get-pip.py --no-check-certificate
执行安装
python get-pip.py
```
至此pip安装成功
![pip安装](https://upload-images.jianshu.io/upload_images/5955013-818d8173611ace24.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

安装tensorflow较为简单
```
pip install tensorflow
```
测试tensorflow是否安装成功
```
python
import tensorflow as tf
tf.__version__
```
![tensorflow](https://upload-images.jianshu.io/upload_images/5955013-f04a81321b33af4e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 2. Caffe
### 1) 配置环境变量
```
vim ~/.bashrc
```
打开后在文件最后加入以下两行内容：
```
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```
保存退出,并使环境变量立即生效
```
source ~/.bashrc
```
### 2)安装CUDA
服务器已配备好CUDA10.1,在此不再赘述。
### 3）安装HDF5
1 从官网下载hdf5，我下的版本是hdf5-1.8.21.tar.gz
2 执行解压
```
tar -xvf hdf5-1.8.21.tar.gz
```
解压后会生成目录hdf5-1.8.21，切换到该目录下
```
 cd  hdf5-1.8.21/
```
3 依次执行
``` 
 ./configure --prefix=/usr/local/hdf5
make
make check
make install
```
4 安装成功后，在安装目录/usr/local下出现hdf5文件夹，打开后
![hdf5](https://upload-images.jianshu.io/upload_images/5955013-face99121de7672c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

5 再切换到该目录下
```
cd /usr/local/hdf5/share/hdf5_examples/c
./run-c-ex.sh
```
6 执行命令
```
h5cc -o h5_extend h5_extend.c
```
若出现
```
-bash: h5cc: command not found
```
则执行
```
apt install hdf5-helpers
再执行
h5cc -o h5_extend h5_extend.c
```
若出现
![](https://upload-images.jianshu.io/upload_images/5955013-d3e7b892e9d32bc9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
则执行
```
apt-get install libhdf5-serial-dev
 再执行
h5cc -o h5_extend h5_extend.c
直到执行后没有错误显示
```
7 执行命令
```
./h5_extend
```
到此HDF5安装完毕

### 4）安装CUDNN

登录官网：[https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download) ，下载对应 cuda 版本且 linux 系统的 cudnn 压缩包，注意官网下载 cudnn 需要注册帐号并登录。

我下的版本是cudnn-10.1-linux-x64-v7.5.0.56.tgz

下载完成后解压，得到一个 cuda 文件夹，该文件夹下include 和 lib64 两个文件夹，命令行进入 cuda/include 路径下，然后进行以下操作：
```
cp cudnn.h /usr/local/cuda/include/ #复制头文件
```
然后命令行进入 cuda/lib64 路径下，运行以下命令：
```
cp lib* /usr/local/cuda/lib64/ #复制动态链接库
cd /usr/local/cuda/lib64/
rm -rf libcudnn.so libcudnn.so.7 #删除原有动态文件
ln -s libcudnn.so.7.5.0 libcudnn.so.7 #生成软衔接
ln -s libcudnn.so.7 libcudnn.so #生成软链接
```
安装nvcc
```
apt-get install nvidia-cuda-toolkit
```
安装完成后可用 nvcc -V 命令验证是否安装成功，若出现以下信息则表示安装成功：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17
```
### 5) 安装caffe
首先在你要安装的路径下 clone 
```
git clone https://github.com/BVLC/caffe.git
```
进入 caffe ，将 Makefile.config.example 文件复制一份并更名为 Makefile.config ，也可以在 caffe 目录下直接调用以下命令完成复制操作 
```
cp Makefile.config.example Makefile.config
```
复制一份的原因是编译 caffe 时需要的是 Makefile.config 文件，而Makefile.config.example 只是caffe 给出的配置文件例子，不能用来编译 caffe。

然后修改 Makefile.config 文件，在 caffe 目录下打开该文件
```
vim Makefile.config
```
修改 Makefile.config 文件内容：
1.应用 cudnn
```
将
#USE_CUDNN := 1
修改成： 
USE_CUDNN := 1
```
2.使用 python 接口
```
将
#WITH_PYTHON_LAYER := 1 
修改为 
WITH_PYTHON_LAYER := 1
```
3.修改 python 路径
```
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib 
修改为： 
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial
```

然后修改 caffe 目录下的 Makefile 文件
```
将：
NVCCFLAGS +=-ccbin=$(CXX) -Xcompiler-fPIC $(COMMON_FLAGS)
替换为：
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
```
```
将：
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5
改为：
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```

接着便可以开始编译了，在 caffe 目录下执行 :
```
make all -j8
```
此时可能报错
![](https://upload-images.jianshu.io/upload_images/5955013-2c464822b2634c05.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
原因是CUDA的版本较高，进入Makefile.config,将下面这两行注释掉
```
-gencode arch=compute_20,code=sm_20 \
-gencode arch=compute_20,code=sm_21 \
```
![](https://upload-images.jianshu.io/upload_images/5955013-5e797d1565a3c387.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
make clean
make all -j8 
```
可能报错如下
![](https://upload-images.jianshu.io/upload_images/5955013-754ce11565c94947.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
解决方案：出现该错误的原因是少了依赖。

在命令行输入：
```
apt-get install --no-install-recommends libboost-all-dev
```
即可解决。
若出现
```
Makefile:637: recipe for target '.build_release/tools/extract_features.bin' failed
make: *** [.build_release/tools/extract_features.bin] Error 1
```
的类似情况，一般是依赖包的问题，请自行排查

编译成功后可运行测试：
```
make runtest -j8
```
若显示结果如图，则证明caffe已经安装成功
![](https://upload-images.jianshu.io/upload_images/5955013-a4dbf12dc61706ce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

测试：
```
python
```
```
import caffe
```
若没有报错则说明caffe的python接口已经编译成功
但也出现以下问题
**问题1**
```
ImportError: No module named caffe
```
**解决办法**
```
在环境变量中加入
export PYTHONPATH=/workspace/caffe/python:$PYTHONPATH
```
**问题2**
```
ImportError: No module named skimage.io
```
**解决办法**
```
pip install -U scikit-image #若没有安装pip: sudo apt install python-pip
```
**问题3**
![](https://upload-images.jianshu.io/upload_images/5955013-ff86d3a9d5e09b4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
**解决办法**
```
apt-get install python-numpy
编译pycaffe
make pycaffe -j16
```
最终结果如图：
![](https://upload-images.jianshu.io/upload_images/5955013-34a58060981cafcd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 3.Matlab
### 下载安装包
链接：[https://pan.baidu.com/s/1wijZCXIWsNXgz0yYYBXHnQ](https://pan.baidu.com/s/1wijZCXIWsNXgz0yYYBXHnQ) 
密码：e8b2

下载完成如下图所示
![image](http://upload-images.jianshu.io/upload_images/5955013-eff9adb60fc682b4?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### 开始安装
1 文件解压
将文件下R2017b_glnxa64以及MATLABR2017b_Linux_Crack两压缩包解压

2.挂载镜像文件
首先需要挂载iso文件
```
mkdir matlab
mount -t auto -o loop R2017b_glnxa64.iso matlab/
```

3.进入文件夹安装
直接在vnc中双击install进入安装
到激活页面选择‘我已有我的许可证的文件安装密钥’
激活码为09806-07443-53955-64350-21751-41297
一直下一步直到完成安装

4.破解
打开MATLABR2017b_Linux_Crack文件夹，在该文件夹下右击打开终端，在终端输入如下代码
```
cp license_standalone.lic /usr/local/MATLAB/R2017b/licenses/ 
cp libmwservices.so /usr/local/MATLAB/R2017b/bin/glnxa64/
```
5.取消挂载
安装完成，这时可以取消前面的文件挂载了，在终端输入以下代码取消挂载
```
umount matlab/
```
6. 运行matlab
```
cd /usr/local/MATLAB/R2017b/bin
./matlab
```
为了方便在任意位置都能直接输入“matlab”启动matlab，可创建一个sh文件，在此以matlab.sh为例
```
vim /workspace/Shell/matlab.sh
```
在matlab.sh中输入
```
cd /usr/local/MATLAB/R2017b/bin
./matlab
```
给matlab.sh赋权限
```
chmod 777 matlab.sh
```
在~/.bashrc的最后添加
```
alias matlab="/workspace/Shell/matlab.sh"
```
最后激活环境变量
```
source ~/.bashrc
```

在任意路径输入matlab,大功告成！
至此matlab安装完成




# 4.Pycharm
首先我们要下载pycharm的安装包，

地址为[https://www.jetbrains.com/pycharm/download/#section=linux](https://www.jetbrains.com/pycharm/download/#section=linux)

解压好后打开终端设备，输入命令
```
mv pycharm-2019.1.1 /opt
```
将pycharm解压包移动到opt文件夹下，然后在终端设置中输入命令给pycharm文件夹权限
```
chmod -R 744 /opt/pycharm-2019.1.1
```
在安装之前请首先修改hosts文件，在终端设备中输入命令
```
vim /etc/hosts
```
在打开的文件中加入0.0.0.0 account.jetbrains.com和0.0.0.0 www.jetbrains.com，如下图
![](https://upload-images.jianshu.io/upload_images/5955013-fdd6566de5a6bac9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


接下来需要把文件JetbrainsCrack-2.6.10-release-enc.jar下载并放到pycharm-2019.1.1/bin文件夹中，文件下载地址为
[ https://pan.baidu.com/s/1suF5f3byC1EoIjQRZNTIuA](https://pan.baidu.com/s/1suF5f3byC1EoIjQRZNTIuA)  密码为：5rp7

然后在进入终端设备，输入命令 
```
cd /opt/pycharm-2019.1.1/bin/
```
进入后输入命令
```
sh ./pycharm.sh
```
启动pycharm 的安装，等待一段时间后会自安装完成并打开
激活时选择第二个激活码激活
![](https://upload-images.jianshu.io/upload_images/5955013-33f3d8ddfa35b6cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

进入[http://idea.lanyus.com/](http://idea.lanyus.com/)获取注册码，注册码有效期至2020.3.
参照matlab的方式可在任意路径输入’pycharm‘启动pycharm
至此Pycharm安装完毕











