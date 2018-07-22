# 计算机视觉和机器学习Final实验报告

## 1. 实验环境

 - Windows10 + VS2015 + python
 - ubuntu + python
 - python有以下环境：
	 - numpy
	 - pandas
	 - tensorflow

## 2. 实验数据

老师和TA提供的写有数字的A4纸照片

## 3. 实验原理

该实验包含了从检测A4纸、A4纸矫正、数字切割、数字识别等一系列操作，原理介绍如下：

 - A4纸检测：主要采用Canny算法进行边缘检测，检测以后通过Hough变换检测A4纸的4条边，根据4条边求出4个角点，从而检测出A4纸，大致过程和之前的作业相同，因此在这里不详细说，主要列举采用的方法：
	 - 对输入的图片进行统一放缩，使得Hough变换的阈值能够适应各种大小的图片。
	   ![scale](http://p8pbukobc.bkt.clouddn.com/scale.PNG)
	 - Canny边缘检测算法
	 - Hough变换
 - A4纸矫正：主要采用仿射变换对原图进行变换；根据A4纸检测中检测到的4个角点进行标准A4纸的矫正，大致过程和之前的作业相同，主要采用的方法：
	 - 利用C++的Eigen库进行仿射变换，原理如下：
	 ![fangshe](http://p8pbukobc.bkt.clouddn.com/fangshe.png)
	 - A4纸进行矫正后存在一定的误差，因此裁剪掉边缘的几个像素，便于后续处理。
	 ```result = result.get_crop(10, 10, result._width - 10, result._height - 10);```
 - 数字切割：这里切割数字先采用了行切割，而后才对每行的单个字符进行数字字符的切割，原因在于最后识别出这些数字时，要能够有序的输出，并且有着9行数字这样的先验知识，因此才用了这样的一种模式。
	 - 预处理：后续识别所用的模型是mnist数据集进行训练的，而mnist训练集是黑底白字，而我们的测试样本均是白底黑字，因此要进行颜色的反转，这里主要采用的是二值化的操作，通过设定某个阈值，把灰度图进行二值化。
	 ![preprocess](http://p8pbukobc.bkt.clouddn.com/preprocess.PNG)
	 - 行切割：采用了像素统计的办法进行行的切割，通过统计每行白色像素的个数从而确定数字所在的大致范围并进行行切割，考虑到图片可能存在噪声，还对白色像素进行了统计，数目过小的堆会被试做噪声而不是数字行，统计图(黑色宽度代表改行白色像素个数)如下：
	 ![tongji](http://p8pbukobc.bkt.clouddn.com/tongji.PNG)
	 - 单个字符切割：在进行行切割以后，将对每行进行单个数字的切割，这里主要采用连通域的算法进行，单个数字的像素必定构成连通域，通过对连通域的搜索，确定了构成单个数字的所有像素，通过这些像素找到该数字的左上角（最小x,y）坐标和右下角（最大x,y）坐标，从而实现单个字符的切割。
		 - 通过连通域进行单个数字的切割可以很好的找出每一个数字，但前提是该数字的所有像素信息没有丢失，由于我们一开始进行过二值化的处理，因此可能造成了某些数字的断裂，导致单个数字的像素不连通，为了使该算法更加有效，在进行连通域分割数字之前，先进行膨化操作，尽可能使得断裂的数字相连。
		 - 这里“切割”并不是单纯的从图里直接截取数字的框，而是对一个新的图像进行像素赋值，原因是mnist是28\*28大小的数据集，因此我们在检测时候也需要输入28\*28大小的图片，但是由于直接切割的数字往往不是正方形，强行进行28\*28的缩放会导致数字变形扭曲，因此这里“切割”时构造一个边长为(width > height ? width : height)的正方形，之后将数字的像素通过遍历赋值的方式存入该正方形中，最后统一缩放到28\*28大小,用于下一步的数字识别。
 - 数字识别：采用卷积神经网络训练mnist数据集，所用平台为ubuntu + python + tensorflow，卷积神经网络结构(卷积均使用1步长，0边距)如下：
	 - 第一层卷积，卷积核大小为5*5，计算出32个特征，特征大小为[28,28,32]，通过relu激活
	 - 最大池化层：大小为2*2，输出32个特征，特征大小为[14,14,32]
	 - 第二层卷积：卷积核大小为5*5，计算得到64个特征，特征大小为[14,14,64]，通过relu激活
	 - 最大池化层：大小为2*2，输出64个特征，特征大小为[7,7,64]
	 - 全连接层：1024个神经元的全连接层，大小为[7 \* 7 \* 64,1024]
	 - softmax层：最后通过一个softmax层输出10个类（0-9）的概率

## 4. 实验结果

这里以15331180.bmp图为例展示算法运行的全过程：

原图展示：

![yuantu](http://p8pbukobc.bkt.clouddn.com/yuantu.PNG)

角点坐标：

![jiaodian](http://p8pbukobc.bkt.clouddn.com/jiaodian.PNG)

A4纸矫正后：

![jiaozheng](http://p8pbukobc.bkt.clouddn.com/jiaozheng.PNG)

微调后：

![weitiao](http://p8pbukobc.bkt.clouddn.com/weitiao.PNG)

二值化结果：

![erzhihua](http://p8pbukobc.bkt.clouddn.com/erzhihua.PNG)

切割单行结果：

![danhang](http://p8pbukobc.bkt.clouddn.com/danhang.PNG)

![danhang1](http://p8pbukobc.bkt.clouddn.com/danhang1.PNG)

连通域结果(不同亮度代表不同的连通域)：

![liantongyu](http://p8pbukobc.bkt.clouddn.com/liantongyu.PNG)

![liantongyu1](http://p8pbukobc.bkt.clouddn.com/liantongyu1.PNG)

切割结果：

![qiegejieguo](http://p8pbukobc.bkt.clouddn.com/qiegejieguo.PNG)

预测结果：

![yucejieguo](http://p8pbukobc.bkt.clouddn.com/yucejieguo.PNG)

精度分析：

从图片可以看出，九行数字分别是15331180、13260831048、442313199803273059重复三遍；
预测的结果可以看出15331180全部预测正确，即24/24；
13260831048预测结果不一，有一个预测了12个数字，显然是在切割数字时，数字断裂造成了一个数字切成了两块，计算正确预测个数，分别为8/11,10/11,10/11，即28/33，这里可以看出，后两行主要的错误是6预测为8；
442313199803273059结果正确率分别为16/18（主要错误为9预测为7，多了一个1，以及9预测为7），16/18,14/18，即46/54。
总的准确率为98/111 = 88.29%
总体来说还是让人满意的。

下面是其他测试图片的一些精度：

![15331159](http://p8pbukobc.bkt.clouddn.com/15331159.PNG)

精度为：97/111 = 87.39%

![15331362](http://p8pbukobc.bkt.clouddn.com/15331362.PNG)

精度为：103/111 = 92.80%（发现8和6特别容易混淆）

![15331009](http://p8pbukobc.bkt.clouddn.com/15331009.PNG)

精度为：99/111 = 89.19%

![15331050](http://p8pbukobc.bkt.clouddn.com/15331050.PNG)

精度为：98/111 = 88.29%







