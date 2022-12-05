<!-- Daily Note For A Week -->

#### 2022.11.13
##### 一、VScode连接跳板机（免密登录）：
1 生成本地的ssh的公钥，并将公钥复制到跳板机和目标服务器上
```linux{.line-numbers}
   ssh-keygen -t rsa -b 4096
```
生成的公钥在C:\Users\XXX\.ssh目录下id_rsa.pub，将其中内容复制到跳板机和目标服务器~/.ssh/authorized_keys中
2 配置VScode

1） 在扩展中安装Remote - SSH插件
2） 配置configure文件
```linux{.line-numbers}
Host JumpMachine             #跳板机名称
    HostName XXX.XXX.XXX.XXX #跳板机IP
    Port XXX                 #跳板机ssh端口
    User root                #跳板机用户名
 
Host TargetMachine           #远程服务器名称
    HostName XXX.XXX.XXX.XXX #远程服务器IP
    Port XXX                 #远程服务器ssh端口
    User root                #远程服务器用户名
    ProxyCommand ssh -W %h:%p JumpMachine
```
##### 二、nn.LSTM函数的用法：
1、长短期记忆(Long short-term memory, LSTM)：
为了解决长序列训练过程中的梯度消失和梯度爆炸问题。
2、torch.nn.lstm的用法：
```python{.line-numbers}
torch.nn.lstm(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)
```
input_size：表示的是输入的矩阵特征数，或者说是输入的维度；
hidden_size：隐藏层的大小（即隐藏层节点数量），输出向量的维度等于隐藏节点数；
num_layers：lstm 隐层的层数，默认为1；
bias：隐层状态是否带 bias，默认为 true；
batch_first：True 或者 False，如果是 True，则 input 为(batch, seq, input_size)，默认值为：False（seq_len, batch, input_size）
dropout：默认值0，除最后一层，每一层的输出都进行dropout；
bidirectional：如果设置为 True, 则表示双向 LSTM，默认为 False。

####2022.11.14
#####一、极线几何（Epipolar Geometry）：
http://t.csdn.cn/mteCt

极点e：分别是左边相机中心在右图像平面上的像，右相机中心在左像平面上的像。
极平面：两个相机中心和空间中某店p形成的平面。
极线l：极平面分别和两个像平面的交线。
对极几何则是描述这几个量之间的对应关系。直观讲，从左图的角度看，如果不知道p点的深度信息，射线op是这个点可能出现的空间位置，因为该射线上的点都会投影到同一个像素点，同时，如果不知道p点的具体位置，那么当在右图的位置看时，极线 l' 就是点p可能出现的位置，即在这条线上的某个地方。如下图所示



待补充

###2022.11.15
####一、在关闭ssh终端后让程序继续运行
在执行命令前加入nohup
```python{.line-numbers}
CUDA_VISIBLE_DEVICES=5 nohup python newcrfs/train.py configs/arguments_train_nyu.txt
```
####二、关闭nohup命令的方法
```shell{.line-numbers}
ps -aux|grep 命令关键字
```
看到后台运行进程id（数字）
```shell{.line-numbers}
kill -9 id
```
####三、查看当前运行显卡状态：
```cmd{.line-numbers}
nvidia-smi
```
四、显卡显存和内存的关系，以一次报错为例：

五、调用GPU进行训练
在命令前加入 例如
```python{line-numbers}
CUDA_VISIBLE_DEVICES=7 python train.py args_train_nyu.txt
```






























###2022.11.16
####一、嵌入（Embedding）
#####1 简单理解
简单来说，embedding就是用一个低维的向量表示一个物体，或是一个词等等。这个embedding向量的性质是**能使距离相近的向量对应的物体有相近的含义**，比如 Embedding(复仇者联盟)和Embedding(钢铁侠)之间的距离就会很接近，但 Embedding(复仇者联盟)和Embedding(乱世佳人)的距离就会远一些。

除此之外Embedding甚至还具有**数学运算**的关系，比如Embedding（马德里）-Embedding（西班牙）+Embedding(法国)≈Embedding(巴黎)

Embedding能够**用低维向量对物体进行编码还能保留其含义**的特点非常适合深度学习。

在传统机器学习模型构建过程中，经常使用one hot encoding对离散特征，特别是id类特征进行编码，但由于one hot encoding的维度等于物体的总数，比如阿里的商品one hot encoding的维度就至少是千万量级的。这样的编码方式对于商品来说是极端稀疏的，甚至用multi hot encoding对用户浏览历史的编码也会是一个非常稀疏的向量。

而深度学习的特点以及工程方面的原因使其**不利于稀疏特征向量的处理**。因此如果能把物体编码为一个**低维稠密向量**再喂给DNN，自然是一个高效的基本操作。
#####2 nn.Embedding原理及使用
https://www.jb51.net/article/254032.htm

###2022.11.17
####一、批标准化（Batch Normalization）
具体见：https://www.cnblogs.com/guoyaohua/p/8724433.html
#####1 问题引入
**1.1 IID独立同分布假设**
即训练数据和测试数据是相同分布，这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障。
**1.2 BN解决的问题**
随着网络深度加深，训练起来越困难，收敛越来越慢。改进方法：ReLU激活函数、Residual Network、BN等。
#####2 “Internal Covariate Shift”问题
**2.1 covariate shift：**
若<X,Y>中的输入值X的分布老是变，不符合IID假设，网络模型很难稳定的学规律。
**2.2 Internal**
在训练过程中，**隐层**的输入分布老是变来变去。
**2.3 BN的基本思想：**
**让每个隐层节点的激活输入分布固定下来**。研究表明，对输入图像进行白化（Whiten）——对输入数据分布变换到0均值，单位方差的正态分布，那么神经网络会较快收敛。可以理解为对深层神经网络**每个隐层神经元的激活值**做简化版本的白化操作。
#####3 BN的本质思想
**3.1 收敛慢的原因**
深层神经网络在做非线性变换前的激活输入值（Z=WX+B，X是输入）随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是**整体分布逐渐往非线性函数的取值区间的上下限两端靠近**（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），**导致反向传播时低层神经网络的梯度消失**。
**3.2 BN做法**
通过一定的规范化手段，把**每层**神经网络任意神经元这个**输入值**的分布强行拉回到**均值为0方差为1的标准正态分布**。使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，进而让梯度变大，①避免梯度消失问题产生，②学习收敛速度快，能大大加快训练速度。
**3.3 举例说明 看图说话**
<img src=/pics/1192699-20180405225246905-37854887.png width="50%" ></img>
假设某个隐层神经元原先的激活输入x取值符合正态分布，正态分布均值是-2，方差是0.5，（浅蓝色），通过BN后转换为均值为0，方差是1的正态分布（深蓝色），意味着输入x的取值正态分布整体右移2（均值的变化），图形曲线更平缓了（方差增大的变化）。这个图的意思是，BN其实就是把每个隐层神经元的激活输入分布从偏离均值为0方差为1的正态分布通过平移均值压缩或者扩大曲线尖锐程度，调整为均值为0方差为1的正态分布。
均值为0，方差为1的标准正态分布代表什么含义：
<img src=https://images2018.cnblogs.com/blog/1192699/201804/1192699-20180405225314624-527885612.png width="50%" ></img>
64%的概率x其值落在[-1,1]的范围内，在两个标准差范围内，也就是说95%的概率x其值落在了[-2,2]的范围内。那么这又意味着什么？我们知道，激活值x=WU+B,U是真正的输入，x是某个神经元的激活值，假设非线性函数是sigmoid，那么看下sigmoid(x)其图形：
<img src=https://images2018.cnblogs.com/blog/1192699/201804/1192699-20180407143109455-1460017374.png width="50%" ></img>
及sigmoid(x)的导数为：G’=f(x)*(1-f(x))，因为f(x)=sigmoid(x)在0到1之间，所以G’在0到0.25之间，其对应的图如下：
<img src=https://images2018.cnblogs.com/blog/1192699/201804/1192699-20180407142351924-124461667.png width="50%" ></img>
假设没有经过BN调整前x的原先正态分布均值是-6，方差是1，那么意味着95%的值落在了[-8,-4]之间，那么对应的Sigmoid（x）函数的值明显接近于0，典型的梯度饱和区。
而假设经过BN后，95%的x值落在了[-2,2]区间内，很明显这一段是sigmoid(x)函数接近于线性变换的区域，意味着x的小变化会导致非线性函数值较大的变化，也即是梯度变化较大，对应导数函数图中明显大于0的区域，就是梯度非饱和区。
#####4 训练阶段BN的具体做法
**4.1 BN的公式** 
<img src=https://images2018.cnblogs.com/blog/1192699/201804/1192699-20180405213859690-1933561230.png width="50%" ></img>
<img src=https://images2018.cnblogs.com/blog/1192699/201804/1192699-20180405213955224-1791925244.png width=50%></img>
在Mini-Batch SGD下做BN怎么做
$\hat x^{(k)} = \frac {x^{(k)}-E{[x^{(k)}]}}{\sqrt {Var[x^{(k)}]}} $
<img src="https://img-blog.csdnimg.cn/20200221215813522.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70" width=50%></img>
某个神经元对应的原始的激活x通过减去mini-Batch内m个实例获得的m个激活x求得的均值E(x)并除以求得的方差Var(x)来进行转换。
**4.2 scale和shift**
normalization 能够让所有layer output distributions 都固定在mean=0, variance=1的特定区域（可以缓解covariance shift problem)但在特定区域的大体框架下，如果可以在垂直方向上有一定程度的放大缩小的自由度（rescale），在水平方向上有一定的移动的自由度（shift)那么，也许所有layer output distributions能够做到更好的相似与稳定（当然，具体值不会相同）从而进一步缓解covariance shift 的问题

每个神经元增加两个调节参数（scale和shift），这两个参数是通过**训练**（在训练中通过反向传递来更新）来学习到的，用来对变换后的激活**反变换**，使得网络表达能力增强：
$ y^{(k)}= \gamma^{(k)}x^{(k)}+\beta^{(k)} $

#####5 理解图片中BN的具体操作
“对于一个拥有d维的输入x，我们将对它的**每一个维度**进行标准化处理。” 
假设输入的x是RGB三通道的彩色图像，d就是输入图像的channels即d=3
<img src="D:\PersonalFiles\Daily Note\pics\QQ截图20221117103514.jpg" ></img>
也即求出每个通道R,G,B的均值和方差，对于每个通道都进行归一化。
####二、Markdown图片导入
#####1 利用md语言：
![测试图片](/pics/1192699-20180405225246905-37854887.png)
开头一个感叹号 !
接着一个方括号，里面放上图片的替代文字
接着一个普通括号，里面放上图片的网址，最后还可以用引号包住并加上选择性的 'title' 属性的文字
缺：无法修改大小
#####2 利用html语言
<img src=/pics/1192699-20180405225246905-37854887.png width="50%" ></img>
缺：代码略复杂
####三、Markdown公式编写
https://blog.csdn.net/qq_42518956/article/details/116795578
建议利用Ctrl+F进行搜索
####四、条件随机场（Conditionsal Random Field）：
课程地址：
 https://www.bilibili.com/video/BV19t411R7QU
课程笔记（VPN）：
https://anxiang1836.github.io/2019/11/05/NLP_From_HMM_to_CRF/

###2022.11.18
####一、Opencv中图片基本操作：
#####1 翻转（Flip）
```python{lines-numbers}
img_new = cv2.flip(img, flipCode)
```
flipcode 小于0（例如-1）代表左右上下颠倒；0代表上下颠倒；大于0（例如1）代表左右颠倒
#####2 旋转（Rotate）
```python{lines-numbers}
img_new = cv2.flip(img, flipCode)
```
flipcode 小于0（例如-1）代表左右上下颠倒；0代表上下颠倒；大于0（例如1）代表左右颠倒
http://t.csdn.cn/ZO7No
**2.1 非90°旋转，角落的像素点如何改变**
####二、矩阵奇异值分解

####三、Markdown不同语言的命令分类


###2022.11.19
####一、python中 datetime的用法

####二、在线评估（Online Evaluation）

####三、ValueError: I/O operation on closed file。
报错原因：
- 是指处理了已经被关闭的数据。一般是语句没有对齐。当python的处理代码不对齐的时候会出现这种情况。
- 使用with方法打开了文件，生成的文件操作实例在with语句之外是无效的，因为with语句之外文件已经关闭了。
解决方法：
注意代码缩进，调用的保存/写入操作要在with语句块里能有效。

###2022.11.20
####一、tensor的切片操作

####二、Vscode编辑Markdown复制粘贴图片
- 安装扩展软件 Markdown Image
- 按照下图修改插件设置
![图 1](images/note/IMG_20221121-093120942.png)  
- 关键在于最后一行
```
${mdname}/IMG_${YY}${MM}${DD}-${HH}${mm}${ss}${mss}
```
功能就是在粘贴图片的同时把图片存放到 md 目录下与 mdname 命名的文件夹内
