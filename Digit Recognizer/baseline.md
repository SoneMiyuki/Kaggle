#Kaggle Digit Recognizer Notebook

##Intro

[Source](https://www.kaggle.com/c/digit-recognizer/data)
The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

### Score by:
Accuracy

## 1.0 baseline
 - 这次主要尝试提交Kaggle的各项功能

### Analysis
这显然是一个分类问题.

### Feature Programming
无

### Data preprocessing
无

### Baseline:
主要用了sklearn库里的比较朴素的MLPClassifier进行学习分类,但是貌似这个case用CNN会好一点.

- 用L-BFGS算法进行近似和迭代
- 100层隐藏层
- 激活函数用relu提升速度


```Python
mlp_hw = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes = [100,100],activation = 'relu',alpha = 1e-5,random_state = 0)
```

但是由于数据集太小,特征工程做得不太好,没有做交叉验证0等原因,得分很可怜.模型太简单,IO的时间都超过了训练模型的时间

算了算了,当作一个渺小的开始吧.

### Score:
0.08100
	
## 2.0 baseline
用MNIST训练了这个模型
我貌似用到了原始数据集,结果这个成绩非常让人满意

### Analysis
这显然是一个分类问题.

### Feature Programming
无

### Data preprocessing
将0-255以线性变化压到0-1的浮点数
这应该是提分的一个点

### Baseline:


### Score : 
0.99992

### 2.1 baseline
修改成了用提供的数据集

### Analysis
这显然是一个分类问题.

### Feature Programming
无

### Data preprocessing
 - 将0-255以线性变化压到0-1的浮点数
 - 这应该是提分的一个点

### Baseline:


### Score : 
0.97178



----

代码主要参考了清华大学出版社 段小手著 深入浅出Python机器学习 神经网络章节