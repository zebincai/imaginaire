This repository re-implement the paper The Conditional Analogy GAN: Swapping Fashion Articles on People Images. The code is based on the repository [imaginaire](https://github.com/NVlabs/imaginaire).

## 一、简介

### 1.1 CAGAN 论文简介

​		CAGAN是有条件GAN的典型应用，主要解决服装行业模特试装成本高、效率低的问题；论文希望通过GAN的技术开发出一个算法，该算法能够将一件衣服试穿到一个模特身上；算法具有以下特点：

- 不需要成对的数据：这里数据现实不存在，及时存在需要付出大量的标注人力；
- 重复利用服装行业数据：服装行列多年来沉淀了海量数据，包含衣服样品照片、模特试穿照片、以及未被试穿的衣服样品照片；
- 泛化性能好：该算法能够泛化到没有没见过的模型和衣服的试穿

### 1.2 生成器

 **生成器目的**

   - 学习出模特和对应衣服的关系（作者从算法的角度看已有的数据可以更加丰富的信息，比直接生成一个模特更简单，得到穿衣效果更真实）

   - 学习出新衣服穿到模式身上，即交换身上衣服和新衣服

   - 在算法实现中，生成器并不直接生成换好衣服的样本；生成器分为两阶段：Unet阶段和凸组合；



**生成器输入**

​	包含三个输入Xi (模特)、Yi(模特身上的衣服)、Yj(新衣服);  

 ![](./image/image-20210411161704510.png)

**生成器实现**

- Unet阶段

  ![image-20210411163253499](C:\Users\10287\AppData\Roaming\Typora\typora-user-images\image-20210411163253499.png)![image-20210411163543055](C:\Users\10287\AppData\Roaming\Typora\typora-user-images\image-20210411163543055.png)

  输入：9通道的RGB数据, 分别是Xi、Yi,、Yj 3张图像

  输出：4通道的feature map；第1个通道为alpha分割mask，第2-3为衣服穿到模特身上的近似样本；

  

- 凸组合：

   再通过第二阶段凸组合合成最后的样本Xij; 

  Xij = alpha * Xij(近似) + （1 - alpha）* xi

   ![image-20210411162642506](./image/image-20210411162642506.png)

  

  **重点1：为了让生成器更好的学习，在生成器每一个中间都添加6个通道数据，为Xi 和Yi缩放到该层网络的尺度数据；**
  
  ​		**有博客对这个实现替换成Xi和Yj取得更好的效果；这个改动是非常有道理，在生成器是要生成模特穿新衣服的效果，原来的衣服Yi会不断减弱，而Yj的信息会不断增强；**
  
  **重点2：输出不直接生成模型样本，而是定义的mask和穿上的效果也是非常有意义**
  
  ​		**通过定义阶段2凸组合定义alpha、Xij、Xi三者直接的关系，同时也是定义了一个规则；约束alpha为分割的mask;这个先验信息的引入能够引导生成器在规则下更好学习；**

### 1.3  判别器

**判别器目的**

- 判别输入的样本是由真实的拍摄图像和生成样本，引导生成生成更加逼真的图像；

- 判别模特身材穿的衣服和和衣服是否是同一件

  ![image-20210411165908243](./image/image-20210411165908243.png)

  

  **判别器输入**

  ​	包含一个模特和一个衣服；具体组合如上图所示

  **判别器实现**

  ![image-20210411170153110](./image/image-20210411170153110.png)

     输入：6通道数据: 3个通道是模特、3个通道是衣服；

     输出：判别不是直接输出一个常数的二分类，而是输出一个 512x8x6 的feature map; 

  

  **重点1：实现时增加一个卷积层将512X8X6 转为 1X8X6 ；同时增加一个sigmoid层把输出约束在[0, 1]**

  **重点2：不直接输出常数的二分类，为了保持local（局部）信息的一致性，特别在大图像，采用patches判别器非常有意义**
  
  **重点3：与生成器的重点1相同**

### 1.4 损失函数

 - 损失函数组成：

   包含3个部分：GAN损失 + ID损失 + cycle损失

   ![image-20210411171431107](./image/image-20210411171431107.png)

   

 - GAN损失

     GAN损失分别对应3种模特和衣服组合的损失之和，包含是否是拍摄数据、是否是生成数据、模特和衣服是否是同一件；

   ![image-20210411171757076](./image/image-20210411171757076.png)

 - ID 损失

     约束生成器对模特的修改只修改衣服部分、而不过分修改非衣服部分；让修改的区域尽可能少

     ![image-20210411172258919](C:\Users\10287\AppData\Roaming\Typora\typora-user-images\image-20210411172258919.png)

 - cycle损失

     这里很好的利用 CycleGAN  对非对称数据的处理特性；约束模特换到新衣服后，再换回旧衣服；还原的图片和原始图片有非常好的一致性；

     ![image-20210411172947968](./image/image-20210411172947968.png)

   **重点1：ID损失文章并没有非常理论的推道，从常理也无法很好理解对alpha通道进行1范数约束能有效的引导；个人角度这个损失非常牵强。**

   **重点2：GAN损失和Cycle损失两个是非常巧妙，在作者的基础上，实现时在计算生成器损失增加一个损失，即由cycle 还原出图像 xji, 也要求xji 、yi的判别损失要像真实图像；**

   ![image-20210411173927258](./image/image-20210411173927258.png)



## 二、 代码实现

- **代码**

  在imaginaire框架下实现CAGAN, 代码提交地址 [github](https://github.com/zebincai/imaginaire) 参考提交记录：

  ​	https://github.com/zebincai/imaginaire/commit/2b7413ab7eb109694e39ca3b0a4a651955edc4af

  主要增加了以下文件

  [configs/projects/cagan/LipMPV/base.yaml](https://github.com/zebincai/imaginaire/commit/2b7413ab7eb109694e39ca3b0a4a651955edc4af#diff-82e4a25754372005263b04967e5e072e3ca4b18daff9e3b8846cd73c45e1612c) 

  [imaginaire/datasets/cagan.py](https://github.com/zebincai/imaginaire/commit/2b7413ab7eb109694e39ca3b0a4a651955edc4af#diff-2ab5add57cbf2cdd72d218b6e9b83e57bdd6119292b8992aa2a91e97d739cfe7) 

  [imaginaire/discriminators/cagan.py](https://github.com/zebincai/imaginaire/commit/2b7413ab7eb109694e39ca3b0a4a651955edc4af#diff-2bcef90c325aeb8db54d9b4e5c54e91a46099d8775933e6b13cc93cf53de1297)

  [imaginaire/generators/cagan.py](https://github.com/zebincai/imaginaire/commit/2b7413ab7eb109694e39ca3b0a4a651955edc4af#diff-77a75440e06d8c5ed36461162b18419a690e755942a33f7a49d3e4cac4ac65b3) 

  [imaginaire/trainers/cagan.py](https://github.com/zebincai/imaginaire/commit/2b7413ab7eb109694e39ca3b0a4a651955edc4af#diff-17c8345fb23defa518c03dc2c448eceba5d19ca570ac8d5303e2aa7d6b2f0e6b) 

- **训练数据**

  [LIP_MPV_256_192](https://competitions.codalab.org/competitions/23471)

  只使用每个样本的半身正面照、和对应正面衣服的成对样本, 分别作为训练的Xi, Yi；训练的时候Yj是随机从数据集中挑选j 不等于i的衣服样本；

- **训练命令**

```python
cd imaginaire

python train.py --config configs\projects\cagan\LipMPV\base.yaml --logdir D:\workspace\output\cagan  --single_gpu
```

- 补充说明： 

  - 代码是按照第一章节理解实现，代码完全跑通；训练参数是用了文章的训练参数；

  - 目前只用了5个样本调通代码，没有实际训练出模型；家里没有服务器，只在个人笔记本跑；



## 三、如何调参

### **3.1 本次任务调参**

​	我仅实现了代码，没有实际调参

### 3.2 过完调参经验

- **学习率**：采用warm up的方式，学习率的初始化值，通常根据paper或者以10位倍测试的出；
- **batchsize**:  通道调至GPU现存占用90%+
- **loss权重**：通常loss较重要的给与更大的权重；

-  **对于训练失败、溢出、none等情况**：确认数据的预处理到模型是正常的，其中通过观测loss组合中的每个loss, 查看哪一个比较异常，进一步定位是否实现有问题；有时候过高的学习率也会导致none的情况；

- **模型收敛性**：采用与训练模型；使用batchnorm;

- **算法优化**：通过输出中间结果，确认哪些效果较差，检测率低、还是召回率低；相应的模型、loss改进方向；

  

## 四、时间分配

​	 

| 序号 | 事项                                       | 时间      |
| ---- | ------------------------------------------ | --------- |
| 1    | 文章阅读、算法理解                         | 2个小时   |
| 2    | 网上资料：论文相关博客、github代码收集理解 | 3个小时   |
| 3    | 数据下载：主要是google drive，首次翻墙等   | 1.5个小时 |
| 4    | 理解imaginaire代码框架                     | 1.5个小时 |
| 5    | 代码实现：代码开发、调试                   | 8个小时   |
| 6    | Readme.md                                  | 3个小时   |



## 五、其他

### 5.1 适配imaginaire花费较多时间：

- 适配框架：理解框架，使用框架已封装的api花费较多时间；

- 框架安装：多第三方库需要安装才能正常跑；

- 运行环境：在笔记本上开发，有些库不支持，对框架部分不支持功能调试，注释掉不支持功能；

- 个人习惯：我自己会实现一个自己的训练框架，通常对于一个新算法有充足实现下我习惯在自己的框架下实现。

### 5.2 调试 debug

- debug通常是采用打日志；
- 日志对关键的流程进行打印，打印关键的参数，如loss, 准确率，召回率、输出中间效果图片
- 对于bug调试，同时会打印变了的shape,  参数值等；

## 六、参考资料

1. [Cloth Swapping with Deep Learning: Implement Conditional Analogy GAN in Keras](https://shaoanlu.wordpress.com/2017/10/26/reimplement-conditional-anology-gan-in-keras/) 
2.  keras实现： [github](https://github.com/shaoanlu/Conditional-Analogy-GAN-keras)
3. pytorch实现： [github](https://github.com/maktu6/CAGAN_v2_pytorch)
4. 数据集：[LIP_MPV_256_192](http://sysu-hcp.net/lip/overview.php)    2. [LIP_MPV_256_192](https://competitions.codalab.org/competitions/23471)

