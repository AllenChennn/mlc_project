# 多文本分类论文复现

## 代码结构

文件目录：

`data/`：存放各种数据集文件

`files/`：存放配置文件及相关文件

`output/`：存放模型保存及各种输出文件

`utils/`：存放自定义的函数类py文件

`models/`：存放模型定义类py文件

主目录下py文件用来存放各种模型的训练实现代码，风格与深度学习实践步骤对应。

> 深度学习模型实践步骤：
> 
> 1. 加载配置项
> 
> 2. 导入数据集
> 
> 3. 定义模型，loss，优化器
> 
> 4. 模型训练，保存

## LSAN

[GitHub - EMNLP2019LSAN/LSAN: Label-Specific Document Representation for Multi-Label Text Classification](https://github.com/EMNLP2019LSAN/LSAN)

将官方代码进行了封装，使代码更好调整和复用。

下载[AAPD.zip - Google 云端硬盘](https://drive.google.com/file/d/1QoqcJkZBHsDporttTxaYWOM_ExSn7-Dz/view)数据，解压到 `data/AAPD`下。

运行 `python train_lsan_on_aapd.py`。