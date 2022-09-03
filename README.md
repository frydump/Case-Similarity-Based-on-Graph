<h1 align="center">Deep Graph Matching Consensus</h1>


--------------------------------------------------------------------------------
##数据介绍
实验数据集来自相似案例匹配数据集，每条数据分为A，B，C三个案例，然后两两比较相似度。将案例数据提取并去重之后，大概可以得到一千多件不同案例，对应一千多个不同的图。

相似案例匹配数据的详细内容可查看[官网网站](http://cail.cipsc.org.cn/)和[官方Github](https://github.com/china-ai-law-challenge/CAIL2019)。
## Requirements

* **[PyTorch](https://pytorch.org/get-started/locally/)** (>=1.2.0)
* **[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)** (>=1.5.0)
* **[KeOps](https://github.com/getkeops/keops)** (>=1.1.0)
* **[networkx](https://github.com/networkx/networkx)** (>=2.4)

##数据处理
```
$ python vector.py
$ python data_create.py
```

##实验代码
从上到下分别是：本文的模型，监督bert，无监督bert，GED，TF-IDF

2019年冠军模型请参考https://github.com/hecongqing/CAIL2019
```
$ python Judicial_case.py
$ python CASE_BERT.py
$ python Bert.py
$ python GED.py
$ python TF-IDF.py
```


