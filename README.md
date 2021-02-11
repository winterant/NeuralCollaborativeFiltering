
# Neural Collaborative Filtering

This is my implementation for the paper:
>Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). Neural Collaborative Filtering. In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

该代码包含GMF、MLP、NeuMF

注：该代码的目标函数和评测指标是均方误差（原文目标函数为交叉熵函数，评测指标为HR和NDCG）。
该代码中模型最后一层没有激活函数（原文中最后一层激活函数为Sigmoid）。

# Environments
+ python 3.8
+ pytorch 1.70

# Dataset

1. Amazon(2014) http://jmcauley.ucsd.edu/data/amazon/links.html
2. Yelp(2020) https://www.yelp.com/dataset

For example:
`data/ratings_Digital_Music.csv` (Amazon Digital Music: rating only)

注：数据集前三列分别为用户id、产品id、评分（1~5）。
运行`main.py`时，数据集按0.8/0.1/0.1的比例划分为训练集、验证集、测试集。
若使用了amazon/yelp数据集json格式，可使用data_preprocess.py预处理。

For example:
```shell script
python data_preprocess.py --data_path Digital_Music_5.json --data_source amazon --save_file amazon_music_ratings.csv
```

# Running
Train and evaluate the model
```
python main.py --dataset_file data/ratings_Digital_Music.csv
```

# Experiment
<table align="center">
    <tr>
        <th>Dataset</th>
        <th>number of users</th>
        <th>number of items</th>
        <th>MSE of NeuMF</th>
    </tr>
    <tr>
        <td><a href="http://files.grouplens.org/datasets/movielens/ml-latest-small.zip">movielens-small</a> (100,836)</td>
        <td>610</td>
        <td>9724</td>
        <td>0.740655</td>
    </tr>
    <tr>
        <td>Amazon music-small (64,706)</td>
        <td>5541</td>
        <td>3568</td>
        <td>0.822472</td>
    </tr>
    <tr>
        <td>Amazon music (836,006)</td>
        <td>478235</td>
        <td>266414</td>
        <td>0.825261</td>
    </tr>
    <tr>
        <td>Amazon Clothing, Shoes and Jewelry (278,677)</td>
        <td>39387</td>
        <td>23033</td>
        <td>1.093927</td>
    </tr>
</table>
