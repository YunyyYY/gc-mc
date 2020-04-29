# Record

Latent variable model

观测一个data generating process的数据，假设一个latent model，推断参数

在求posterior probability的时候如果涉及积分，需要很大的复杂度求closed form，用variatial inference逼近posterior

likelihood lower bound

计算lower bound不需要知道

KL divergence 需要知道posterior

通过 maximize lower bound，逼近posterior



## GCMC tensorflow benchmarks

### gcmc-tf

#### Douban

```shell
python train.py -d douban --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing 

# output
Optimization Finished!
best validation score = 0.6832775 at iteration 193
test loss =  1.0493292
test rmse =  0.7347704
polyak test loss =  1.0564681
polyak test rmse =  0.7353999

{"features": true, "accumulation": "stack", "best_epoch": 193, "feat_hidden": 64, "learning_rate": 0.01, "testing": true, "dataset": "douban", "epochs": 200, "num_basis_functions": 2, "summaries_dir": "logs/2020-04-08_10:35:19.657006", "dropout": 0.7, "best_val_score": 0.6832774877548218, "hidden": [500, 75], "norm_symmetric": false, "data_seed": 1234, "write_summary": true}
```



#### Flixster

```shell
python train.py -d flixster --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing

# output
Optimization Finished!
best validation score = 0.63396037 at iteration 199
test loss =  1.9749713
test rmse =  0.92655355
polyak test loss =  1.9442751
polyak test rmse =  0.92089385

global seed =  1586356352
{"features": true, "accumulation": "stack", "best_epoch": 199, "feat_hidden": 64, "learning_rate": 0.01, "testing": true, "dataset": "flixster", "epochs": 200, "num_basis_functions": 2, "summaries_dir": "logs/2020-04-08_10:32:32.507787", "dropout": 0.7, "best_val_score": 0.6339603662490845, "hidden": [500, 75], "norm_symmetric": false, "data_seed": 1234, "write_summary": true}
```



####Yahoo Music

```shell
python train.py -d yahoo_music --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing

# output
Optimization Finished!
best validation score = 14.666101 at iteration 198
test loss =  1.7242996
test rmse =  19.17603
polyak test loss =  1.7287583
polyak test rmse =  19.277351

global seed =  1586356288
{"features": true, "accumulation": "stack", "best_epoch": 175, "feat_hidden": 64, "learning_rate": 0.01, "testing": true, "dataset": "yahoo_music", "epochs": 200, "num_basis_functions": 2, "summaries_dir": "logs/2020-04-08_10:31:28.855005", "dropout": 0.7, "best_val_score": 17.137279510498047, "hidden": [497, 75], "norm_symmetric": false, "data_seed": 1234, "write_summary": true}
```



Reimplement Figure 3:

|         | feature | 0          | 50         | 100        | 150        |
| ------- | ------- | ---------- | ---------- | ---------- | ---------- |
| Nr = 1  | with    | 0.9021618  | 0.91765934 | 0.9379368  | 0.95108765 |
|         | without | 0.91341203 | 0.92298335 | 0.94107444 | 0.95593945 |
| Nr = 5  | with    | 0.9021618  | 0.91181445 | 0.91912806 | 0.9262069  |
|         | without | 0.9086468  | 0.9209418  | 0.92117655 | 0.9419526  |
| Nr = 10 | with    | 0.9021618  | 0.9098966  | 0.913428   | 0.92243236 |
|         | without | 0.9086468  | 0.9144648  | 0.9166853  | 0.9225823  |



#### Movielens 100K on official split with features

```shell
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing

# output
Optimization Finished!
best validation score = 0.7550453 at iteration 998
test loss =  1.2138525
test rmse =  0.9021618
polyak test loss =  1.2122221
polyak test rmse =  0.90033966

{"features": true, "accumulation": "stack", "best_epoch": 998, "feat_hidden": 10, "learning_rate": 0.01, "testing": true, "dataset": "ml_100k", "epochs": 1000, "num_basis_functions": 2, "summaries_dir": "logs/2020-04-08_10:44:52.076100", "dropout": 0.7, "best_val_score": 0.7550452947616577, "hidden": [500, 75], "norm_symmetric": false, "data_seed": 1234, "write_summary": true}
```



#### Movielens 100K on official split without features

```shell
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --testing

# output
Optimization Finished!
best validation score = 0.7730895 at iteration 997
test loss =  1.2198563
test rmse =  0.9086468
polyak test loss =  1.2162966

{"features": false, "accumulation": "stack", "best_epoch": 997, "feat_hidden": 64, "learning_rate": 0.01, "testing": true, "dataset": "ml_100k", "epochs": 1000, "num_basis_functions": 2, "summaries_dir": "logs/2020-04-08_10:48:25.773115", "dropout": 0.7, "best_val_score": 0.7730895280838013, "hidden": [500, 75], "norm_symmetric": false, "data_seed": 1234, "write_summary": true}
```





#### Movielens 1M

```shell
python train.py -d ml_1m --data_seed 1234 --accum sum -do 0.7 -nsym -nb 2 -e 180 --testing

Optimization Finished!
best validation score = 0.891593 at iteration 174
test loss =  1.220517
test rmse =  0.9042224
polyak test loss =  1.2328982
polyak test rmse =  0.9030332

{"features": false, "accumulation": "sum", "best_epoch": 174, "feat_hidden": 64, "learning_rate": 0.01, "testing": true, "dataset": "ml_1m", "epochs": 180, "num_basis_functions": 2, "summaries_dir": "logs/2020-04-08_11:46:21.588444", "dropout": 0.7, "best_val_score": 0.8915929794311523, "hidden": [500, 75], "norm_symmetric": true, "data_seed": 1234, "write_summary": true}
```





### Summary

1. 
2. ML100K: <font color='red'>**With or without feature not much difference, in paper, with feature is slightly better.**</font>
3. ML1M: <font color='red'>**提前终止了，每次都被kill**</font>
4. ML10M: 没跑

