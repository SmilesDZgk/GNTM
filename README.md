## Graph Neural Topic Model (GNTM)
This is the pytorch implementation of the paper "Topic Modeling Revisited: A Document Graph-based Neural Network Perspective"

### Requirements
* Python >= 3.6
* Pytorch == 1.6.0
* torch-geometric == 1.7.0
* torch-scatter == 2.0.6
* torch-sparse == 0.6.9

### Dataset
The links of the datasets can be found in the following:

* [20 News Group](http://qwone.com/~jason/20Newsgroups/)
* [Tag My News](http://acube.di.unipi.it/tmn-dataset/)
* [British National Corpus](https://www.sketchengine.eu/british-national-corpus/)
* [Reuters](https://trec.nist.gov/data/reuters/reuters.html)

The Glove word embeddings can be download from theis [link](https://nlp.stanford.edu/projects/glove/).

The datasets and word embedings should be placed with the guide of the paths in the `settings.py`.

### Usage
Before training GNTM, we first need to preprocess the data by the following scripts (need  adjust some parameters based on the description in our paper for different datasets.):
```angular2
cd dataPrepare
python preprocess.py
python graph_data.py
```

Example script to train GNTM:
```angular2
python main.py \
--device cuda:0 \
--dataset News20 \
--model GDGNNMODEL \
--num_topic 20 \
--num_epoch 400 \
--ni 300  \
--word \
--taskid 0 \
--nwindow  3
```

Here,
* `--dataset` specifies the dataset name, currently it supports `News20`, `TMN`, `BNC` and `Reuters` for `20 News Group`, `Tag My News`, `British National Corpus` and `Reuters`, respectively.
* `--device` represents computation device, such as `cpu` or `cuda:0`.
* `--model` represents the used model, `GDGNNMODEL` is corresponding to `GNTM`
* `--num_topic` represents the number of topics.
* `--num_epoch` represents the maximized number of  training epochs.
* `--ni` represents the dimension of word embeddings.
* `--taskid` is corresponding to the random seed.
* `--nwindow` represents the window size to construct dpcument graphs.  







### Reference
If you find our methods or code helpful, please kindly cite the paper: 
```angular2
@inproceedings{shen2021topic,
  title={Topic Modeling Revisited: A Document Graph-based Neural Network Perspective},
  author={Shen, Dazhong and Qin, Chuan and Wang, Chao and Dong, Zheng and Zhu, Hengshu and Xiong, Hui},
  booktitle={Proceedings of Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS-2021)},
  year={2021}
}
``` 
