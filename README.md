# News Recommendation

Implementation of several news recommendation methods in Pytorch.

Support multi-GPU training and testing.



> **WARNING**: Multi-GPU training and testing only work on Linux because currently `torch.distributed` only support Linux.



### Requirements

- python
- pytorch
- numpy
- scikit-learn
- nltk
- tqdm

You can create a `conda` virtual environment named `nr` with the provided `env.yaml`.

```bash
conda env create -f env.yaml			# Maybe only work on Linux
```



### Usage

- **Clone this repository**	

  ```bash
  git clone https://github.com/yflyl613/NewsRecommendation.git
  cd NewsRecommendation
  ```
  
- **Prepare data**

  A scirpt `download_data.sh` is provided to download and unzip all the required data. It will create a new folder `data/` under `NewsRecommendation/`.
  
  ```bash
  # In NewsRecommendation/
  chmod +x download_data.sh
  ./download_data.sh
  ```
  
- **Start training**

  A script `demo.sh` is provied for model training and testing, in which you can modify parameters for the experiment. Please refer to `parameters.py` for more details.
  
  ```bash
  # In NewsRecommendation/data/
  cd ../src
  chmod +x demo.sh
  
  # train
  ./demo.sh train
  
  # test
  ./demo.sh test <checkpoint name>
  # E.g. ./demo.sh test epoch-1.pt
  ```



### Results on MIND-small validation set<sup>[1]</sup>

- **NAML<sup>[2]</sup>**

  |         News information         |  AUC  |  MRR  | nDCG@5 | nDCG@10 |                  Configuration                  |
  | :------------------------------: | :---: | :---: | :----: | :-----: | :---------------------------------------------: |
  | subcategory<br>category<br>title | 66.24 | 32.08 | 35.36  |  41.56  | batch size 128 (32*4)<br> 5 epochs<br>lr 3e-4 |
  
- **NRMS<sup>[3]</sup>**

  | News information |  AUC  |  MRR  | nDCG@5 | nDCG@10 |                 Configuration                 |
  | :--------------: | :---: | :---: | :----: | :-----: | :-------------------------------------------: |
  |      title       | 66.61 | 31.86 | 35.19  |  41.46  | batch size 128 (32*4)<br> 4 epochs<br>lr 3e-4 |




> **Please feel free to contact me by opening an [issue](https://github.com/yflyl613/NewsRecommendation/issues) if you have any problem or find a bug :)**



### Reference

[1] Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu and Ming Zhou. [MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf). ACL 2020.

[2] Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, and Xing Xie. [Neural News Recommendation with Attentive Multi-View Learning](https://www.ijcai.org/Proceedings/2019/0536.pdf). IJCAI. 2019.

[3] Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie. [Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671.pdf). EMNLP-IJCNLP. 2019.

