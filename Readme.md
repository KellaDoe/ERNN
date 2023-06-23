# Evidence Reconciled Neural Network

Out-of-Distribution detection method.



## Data preparation

The ISIC2019 dataset can be download at [https://challenge.isic-archive.com/data/#2019](https://challenge.isic-archive.com/data/#2019)

Please change your own path for ISIC2019 dataset in **evaluate.py** and **train.py**

```python
p_train_img = '/mnt/mnt_data/ISIC_2019/ISIC_2019_Training_Input'
p_train_label = '/mnt/mnt_data/ISIC_2019/ISIC_2019_Training_GroundTruth.csv'
```



## Code

**evaluate.py**

To evaluate ID classification and OOD detection:

```
python evaluate.py
```

**train.py**

To train the proposed ERNN on existing dataset

```
python evaluate.py
```

**dataset.py**

Load dataset, split ood categories, split data for five-fold for id categories.

**models.py**

Implementation for proposed ERNN.

**evaluation.py**

In line with the work  "Out-of-Distribution Detection for Long-tailed and Fine-grained Skin Lesion Images" (https://arxiv.org/abs/2206.15186) in MICCAI 2022.

