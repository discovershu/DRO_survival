## Distributionally Robust Survival Analysis: A Novel Fairness Loss Without Demographics
Shu Hu, George H. Chen

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)

_________________

Some of codes are extracted from  [FairSurv](https://github.com/kkeya1/FairSurv).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## How to run Dro-COX code
### 1. To run Dro-COX on FLC
#### For FLC (Linear), 
 ```test
python -u run_dro_cox.py --dataset FLC --model Linear --eps 0.15 --seed 7 > FLC_Linear_joint_dro_split.log 2>&1 & 
```
#### For FLC (MLP), 
 ```test
python -u run_dro_cox.py --dataset FLC --model MLP --eps 0.3 --seed 7 > FLC_MLP_joint_dro_split.log 2>&1 &  
```
### 2. To run Dro-COX on SUPPORT
#### For SUPPORT (Linear), 
 ```test
python -u run_dro_cox.py --dataset SUPPORT --model Linear --eps 0.15 --seed 7 > SUPPORT_Linear_joint_dro_split.log 2>&1 & 
```
#### For SUPPORT (MLP), 
 ```test
python -u run_dro_cox.py --dataset SUPPORT --model MLP --eps 0.5 --seed 7 > SUPPORT_MLP_joint_dro_split.log 2>&1 & 
```
### 3. To run Dro-COX on SEER
#### For SEER (Linear), 
 ```test
python -u run_dro_cox.py --dataset SEER --model Linear --eps 0.2 --seed 7 > SEER_Linear_joint_dro_split.log 2>&1 &
```
#### For SEER (MLP), 
 ```test
python -u run_dro_cox.py --dataset SEER --model MLP --eps 0.15 --seed 7 > SEER_MLP_joint_dro_split.log 2>&1 & 
```

## How to run Dro-COX (SPLIT) code
### 1. To run Dro-COX (SPLIT) on FLC
#### For FLC (Linear), 
 ```test
python -u run_dro_cox_split.py --dataset FLC --model Linear --eps 0.1 --seed 7 > FLC_Linear_joint_dro_split.log 2>&1 & 
```
#### For FLC (MLP), 
 ```test
python -u run_dro_cox_split.py --dataset FLC --model MLP --lr 0.0001 --eps 0.05 --seed 7 > FLC_MLP_joint_dro_split.log 2>&1 &  
```
### 2. To run Dro-COX (SPLIT) on SUPPORT
#### For SUPPORT (Linear), 
 ```test
python -u run_dro_cox_split.py --dataset SUPPORT --model Linear --eps 0.15 --seed 7 > SUPPORT_Linear_joint_dro_split.log 2>&1 & 
```
#### For SUPPORT (MLP), 
 ```test
python -u run_dro_cox_split.py --dataset SUPPORT --model MLP --lr 0.0001 --eps 0.2 --seed 7 > SUPPORT_MLP_joint_dro_split.log 2>&1 & 
```
### 3. To run Dro-COX (SPLIT) on SEER
#### For SEER (Linear), 
 ```test
python -u run_dro_cox_split.py --dataset SEER --model Linear --eps 0.15 --seed 7 > SEER_Linear_joint_dro_split.log 2>&1 &
```
#### For SEER (MLP), 
 ```test
python -u run_dro_cox_split.py --dataset SEER --model MLP --lr 0.0001 --eps 0.2 --seed 7 > SEER_MLP_joint_dro_split.log 2>&1 & 
```



## How to plot Figure 1 ( Dro-COX )
Setting dataset = 'FLC' in plot.py
#### For FLC,
 ```test
python plot.py
```

Setting dataset = 'SUPPORT' in plot.py
#### For SUPPORT,
 ```test
python plot.py
```

Setting dataset = 'SEER' in plot.py
#### For SEER,
 ```test
python plot.py
```

## Citation
Please kindly consider citing our paper in your publications. 
```bash
@inproceedings{hu2022distributionally,
  title={Distributionally Robust Survival Analysis: A Novel Fairness Loss Without Demographics},
  author={Hu, Shu and Chen, George H},
  booktitle={Machine Learning for Health},
  pages={62--87},
  year={2022},
  organization={PMLR}
}
```
