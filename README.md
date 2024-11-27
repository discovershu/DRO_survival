## Paper 1. Distributionally Robust Survival Analysis: A Novel Fairness Loss Without Demographics

Shu Hu, George H. Chen

## Paper 2. Fairness in Survival Analysis with Distributionally Robust Optimization

Shu Hu, George H. Chen

[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)

This repository is the official implementation of our two papers "[Distributionally Robust Survival Analysis: A Novel Fairness Loss Without Demographics](https://proceedings.mlr.press/v193/hu22a.html)" (ML4H, 2022) and  "[Fairness in Survival Analysis with Distributionally Robust Optimization] (https://www.jmlr.org/papers/v25/23-0888.html)" (JMLR, 2024).

_________________

Some of the codes are extracted from  [FairSurv](https://github.com/kkeya1/FairSurv) and [SODEN](https://github.com/jiaqima/SODEN).

If you would like to use the SEER dataset, you should request access from https://seer.cancer.gov/data/.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Please also consider to install necessary packages from the SODEN. 

## How to run Dro-COX code
### 1. To run Dro-COX on FLC
#### For FLC (Linear), 
 ```test
python -u run_dro_cox.py --dataset FLC --model Linear --eps 0.15 --seed 7 > FLC_Linear_joint_dro.log 2>&1 & 
```
#### For FLC (MLP), 
 ```test
python -u run_dro_cox.py --dataset FLC --model MLP --eps 0.3 --seed 7 > FLC_MLP_joint_dro.log 2>&1 &  
```
### 2. To run Dro-COX on SUPPORT
#### For SUPPORT (Linear), 
 ```test
python -u run_dro_cox.py --dataset SUPPORT --model Linear --eps 0.15 --seed 7 > SUPPORT_Linear_joint_dro.log 2>&1 & 
```
#### For SUPPORT (MLP), 
 ```test
python -u run_dro_cox.py --dataset SUPPORT --model MLP --eps 0.5 --seed 7 > SUPPORT_MLP_joint_dro.log 2>&1 & 
```
### 3. To run Dro-COX on SEER
#### For SEER (Linear), 
 ```test
python -u run_dro_cox.py --dataset SEER --model Linear --eps 0.2 --seed 7 > SEER_Linear_joint_dro.log 2>&1 &
```
#### For SEER (MLP), 
 ```test
python -u run_dro_cox.py --dataset SEER --model MLP --eps 0.15 --seed 7 > SEER_MLP_joint_dro.log 2>&1 & 
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

## How to run Dro-DeepHit code
### 1. To run Dro-DeepHit on FLC
#### For FLC (MLP), 
 ```test
python -u run_dro_deephit.py --dataset FLC --model MLP --eps 0.3 --seed 7 > FLC_MLP_joint_dro_deephit.log 2>&1 &  
```
### 2. To run Dro-DeepHit on SUPPORT
#### For SUPPORT (MLP), 
 ```test
python -u run_dro_deephit.py --dataset SUPPORT --model MLP --eps 0.5 --seed 7 > SUPPORT_MLP_joint_dro_deephit.log 2>&1 & 
```
### 3. To run Dro-DeepHit on SEER
#### For SEER (MLP), 
 ```test
python -u run_dro_deephit.py --dataset SEER --model MLP --eps 0.15 --seed 7 > SEER_MLP_joint_dro_deephit.log 2>&1 & 
```

## How to run Dro-DeepHit (SPLIT) code
### 1. To run Dro-DeepHit (SPLIT) on FLC
#### For FLC (MLP), 
 ```test
python -u run_dro_deephit_split.py --dataset FLC --model MLP --lr 0.0001 --eps 0.05 --seed 7 > FLC_MLP_joint_dro_deephit_split.log 2>&1 &  
```
### 2. To run Dro-DeepHit (SPLIT) on SUPPORT
#### For SUPPORT (MLP), 
 ```test
python -u run_dro_deephit_split.py --dataset SUPPORT --model MLP --lr 0.0001 --eps 0.2 --seed 7 > SUPPORT_MLP_joint_dro_deephit_split.log 2>&1 & 
```
### 3. To run Dro-DeepHit (SPLIT) on SEER
#### For SEER (MLP), 
 ```test
python -u run_dro_deephit_split.py --dataset SEER --model MLP --lr 0.0001 --eps 0.2 --seed 7 > SEER_MLP_joint_dro_deephit_split.log 2>&1 & 
```

## How to run Dro-SODEN code
Please first read the settings of the original SODEN code and then use our method.

### 1. To run Dro-SODEN on FLC
#### For FLC (MLP), 
 ```test
python -u ./SODEN/main_DRO_COX_ODE.py --dataset flc > FLC_MLP_joint_dro_SODEN.log 2>&1 &  
```
### 2. To run Dro-SODEN on SUPPORT
#### For SUPPORT (MLP), 
 ```test
python -u ./SODEN/main_DRO_COX_ODE.py --dataset support > SUPPORT_MLP_joint_dro_SODEN.log 2>&1 & 
```
### 3. To run Dro-SODEN on SEER
#### For SEER (MLP), 
 ```test
python -u ./SODEN/main_DRO_COX_ODE.py --dataset seer > SEER_MLP_joint_dro_SODEN.log 2>&1 & 
```

## How to run the Exact DRO Cox code

### 1. To run the Exact DRO Cox on FLC
#### For FLC (MLP), 
 ```test
python -u run_dro_cox_full.py --dataset FLC > FLC_MLP_joint_dro_cox_full.log 2>&1 &  
```
### 2. To run the Exact DRO Cox on SUPPORT
#### For SUPPORT (MLP), 
 ```test
python -u run_dro_cox_full.py --dataset support > SUPPORT_MLP_joint_dro_cox_full.log 2>&1 & 
```
### 3. To run the Exact DRO Cox on SEER
#### For SEER (MLP), 
 ```test
python -u run_dro_cox_full.py --dataset seer > SEER_MLP_joint_dro_cox_full.log 2>&1 & 
```




## Citation
Please kindly consider citing our paper in your publications. 
```bash
@inproceedings{hu2022distributionally,
  title={Distributionally robust survival analysis: A novel fairness loss without demographics},
  author={Hu, Shu and Chen, George H},
  booktitle={Machine Learning for Health},
  pages={62--87},
  year={2022},
  organization={PMLR}
}
```
