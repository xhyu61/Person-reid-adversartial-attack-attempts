# Person-reid-adversartial-attack-attempts
![Static Badge](https://img.shields.io/badge/Lang-Python-blue)
![Static Badge](https://img.shields.io/badge/Status-InProgress-red)

Harbin Engineering University Graduation Project


Framework:
[Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)

Add the files from this repository to the framework, and download the dataset Market, place in the same folder as framework project.

Use `prepare.py` to preprocess the dataset.

Your folder will be the form like:

```text
yourpath/Person_reID_baseline_pytorch
yourpath/Market/pytorch
```

## get model
### 1.ft_ResNet50
use the command from the original work
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32
```

## test model
In `adv.py`, comment out line 295-297, uncomment line 299, use the command below to test the model's performance
```bash
python adv.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch "last"
```
Change model name after `--name` to yours.

## use attack method
### 1.FGSM
In `adv.py`, comment out line 299, uncomment line 295-297, use the command below to test the model's performance
```bash
python adv.py --gpu_ids 0 --name ft_ResNet50 --batchsize 32 --which_epoch "last" --use_FGSM
```
Change model name after `--name` to yours.
