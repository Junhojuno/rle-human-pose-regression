# Human Pose Regression Baseline
The main goal of this repository is to rewrite the human pose regression with a Tensorflow and better structure for better portability and adaptability to apply new experimental methods. The human pose regression pipeline is based on [Human Pose Regression with Residual Log-likelihood Estimation](https://arxiv.org/abs/2107.11291). <br>

## Performance & Trained model files
| Model | flip test | # Params | FLOPs | AP | AP50 | AP75 |
| :------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| ResNet-50 | X | ... | ... | 0.682 | 0.892 | 0.756 |
| ResNet-50 | O | ... | ... | 0.695 | 0.903 | 0.769 |
| MobileNetV2 | X | ... | ... | 0.598 | 0.852 | 0.661 |
| MobileNetV2 | O | ... | ... | 0.613 | 0.862 | 0.682 |

## Environment
- docker
- Tensorflow 2.9+
- Tensorflow-Probability 0.17.0+
- hydra
- wandb

## Usage
### Convert to TFRecord
```python
python write_tfrecords.py
```

### Train / Evaluation
```python
sh train.sh
```

## TO-DO
- [ ] hydra 대신 일반 yaml 파일로 학습 파이프라인 수정
- [ ] tfrecord 만드는 코드 추가


## References
- [Jeff-sjtu/res-loglikelihood-regression](https://github.com/Jeff-sjtu/res-loglikelihood-regression)
