# Human Pose Regression Baseline
The main goal of this repository is to rewrite the human pose regression with a Tensorflow and better structure for better portability and adaptability to apply new experimental methods. The human pose regression pipeline is based on [Human Pose Regression with Residual Log-likelihood Estimation](https://arxiv.org/abs/2107.11291). <br>

## Performance & Trained model files
| Model | flip test | #Params | GFLOPs | AP | AP50 | AP75 |
| :------------- | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| Benchmark(ResNet50) | O | 23.6 | 4.0G | 0.713 | 0.889 | 0.783 |
| ResNet50 | X | ... | ... | 0.682 | 0.892 | 0.756 |
| ResNet50 | O | ... | ... | 0.695 | 0.903 | 0.769 |
| MobileNetV2 | X | ... | ... | 0.598 | 0.852 | 0.661 |
| MobileNetV2 | O | ... | ... | 0.613 | 0.862 | 0.682 |

## Environment
after downloading `docker` and `nvidia-docker` and clone this repo, just run the below script on the cmd at current dir. <br>
feel free to name the image.
```bash
docker build -t rle:tf .
```

## Usage
### Convert to TFRecord
the dataset folder should be located in the parent of the current dir. if not, you have to change the lots of lines of code.
```python
python write_tfrecords.py
```

### Train / Evaluation
```python
python train.py -c config/256x192_res50_regress-flow.yaml
```

## TO-DO
- [x] hydra 대신 일반 yaml 파일로 학습 파이프라인 수정
- [x] tfrecord 만드는 코드 추가
- [ ] 현재 모델들의 #params, GFLOPs 확인하여 기록
- [ ] pflayer 기반 ResNet 구성


## References
- [Jeff-sjtu/res-loglikelihood-regression](https://github.com/Jeff-sjtu/res-loglikelihood-regression)
