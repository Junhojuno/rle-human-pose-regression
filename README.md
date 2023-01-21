# Human Pose Regression Baseline
The main goal of this repository is to rewrite the human pose regression with a Tensorflow and better structure for better portability and adaptability to apply new experimental methods. The human pose regression pipeline is based on ['Human Pose Regression with Residual Log-likelihood Estimation'](https://arxiv.org/abs/2107.11291). <br>

## Performance & Trained model files
| Model | # Params | FLOPs | AP | AP50 | AP75 | link |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
| ResNet-50(w/o flip) | ... | ... | 0.682 | 0.892 | 0.756 | [model]() |
| ResNet-50(w/ flip) | ... | ... | 0.695 | 0.903 | 0.769 | [model]() |
| MobileNetV2(w/o flip) | ... | ... | 0.598 | 0.852 | 0.661 | [model]() |
| MobileNetV2(w/ flip) | ... | ... | 0.613 | 0.862 | 0.682 | [model]() |

## ✔️ Set environment
- conda env / docker
- Tensorflow 2.9+
- Tensorflow-Probability 0.17.0+

## ✔️ Convert data to tfrecord
- download MSCOCO
- convert to tfrecord

## ✔️ Train / Evaluation
```python
sh train.sh
```

## ✔️ Advanced
- Apply Automatic Mixed-Precision Training
- 


## :thinking: TO-DO
  - [x] set of stable training with hydra / wandb
  - [x] add MobileNetV2
  - [ ] add HRNet-W32/W48
  - [x] demo webcam / video
  - [ ] export modules for web and mobile 
  - [x] add evaluation module
  - [ ] 

## ✔️ References

