# Efficient and Separate Authentication Image Steganography Network

This repo contains the codebase of **《Efficient and Separate Authentication Image Steganography Network》**, Forty-second International Conference on Machine Learning (2025) Spotlight Poster.

## Environment

Run ```pip install -r requirements.txt```

## Dataset

Prepare the public dataset through:

**ImageNet:** [ILSVRC2012](https://image-net.org/)

**DIV2K:** [High Resolution Images](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

You can also use your own private dataset.

## Train

Modify **config.py**, replace all the paths with the paths in your environment.

Run ```python train.py```

## Test

Run ```python test.py ```

## Checkpoint

For quick test, you can download the checkpoint from the following links:

[Baidu Netdisk](https://pan.baidu.com/s/1DXqTYEZYzfpX2F2HulD66A?pwd=0916)

[Google Drive](https://drive.google.com/drive/folders/1P7eQ3W7qbmL6kSmSG6ezNVvrmshd1TkK?usp=drive_link)

The file name represents the settings of dataset and number of secret images. For example, **model_DIV2K_3.pt** means hiding 3 secret images, trained and tested on DIV2K dataset.

## Citation
If you use this code in your research, please kindly cite the following papers

```bash
@inproceedings{zhou2025ais,
    title={Efficient and Separate Authentication Image Steganography Network},
    author={Zhou, Junchao and Lu Yao and Pei Wenjie and Lu Guangming},
    booktitle={International Conference on Machine Learning},
    year={2025}
}
```
