# DC-Net
This is the official repo for our paper: "DC-Net: Divide-and-Conquer for Salient Object Detection"

[Jiayi Zhu][Xuebin Qin][Abdulmotaleb Elsaddik]

__Contact__: zjyzhujiayi55@gmail.com

## Usage
1. Clone this repo.
```
git clone https://github.com/PiggyJerry/DC-Net.git
```
2. Download the pre-trained model from here:

| name | pretrain | backbone | resolution | #params | FPS | download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DC-Net-R | DUTS-TR | ResNet-34 | 352*352 | 356.3MB | 60 | GoogleDrive() |
| DC-Net-S | DUTS-TR | Swin-B | 384*384 | 1495.0MB | 29 | GoogleDrive() |
| DC-Net-R-HR | DIS5K | ResNet-34 | 1024*1024 | 356.3MB | 55 | GoogleDrive() |
3. Unzip `apex.zip` to the directory 'DC-Net'.
4. Train the model.

Cd to the directory 'DC-Net', run the train process by command: ```python main-DC-Net-R.py``` for DC-Net-R or ```python main-DC-Net-S.py``` for DC-Net-S respectively. 

5. Inference the model.

Cd to the directory 'DC-Net', run the inference process by command: ```python Inference-R.py``` for DC-Net-R or ```python Inference-S.py``` for DC-Net-S respectively. 

6. We also provide the predicted saliency maps:

| name | Low-Resolution Datasets | High-Resolution Datasets |
| :---: | :---: | :---: |
| DC-Net-R | GoogleDrive() | GoogleDrive() |
| DC-Net-S | GoogleDrive() | GoogleDrive() |
| DC-Net-R-HR | GoogleDrive() | GoogleDrive() |
