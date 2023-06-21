# DC-Net
This is the official repo for our paper: "DC-Net: Divide-and-Conquer for Salient Object Detection"

Authors: Jiayi Zhu, Xuebin Qin, Abdulmotaleb Elsaddik

__Contact__: zjyzhujiayi55@gmail.com

## Usage
1. Clone this repo.
```
git clone https://github.com/PiggyJerry/DC-Net.git
```

2. Download the pre-trained model and put the model file to the directory `DC-Net/saved_models`:

| name | pretrain | backbone | resolution | #params | FPS | download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DC-Net-R | DUTS-TR | ResNet-34 | 352*352 | 356.3MB | 60 | [GoogleDrive](https://drive.google.com/file/d/17-yqt_aEorTpKOEzMgobpBIjHZbrRbov/view?usp=sharing) |
| DC-Net-S | DUTS-TR | Swin-B | 384*384 | 1495.0MB | 29 | [GoogleDrive](https://drive.google.com/file/d/1HNeIH-pmwaf7V6RaAPOu6Gda4dR7CjNL/view?usp=sharing) |
| DC-Net-R-HR | DIS5K | ResNet-34 | 1024*1024 | 356.3MB | 55 | [GoogleDrive](https://drive.google.com/file/d/1At4I-TXSOZOrOth4PrNF_oUAo3Yz5z8f/view?usp=sharing) |

3. Download checkpoint from [GoogleDrive](https://drive.google.com/file/d/1xvdXwN27a4YjOemWBtgxexnyK2_sS_cK/view?usp=sharing) and put it to the directory `DC-Net/checkpoint`.
   
4. Unzip `apex.zip` to the directory 'DC-Net'.
   
5. Train the model.

First, download the datasets to the directory `DC-Net/datasets`, then cd to the directory 'DC-Net', run the train process by command: ```python main-DC-Net-R.py``` for DC-Net-R or ```python main-DC-Net-S.py``` for DC-Net-S respectively. 

6. Inference the model.

First, put test images to the directory `DC-Net/testImgs`, then cd to the directory 'DC-Net', run the inference process by command: ```python Inference-R.py``` for DC-Net-R or ```python Inference-S.py``` for DC-Net-S respectively. 

7. We also provide the predicted saliency maps:

| name | Low-Resolution Datasets | High-Resolution Datasets |
| :---: | :---: | :---: |
| DC-Net-R | GoogleDrive() | GoogleDrive() |
| DC-Net-S | GoogleDrive() | GoogleDrive() |
| DC-Net-R-HR | GoogleDrive() | GoogleDrive() |
