# DC-Net
This is the official repo for our paper: "DC-Net: Divide-and-Conquer for Salient Object Detection".

Authors: Jiayi Zhu, Xuebin Qin and Abdulmotaleb Elsaddik

__Contact__: zjyzhujiayi55@gmail.com

## Usage
1. Clone this repo.
```
git clone https://github.com/PiggyJerry/DC-Net.git
```

2. Download the pre-trained model and put the model file to the directory `DC-Net/saved_models`:

| name | pretrain | backbone | resolution | #params | FPS | download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DC-Net-R | DUTS-TR | ResNet-34 | 352*352 | 356.3MB | 60 | [GoogleDrive](https://drive.google.com/file/d/17-yqt_aEorTpKOEzMgobpBIjHZbrRbov/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/1WqXHf_GmQcJ_6V8S0xHYyA?pwd=1sq1) |
| DC-Net-S | DUTS-TR | Swin-B | 384*384 | 1495.0MB | 29 | [GoogleDrive](https://drive.google.com/file/d/1HNeIH-pmwaf7V6RaAPOu6Gda4dR7CjNL/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/1SiUvnxBBzHNaEDhPtkJGUw?pwd=p9he) |
| DC-Net-R-HR | DIS5K | ResNet-34 | 1024*1024 | 356.3MB | 55 | [GoogleDrive](https://drive.google.com/file/d/1At4I-TXSOZOrOth4PrNF_oUAo3Yz5z8f/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/1wiYcozN5JFdicooGr_lnAQ?pwd=ur1k) |

3. Download checkpoint from [GoogleDrive](https://drive.google.com/file/d/1xvdXwN27a4YjOemWBtgxexnyK2_sS_cK/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/167L7qRDpWDyx41WWcUjfHQ?pwd=hhgu) and put it to the directory `DC-Net/checkpoint`.
   
4. Unzip `apex.zip` to the directory 'DC-Net'.
   
5. Train the model.

First, download the datasets to the directory `DC-Net/datasets`, then cd to the directory 'DC-Net', run the train process by command: ```python main-DC-Net-R.py``` for DC-Net-R or ```python main-DC-Net-S.py``` for DC-Net-S respectively. 

6. Inference the model.

First, put test images to the directory `DC-Net/testImgs`, then cd to the directory 'DC-Net', run the inference process by command: ```python Inference-R.py``` for DC-Net-R or ```python Inference-S.py``` for DC-Net-S respectively. 

## Predicted saliency maps

For DC-Net-R and DC-Net-S we provide the predicted saliency maps for low-resolution datasets DUTS-TE, DUT-OMRON, HKU-IS, ECSSD and PASCAL-S.

For DC-Net-R-HR we also provide the predicted saliency maps for high-resolution datasets DIS-TE, ThinObject5K, UHRSD, HRSOD and DAVIS-S.

| name | predicted saliency maps |
| :---: | :---: |
| DC-Net-R | [GoogleDrive](https://drive.google.com/file/d/1nUvXLkUovfutIRxTsKQ2csuGpRUMhYxv/view?usp=share_link)/[Baidu Pan](https://pan.baidu.com/s/1yrbikTVf_uLdcHsbMKTdMg?pwd=i4r9) |
| DC-Net-S | [GoogleDrive](https://drive.google.com/file/d/1CoCNZzNC7g4EymLQlZ0vHcoh8qONdRob/view?usp=share_link)/[Baidu Pan](https://pan.baidu.com/s/1FHq6iEiuBBwpHeI0LfxEvA?pwd=3gld) |
| DC-Net-R-HR | [GoogleDrive](https://drive.google.com/file/d/1Io_aKlke9UdB2xv8PJyvEINjzWZjCJlw/view?usp=share_link)/[Baidu Pan](https://pan.baidu.com/s/12sl7yWcN1Zdvdr5LyoT2CQ?pwd=ww0a) |

## How to modify the edge width of the edge map?
You just need to modify the 330 line of `data_loader_cache.py`, where the last hyperparameter $thickness$ of `cv2.drawContours` means the bilateral edge pixel, after processing by line 332, the bilateral edge pixel becomes inter unilateral edge pixel $edge\ width$, which is what we want. $edge\ width$=($thickness$+1)/2.

## How to use Parallel-ResNet and Parallel-Swin-Transformer?
Same as the original ResNet and Swin-Transformer, you just need to modify the new hyperparameter `parallel` to how many encoders you want. 

## Citation
```
@article{zhu2023dc,
  title={DC-Net: Divide-and-Conquer for Salient Object Detection},
  author={Zhu, Jiayi and Qin, Xuebin and Elsaddik, Abdulmotaleb},
  journal={arXiv preprint arXiv:2305.14955},
  year={2023}
}
```
