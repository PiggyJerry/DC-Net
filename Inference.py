import os
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from DC_Net import Inference_DC_Net

from einops import rearrange
from reparameterize import Reparameterize

if __name__ == "__main__":
    type='R'
    dataset_path="./datasets/DUTS-TE/im"  #Your dataset path
    model_path="./saved_models/"+"DC-Net-"+type+".pth"  # the model path
    result_path="./results/"+"DC-Net-"+type+"/DUTS-TE"  #The folder path that you want to save the results
    if type=='R':
        input_size=[352,352]
    elif type=='S':
        input_size=[384,384]
    net=Inference_DC_Net(parallel=2,type=type)
    if torch.cuda.is_available():
        state_dict=torch.load(model_path)
    else:
        state_dict=torch.load(model_path,map_location="cpu")

    if not os.path.exists(result_path): # create the folder for cache
        os.makedirs(result_path)

    if state_dict is not None:
        net=Reparameterize(net,state_dict,type=type)
        if torch.cuda.is_available():
            net=net.cuda()
    net.eval()

    im_list = glob(dataset_path+"/*.jpg")+glob(dataset_path+"/*.JPG")+glob(dataset_path+"/*.jpeg")+glob(dataset_path+"/*.JPEG")+glob(dataset_path+"/*.png")+glob(dataset_path+"/*.PNG")+glob(dataset_path+"/*.bmp")+glob(dataset_path+"/*.BMP")+glob(dataset_path+"/*.tiff")+glob(dataset_path+"/*.TIFF")
    with torch.no_grad():
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            print("im_path: ", im_path)
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(torch.squeeze(F.upsample(result[0],im_shp,mode='bilinear'),0),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            im_name=im_path.split('/')[-1].split('.')[0]
            io.imsave(os.path.join(result_path,im_name+".png"),(result*255).cpu().data.numpy().astype(np.uint8))
