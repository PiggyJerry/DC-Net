import os
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from DC_Net_S import Inference_DC_Net

from einops import rearrange


if __name__ == "__main__":
    dataset_path="./testImgs"  #Your dataset path
    model_path="./saved_models/DC-Net-S.pth"  # the model path
    result_path="./results/testImgs"  #The folder path that you want to save the results
    input_size=[384,384]
    net=Inference_DC_Net()
    if torch.cuda.is_available():
        state_dict=torch.load(model_path)
    else:
        state_dict=torch.load(model_path,map_location="cpu")

    if not os.path.exists(result_path): # create the folder for cache
        os.makedirs(result_path)

    if state_dict is not None:
        test_state_dict=net.state_dict()
        for weight in test_state_dict:
            # print(weight,test_state_dict[weight].shape)
            if weight.split(".")[0][-1]=="d" or weight.split(".")[0][-1]=="B":
                # print(weight)
                if weight.split(".")[1]=="rebn2":
                    if len(test_state_dict[weight].shape)==0:#num_batches_tracked
                        test_state_dict[weight]=state_dict[weight1]==state_dict[weight2]
                    else:
                        test_state_dict[weight]=rearrange(state_dict[weight],"(n g c)->(g n c)",g=4,n=4)
                    
                elif weight in state_dict.keys():
                    
                    test_state_dict[weight]=state_dict[weight]

                elif len(test_state_dict[weight].shape)>0:
                    if weight.split(".")[1][-1]=="2":
                        weight1_1=".".join([weight.split(".")[0],weight.split(".")[1]+"_1_1",*weight.split(".")[2:]])
                        weight1_2=".".join([weight.split(".")[0],weight.split(".")[1]+"_1_2",*weight.split(".")[2:]])
                        weight1_4=".".join([weight.split(".")[0],weight.split(".")[1]+"_1_4",*weight.split(".")[2:]])
                        weight1_8=".".join([weight.split(".")[0],weight.split(".")[1]+"_1_8",*weight.split(".")[2:]])

                        weight2_1=".".join([weight.split(".")[0],weight.split(".")[1]+"_2_1",*weight.split(".")[2:]])
                        weight2_2=".".join([weight.split(".")[0],weight.split(".")[1]+"_2_2",*weight.split(".")[2:]])
                        weight2_4=".".join([weight.split(".")[0],weight.split(".")[1]+"_2_4",*weight.split(".")[2:]])
                        weight2_8=".".join([weight.split(".")[0],weight.split(".")[1]+"_2_8",*weight.split(".")[2:]])

                        weight4_1=".".join([weight.split(".")[0],weight.split(".")[1]+"_4_1",*weight.split(".")[2:]])
                        weight4_2=".".join([weight.split(".")[0],weight.split(".")[1]+"_4_2",*weight.split(".")[2:]])
                        weight4_4=".".join([weight.split(".")[0],weight.split(".")[1]+"_4_4",*weight.split(".")[2:]])
                        weight4_8=".".join([weight.split(".")[0],weight.split(".")[1]+"_4_8",*weight.split(".")[2:]])

                        weight8_1=".".join([weight.split(".")[0],weight.split(".")[1]+"_8_1",*weight.split(".")[2:]])
                        weight8_2=".".join([weight.split(".")[0],weight.split(".")[1]+"_8_2",*weight.split(".")[2:]])
                        weight8_4=".".join([weight.split(".")[0],weight.split(".")[1]+"_8_4",*weight.split(".")[2:]])
                        weight8_8=".".join([weight.split(".")[0],weight.split(".")[1]+"_8_8",*weight.split(".")[2:]])
                        if len(test_state_dict[weight].shape)==4:
                            test_state_dict[weight]=torch.cat([state_dict[weight1_1],state_dict[weight2_1],state_dict[weight4_1],state_dict[weight8_1],
                                                                state_dict[weight1_2],state_dict[weight2_2],state_dict[weight4_2],state_dict[weight8_2],
                                                                state_dict[weight1_4],state_dict[weight2_4],state_dict[weight4_4],state_dict[weight8_4],
                                                                state_dict[weight1_8],state_dict[weight2_8],state_dict[weight4_8],state_dict[weight8_8]],0)
                        elif len(test_state_dict[weight].shape)==1:
                            test_state_dict[weight]=torch.cat([state_dict[weight1_1],state_dict[weight2_1],state_dict[weight4_1],state_dict[weight8_1],
                                                                state_dict[weight1_2],state_dict[weight2_2],state_dict[weight4_2],state_dict[weight8_2],
                                                                state_dict[weight1_4],state_dict[weight2_4],state_dict[weight4_4],state_dict[weight8_4],
                                                                state_dict[weight1_8],state_dict[weight2_8],state_dict[weight4_8],state_dict[weight8_8]],0)                                        
                    else:
                        weight1=".".join([weight.split(".")[0],weight.split(".")[1]+"_1",*weight.split(".")[2:]])
                        weight2=".".join([weight.split(".")[0],weight.split(".")[1]+"_2",*weight.split(".")[2:]])
                        weight4=".".join([weight.split(".")[0],weight.split(".")[1]+"_4",*weight.split(".")[2:]])
                        weight8=".".join([weight.split(".")[0],weight.split(".")[1]+"_8",*weight.split(".")[2:]])
                        if len(test_state_dict[weight].shape)==4:
                            test_state_dict[weight]=torch.cat([state_dict[weight1],state_dict[weight2],state_dict[weight4],state_dict[weight8]],0)
                        elif len(test_state_dict[weight].shape)==1:
                            test_state_dict[weight]=torch.cat([state_dict[weight1],state_dict[weight2],state_dict[weight4],state_dict[weight8]],0)
                elif len(test_state_dict[weight].shape)==0:#num_batches_tracked
                    test_state_dict[weight]=state_dict[weight1]==state_dict[weight2]
            else:
                # print(weight)
                if weight in state_dict.keys():
                    test_state_dict[weight]=state_dict[weight]
                elif "relative_position_index" in weight:
                    test_state_dict[weight]=state_dict[".".join([weight.split(".")[0]+"_1",*weight.split(".")[1:]])]
                elif "attn_mask" in weight:
                    test_state_dict[weight]=state_dict[".".join([weight.split(".")[0]+"_1",*weight.split(".")[1:]])]
                else:
                    weights=[]
                    names=locals()
                    #parallel=2
                    for i in range(1,2+1):
                        names['weight'+str(i)]=".".join([weight.split(".")[0]+"_"+str(i),*weight.split(".")[1:]])
                        weights.append(names['weight'+str(i)])
                    if "relative_position_bias_table" in weight:
                        test_state_dict[weight]=torch.stack([state_dict[_] for _ in weights])
                    elif len(test_state_dict[weight].shape)==4:
                        test_state_dict[weight]=torch.cat([state_dict[_] for _ in weights],0)
                    elif len(test_state_dict[weight].shape)==2:
                        test_state_dict[weight]=torch.block_diag(*[state_dict[_] for _ in weights])
                    elif len(test_state_dict[weight].shape)==1:
                        test_state_dict[weight]=torch.cat([state_dict[_] for _ in weights],0)
                    elif len(test_state_dict[weight].shape)==0:#num_batches_tracked
                        test_state_dict[weight]=state_dict[weights[0]]==state_dict[weights[1]]
        net.load_state_dict(test_state_dict)  
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
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            result=net(image)
            result=torch.squeeze(F.upsample(result[0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            im_name=im_path.split('/')[-1].split('.')[0]
            io.imsave(os.path.join(result_path,im_name+".png"),(result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8))
