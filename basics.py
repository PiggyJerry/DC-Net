import os

from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

import time

def mae_torch(pred,gt):

	h,w = gt.shape[0:2]
	sumError = torch.sum(torch.absolute(torch.sub(pred.float(), gt.float())))
	maeError = torch.divide(sumError,float(h)*float(w)*255.0+1e-4)

	return maeError

def f1score_torch(pd,gt):

	gtNum = torch.sum((gt>128).float()*1) ## number of ground truth pixels

	pp = pd[gt>128]
	nn = pd[gt<=128]
	#返回直方图
	pp_hist =torch.histc(pp,bins=255,min=0,max=255)
	nn_hist = torch.histc(nn,bins=255,min=0,max=255)

	#上下方向翻转tensor,因为histc返回的是一个1-D向量，因此这里是将向量前后翻转，即intensity越大的越前
	pp_hist_flip = torch.flipud(pp_hist)
	nn_hist_flip = torch.flipud(nn_hist)

	#cumsum就是累积求和，返回的tensor大小和原tensor大小一样
	pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
	nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

	precision = (pp_hist_flip_cum)/(pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)#torch.divide(pp_hist_flip_cum,torch.sum(torch.sum(pp_hist_flip_cum, nn_hist_flip_cum), 1e-4))
	recall = (pp_hist_flip_cum)/(gtNum + 1e-4)
	
	f1 = (1+0.3)*precision*recall/(0.3*precision+recall + 1e-4)

	return torch.reshape(precision,(1,precision.shape[0])),torch.reshape(recall,(1,recall.shape[0])),torch.reshape(f1,(1,f1.shape[0]))


def f1_mae_torch(pred, gt, valid_dataset, idx, mybins, hypar):
	import time
	tic = time.time()

	if(len(gt.shape)>2):
		gt = gt[:,:,0]
	
	pre, rec, f1 = f1score_torch(pred,gt)
	mae = mae_torch(pred,gt)

	#add below 2 lines to generate binary image
	# pred[pred<128]=0
	# pred[pred>=128]=255
	# hypar["valid_out_dir"] = hypar["valid_out_dir"]+"-eval" ###

	# save predicted saliency maps
	if(hypar["valid_out_dir"]!=""):
		if(not os.path.exists(hypar["valid_out_dir"])):
			os.mkdir(hypar["valid_out_dir"])
		dataset_folder = os.path.join(hypar["valid_out_dir"],valid_dataset.dataset["data_name"][idx])
		if(not os.path.exists(dataset_folder)):
			os.mkdir(dataset_folder)
		io.imsave(os.path.join(dataset_folder,valid_dataset.dataset["im_name"][idx]+".png"),pred.cpu().data.numpy().astype(np.uint8))
	print(valid_dataset.dataset["im_name"][idx]+".png")
	print("time for evaluation : ", time.time()-tic)

	return pre.cpu().data.numpy(), rec.cpu().data.numpy(), f1.cpu().data.numpy(), mae.cpu().data.numpy()

def compute_IoU(d,lbl_name_list,imidx_val,d_dir,mybins):
    predict = d
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    i_test = imidx_val.data.numpy()[0]
    gt = io.imread(lbl_name_list[i_test[0]])
    if len(gt.shape)>2:
        gt = gt[:,:,0]

    imo = im.resize((gt.shape[1],gt.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)#np.resize(predict_np*255,(image.shape[0],image.shape[1]))

    pb_np = (pb_np[:,:,0]+1e-8)/(np.amax(pb_np[:,:,0])+1e-8)
    gt = (gt+1e-8)/(np.amax(gt)+1e-8)

    pb_bw = pb_np > 0.5
    gt_bw = gt > 0.5

    pb_and_gt = np.logical_and(pb_bw,gt_bw)
    numerator = np.sum(pb_and_gt.astype(np.float))+1e-8
    demoninator = np.sum(pb_bw.astype(np.float))+np.sum(gt_bw.astype(np.float))-numerator+1e-8

    return numerator/demoninator

