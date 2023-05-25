# data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
import json
from tqdm import tqdm
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
import cv2
from skimage import morphology

def _upsample_like(src,size):

    src = F.upsample(src,size=size,mode='bilinear')

    return src
#### --------------------- GOS dataloader cache ---------------------####

### collect im and gt name lists
def get_im_gt_name_dict(datasets, flag='train'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []
    #其实长度就只有1
    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list, tmp_edge_list, tmp_main_list = [], [], [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])

        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            tmp_edge_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
    
            tmp_body_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            tmp_detail_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))

        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "edge_path":tmp_edge_list,
                                "body_path":tmp_body_list,
                                "detail_path":tmp_detail_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"],
                                "cache_dir":datasets[i]["cache_dir"]})

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, cache_size=[], cache_boost=True, my_transforms=[], batch_size=1, shuffle=False, collate=False):


    gos_dataloaders = []
    gos_datasets = []
    # if(mode=="train"):
    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8

    for i in range(0,len(name_im_gt_list)):
        gos_dataset = GOSDatasetCache([name_im_gt_list[i]],
                                      cache_size = cache_size,
                                      cache_path = name_im_gt_list[i]["cache_dir"],
                                      cache_boost = cache_boost,
                                      transform = transforms.Compose(my_transforms))
        if collate==True:
            gos_dataloaders.append(DataLoader(gos_dataset, collate_fn=gos_dataset.collate,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers_))
        else:
            gos_dataloaders.append(DataLoader(gos_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers_))
        gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

def im_reader(im_path):
    return io.imread(im_path)

def im_preprocess(im,size):

    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
    im_tensor = torch.transpose(torch.transpose(im_tensor,1,2),0,1)
    if(len(size)<2):
        return im_tensor, im.shape[0:2]
    else:
        im_tensor = torch.unsqueeze(im_tensor,0)
        im_tensor = F.upsample(im_tensor, size, mode="bilinear")
        im_tensor = torch.squeeze(im_tensor,0)

    return im_tensor.type(torch.uint8), im.shape[0:2]

def gt_preprocess(gt,size):

    if len(gt.shape) > 2:
        gt = gt[:, :, 0]
  
    
    gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.uint8),0)

    if(len(size)<2):
        return gt_tensor.type(torch.uint8), gt.shape[0:2]
    else:
        gt_tensor = torch.unsqueeze(torch.tensor(gt_tensor, dtype=torch.float32),0)
        gt_tensor = F.upsample(gt_tensor, size, mode="bilinear")
        gt_tensor = torch.squeeze(gt_tensor,0)

    return gt_tensor.type(torch.uint8), gt.shape[0:2]
    # return gt_tensor, gt.shape[0:2]

class GOSRandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, edge, body, detail, shape =  sample['imidx'], sample['image'], sample['label'],sample['edge'],sample['body'], sample['detail'],sample['shape']
        
        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])
            edge = torch.flip(edge,dims=[2])
            body = torch.flip(body,dims=[2])
            detail = torch.flip(detail,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label,'edge':edge, 'body':body,'detail':detail,'shape':shape}

class GOSResize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, edge, body, detail, shape =  sample['imidx'], sample['image'], sample['label'],sample['edge'],sample['body'], sample['detail'],sample['shape']

        # import time
        # start = time.time()

        image = torch.squeeze(F.upsample(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.upsample(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)
        edge = torch.squeeze(F.upsample(torch.unsqueeze(edge,0),self.size,mode='bilinear'),dim=0)
        body = torch.squeeze(F.upsample(torch.unsqueeze(body,0),self.size,mode='bilinear'),dim=0)
        detail = torch.squeeze(F.upsample(torch.unsqueeze(detail,0),self.size,mode='bilinear'),dim=0)

        return {'imidx':imidx,'image':image, 'label':label,'edge':edge, 'body':body,'detail':detail,'shape':shape}
class GOSRandomCrop(object):
    def __call__(self, sample):
        imidx, image, label, edge,body, detail, shape =  sample['imidx'], sample['image'], sample['label'],sample['edge'],sample['body'], sample['detail'],sample['shape']
        H,W  = image.shape[1:]
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if label is None:
            return image[:,p0:p1,p2:p3]
        return {'imidx':imidx,'image':image[:,p0:p1,p2:p3], 'label':label[:,p0:p1,p2:p3],'edge':edge[:,p0:p1,p2:p3], 'body':body[:,p0:p1,p2:p3],'detail':detail[:,p0:p1,p2:p3],'shape':shape}


class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, edge, body, detail, shape =  sample['imidx'], sample['image'], sample['label'],sample['edge'], sample['body'], sample['detail'],sample['shape']

        image = normalize(image,self.mean,self.std)
    
        return {'imidx':imidx,'image':image, 'label':label,'edge':edge,'body':body,'detail':detail,'shape':shape}


class GOSDatasetCache(Dataset):

    def __init__(self, name_im_gt_list, cache_size=[], cache_path='./cache', cache_file_name='dataset.json', cache_boost=False, transform=None):


        self.cache_size = cache_size
        self.cache_path = cache_path
        self.cache_file_name = cache_file_name
        self.cache_boost_name = ""

        self.cache_boost = cache_boost
        # self.ims_npy = None
        # self.gts_npy = None

        ## cache all the images and ground truth into a single pytorch tensor
        self.ims_pt = None
        self.gts_pt = None
        self.edge_pt = None

        self.body_pt = None
        self.detail_pt = None

        ## we will cache the npy as well regardless of the cache_boost
        # if(self.cache_boost):
        self.cache_boost_name = cache_file_name.split('.json')[0]

        self.transform = transform

        self.dataset = {}

        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        edge_path_list = [] # im path

        body_path_list = [] # gt path
        detail_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            edge_path_list.extend(name_im_gt_list[i]["edge_path"])

            body_path_list.extend(name_im_gt_list[i]["body_path"])
            detail_path_list.extend(name_im_gt_list[i]["detail_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_shp"] = []
        self.dataset["gt_shp"] = []
        self.dataset["edge_path"] = edge_path_list
        self.dataset["edge_shp"] = []
        self.dataset["body_path"] = body_path_list
        self.dataset["body_shp"] = []
        self.dataset["detail_path"] = detail_path_list
        self.dataset["detail_shp"] = []
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        # self.dataset["ims_npy_dir"] = ""
        # self.dataset["gts_npy_dir"] = ""

        self.dataset["ims_pt_dir"] = ""
        self.dataset["gts_pt_dir"] = ""
        self.dataset["edge_pt_dir"] = ""
        self.dataset["body_pt_dir"] = ""
        self.dataset["detail_pt_dir"] = ""

        # self.dataset["ims_npy"] = None
        # self.dataset["gts_npy"] = None

        self.dataset = self.manage_cache(dataset_names)

    def manage_cache(self,dataset_names):
        if not os.path.exists(self.cache_path): # create the folder for cache
            os.mkdir(self.cache_path)
        cache_folder = os.path.join(self.cache_path, "_".join(dataset_names)+"_"+"x".join([str(x) for x in self.cache_size]))
        if not os.path.exists(cache_folder): # check if the cache files are there, if not then cache
            return self.cache(cache_folder)
        return self.load_cache(cache_folder)

    def cache(self,cache_folder):
        os.mkdir(cache_folder)
        cached_dataset = deepcopy(self.dataset)

        # ims_list = []
        # gts_list = []
        ims_pt_list = []
        gts_pt_list = []
        edge_pt_list = []

        body_pt_list = []
        detail_pt_list = []

        for i, im_path in tqdm(enumerate(self.dataset["im_path"]), total=len(self.dataset["im_path"])):


            im_id = cached_dataset["im_name"][i]

            im = im_reader(im_path)
            im, im_shp = im_preprocess(im,self.cache_size)
            im_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_im.pt")
            torch.save(im,im_cache_file)

            cached_dataset["im_path"][i] = im_cache_file
            if(self.cache_boost):
                ims_pt_list.append(torch.unsqueeze(im,0))
            # ims_list.append(im.cpu().data.numpy().astype(np.uint8))

            gt = im_reader(self.dataset["gt_path"][i])
            gt, gt_shp = gt_preprocess(gt,self.cache_size)

            gt_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_gt.pt")
            torch.save(gt,gt_cache_file)
            cached_dataset["gt_path"][i] = gt_cache_file
            if(self.cache_boost):
                gts_pt_list.append(torch.unsqueeze(gt,0))


            
            binary_mask=np.array(gt.squeeze(0),dtype='uint8')
            # calculate the edge map with width 4, where (7+1)/2=4
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            instance_contour = np.zeros(binary_mask.shape)
            edge=cv2.drawContours(instance_contour, contours, -1, 255, 7)
            edge=np.array(edge,dtype='uint8')
            edge[binary_mask==0]=0
            # calculate body and bedtail maps
            instance_contour_ = np.zeros(binary_mask.shape)
            edge_=cv2.drawContours(instance_contour_, contours, -1, 255, 1)
            edge_=np.array(edge_,dtype='uint8')
            dis = cv2.distanceTransform(src=edge_.max()-edge_, distanceType=cv2.DIST_L2, maskSize=0)
            signed_dis=dis.copy()
            ma=signed_dis.max()
            mi=signed_dis.min()
            signed_dis=((signed_dis-mi)/(ma-mi))
            signed_dis[binary_mask==0]=0
            body=((signed_dis-signed_dis.min())/(signed_dis.max()-signed_dis.min()))*255.0
            
            detail=binary_mask-body
            detail=((detail-detail.min())/(detail.max()-detail.min()))*255.0

            edge_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_edge.pt")
            edge=torch.tensor(edge,dtype=torch.uint8).unsqueeze(0)
            torch.save(edge,edge_cache_file)
            cached_dataset["edge_path"][i] = edge_cache_file
            if(self.cache_boost):
                edge_pt_list.append(torch.unsqueeze(edge,0))
            # gts_list.append(gt.cpu().data.numpy().astype(np.uint8))


            body_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_body.pt")
            body=torch.tensor(body,dtype=torch.uint8).unsqueeze(0)
            torch.save(body,body_cache_file)
            cached_dataset["body_path"][i] = body_cache_file
            if(self.cache_boost):
                body_pt_list.append(torch.unsqueeze(body,0))
            
            detail_cache_file = os.path.join(cache_folder,self.dataset["data_name"][i]+"_"+im_id + "_detail.pt")
            detail=torch.tensor(detail,dtype=torch.uint8).unsqueeze(0)
            torch.save(detail,detail_cache_file)
            cached_dataset["detail_path"][i] = detail_cache_file
            if(self.cache_boost):
                detail_pt_list.append(torch.unsqueeze(detail,0))

            # im_shp_cache_file = os.path.join(cache_folder,im_id + "_im_shp.pt")
            # torch.save(gt_shp, shp_cache_file)
            cached_dataset["im_shp"].append(im_shp)
            # self.dataset["im_shp"].append(im_shp)

            # shp_cache_file = os.path.join(cache_folder,im_id + "_gt_shp.pt")
            # torch.save(gt_shp, shp_cache_file)
            cached_dataset["gt_shp"].append(gt_shp)
            # self.dataset["gt_shp"].append(gt_shp)

        if(self.cache_boost):
            cached_dataset["ims_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_ims.pt')
            cached_dataset["gts_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_gts.pt')
            cached_dataset["edge_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_edge.pt')
        
            cached_dataset["body_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_body.pt')
            cached_dataset["detail_pt_dir"] = os.path.join(cache_folder, self.cache_boost_name+'_detail.pt')

            self.ims_pt = torch.cat(ims_pt_list,dim=0)
            self.gts_pt = torch.cat(gts_pt_list,dim=0)
            self.edge_pt = torch.cat(edge_pt_list,dim=0)
            self.body_pt = torch.cat(body_pt_list,dim=0)
            self.detail_pt = torch.cat(detail_pt_list,dim=0)
      
            torch.save(torch.cat(ims_pt_list,dim=0),cached_dataset["ims_pt_dir"])
            torch.save(torch.cat(gts_pt_list,dim=0),cached_dataset["gts_pt_dir"])
            torch.save(torch.cat(edge_pt_list,dim=0),cached_dataset["edge_pt_dir"])
            torch.save(torch.cat(body_pt_list,dim=0),cached_dataset["body_pt_dir"])
            torch.save(torch.cat(detail_pt_list,dim=0),cached_dataset["detail_pt_dir"])


        try:
            json_file = open(os.path.join(cache_folder, self.cache_file_name),"w")
            json.dump(cached_dataset, json_file)
            json_file.close()
        except Exception:
            raise FileNotFoundError("Cannot create JSON")
        return cached_dataset

    def load_cache(self, cache_folder):
        json_file = open(os.path.join(cache_folder,self.cache_file_name),"r")
        dataset = json.load(json_file)
        json_file.close()
        ## if cache_boost is true, we will load the image npy and ground truth npy into the RAM
        ## otherwise the pytorch tensor will be loaded
        if(self.cache_boost):
            # self.ims_npy = np.load(dataset["ims_npy_dir"])
            # self.gts_npy = np.load(dataset["gts_npy_dir"])
            self.ims_pt = torch.load(dataset["ims_pt_dir"], map_location='cpu')
            self.gts_pt = torch.load(dataset["gts_pt_dir"], map_location='cpu')
            self.edge_pt = torch.load(dataset["edge_pt_dir"], map_location='cpu')

            self.body_pt = torch.load(dataset["body_pt_dir"], map_location='cpu')
            self.detail_pt = torch.load(dataset["detail_pt_dir"], map_location='cpu')

        return dataset

    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):

        im = None
        gt = None
        edge = None
        body = None
        detail = None

        if(self.cache_boost and self.ims_pt is not None):


            # start = time.time()
            im = self.ims_pt[idx]#.type(torch.float32)
            gt = self.gts_pt[idx]#.type(torch.float32)
            edge = self.edge_pt[idx]
            body = self.body_pt[idx]
            detail = self.detail_pt[idx]

            # print(idx, 'time for pt loading: ', time.time()-start)

        else:
            # import time
            # start = time.time()
            # print("tensor***")
            im_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["im_path"][idx].split(os.sep)[-2:]))
            im = torch.load(im_pt_path)#(self.dataset["im_path"][idx])
            gt_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["gt_path"][idx].split(os.sep)[-2:]))
            gt = torch.load(gt_pt_path)#(self.dataset["gt_path"][idx])
            edge_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["edge_path"][idx].split(os.sep)[-2:]))
            edge = torch.load(edge_pt_path)#(self.dataset["gt_path"][idx])
            body_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["body_path"][idx].split(os.sep)[-2:]))
            body = torch.load(body_pt_path)#(self.dataset["gt_path"][idx])
            detail_pt_path = os.path.join(self.cache_path,os.sep.join(self.dataset["detail_path"][idx].split(os.sep)[-2:]))
            detail = torch.load(detail_pt_path)#(self.dataset["gt_path"][idx])
         


        im_shp = self.dataset["im_shp"][idx]

        im = torch.divide(im,255.0)
        gt = torch.divide(gt,255.0)
        if len(edge.shape)<3:
            edge = torch.divide(edge,255.0).unsqueeze(0)
            body = torch.divide(body,255.0).unsqueeze(0)
            detail = torch.divide(detail,255.0).unsqueeze(0)
        else:
            edge = torch.divide(edge,255.0)
            body = torch.divide(body,255.0)
            detail = torch.divide(detail,255.0)
        
        
        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "label": gt,
        "edge": edge,
        "body": body,
        "detail": detail,
        "shape": torch.from_numpy(np.array(im_shp)),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]

        imidx=[]
        image=[]
        label=[]
        edge=[]
        body=[]
        detail=[]
        shape=[]
        for item in batch:
            imidx.append(item["imidx"])
            image.append(_upsample_like(item["image"].unsqueeze(0), size=(size, size)))
            label.append(_upsample_like(item["label"].unsqueeze(0), size=(size, size)))
            edge.append(_upsample_like(item["edge"].unsqueeze(0), size=(size, size)))
            body.append(_upsample_like(item["body"].unsqueeze(0), size=(size, size)))
            detail.append(_upsample_like(item["detail"].unsqueeze(0), size=(size, size)))
            shape.append(item["shape"])
            
        imidx   = torch.stack(imidx, axis=0)
        image  = torch.stack(image, axis=0).squeeze(1)
        label   = torch.stack(label, axis=0).squeeze(1)
        edge   = torch.stack(edge, axis=0).squeeze(1)
        body   = torch.stack(body, axis=0).squeeze(1)
        detail = torch.stack(detail, axis=0).squeeze(1)
        shape = torch.stack(shape, axis=0)

        sample = {
        "imidx": imidx,
        "image": image,
        "label": label,
        "edge": edge,
        "body": body,
        "detail": detail,
        "shape": shape,
        }
        return sample

