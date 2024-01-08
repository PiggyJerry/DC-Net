
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from einops import rearrange
import parallel_resnet
from swin_transformer import SwinTransformer
from parallel_swin_transformer import parallel_swin_transformer
from ResASPP2 import ResASPP2_train, ResASPP2_test

import matplotlib.pyplot as plt

def _iou(pred, target, size_average = True):
    smooth=1e-6
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = (Iand1+smooth)/(Ior1+smooth)
        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

bce_loss = nn.BCELoss(size_average=True)
iou_loss = IOU(size_average=True)
def multi_loss_fusion(preds, target):
    loss0 = 0.0
    loss = 0.0
    #saliency
    for i in range(0,5):
        if(preds[i].shape[2]!=target[0].shape[2] or preds[i].shape[3]!=target[0].shape[3]):
            tmp_target = F.interpolate(target[0], size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)+ iou_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target[0])+ iou_loss(preds[i],target[0])
        if(i==0):
            loss0 = loss
    #edge4
    for i in range(5,9):
        if(preds[i].shape[2]!=target[1].shape[2] or preds[i].shape[3]!=target[1].shape[3]):
            tmp_target = F.interpolate(target[1], size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target[1])
    #location
    for i in range(9,13):
        if(preds[i].shape[2]!=target[2].shape[2] or preds[i].shape[3]!=target[2].shape[3]):
            tmp_target = F.interpolate(target[2], size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target[2])
    return loss0, loss

def test_loss(preds, target):
    loss = 0.0
    #saliency
    for i in range(0,1):
        if(preds[i].shape[2]!=target[0].shape[2] or preds[i].shape[3]!=target[0].shape[3]):
            tmp_target = F.interpolate(target[0], size=preds[i].size()[2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds[i],tmp_target)+ iou_loss(preds[i],tmp_target)
        else:
            loss = loss + bce_loss(preds[i],target[0])+ iou_loss(preds[i],target[0])
    return loss

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class Training_DC_Net(nn.Module):

    def __init__(self,in_ch=3,out_ch=1,type='R'):
        super(Training_DC_Net,self).__init__()
        self.type=type
        if type=='R':
            resnet1 = models.resnet34(pretrained=True)
            resnet2 = models.resnet34(pretrained=True)
            ## -------------Encoder--------------
            self.conv_in_1 = nn.Conv2d(in_ch,64,3,stride=2,padding=1)
            self.conv_in_2 = nn.Conv2d(in_ch,64,3,stride=2,padding=1)

            #stage 1
            self.stage1_1 = resnet1.layer1
            self.stage1_2 = resnet2.layer1

            #stage 2
            self.stage2_1 = resnet1.layer2
            self.stage2_2 = resnet2.layer2

            #stage 3
            self.stage3_1 = resnet1.layer3
            self.stage3_2 = resnet2.layer3

            #stage 4
            self.stage4_1 = resnet1.layer4
            self.stage4_2 = resnet2.layer4

            self.pool4_1 = nn.MaxPool2d(2,2,ceil_mode=True)
            self.pool4_2 = nn.MaxPool2d(2,2,ceil_mode=True)

            # decoder
            self.stageB = ResASPP2_train(1024,512,1024)

            self.stage4d = ResASPP2_train(2048,256,512)

            self.stage3d = ResASPP2_train(1024,128,256)

            self.stage2d = ResASPP2_train(512,64,128)

            self.stage1d = ResASPP2_train(256,32,64)

            self.side1d = nn.Conv2d(64,out_ch,3,padding=1)
            self.side2d = nn.Conv2d(128,out_ch,3,padding=1)
            self.side3d = nn.Conv2d(256,out_ch,3,padding=1)
            self.side4d = nn.Conv2d(512,out_ch,3,padding=1)
            self.sideB = nn.Conv2d(1024,out_ch,3,padding=1)

            self.side1_1 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side2_1 = nn.Conv2d(128,out_ch,3,padding=1)
            self.side3_1 = nn.Conv2d(256,out_ch,3,padding=1)
            self.side4_1 = nn.Conv2d(512,out_ch,3,padding=1)

            self.side1_2 = nn.Conv2d(64,in_ch,3,padding=1)
            self.side2_2 = nn.Conv2d(128,in_ch,3,padding=1)
            self.side3_2 = nn.Conv2d(256,in_ch,3,padding=1)
            self.side4_2 = nn.Conv2d(512,in_ch,3,padding=1)

        elif type=='S':
            self.encoder_1=SwinTransformer(img_size=384, 
                                        embed_dim=128,
                                        depths=[2,2,18,2],
                                        num_heads=[4,8,16,32],
                                        window_size=12)

            pretrained_dict = torch.load('/home/jiayi/DC-Net/checkpoint/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder_1.state_dict()}
            self.encoder_1.load_state_dict(pretrained_dict)

            self.encoder_2=SwinTransformer(img_size=384, 
                                        embed_dim=128,
                                        depths=[2,2,18,2],
                                        num_heads=[4,8,16,32],
                                        window_size=12)

            pretrained_dict = torch.load('/home/jiayi/DC-Net/checkpoint/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder_2.state_dict()}
            self.encoder_2.load_state_dict(pretrained_dict)

            self.pool4_1 = nn.MaxPool2d(2,2,ceil_mode=True)
            self.pool4_2 = nn.MaxPool2d(2,2,ceil_mode=True)

            # decoder
            self.stageB = ResASPP2_train(2048,1024,2048)

            self.stage4d = ResASPP2_train(4096,512,1024)

            self.stage3d = ResASPP2_train(2048,256,512)

            self.stage2d = ResASPP2_train(1024,128,256)

            self.stage1d = ResASPP2_train(512,64,128)

            self.side1d = nn.Conv2d(128,out_ch,3,padding=1)
            self.side2d = nn.Conv2d(256,out_ch,3,padding=1)
            self.side3d = nn.Conv2d(512,out_ch,3,padding=1)
            self.side4d = nn.Conv2d(1024,out_ch,3,padding=1)
            self.sideB = nn.Conv2d(2048,out_ch,3,padding=1)

            self.side1_1 = nn.Conv2d(128,out_ch,3,padding=1)
            self.side2_1 = nn.Conv2d(256,out_ch,3,padding=1)
            self.side3_1 = nn.Conv2d(512,out_ch,3,padding=1)
            self.side4_1 = nn.Conv2d(1024,out_ch,3,padding=1)

            self.side1_2 = nn.Conv2d(128,in_ch,3,padding=1)
            self.side2_2 = nn.Conv2d(256,in_ch,3,padding=1)
            self.side3_2 = nn.Conv2d(512,in_ch,3,padding=1)
            self.side4_2 = nn.Conv2d(1024,in_ch,3,padding=1)

    def compute_loss(self, preds, targets):

        return multi_loss_fusion(preds,targets)

    def compute_test_loss(self, preds, targets):

        return test_loss(preds,targets)

    def forward(self,x):

        hx = x
        if self.type=='R':
            hxin1 = self.conv_in_1(hx)
            hxin2 = self.conv_in_2(hx)
        
            #stage 1
            hx1_1 = self.stage1_1(hxin1)
            hx1_2 = self.stage1_2(hxin2)

            #stage 2
            hx2_1 = self.stage2_1(hx1_1)
            hx2_2 = self.stage2_2(hx1_2)

            #stage 3
            hx3_1 = self.stage3_1(hx2_1)
            hx3_2 = self.stage3_2(hx2_2)

            # #stage 4
            hx4_1 = self.stage4_1(hx3_1)
            hx4_2 = self.stage4_2(hx3_2)

        elif self.type=='S':
            hxin1,hx1_1,hx2_1,hx3_1,hx4_1=self.encoder_1(hx)
            hxin2,hx1_2,hx2_2,hx3_2,hx4_2=self.encoder_2(hx)

        hx1 = self.pool4_1(hx4_1)
        hx2 = self.pool4_2(hx4_2)
        
        # #Bridge
        hxB= self.stageB(torch.cat([hx1,hx2],1))
        hxBup = _upsample_like(hxB,hx4_1)

        hx4d = self.stage4d(torch.cat([hxBup,hx4_1,hx4_2],1))
        hx4dup = _upsample_like(hx4d,hx3_1)

        hx3d = self.stage3d(torch.cat([hx4dup,hx3_1,hx3_2],1))
        hx3dup = _upsample_like(hx3d,hx2_1)

        hx2d = self.stage2d(torch.cat([hx3dup,hx2_1,hx2_2],1))
        hx2dup = _upsample_like(hx2d,hx1_1)

        hx1d = self.stage1d(torch.cat([hx2dup,hx1_1,hx1_2],1))

        #side output
        d1d = self.side1d(hx1d)
        d1d = _upsample_like(d1d,x)

        d2d = self.side2d(hx2d)
        d2d = _upsample_like(d2d,x)

        d3d = self.side3d(hx3d)
        d3d = _upsample_like(d3d,x)

        d4d = self.side4d(hx4d)
        d4d = _upsample_like(d4d,x)

        dB = self.sideB(hxB)
        dB = _upsample_like(dB,x)

        d1_1 = self.side1_1(hx1_1)
        d1_1 = _upsample_like(d1_1,x)

        d2_1 = self.side2_1(hx2_1)
        d2_1 = _upsample_like(d2_1,x)

        d3_1 = self.side3_1(hx3_1)
        d3_1 = _upsample_like(d3_1,x)

        d4_1 = self.side4_1(hx4_1)
        d4_1 = _upsample_like(d4_1,x)

        d1_2 = self.side1_2(hx1_2)
        d1_2 = _upsample_like(d1_2,x)

        d2_2 = self.side2_2(hx2_2)
        d2_2 = _upsample_like(d2_2,x)

        d3_2 = self.side3_2(hx3_2)
        d3_2 = _upsample_like(d3_2,x)

        d4_2 = self.side4_2(hx4_2)
        d4_2 = _upsample_like(d4_2,x)

        return [torch.sigmoid(d1d), torch.sigmoid(d2d), torch.sigmoid(d3d), torch.sigmoid(d4d), torch.sigmoid(dB),
        torch.sigmoid(d1_1), torch.sigmoid(d2_1), torch.sigmoid(d3_1), torch.sigmoid(d4_1),
        torch.sigmoid(d1_2), torch.sigmoid(d2_2), torch.sigmoid(d3_2), torch.sigmoid(d4_2)]

class Inference_DC_Net(nn.Module):

    def __init__(self,in_ch=3,out_ch=1,parallel=2,type='R'):
        super(Inference_DC_Net,self).__init__()
        self.parallel=parallel
        self.type=type
        if type=='R':
            resnet = parallel_resnet.parallel_resnet34(pretrained=False,parallel=parallel)

            ## -------------Encoder--------------
            self.conv_in = nn.Conv2d(in_ch*2,64*2,3,stride=2,padding=1,groups=parallel)

            #stage 1
            self.stage1 = resnet.layer1 #224

            #stage 2
            self.stage2 = resnet.layer2 #112

            #stage 3
            self.stage3 = resnet.layer3 #56

            #stage 4
            self.stage4 = resnet.layer4 #28

            self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

            # decoder
            self.stageB = ResASPP2_test(1024,512,1024)

            self.stage4d = ResASPP2_test(2048,256,512)

            self.stage3d = ResASPP2_test(1024,128,256)

            self.stage2d = ResASPP2_test(512,64,128)

            self.stage1d = ResASPP2_test(256,32,64)

            self.side1d = nn.Conv2d(64,out_ch,3,padding=1)
            self.side2d = nn.Conv2d(128,out_ch,3,padding=1)
            self.side3d = nn.Conv2d(256,out_ch,3,padding=1)
            self.side4d = nn.Conv2d(512,out_ch,3,padding=1)
            self.sideB = nn.Conv2d(1024,out_ch,3,padding=1)

            self.side1_1 = nn.Conv2d(64,out_ch,3,padding=1)
            self.side2_1 = nn.Conv2d(128,out_ch,3,padding=1)
            self.side3_1 = nn.Conv2d(256,out_ch,3,padding=1)
            self.side4_1 = nn.Conv2d(512,out_ch,3,padding=1)

            self.side1_2 = nn.Conv2d(64,in_ch,3,padding=1)
            self.side2_2 = nn.Conv2d(128,in_ch,3,padding=1)
            self.side3_2 = nn.Conv2d(256,in_ch,3,padding=1)
            self.side4_2 = nn.Conv2d(512,in_ch,3,padding=1)

        elif type=='S':
            self.encoder=parallel_swin_transformer(img_size=384, 
                                        embed_dim=128,
                                        depths=[2,2,18,2],
                                        num_heads=[4,8,16,32],
                                        window_size=12,
                                        parallel=parallel,
                                        pretrained=False)
            self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

            # decoder
            self.stageB = ResASPP2_test(2048,1024,2048)

            self.stage4d = ResASPP2_test(4096,512,1024)

            self.stage3d = ResASPP2_test(2048,256,512)

            self.stage2d = ResASPP2_test(1024,128,256)

            self.stage1d = ResASPP2_test(512,64,128)

            self.side1d = nn.Conv2d(128,out_ch,3,padding=1)
            self.side2d = nn.Conv2d(256,out_ch,3,padding=1)
            self.side3d = nn.Conv2d(512,out_ch,3,padding=1)
            self.side4d = nn.Conv2d(1024,out_ch,3,padding=1)
            self.sideB = nn.Conv2d(2048,out_ch,3,padding=1)

            self.side1_1 = nn.Conv2d(128,out_ch,3,padding=1)
            self.side2_1 = nn.Conv2d(256,out_ch,3,padding=1)
            self.side3_1 = nn.Conv2d(512,out_ch,3,padding=1)
            self.side4_1 = nn.Conv2d(1024,out_ch,3,padding=1)

            self.side1_2 = nn.Conv2d(128,in_ch,3,padding=1)
            self.side2_2 = nn.Conv2d(256,in_ch,3,padding=1)
            self.side3_2 = nn.Conv2d(512,in_ch,3,padding=1)
            self.side4_2 = nn.Conv2d(1024,in_ch,3,padding=1)


    def compute_loss(self, preds, targets):

        return multi_loss_fusion(preds,targets)

    def compute_test_loss(self, preds, targets):

        return test_loss(preds,targets)

    def forward(self,x):

        hx = x
        if self.type=='R':
            hxin = self.conv_in(hx.repeat(1,self.parallel,1,1))

            #stage 1
            hx1 = self.stage1(hxin)
            
            #stage 2
            hx2 = self.stage2(hx1)

            #stage 3
            hx3 = self.stage3(hx2)

            # #stage 4
            hx4 = self.stage4(hx3)

        elif self.type=='S':
            hxin,hx1,hx2,hx3,hx4=self.encoder(hx.repeat(1,self.parallel,1,1))

        hx = self.pool4(hx4)
        
        # #Bridge
        hxB= self.stageB(hx)
        hxBup = _upsample_like(hxB,hx4)

        hx4d = self.stage4d(torch.cat([hxBup,hx4],1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat([hx4dup,hx3],1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat([hx3dup,hx2],1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat([hx2dup,hx1],1))

        #side output
        d1d = self.side1d(hx1d)
        d1d = _upsample_like(d1d,x)

        return [torch.sigmoid(d1d)]
