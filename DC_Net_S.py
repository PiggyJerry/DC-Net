from readline import insert_text
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import time
from einops import rearrange
import math
from swin_transformer import SwinTransformer
from parallel_swin_transformer import parallel_swin_transformer

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

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1,stride=1,groups=1):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate,stride=stride,groups=groups)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
class REBN(nn.Module):
    def __init__(self,in_ch=3):
        super(REBN,self).__init__()

        self.bn_s1 = nn.BatchNorm2d(in_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(hx))

        return xout

def _upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

def insert_cat(x,groups=1):
    out=rearrange(torch.cat(x,dim=1),"b (n g c) h w -> b (g n c) h w",n=len(x),g=groups)
    return out

class MergeConv2d(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=[1,6,12,18],dilation=[1,6,12,18],stride=1,groups=4,bias=True):
        super(MergeConv2d,self).__init__()

        self.kernel_size=kernel_size if isinstance(kernel_size,tuple) else (kernel_size,kernel_size)
        self.padding=padding
        self.stride=stride
        self.dilation=dilation
        self.groups=groups
        self.out_ch=out_ch
        self.weight = nn.Parameter(torch.Tensor(self.groups*out_ch//self.groups,in_ch//self.groups,self.kernel_size[0],self.kernel_size[1]))
        self.bias_=bias
        if self.bias_ is True:
            self.bias   = nn.Parameter(torch.zeros(out_ch))
        self._initialize()
    def forward(self,x):
        hx = x
        b,c,h,w=hx.shape
        kh,kw=self.kernel_size
        out_padding=((kh-1)//2,(kw-1)//2)
        fea=[]
        hx_g=hx.reshape(b,len(self.dilation),-1,h,w)
        for i in range(len(self.dilation)):
            X_unfold=F.unfold(hx_g[:,i],kernel_size=(kh,kw),padding=self.padding[i],dilation=self.dilation[i],stride=self.stride)
            fea.append(X_unfold)
        input=torch.cat(fea,dim=1).transpose(1,2)
        weight=self.weight.reshape(self.groups,self.out_ch//self.groups,self.weight.shape[1],self.weight.shape[2],self.weight.shape[3]).type(input.type())
        input=input.reshape(input.shape[0],input.shape[1],self.groups,weight.shape[2],weight.shape[3],weight.shape[4])
        conv=torch.einsum('bxgihw,goihw->bxgo',input,weight).flatten(2).transpose(1,2)
        o_h,o_w=(h-kh+out_padding[0]*2)//self.stride+1,(w-kw+out_padding[1]*2)//self.stride+1
        if self.bias_:
            out=F.fold(conv,output_size=(o_h,o_w),kernel_size=(1,1))+self.bias.type(input.type()).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        else:
            out=F.fold(conv,output_size=(o_h,o_w),kernel_size=(1,1))
        return out
    def _initialize(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_ is True:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
#ResASPP2 used in Inference-DC-Net
class ResASPP2_test(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(ResASPP2_test,self).__init__()
        self.dilation=[1,3,5,7]
        self.padding=[1,3,5,7]

        self.residual = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconvin = REBNCONV(out_ch,mid_ch,dirate=1)
        self.scale1 = MergeConv2d(mid_ch*4,mid_ch,padding=self.padding,dilation=self.dilation,groups=4)
        self.rebn1=REBN(mid_ch)

        self.scale2 = MergeConv2d(mid_ch*4,mid_ch,padding=self.padding,dilation=self.dilation,groups=16)
        self.rebn2=REBN(mid_ch)

        self.scale1d = MergeConv2d(mid_ch*2,mid_ch,padding=self.padding,dilation=self.dilation,groups=4)
        self.rebn1d=REBN(mid_ch)

        self.rebnconvout = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        residual=self.residual(hx)
        hxin=self.rebnconvin(residual)
        scale1=self.rebn1(self.scale1(hxin.repeat(1,4,1,1)))
        scale2=self.rebn2(self.scale2(scale1.repeat(1,4,1,1)))
        scale1d=self.rebn1d(self.scale1d(insert_cat([scale1,rearrange(scale2,"b (n g c) h w -> b (g n c) h w",g=4,n=4)],groups=4)))
        hxout=self.rebnconvout(torch.cat([hxin,scale1d],1))

        return hxout+residual
#ResASPP2 used in Training-DC-Net
class ResASPP2_train(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(ResASPP2_train,self).__init__()
        self.dilation=[1,3,5,7]
        self.padding=[1,3,5,7]
        self.residual = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconvin = REBNCONV(out_ch,mid_ch,dirate=1)
        self.scale1_1 = nn.Conv2d(mid_ch,mid_ch//4,3,padding=self.padding[0],dilation=self.dilation[0])
        self.scale1_2 = nn.Conv2d(mid_ch,mid_ch//4,3,padding=self.padding[1],dilation=self.dilation[1])
        self.scale1_4 = nn.Conv2d(mid_ch,mid_ch//4,3,padding=self.padding[2],dilation=self.dilation[2])
        self.scale1_8 = nn.Conv2d(mid_ch,mid_ch//4,3,padding=self.padding[3],dilation=self.dilation[3])
        self.rebn1=REBN(mid_ch)

        self.scale2_1_1 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[0],dilation=self.dilation[0])
        self.scale2_1_2 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[1],dilation=self.dilation[1])
        self.scale2_1_4 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[2],dilation=self.dilation[2])
        self.scale2_1_8 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[3],dilation=self.dilation[3])

        self.scale2_2_1 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[0],dilation=self.dilation[0])
        self.scale2_2_2 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[1],dilation=self.dilation[1])
        self.scale2_2_4 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[2],dilation=self.dilation[2])
        self.scale2_2_8 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[3],dilation=self.dilation[3])

        self.scale2_4_1 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[0],dilation=self.dilation[0])
        self.scale2_4_2 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[1],dilation=self.dilation[1])
        self.scale2_4_4 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[2],dilation=self.dilation[2])
        self.scale2_4_8 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[3],dilation=self.dilation[3])

        self.scale2_8_1 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[0],dilation=self.dilation[0])
        self.scale2_8_2 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[1],dilation=self.dilation[1])
        self.scale2_8_4 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[2],dilation=self.dilation[2])
        self.scale2_8_8 = nn.Conv2d(mid_ch//4,mid_ch//16,3,padding=self.padding[3],dilation=self.dilation[3])
        self.rebn2=REBN(mid_ch)

        self.scale1d_1 = nn.Conv2d(mid_ch//2,mid_ch//4,3,padding=self.padding[0],dilation=self.dilation[0])
        self.scale1d_2 = nn.Conv2d(mid_ch//2,mid_ch//4,3,padding=self.padding[1],dilation=self.dilation[1])
        self.scale1d_4 = nn.Conv2d(mid_ch//2,mid_ch//4,3,padding=self.padding[2],dilation=self.dilation[2])
        self.scale1d_8 = nn.Conv2d(mid_ch//2,mid_ch//4,3,padding=self.padding[3],dilation=self.dilation[3])
        self.rebn1d=REBN(mid_ch)

        self.rebnconvout = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x
        residual=self.residual(hx)
        hxin=self.rebnconvin(residual)
        scale1_1=self.scale1_1(hxin)
        scale1_2=self.scale1_2(hxin)
        scale1_4=self.scale1_4(hxin)
        scale1_8=self.scale1_8(hxin)
        scale1=self.rebn1(torch.cat([scale1_1,scale1_2,scale1_4,scale1_8],1))

        scale1_1,scale1_2,scale1_4,scale1_8=scale1.chunk(4,1)
        scale2_1_1=self.scale2_1_1(scale1_1)
        scale2_1_2=self.scale2_1_2(scale1_1)
        scale2_1_4=self.scale2_1_4(scale1_1)
        scale2_1_8=self.scale2_1_8(scale1_1)

        scale2_2_1=self.scale2_2_1(scale1_2)
        scale2_2_2=self.scale2_2_2(scale1_2)
        scale2_2_4=self.scale2_2_4(scale1_2)
        scale2_2_8=self.scale2_2_8(scale1_2)

        scale2_4_1=self.scale2_4_1(scale1_4)
        scale2_4_2=self.scale2_4_2(scale1_4)
        scale2_4_4=self.scale2_4_4(scale1_4)
        scale2_4_8=self.scale2_4_8(scale1_4)

        scale2_8_1=self.scale2_8_1(scale1_8)
        scale2_8_2=self.scale2_8_2(scale1_8)
        scale2_8_4=self.scale2_8_4(scale1_8)
        scale2_8_8=self.scale2_8_8(scale1_8)
        scale2=self.rebn2(torch.cat([scale2_1_1,scale2_1_2,scale2_1_4,scale2_1_8,
                                    scale2_2_1,scale2_2_2,scale2_2_4,scale2_2_8,
                                    scale2_4_1,scale2_4_2,scale2_4_4,scale2_4_8,
                                    scale2_8_1,scale2_8_2,scale2_8_4,scale2_8_8],1))

        scale2_1,scale2_2,scale2_4,scale2_8=scale2.chunk(4,1)  

        scale1d_1=self.scale1d_1(torch.cat([scale1_1,scale2_1],1))
        scale1d_2=self.scale1d_2(torch.cat([scale1_2,scale2_2],1))
        scale1d_4=self.scale1d_4(torch.cat([scale1_4,scale2_4],1))
        scale1d_8=self.scale1d_8(torch.cat([scale1_8,scale2_8],1))                    
        scale1d=self.rebn1d(torch.cat([scale1d_1,scale1d_2,scale1d_4,scale1d_8],1))
        hxout=self.rebnconvout(torch.cat([hxin,scale1d],1))

        return hxout+residual

class Training_DC_Net(nn.Module):

    def __init__(self,in_ch=3,out_ch=1):
        super(Training_DC_Net,self).__init__()
        ## -------------Encoder--------------
        self.encoder_1=SwinTransformer(img_size=384, 
                                    embed_dim=128,
                                    depths=[2,2,18,2],
                                    num_heads=[4,8,16,32],
                                    window_size=12)

        pretrained_dict = torch.load('./checkpoint/swin_base_patch4_window12_384_22k.pth')["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder_1.state_dict()}
        self.encoder_1.load_state_dict(pretrained_dict)

        self.encoder_2=SwinTransformer(img_size=384, 
                                    embed_dim=128,
                                    depths=[2,2,18,2],
                                    num_heads=[4,8,16,32],
                                    window_size=12)

        pretrained_dict = torch.load('./checkpoint/swin_base_patch4_window12_384_22k.pth')["model"]
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

    def __init__(self,in_ch=3,out_ch=1):
        super(Inference_DC_Net,self).__init__()

        self.encoder=parallel_swin_transformer(img_size=384, 
                                    embed_dim=128,
                                    depths=[2,2,18,2],
                                    num_heads=[4,8,16,32],
                                    window_size=12,
                                    parallel=2,
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

        hxin,hx1,hx2,hx3,hx4=self.encoder(hx.repeat(1,2,1,1))
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
