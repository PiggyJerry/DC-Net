import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

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

def insert_cat(x,groups=1):
    out=rearrange(torch.cat(x,dim=1),"b (n g c) h w -> b (g n c) h w",n=len(x),g=groups)
    return out

class MergedConv2d(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,padding=[1,6,12,18],dilation=[1,6,12,18],stride=1,groups=4,bias=True):
        super(MergedConv2d,self).__init__()

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
class ResASPP2_test(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(ResASPP2_test,self).__init__()
        self.dilation=[1,3,5,7]
        self.padding=[1,3,5,7]

        self.residual = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconvin = REBNCONV(out_ch,mid_ch,dirate=1)
        self.scale1 = MergedConv2d(mid_ch*4,mid_ch,padding=self.padding,dilation=self.dilation,groups=4)
        self.rebn1=REBN(mid_ch)

        self.scale2 = MergedConv2d(mid_ch*4,mid_ch,padding=self.padding,dilation=self.dilation,groups=16)
        self.rebn2=REBN(mid_ch)

        self.scale1d = MergedConv2d(mid_ch*2,mid_ch,padding=self.padding,dilation=self.dilation,groups=4)
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
class ResASPP2_train(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
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

#please modify this function according to your model code
def reparametrize(state_dict,pretrained_state_dict):
    for weight in state_dict:
        # for stage1d, stage2d, stage3d, stage4d, stageB in DC_Net
        if weight.split(".")[0][-1]=="d" or weight.split(".")[0][-1]=="B":
            if weight.split(".")[1]=="rebn2":
                if len(state_dict[weight].shape)==0:#num_batches_tracked
                    state_dict[weight]=pretrained_state_dict[weight1]==pretrained_state_dict[weight2]
                else:
                    state_dict[weight]=rearrange(pretrained_state_dict[weight],"(n g c)->(g n c)",g=4,n=4)
                
            elif weight in pretrained_state_dict.keys():
                
                state_dict[weight]=pretrained_state_dict[weight]

            elif len(state_dict[weight].shape)>0:
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
                    if len(state_dict[weight].shape)==4:
                        state_dict[weight]=torch.cat([pretrained_state_dict[weight1_1],pretrained_state_dict[weight2_1],pretrained_state_dict[weight4_1],pretrained_state_dict[weight8_1],
                                                            pretrained_state_dict[weight1_2],pretrained_state_dict[weight2_2],pretrained_state_dict[weight4_2],pretrained_state_dict[weight8_2],
                                                            pretrained_state_dict[weight1_4],pretrained_state_dict[weight2_4],pretrained_state_dict[weight4_4],pretrained_state_dict[weight8_4],
                                                            pretrained_state_dict[weight1_8],pretrained_state_dict[weight2_8],pretrained_state_dict[weight4_8],pretrained_state_dict[weight8_8]],0)
                    elif len(state_dict[weight].shape)==1:
                        state_dict[weight]=torch.cat([pretrained_state_dict[weight1_1],pretrained_state_dict[weight2_1],pretrained_state_dict[weight4_1],pretrained_state_dict[weight8_1],
                                                            pretrained_state_dict[weight1_2],pretrained_state_dict[weight2_2],pretrained_state_dict[weight4_2],pretrained_state_dict[weight8_2],
                                                            pretrained_state_dict[weight1_4],pretrained_state_dict[weight2_4],pretrained_state_dict[weight4_4],pretrained_state_dict[weight8_4],
                                                            pretrained_state_dict[weight1_8],pretrained_state_dict[weight2_8],pretrained_state_dict[weight4_8],pretrained_state_dict[weight8_8]],0)                                        
                else:
                    weight1=".".join([weight.split(".")[0],weight.split(".")[1]+"_1",*weight.split(".")[2:]])
                    weight2=".".join([weight.split(".")[0],weight.split(".")[1]+"_2",*weight.split(".")[2:]])
                    weight4=".".join([weight.split(".")[0],weight.split(".")[1]+"_4",*weight.split(".")[2:]])
                    weight8=".".join([weight.split(".")[0],weight.split(".")[1]+"_8",*weight.split(".")[2:]])
                    if len(state_dict[weight].shape)==4:
                        state_dict[weight]=torch.cat([pretrained_state_dict[weight1],pretrained_state_dict[weight2],pretrained_state_dict[weight4],pretrained_state_dict[weight8]],0)
                    elif len(state_dict[weight].shape)==1:
                        state_dict[weight]=torch.cat([pretrained_state_dict[weight1],pretrained_state_dict[weight2],pretrained_state_dict[weight4],pretrained_state_dict[weight8]],0)
            elif len(state_dict[weight].shape)==0:#num_batches_tracked
                state_dict[weight]=pretrained_state_dict[weight1]==pretrained_state_dict[weight2]
    return state_dict