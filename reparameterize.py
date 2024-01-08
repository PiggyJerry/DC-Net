import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from einops import rearrange

def Reparameterize(net,state_dict,type='R'):
    if type=='R':
        test_state_dict=net.state_dict()
        for weight in test_state_dict:
            # print(weight,test_state_dict[weight].shape)
            if weight.split(".")[0][-1]=="d" or weight.split(".")[0][-1]=="B":
                if weight.split(".")[1]=="rebn2":
                    if len(test_state_dict[weight].shape)==0:#num_batches_tracked
                        test_state_dict[weight]=state_dict[weight1]
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
                    test_state_dict[weight]=state_dict[weight1]
            else:
                if weight in state_dict.keys():
                    test_state_dict[weight]=state_dict[weight]
                else:
                    weights=[]
                    names=locals()
                    #parallel=2
                    for i in range(1,2+1):
                        names['weight'+str(i)]=".".join([weight.split(".")[0]+"_"+str(i),*weight.split(".")[1:]])
                        weights.append(names['weight'+str(i)])
                    if len(test_state_dict[weight].shape)>0:
                        test_state_dict[weight]=torch.cat([state_dict[_] for _ in weights],0)
                    elif len(test_state_dict[weight].shape)==0:#num_batches_tracked
                        test_state_dict[weight]=state_dict[weights[0]]
        net.load_state_dict(test_state_dict)  
    elif type=='S':
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
                    test_state_dict[weight]=state_dict[weight1]
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
    return net