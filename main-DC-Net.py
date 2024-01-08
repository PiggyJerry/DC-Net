import os
import time
import numpy as np
from skimage import io
import time

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from reparameterize import Reparameterize
import sys
sys.path.insert(0, '.')
from data_loader_cache import get_im_gt_name_dict, GOSDatasetCache, create_dataloaders, GOSRandomHFlip, GOSResize, GOSRandomCrop, GOSNormalize
from basics import f1score_torch, f1_mae_torch #normPRED, GOSPRF1ScoresCache,
from DC_Net import Training_DC_Net, Inference_DC_Net
import logging
import sys
from apex import amp

def train(net_train,state_dict,net_test, optimizer, train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar): #model_path, model_save_fre, max_ite=1000000):

    model_path = hypar["model_path"]
    model_save_fre = hypar["model_save_fre"]
    max_ite = hypar["max_ite"]
    batch_size_train = hypar["batch_size_train"]
    batch_size_valid = hypar["batch_size_valid"]

    if(not os.path.exists(model_path)):
        os.mkdir(model_path)

    ite_num = hypar["start_ite"] # count the toal iteration number
    ite_num4val = 0 #
    running_loss = 0.0 # count the toal loss
    running_tar_loss = 0.0 # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets[0].__len__()
    if(hypar["restore_model"]!=""):
        net_train.load_state_dict(state_dict)

    net_train.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders[0]
    epoch_num = hypar["max_epoch_num"]
    notgood_cnt = 0
    for epoch in range(epoch_num): ## set the epoch num as 100000
        # optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(epoch_num+1)*2-1))*hypar['lr']

        for i, data in enumerate(gos_dataloader):

            if(ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, labels, edge= data['image'], data['label'], data['edge']

            if(hypar["model_digit"]=="full"):
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                edge = edge.type(torch.FloatTensor)

            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)
                edge = edge.type(torch.HalfTensor)


            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v, edge_v= Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False),\
                Variable(edge.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v, edge_v= Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False),\
                Variable(edge, requires_grad=False)
            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad() ##########
           
            ds = net_train(inputs_v)

            ma = torch.max(inputs_v)
            mi = torch.min(inputs_v)
            inp = (inputs_v-mi)/(ma-mi) # max = 1
            loss2, loss = net_train.compute_loss(ds, [labels_v,edge_v,inp*labels_v])
            ## "loss2" is computed based on the final output of our model, "loss" is the sum of all the outputs including side outputs for dense supervision
 
            if hypar['use_amp']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()#########
            optimizer.step()#########
          
            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            del ds, loss2, loss, scaled_loss
            end_inf_loss_back = time.time()-start_inf_loss_back

            print(">>>"+model_path.split('/')[-1]+" - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f, best_f1:%3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, time.time()-start_last, time.time()-start_last-end_inf_loss_back,last_f1[-1]))
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                # net.eval()
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid(net_test,net_train.state_dict(), valid_dataloaders, valid_datasets, hypar, epoch)
                logging.info(
                'iteration %d : f1 : %f mae : %f' % (ite_num, tmp_f1[-1], tmp_mae[-1]))
                # net.train()  # resume train

                tmp_out = 0
                print("last_f1:",last_f1)
                print("tmp_f1:",tmp_f1)
                for fi in range(len(last_f1)):
                    if(tmp_f1[fi]>last_f1[fi]):
                        tmp_out = 1
                print("tmp_out:",tmp_out)
                if(tmp_out):
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x,4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx,4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/gpu_itr_"+str(ite_num)+\
                                "_traLoss_"+str(np.round(running_loss / ite_num4val,4))+\
                                "_traTarLoss_"+str(np.round(running_tar_loss / ite_num4val,4))+\
                                "_valLoss_"+str(np.round(val_loss /(i_val+1),4))+\
                                "_valTarLoss_"+str(np.round(tar_loss /(i_val+1),4)) + \
                                "_maxF1_" + maxf1 + \
                                "_mae_" + meanM + \
                                "_time_" + str(np.round(np.mean(np.array(tmp_time))/batch_size_valid,6))+".pth"
                    torch.save(net_train.state_dict(), model_path + model_name)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if(notgood_cnt >= hypar["early_stop"]):
                    print("No improvements in the last "+str(notgood_cnt)+" validation periods, so training stopped !")
                    exit()

    print("Training Reaches The Maximum Epoch Number")
@torch.no_grad()
def valid(net_test,state_dict, valid_dataloaders, valid_datasets, hypar, epoch=0):
    if state_dict is not None:
        net_test=Reparameterize(net_test,state_dict,type=hypar['type'])
    net_test.eval()
    print("Validating...")
    epoch_num = hypar["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()
    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders[k]
        valid_dataset = valid_datasets[k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0,256)
        PRE = np.zeros((val_num,len(mybins)-1))
        REC = np.zeros((val_num,len(mybins)-1))
        F1 = np.zeros((val_num,len(mybins)-1))
        MAE = np.zeros((val_num))


        for i_val, data_val in enumerate(valid_dataloader):
            val_cnt = val_cnt + 1.0

            imidx_val, inputs_val, labels_val,edge_val, shapes_val = data_val['imidx'], data_val['image'], data_val['label'],data_val['edge'], data_val['shape']

            if(hypar["model_digit"]=="full"):
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
                edge_val = edge_val.type(torch.FloatTensor)

            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)
                edge_val = edge_val.aype(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_val_v, labels_val_v, edge_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(labels_val.cuda(), requires_grad=False),\
                Variable(edge_val.cuda(), requires_grad=False)
            else:
                inputs_val_v, labels_val_v, edge_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val, requires_grad=False),\
                Variable(edge_val, requires_grad=False)

            t_start = time.time()
            ds_val = net_test(inputs_val_v)

            t_end = time.time()-t_start
            tmp_time.append(t_end)

            loss_val = net_test.compute_test_loss(ds_val, [labels_val_v])

            # compute F measure
            for t in range(hypar["batch_size_valid"]):

                i_test = imidx_val[t].data.numpy()

                pred_val = ds_val[0][t,:,:,:] # B x 1 x H x W

                ## recover the prediction spatial size to the orignal image size

                pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[t][0],shapes_val[t][1]),mode='bilinear'))

                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val-mi)/(ma-mi) # max = 1
  
                gt = np.squeeze(io.imread(valid_dataset.dataset["ori_gt_path"][i_test])) # max = 255
                with torch.no_grad():
                    gt = torch.tensor(gt).cuda()

                pre,rec,f1,mae = f1_mae_torch(pred_val*255, gt, valid_dataset, i_test, mybins, hypar)

                PRE[i_test,:]=pre
                REC[i_test,:] = rec
                F1[i_test,:] = f1
                MAE[i_test] = mae

                del ds_val, gt
                gc.collect()
                torch.cuda.empty_cache()

            tar_loss += loss_val.item()
           
            print("[validating: %5d/%5d] tar_ls: %f, f1: %f, mae: %f, time: %f"% (i_val, val_num, tar_loss / (i_val + 1), np.amax(F1[i_test,:]), MAE[i_test],t_end))

            del loss_val

        print('============================')
        PRE_m = np.mean(PRE,0)
        REC_m = np.mean(REC,0)
        f1_m = (1+0.3)*PRE_m*REC_m/(0.3*PRE_m+REC_m+1e-8)
        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))
        print("The max F1 Score: %f"%(np.max(f1_m)))
        print("The MAE Score: %f"%(np.mean(MAE)))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time

def main(train_datasets,
         valid_datasets,
         hypar): # model: "train", "test"

    ### --- Step 1: Build datasets and dataloaders ---

    if(hypar["mode"]=="train"):
        print("--- create training dataloader ---")
        ## collect training dataset
        train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        ## build dataloader for training datasets
        train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
                                                             cache_size = hypar["cache_size"],
                                                             cache_boost = hypar["cache_boost_train"],
                                                             my_transforms = [
                                                                             GOSRandomHFlip(),
                                                                             # GOSResize(hypar["input_size"]),
                                                                             GOSRandomCrop(),
                                                                              GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                              ],
                                                             batch_size = hypar["batch_size_train"],
                                                             shuffle = True,
                                                             collate=True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    ## build dataloader for validation or testing
    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    ## build dataloader for training datasets
    valid_dataloaders, valid_datasets = create_dataloaders(valid_nm_im_gt_list,
                                                          cache_size = hypar["cache_size"],
                                                          cache_boost = hypar["cache_boost_valid"],
                                                          my_transforms = [
                                                                           GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0]),
                                                                           # GOSResize(hypar["input_size"])
                                                                           ],
                                                          batch_size=hypar["batch_size_valid"],
                                                          shuffle=False,
                                                          collate=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    # print(valid_datasets[0]["data_name"])

    ### --- Step 2: Build Model and Optimizer ---
    print("--- build model ---")
    net_train = hypar["model_train"]#GOSNETINC(3,1)
    net_test = hypar["model_test"]
    pytorch_total_params = sum(p.numel() for p in  net_train.parameters())
    print('Number of parameters: {0}'.format(pytorch_total_params))
    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net_train.half()
        net_test.half()
        for layer in net_train.modules():
          if isinstance(layer, nn.BatchNorm2d):
            layer.float()
        for layer in net_test.modules():
          if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    if torch.cuda.is_available():
        net_train.cuda()
        net_test.cuda()

    if(hypar["restore_model"]!=""):
        print("restore model from:")
        print(hypar["model_path"]+"/"+hypar["restore_model"])
        if torch.cuda.is_available():
            state_dict=torch.load(hypar["restore_model"])
        else:
            state_dict=torch.load(hypar["restore_model"],map_location="cpu")
    else:
        state_dict=None

    print("--- define optimizer ---")
    optimizer = optim.SGD(net_train.parameters(), lr=hypar['lr'], momentum=hypar['momentum'], weight_decay=hypar['weight_decay'])
    net_train, optimizer = amp.initialize(net_train, optimizer, opt_level='O1') 
    ### --- Step 3: Train or Valid Model ---
    if(hypar["mode"]=="train"):
        train(net_train,
              state_dict,
              net_test,
              optimizer,
              train_dataloaders,
              train_datasets,
              valid_dataloaders,
              valid_datasets,
              hypar)
    else:
        valid(net_test,
              state_dict,
              valid_dataloaders,
              valid_datasets,
              hypar)


if __name__ == "__main__":

    ### --- hyperparamters of training ---
    hypar = {}
    hypar['type']='S'#'R' denotes ResNet34, 'S' denotes Swin-Transformer

    hypar["random_flip_h"] = 1
    hypar["random_flip_v"] = 0
    hypar['parallel_num']=2
    hypar["cache_boost_train"] = False
    hypar["cache_boost_valid"] = False
    ##  cache_boost==True: we will loaded the cached .pt file

    if hypar['type']=='R':
        hypar["cache_size"] = [352, 352]
        hypar["input_size"] = [352, 352]
        hypar['lr']=1e-2
        hypar['momentum']=0.9
        hypar['weight_decay']=1e-4 

        hypar["model_name"] = "DC-Net-R"
        print("Model Name: ", hypar["model_name"])

        hypar["batch_size_train"] = 32

    if hypar['type']=='S':
        hypar["cache_size"] = [384, 384]
        hypar["input_size"] = [384, 384]
        hypar['lr']=1e-3
        hypar['momentum']=0.9
        hypar['weight_decay']=1e-4

        hypar["model_name"] = "DC-Net-S"
        print("Model Name: ", hypar["model_name"])

        hypar["batch_size_train"] = 8

    hypar['use_amp']=True

    print("building model...")
    hypar["model_train"] = Training_DC_Net(type=hypar['type'])
    hypar["model_test"] = Inference_DC_Net(parallel=hypar['parallel_num'],type=hypar['type'])

    snapshot_path = "./log/{}".format(
        hypar["model_name"])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    hypar["model_path"] = ""
    hypar["early_stop"] = 10000 ## stop the training when no improvement in the past 100 validation periods
    hypar["model_save_fre"] = 1000 ## validate and output the model weights every 10 iterations

    #put checkpoints here
    hypar["restore_model"] = ""
    
    if hypar["restore_model"]!="":
        hypar["start_ite"] = 0
    else:
        hypar["start_ite"] = 0

    
    hypar["batch_size_valid"] = 1
    print("batch size: ", hypar["batch_size_train"])

    ## train or testing mode
    hypar["mode"] = "train" ## or "valid"
    hypar["model_digit"] = "full" # "half"

    ## maximum iteration number
    hypar["max_ite"] = 10000000
    hypar["max_epoch_num"] = 1000000
    
    dataset_train1 = {"name": "DUTS-TR",
            "im_dir": "./datasets/DUTS-TR/im",
            "gt_dir": "./datasets/DUTS-TR/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png",
            "cache_dir":"./datasets/DUTS-TR/cache_"+str(hypar["cache_size"][0])}

    dataset_valid1 = {"name": "DUTS-TE",
                    "im_dir": "./datasets/DUTS-TE/im",
                    "gt_dir": "./datasets/DUTS-TE/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png",
                    "cache_dir":"./datasets/DUTS-TE/cache_"+str(hypar["cache_size"][0])}

    hypar["model_path"] = "./saved_models/"+hypar["model_name"]+"-"+"x".join([str(s) for s in hypar["cache_size"]])
    hypar["valid_out_dir"] = hypar["model_path"]

    train_datasets, valid_datasets = [], []
    ### train datasets
    train_datasets = [dataset_train1]
    ### valid or test datasets
    valid_datasets = [dataset_valid1]

    main(train_datasets,
         valid_datasets,
         hypar=hypar)
