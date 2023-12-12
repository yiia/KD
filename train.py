import torch
import os
from DataLoader import (Caseset,AliginCentralCropTensor,CollateByDevice,Dataset,RandomAffineTransform)
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tensorboardX as tbx
import loss
import torchvision
import Pspnet
import time
from loss import CriterionKD,AEPT,NewAIFV


print(torch.cuda.device_count())
print(torch.cuda.is_available())


def init_device():
    torch.set_num_threads(1)
    return torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def getTrainDataloader(train_list_dir, pair_dir,device):
    dataset = Dataset(train_list_dir,pair_dir)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=CollateByDevice(device, transforms=transforms.Compose([
                RandomAffineTransform(device),
                AliginCentralCropTensor((200, 200), (160, 160))])),
            pin_memory=False)
    return train_dataloader

def getDataloader(train_list_dir, pair_dir,device):
    dataset = Dataset(train_list_dir,pair_dir)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=CollateByDevice(device, transforms=transforms.Compose([
            AliginCentralCropTensor((200, 200), (160, 160))])), 
        pin_memory=False)
    return train_dataloader

def getTestDataloader( data_list_path: str,pair_dir,resolution_dir):
    dataset = Caseset(data_list_path,pair_dir, resolution_dir)
    test_dataloader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        num_workers=0)
    return test_dataloader


def init_MM_dataset_dir():
    # dataset path
    dataset_dir = os.path.join(os.path.abspath('/home/Dataset/ACDC/'), 'dataset_ACDC')
    all_data_dir = os.path.join(dataset_dir, 'all_pair.txt')
    train_data_dir = os.path.join(dataset_dir, 'training_pair.txt')
    validation_data_dir = os.path.join(dataset_dir, 'validation_pair.txt')
    test_data_all_dir = os.path.join(dataset_dir, 'testing_pair.txt')
    test_data_large_target_dir = os.path.join(dataset_dir, 'testing_pair_CenterPair.txt')
    pair_dir = os.path.join(dataset_dir, '2Dwithoutcenter1')
    resolution_dir = os.path.join(dataset_dir, 'resolution.txt')
    return {
        "all_data_dir": all_data_dir,
        "train_data_dir": train_data_dir,
        "validation_data_dir": validation_data_dir,
        "test_data_all_dir": test_data_all_dir,
        "test_data_large_target_dir": test_data_large_target_dir,
        "pair_dir": pair_dir,
        "resolution_dir": resolution_dir,
    }


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def main():
    device='cuda'
    ceLoss=torch.nn.CrossEntropyLoss()
    Kdloss = CriterionKD()
    wiiaeloss = AEPT()  
    inter = NewAIFV()    

    data_acdc_paras = init_MM_dataset_dir()
    device = init_device()
    loader1 = getTrainDataloader(data_acdc_paras["train_data_dir"], data_acdc_paras["pair_dir"],device)
    Loader = getDataloader(data_acdc_paras["test_data_all_dir"], data_acdc_paras["pair_dir"],device) 

    # Student Setup
    backbone= 'resnet18' 
    base_channel = 2   
    segNet = Pspnet.PSPNet(in_channel=1,out_channel=4,base_channel=base_channel,backend=backbone).to(device)   

    #Teacher model
    Teacher_model_path =  "/home/Dk/ACDC/mm/model.pkl" 
    backbone= 'resnet50' 
    base_channel = 64   
    Tea_segNet = Pspnet.PSPNet(in_channel=1,out_channel=4,base_channel=base_channel,backend=backbone).to(device)  

    Tea_segNet.load_state_dict(torch.load(Teacher_model_path))  
    Tea_segNet.eval()

    opt = torch.optim.Adam(segNet.parameters(), lr= 0.001, weight_decay=1e-4)

    #model_reload_path = "/home/Dk/modelLast.pkl"      
    if 0:
        segNet.load_state_dict(torch.load(model_reload_path))    
        opt = torch.optim.Adam([{'params':segNet.parameters()},{'params':inter.parameters()}], lr= 1.0e-3, weight_decay=1e-4)
        start_epoch = 0  
        Flag =False
        print('loading model start_epoch {} successful'.format(start_epoch))
    else:
        start_epoch = 0
        Flag =False
        print('No saving model, please start training!')

    EPOCH = 5000
    log_path = "./ACDC/" + "ceshi" 
    log_path_img = log_path+'/'+ 'train/img'
    log_path_initSeg = log_path+'/'+ 'train/initSeg'
    log_path_mask = log_path+'/'+ 'train/mask'
    
    log_path_img_val = log_path+'/'+ 'test/img'
    log_path_initSeg_val = log_path+'/'+ 'test/initSeg'
    log_path_mask_val = log_path+'/'+ 'test/mask'    
    
    model_save_path = log_path +'/'+  "model.pkl"
    model_last_save_path = log_path +'/'+  "modelLast.pkl"
    # model_last_save_path_transStu = log_path +'/'+  "modelLasttransStu.pkl"
    # model_last_save_path_transTea = log_path +'/'+  "modelLasttransTea.pkl"    

    writer=tbx.SummaryWriter(logdir=log_path)
    ceLoss=torch.nn.CrossEntropyLoss().to(device)

    initSegIndex={}
    base_lr = 0.005 
    power = 0.8
    lam_wii = 1000  
    lam_wae = 1000  
    lam_inter = 10 
    min_acc = 0
    for epoch in range(EPOCH):
        if Flag ==False:
            epoch = epoch + start_epoch
            
        segNet.train()
        learning_rate = lr_poly(base_lr, epoch, EPOCH, power)                                                          
        print("learning_rate",learning_rate)                           
        for param_group in opt.param_groups:
            param_group['lr'] = learning_rate        

        initSegIndex['train_sumLoss']=0
        initSegIndex['sumDice']=0

        count=0
        for index,dataBatch in enumerate(loader1):
            ed_img  = dataBatch['src']['img']
            ed_mask = dataBatch['src']['seg']
            es_img  = dataBatch['tgt']['img']
            es_mask = dataBatch['tgt']['seg'] 
            img  = torch.concat((ed_img,es_img),dim=0).float()
            mask =  torch.concat((ed_mask,es_mask),dim=0)
            if img.shape[0] == 0:
                continue   
            img  = torch.concat((ed_img,es_img),dim=0).float()[0].unsqueeze(0)
            mask =  torch.concat((ed_mask,es_mask),dim=0)[0].unsqueeze(0)

            unSigSeg,f_stu=segNet(img)
            seg = torch.nn.functional.softmax(unSigSeg,dim=1)
            with torch.no_grad():
                tea_seg,f_tea = Tea_segNet(img)
            loss_train=ceLoss(unSigSeg,mask.round().long()[:,0,:,:])    
            kd_loss = Kdloss(unSigSeg,tea_seg)
            loss_train += kd_loss
                
            if  lam_wii>0 or lam_wae >0:
                wii_loss,wae_loss = wiiaeloss(tea_seg,mask,f_stu,f_tea)
                loss_train += lam_wii*wii_loss  
                loss_train += lam_wae*wae_loss 

            if  lam_inter>0 :
                inter_loss= inter(f_stu,f_tea,unSigSeg,tea_seg,mask)
                loss_train += lam_inter*inter_loss

            loss_train.backward()
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                initSegIndex['train_sumLoss']+=loss_train*img.shape[0]
                count+=img.shape[0]
                              
        writer.add_scalar('loss',initSegIndex['train_sumLoss']/count,epoch)
        print("i:",epoch)

        model_last_save_path = log_path +'/'+  "modelLast.pkl"
        torch.save(segNet.state_dict(),model_last_save_path)

        # model_last_save_path_transStu = log_path +'/'+  "modelLasttransStu.pkl"
        # model_last_save_path_transTea = log_path +'/'+  "modelLasttransTea.pkl"         
        # inter.save_model(model_last_save_path_transStu,model_last_save_path_transTea) ##ifv don't need this
    
        if(epoch%5==0 or epoch==EPOCH-1):
            with torch.no_grad():   
                segNet.eval()
                testCount=0
                initSegIndex['sumTestDice3']=0
                initSegIndex['sumTestDice2']=0
                initSegIndex['sumTestDice1']=0
                initSegIndex['val_sumLoss']=0
                RvDice= []
                LvDice= []
                MyDice= []  
                for i,dataBatch in enumerate(Loader):
                    ed_img  = dataBatch['src']['img']
                    ed_mask = dataBatch['src']['seg']
                    es_img  = dataBatch['tgt']['img']
                    es_mask = dataBatch['tgt']['seg']                   
                    img  = torch.concat((ed_img,es_img),dim=0).float()
                    mask =  torch.concat((ed_mask,es_mask),dim=0) 
                    if img.shape[0] == 0:
                        continue
                    unSigSeg=segNet(img)
                    seg = torch.nn.functional.softmax(unSigSeg[0],dim=1)                    
                    loss_val1=ceLoss(unSigSeg[0],mask.round().long()[:,0,:,:])
                    initSegIndex['sumTestDice3']+=loss.dice3d(seg[:,3],(mask[:,0]==3).float())
                    initSegIndex['sumTestDice2']+=loss.dice3d(seg[:,2],(mask[:,0]==2).float())
                    initSegIndex['sumTestDice1']+=loss.dice3d(seg[:,1],(mask[:,0]==1).float())
                    RvDice3d=loss.dice3d(seg[:,3],(mask[:,0]==3).float())
                    MyDice3d=loss.dice3d(seg[:,2],(mask[:,0]==2).float())
                    LvDice3d=loss.dice3d(seg[:,1],(mask[:,0]==1).float())
                    RvDice.append(RvDice3d.detach().cpu().numpy())
                    LvDice.append(LvDice3d.detach().cpu().numpy())
                    MyDice.append(MyDice3d.detach().cpu().numpy())
                    testCount += int(img.shape[0]/2.0)
                    initSegIndex['val_sumLoss']+=loss_val1                               
                RvDice = np.mean(RvDice) 
                LvDice = np.mean(LvDice)
                MyDice = np.mean(MyDice)
                acc_avg = (RvDice + LvDice + MyDice)/3   
                print("Classes:       RV       LV      MY      AVG")
                print("DICE:   ",RvDice,"    ",LvDice,"    ",MyDice,"    ",acc_avg)
                writer.add_scalars('Dice',{'test3':RvDice},epoch)
                writer.add_scalars('Dice',{'test2':LvDice},epoch)
                writer.add_scalars('Dice',{'test1':MyDice},epoch)
                writer.add_scalar('Valloss',initSegIndex['val_sumLoss']/testCount,epoch)                
                if acc_avg >min_acc:
                    print("Best acc_avg:",acc_avg)
                    min_acc = acc_avg
                    torch.save(segNet.state_dict(),model_save_path)
                if img.shape[0] != 0:
                    imgGrid=torchvision.utils.make_grid(img[0:32],normalize=True)
                    maskGrid=torchvision.utils.make_grid(mask[0:32],normalize=True)
                    segGrid=torchvision.utils.make_grid(seg[0:32,1:],normalize=True)
                    writer.add_image(log_path_img_val,imgGrid,epoch)
                    writer.add_image(log_path_initSeg_val,segGrid,epoch)
                    writer.add_image(log_path_mask_val,maskGrid,epoch)
                #torch.save(segNet.state_dict(),model_save_path)
    writer.close()


if __name__ == '__main__':
    main()