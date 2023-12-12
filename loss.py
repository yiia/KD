from numpy.lib import index_tricks
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from trans import VisionTransformer
import  configs 
import matplotlib.pyplot as plt

def dice(X,Y):
    X=torch.round(X)
    Y=torch.round(Y)
    s=torch.sum(X*Y,dim=[2,3])*2+1e-6
    s/=(torch.sum(X,dim=[2,3])+torch.sum(Y,dim=[2,3])+1e-6)
    return s.mean(),s.std()

def weighted_dice_loss(X,Y,weight=None):
    s=torch.sum(X*Y,dim=[2,3])*2+1e-6
    s/=(torch.sum(X,dim=[2,3])+torch.sum(Y,dim=[2,3])+1e-6)
    if(weight is not None):
        for i in range(len(weight)):
            s[:,i]*=weight[i]*len(weight)
    return (1-s).mean()

def oneHotCeLoss(X,Y,weight=None):
    c=Y.shape[1]
    oneHotX=torch.log_softmax(X,dim=1)
    yLogpMean=torch.mean(Y*oneHotX,dim=(0,2,3))
    if(weight is not None):
        for i in range(c):
            yLogpMean[i]*=weight[i]*c

    loss=-yLogpMean.mean()
    return loss

def dice3d(X,Y):
    X=X.round()
    Y=Y.round()
    s=torch.sum(X*Y)*2+1e-6
    s/=(torch.sum(X)+torch.sum(Y)+1e-6)
    return s

class CriterionKD(nn.Module):
    def __init__(self, upsample=False, temperature=1):
        super(CriterionKD, self).__init__()
        self.upsample = upsample
        self.temperature = temperature
        self.criterion_kd = torch.nn.KLDivLoss()

    def forward(self, pred, soft):
        scale_pred = pred
        scale_soft = soft       
        loss = self.criterion_kd(F.log_softmax(scale_pred / self.temperature, dim=1), F.softmax(scale_soft / self.temperature, dim=1))
        return loss

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y,score):
    return (at(x)*score - at(y)*score).pow(2).mean()

class AEPT(nn.Module):
    def __init__(self,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0,c8=1):
        super(AEPT, self).__init__()
        self.c1 =c1
        self.c2 =c2        
        self.c3 =c3        
        self.c4 =c4
        self.c5 =c5
        self.c6 =c6
        self.c7 =c7                
        self.c8 =c8
        self.num_classes = 4

    def save_model(self,path1,path2):
        torch.save(self.transform1.state_dict(),path1)
        torch.save(self.transform2.state_dict(),path2)

    def load_model(self,path1,path2):
        self.transform1.load_state_dict(torch.load(path1))
        self.transform2.load_state_dict(torch.load(path2))
 
    def forward(self,Pred_T, target,Stu8,Tea8):     
        Score_T = torch.nn.functional.softmax(Pred_T,dim=1)
        batchsize,c,h,w= Tea8.shape  
        one_hot = torch.zeros(Stu8.shape[0], self.num_classes,target.shape[2], target.shape[3])
        one_hot = one_hot.scatter_(1, target.detach().cpu().long(), 1)
        one_hot = one_hot.cuda()
        wei_t8 = Score_T * one_hot
        wei_t8 = wei_t8.sum(1) 
       
        wei_t8_up = wei_t8.unsqueeze(0)      
        wei_t8 = torch.reshape(wei_t8,(batchsize,Stu8.shape[2]*Stu8.shape[3]))

        size_8 = (Stu8.shape[2], Stu8.shape[3])        
        wei_t8 = F.interpolate(wei_t8_up, size=size_8)
        wei_t8 = wei_t8.squeeze(0)                
        wei_t8 = torch.reshape(wei_t8,(batchsize,Stu8.shape[2]*Stu8.shape[3]))
        wei_t8_tmp = 2*wei_t8 -1
        mask_t8 = (wei_t8_tmp >= 0).float()
        loss_PI =   self.c8 * at_loss(Stu8,Tea8,wei_t8_tmp*mask_t8)*size_8[0]*size_8[1]/mask_t8.sum()
        loss_AE =  -self.c8 * at_loss(Stu8,Tea8,(1-mask_t8)*wei_t8_tmp)*size_8[0]*size_8[1]/(1-mask_t8).sum()
        return loss_PI,loss_AE
    

class NewAIFV(nn.Module):
    def __init__(self, classes=4):
        super(NewAIFV, self).__init__()
        self.num_classes = classes
        self.img_size=160
        self.img_pathch_size=16
        self.img_pathch_num= int(self.img_size/self.img_pathch_size)*int(self.img_size/self.img_pathch_size)
        self.transform1 = VisionTransformer(configs.config_0, img_size=self.img_size).cuda()
        self.transform2 = VisionTransformer(configs.config_1, img_size=self.img_size).cuda()
        self.index_matrix = torch.arange(256*self.img_pathch_num)
        self.index_init()
        self.index_matrix= self.index_matrix.cuda()

    def index_init(self):
        ind = torch.arange(256)
        start_ind = ((ind/16).int())*self.img_size + (ind%16)*16
        for i in range(self.img_pathch_num):
            cur_ind = start_ind + (int(i/16))*16*self.img_size+ int(i%16)
            self.index_matrix[i*256:i*256+256] = cur_ind

    def save_model(self,path1,path2):
        torch.save(self.transform1.state_dict(),path1)
        torch.save(self.transform2.state_dict(),path2)

    def load_model(self,path1,path2):
        self.transform1.load_state_dict(torch.load(path1))
        self.transform2.load_state_dict(torch.load(path2))


    def forward(self, feat_S, feat_T,Pred_S,Pred_T, target):
        S_atten = self.transform1(feat_S)[int(configs.config_0.num_layers_enc) -1].mean(1)
        T_atten = self.transform2(feat_T)[int(configs.config_0.num_layers_enc) -1].mean(1)
        batchsize,c_S,h,w= feat_S.shape    
        target_init = target.view(batchsize,1,h*w)
        feat_S = feat_S.view(batchsize,c_S,h*w)
        feat_T = feat_T.view(batchsize,c_T,h*w)  
        Trans_S = torch.zeros(batchsize,c_S,h*w).cuda()
        Trans_T = torch.zeros(batchsize,c_T,h*w).cuda()
        Cur_target = torch.index_select(target_init, 2,self.index_matrix)     
        Cur_pathch_S = torch.index_select(feat_S, 2,self.index_matrix)
        Cur_pathch_T =torch.index_select(feat_T, 2,self.index_matrix)         
        for i in range(256):
            Trans_S[:,:,i*self.img_pathch_num:(i+1)*self.img_pathch_num] =  torch.matmul(Cur_pathch_S[:,:,i*self.img_pathch_num:(i+1)*self.img_pathch_num], S_atten)
            Trans_T[:,:,i*self.img_pathch_num:(i+1)*self.img_pathch_num] =  torch.matmul(Cur_pathch_T[:,:,i*self.img_pathch_num:(i+1)*self.img_pathch_num], T_atten)
        feat_S = Trans_S.view(batchsize,c_S,h,w)
        feat_T = Trans_T.view(batchsize,c_T,h,w)
        target = Cur_target.view(batchsize,1,h,w)
        Score_S = torch.nn.functional.softmax(Pred_S,dim=1)
        Score_T = torch.nn.functional.softmax(Pred_T,dim=1)       
        Score_S_Trans = Score_S.view(batchsize,4,h*w)
        Score_T_Trans = Score_T.view(batchsize,4,h*w)        
        Cur_Score_S = torch.index_select(Score_S_Trans, 2,self.index_matrix)
        Cur_Score_T= torch.index_select(Score_T_Trans, 2,self.index_matrix)        
        Score_S = Cur_Score_S.view(batchsize,4,h,w)
        Score_T = Cur_Score_T.view(batchsize,4,h,w)
        feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        one_hot = torch.zeros(feat_S.shape[0], self.num_classes,feat_S.shape[2], feat_S.shape[3])
        one_hot = one_hot.scatter_(1, target.detach().cpu().long(), 1)
        one_hot = one_hot.cuda()
        wei_s = Score_S * one_hot
        wei_t = Score_T * one_hot
        wei_s = wei_s.sum(1)
        wei_t = wei_t.sum(1)
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_T.size())
        center_feat_S_0 = feat_S.clone()
        center_feat_T_0 = feat_T.clone()
        center_feat_S_1 = feat_S.clone()
        center_feat_T_1 = feat_T.clone()
        center_feat_S_2 = feat_S.clone()
        center_feat_T_2 = feat_T.clone()        
        one_hot = torch.zeros(feat_S.shape[0], self.num_classes,feat_S.shape[2], feat_S.shape[3])
        one_hot = one_hot.scatter_(1, target.detach().cpu().long(), 1)
        one_hot = one_hot.cuda()
        wei_s = Score_S * one_hot
        wei_t = Score_T * one_hot
        wei_s = wei_s.sum(1) 
        wei_t = wei_t.sum(1)
        class_list = [1,0,3,2]
        for i in range(self.num_classes):
            tmp = Score_T[:,class_list[i],:,:] 
            tmp = tmp.unsqueeze(1) 
            Score_Tmp = nn.Upsample(size_f, mode='nearest')(tmp.float()).expand(feat_T.size()) 
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float() 
            mask_feat_S_prto = (tar_feat_S == class_list[i]).float()
            mask_feat_T_prto = (tar_feat_T == class_list[i]).float() 
            center_feat_S_0 = (1 - mask_feat_S) * center_feat_S_0 + mask_feat_S * ((mask_feat_S_prto * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T_0 = (1 - mask_feat_T) * center_feat_T_0 + mask_feat_T * ((mask_feat_T_prto * feat_T*Score_Tmp).sum(-1).sum(-1) / ((Score_Tmp * mask_feat_T).sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
        class_list = [2,3,0,1]
        for i in range(self.num_classes):
            tmp = Score_T[:,class_list[i],:,:]
            tmp = tmp.unsqueeze(1)
            Score_Tmp = nn.Upsample(size_f, mode='nearest')(tmp.float()).expand(feat_T.size())

            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float() 
            mask_feat_S_prto = (tar_feat_S == class_list[i]).float()
            mask_feat_T_prto = (tar_feat_T == class_list[i]).float()
            center_feat_S_1 = (1 - mask_feat_S) * center_feat_S_1 + mask_feat_S * ((mask_feat_S_prto * feat_S).sum(-1).sum(-1) / (mask_feat_S_prto.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T_1 = (1 - mask_feat_T) * center_feat_T_1 + mask_feat_T * ((mask_feat_T_prto * feat_T*Score_Tmp).sum(-1).sum(-1) / ((Score_Tmp * mask_feat_T_prto).sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
        class_list = [3,2,1,0]
        for i in range(self.num_classes):
            tmp = Score_T[:,class_list[i],:,:]
            tmp = tmp.unsqueeze(1)
            Score_Tmp = nn.Upsample(size_f, mode='nearest')(tmp.float()).expand(feat_T.size())
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float() 
            mask_feat_S_prto = (tar_feat_S == class_list[i]).float()
            mask_feat_T_prto = (tar_feat_T == class_list[i]).float() 
            center_feat_S_2 = (1 - mask_feat_S) * center_feat_S_2 + mask_feat_S * ((mask_feat_S_prto * feat_S).sum(-1).sum(-1) / (mask_feat_S_prto.sum(-1).sum(-1) + 1)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T_2 = (1 - mask_feat_T) * center_feat_T_2 + mask_feat_T * ((mask_feat_T_prto * feat_T*Score_Tmp).sum(-1).sum(-1) / ((Score_Tmp * mask_feat_T_prto).sum(-1).sum(-1) + 1)).unsqueeze(-1).unsqueeze(-1)
        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        mse = nn.MSELoss()
        pcsim_feat_S_0 = (1-wei_s)*wei_t*cos(feat_S, center_feat_S_0) 
        pcsim_feat_T_0 = (1-wei_s)*wei_t*cos(feat_T, center_feat_T_0)
        pcsim_feat_S_1 = (1-wei_s)*wei_t*cos(feat_S, center_feat_S_1)
        pcsim_feat_T_1 = (1-wei_s)*wei_t*cos(feat_T, center_feat_T_1)
        pcsim_feat_S_2 = (1-wei_s)*wei_t*cos(feat_S, center_feat_S_2)
        pcsim_feat_T_2 = (1-wei_s)*wei_t*cos(feat_T, center_feat_T_2)
        vec_s = torch.cat((pcsim_feat_S_0.unsqueeze(1), pcsim_feat_S_1.unsqueeze(1), pcsim_feat_S_2.unsqueeze(1)), dim=1)
        vec_t = torch.cat((pcsim_feat_T_0.unsqueeze(1), pcsim_feat_T_1.unsqueeze(1), pcsim_feat_T_2.unsqueeze(1)), dim=1)
        loss = mse(vec_s,vec_t)
        return loss