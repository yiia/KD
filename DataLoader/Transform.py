import numpy as np
import torch
import torch.nn.functional as F
import cv2
from .Utils import randomTransform


class RandomAffineTransform(object):
    def __init__(self, device,
                 rotation_degree=[-45, 45],
                 translate_interval=[-10, 10],
                 scale_interval=[0.8, 1.2],
                 image_size=[160, 160]):
        self.device = device
        self.rotation_degree = rotation_degree
        self.translate_interval = translate_interval
        self.scale_interval = scale_interval
        self.image_size = image_size

    def getAffineMatrix(self, batch_size):
        affine_matrix_list = []
        for i in range(batch_size):
            affine_matrix_list.append(
                randomTransform(self.rotation_degree, self.translate_interval,
                                self.scale_interval, self.image_size))
        return torch.tensor(np.array(affine_matrix_list), dtype=torch.float)

    def __call__(self, pair):
        batch_size = pair['src']['img'].size()[0]
        theta = self.getAffineMatrix(batch_size).to(self.device)
        phi = F.affine_grid(theta,
                            pair['src']['img'].size(),
                            align_corners=True)
        pair['src']['img'] = F.grid_sample(pair['src']['img'],
                                           phi,
                                           mode='bilinear',
                                           align_corners=True)
        pair['tgt']['img'] = F.grid_sample(pair['tgt']['img'],
                                           phi,
                                           mode='bilinear',
                                           align_corners=True)

        pair['src']['seg'] = F.grid_sample(pair['src']['seg'],
                                           phi,
                                           mode='bilinear',
                                           align_corners=True)
        pair['tgt']['seg'] = F.grid_sample(pair['tgt']['seg'],
                                           phi,
                                           mode='bilinear',
                                           align_corners=True)
        return pair


class CentralCropTensor(object):
    def __init__(self, src_img_size, tgt_img_size):
        self.center = (src_img_size[1] // 2, src_img_size[0] // 2)
        self.halfheight = tgt_img_size[0] // 2
        self.halfwidth = tgt_img_size[1] // 2

    def crop(self, img):
        cropped_img = img[:, :, self.center[1] -
                          self.halfheight:self.center[1] + self.halfheight,
                          self.center[0] - self.halfwidth:self.center[0] +
                          self.halfwidth]
        return cropped_img

    def __call__(self, pair):
        pair['src']['img'] = self.crop(pair['src']['img'])
        pair['tgt']['img'] = self.crop(pair['tgt']['img'])
        pair['src']['seg'] = self.crop(pair['src']['seg'])
        pair['tgt']['seg'] = self.crop(pair['tgt']['seg'])
        return pair


class NormalizeTensor(object):
    def normalize(self, tensor):
        return (tensor - torch.min(tensor, dim=0)[0]) / (
            torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0])


    def __call__(self, pair):
        pair['src']['img'] = self.normalize(pair['src']['img'])
        pair['tgt']['img'] = self.normalize(pair['tgt']['img'])
        return pair


class AliginCentralCropTensor(object):
    def __init__(self, src_img_size, tgt_img_size):
        self.center = (src_img_size[1] // 2, src_img_size[0] // 2)
        self.halfheight = tgt_img_size[0] // 2
        self.halfwidth = tgt_img_size[1] // 2
        self.tar_size =  tgt_img_size
    def getCenter(self,stdImg):
        h,w= stdImg.shape
        tempMatrix=np.expand_dims(np.arange(w),axis=0).repeat(h,axis=0)
        wCenter=np.sum(tempMatrix*stdImg)/np.sum(stdImg)
        tempMatrix=np.expand_dims(np.arange(h),axis=1).repeat(w,axis=1)
        hCenter=np.sum(tempMatrix*stdImg)/np.sum(stdImg)
        return  round(hCenter),round(wCenter)


    def crop(self,seg3D, center, size):
        max_height = seg3D.shape[0]
        max_width = seg3D.shape[1]
        start_x = center[0] - size[1] // 2
        start_x = start_x if start_x >= 0 else 0
        start_x = start_x if start_x + size[0] < max_width else max_width - size[0]

        start_y = center[1] - size[0] // 2
        start_y = start_y if start_y >= 0 else 0
        start_y = start_y if start_y + size[1] < max_height else max_height - size[1]

        return seg3D[start_y:start_y + size[1], start_x:start_x + size[0]]

    def __call__(self, pair):
        img =   pair['src']['img'].cpu().numpy()
        es_gt = pair['src']['seg'].cpu().numpy()

        img_es =  []
        gt_es = []

        for i in range(es_gt.shape[0]):
            cur_gt = es_gt[i,0,:,:]
            cur_img = img[i,0,:,:]
            if cur_gt.sum() == 0:
                continue
            Hc,Wc =self.getCenter(es_gt[i,0,:,:]) 
            crop_img = self.crop(cur_img, [Hc,Wc],self.tar_size )
            crop_gt = self.crop(cur_gt, [Hc,Wc],self.tar_size )
            img_es.append(crop_img[None,:,:])
            gt_es.append(crop_gt[None,:,:])

        img_es=np.array(img_es)
        gt_es=np.array(gt_es)     

        img =   pair['tgt']['img'].cpu().numpy()
        ed_gt = pair['tgt']['seg'].cpu().numpy()

        img_ed =  []
        gt_ed = []

        for i in range(ed_gt.shape[0]):
            cur_gt = ed_gt[i,0,:,:]
            cur_img = img[i,0,:,:]
            if cur_gt.sum() == 0:
                continue
            Hc,Wc = self.getCenter(ed_gt[i,0,:,:]) 
            crop_img = self.crop(cur_img, [Hc,Wc],self.tar_size )
            crop_gt = self.crop(cur_gt, [Hc,Wc],self.tar_size )
            img_ed.append(crop_img[None,:,:])
            gt_ed.append(crop_gt[None,:,:])

        img_ed=np.array(img_ed)
        gt_ed=np.array(gt_ed)      
        pair['src']['img'] = torch.tensor(np.array(img_es)).cuda() 
        pair['tgt']['img'] = torch.tensor(np.array(img_ed)).cuda()
        pair['src']['seg'] = torch.tensor(np.array(gt_es)).cuda()
        pair['tgt']['seg'] = torch.tensor(np.array(gt_ed)).cuda()
        return pair


class NormalizeTensor(object):
    def normalize(self, tensor):
        return (tensor - torch.min(tensor, dim=0)[0]) / (
            torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0])

    def __call__(self, pair):
        pair['src']['img'] = self.normalize(pair['src']['img'])
        pair['tgt']['img'] = self.normalize(pair['tgt']['img'])
        return pair


class RandomMirrorTensor2D(object):
    @staticmethod
    def getParams():
        xflip = np.random.choice([0, 1])
        yflip = np.random.choice([0, 1])
        return xflip, yflip

    @staticmethod
    def flip(img: torch.Tensor, xflip, yflip) -> torch.Tensor:
        if not xflip and not yflip:
            return img
        dims = []
        if xflip:
            dims.append(2)
        if yflip:
            dims.append(3)
        return torch.flip(img, dims)

    def __call__(self, pair):
        xflip, yflip = self.getParams()
        pair['src']['img'] = self.flip(pair['src']['img'], xflip, yflip)
        pair['tgt']['img'] = self.flip(pair['tgt']['img'], xflip, yflip)
        pair['src']['seg'] = self.flip(pair['src']['seg'], xflip, yflip)
        pair['tgt']['seg'] = self.flip(pair['tgt']['seg'], xflip, yflip)
        return pair
