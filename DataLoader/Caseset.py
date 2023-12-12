import os

import numpy as np
import torch

from torchvision import transforms

from .Utils import crop, normalize



class refHist():
    def __init__(self):

        self.values,self.quantiles=self.get_ref_Hist()
    def get_ref_Hist(self):

        ref_paths=self.getPath_hist('/home/Dk/ref/')
        values_ref,quantiles_ref=[],[]
        for classes_index in range(2):
            img_time=[]
            for index in range(len(ref_paths[0][0])):
                img_path,label_path,types=ref_paths[classes_index][0][index],ref_paths[classes_index][1][index],ref_paths[classes_index][2]                
                assert self.paths_match(label_path,img_path),"The label_path %s and img_path %s don't match."%(label_path,img_path)
                img=np.load(img_path)
                img_time.append(img)
            img_time=np.concatenate(img_time,axis=0)
            values, tmpl_counts = np.unique(img_time.ravel(), return_counts=True)
            quantiles = np.cumsum(tmpl_counts) / img_time.size
            values_ref.append(values)
            quantiles_ref.append(quantiles)

        return values_ref,quantiles_ref

    def paths_match(self,label_path,img_path):
        filename1_without_ext=os.path.splitext(os.path.basename(label_path))[0]
        filename2_without_ext=os.path.splitext(os.path.basename(img_path))[0]    
        return filename1_without_ext==filename2_without_ext

    def getPath_hist(self,root):
        phase='train'

        classes=['ED/','ES/']
        paths=[]
        for index in range(2):
            img_dir=os.path.join(root+classes[index],'%s_img'%phase)
            img_paths=make_dataset(img_dir,recursive=False,read_cache=True)

            label_dir=os.path.join(root+classes[index],'%s_label'%phase)
            label_paths=make_dataset(label_dir,recursive=False,read_cache=True)

            img_paths=sorted(img_paths)
            label_paths=sorted(label_paths)

            paths.append([img_paths,label_paths,[index]])
        
        return paths
        
    def histMatch(self,img,classes):
        _,src_unique_indices,src_counts=np.unique(img.ravel(),return_inverse=True,return_counts=True)
        src_quantiles=np.cumsum(src_counts)/img.size
        interp_a_values=np.interp(src_quantiles,self.quantiles[classes],self.values[classes])
        return interp_a_values[src_unique_indices].reshape(img.shape)





class Caseset_validation(torch.utils.data.Dataset):
    def __init__(self,
                 pair_list_path,
                 pair_dir,
                 resolution_path,
                 outimg_size=(128, 128)):
        pair_list = np.loadtxt(pair_list_path).astype(np.int)
        self.pair_dir = pair_dir
        self.outimg_size = outimg_size
        resolutions = np.loadtxt(resolution_path)  # size (228, 2)
        

        self.hist = refHist()
        self.case_list = []
        self.packed_case_pair = {}
        for c_no, s, ed, es in pair_list:
            ed_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, ed, s)))
            es_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, es, s)))
            src_seg = ed_unit['seg']
            tgt_seg = es_unit['seg']
            if np.sum(src_seg) == 0 or np.sum(tgt_seg) == 0:
                continue
            # for old version, need to #
            if len(np.unique(src_seg)) is not self.getDatasetName(c_no) or len(
                    np.unique(tgt_seg)) is not self.getDatasetName(c_no):
                continue
            src_img = normalize(ed_unit['img'])
            tgt_img = normalize(es_unit['img'])

            src_img = self.hist.histMatch(src_img,0)
            tgt_img = self.hist.histMatch(tgt_img,1)

            inimg_size = src_img.shape
            center = (inimg_size[1] // 2, inimg_size[0] // 2)

            src_img = crop(src_img, center, self.outimg_size)
            tgt_img = crop(tgt_img, center, self.outimg_size)
            src_seg = crop(src_seg, center, self.outimg_size)
            tgt_seg = crop(tgt_seg, center, self.outimg_size)

            src_img = torch.tensor(src_img)
            tgt_img = torch.tensor(tgt_img)
            src_seg = torch.tensor(src_seg)
            tgt_seg = torch.tensor(tgt_seg)

            imgs = []
            imgs.append(src_img)
            imgs.append(tgt_img)
            imgs.append(src_seg)
            imgs.append(tgt_seg)
            imgs = torch.stack(imgs, dim=0)
            imgs = self.randomTransforms(imgs)

            src_img = imgs[0].unsqueeze(0).unsqueeze(0)
            tgt_img = imgs[1].unsqueeze(0).unsqueeze(0)
            src_seg = imgs[2].unsqueeze(0).unsqueeze(0)
            tgt_seg = imgs[3].unsqueeze(0).unsqueeze(0)

            if c_no not in self.case_list:
                self.packed_case_pair[c_no] = {
                    's_list': [],
                    'src_img': [],
                    'tgt_img': [],
                    'src_seg': [],
                    'tgt_seg': [],
                    'resolution': resolutions[c_no - 1][0]
                }
                self.case_list.append(c_no)
            self.packed_case_pair[c_no]['s_list'].append(s)
            self.packed_case_pair[c_no]['src_img'].append(src_img)
            self.packed_case_pair[c_no]['tgt_img'].append(tgt_img)
            self.packed_case_pair[c_no]['src_seg'].append(src_seg)
            self.packed_case_pair[c_no]['tgt_seg'].append(tgt_seg)
        self.case_list.sort()

    def randomTransforms(self, imgs):
        imgs = transforms.RandomAffine(degrees=(-45, 45), scale=(0.8, 1.2))(imgs)
        return imgs

    def getDatasetName(self, case_no):
        if case_no <= 33:
            return 3
        elif case_no > 33 and case_no <= 78:
            return 2
        else:
            return 4

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        c_no = self.case_list[idx]
        return self.getByCaseNo(c_no)

    def getByCaseNo(self, c_no):
        packed_src_img = torch.cat(self.packed_case_pair[c_no]['src_img'], 0)
        packed_tgt_img = torch.cat(self.packed_case_pair[c_no]['tgt_img'], 0)
        packed_src_seg = torch.cat(self.packed_case_pair[c_no]['src_seg'], 0)
        packed_tgt_seg = torch.cat(self.packed_case_pair[c_no]['tgt_seg'], 0)
        resolution = self.packed_case_pair[c_no]['resolution']

        return {
            'src': packed_src_img,
            'tgt': packed_tgt_img,
            'src_seg': packed_src_seg,
            'tgt_seg': packed_tgt_seg,
            'case_no': c_no,
            'slice': self.packed_case_pair[c_no]['s_list'],
            'resolution': resolution
        }


class Caseset(torch.utils.data.Dataset):
    def __init__(self,
                 pair_list_path,
                 pair_dir,
                 resolution_path,
                 outimg_size=(128, 128)):
        pair_list = np.loadtxt(pair_list_path).astype(np.int)
        self.pair_dir = pair_dir
        self.outimg_size = outimg_size
        print("resolution_path  ",resolution_path)
        resolutions = np.loadtxt(resolution_path)  

        self.case_list = []
        self.packed_case_pair = {}
        for c_no, s, ed, es in pair_list:
            ed_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, ed, s)))
            es_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, es, s)))
            src_seg = ed_unit['seg']
            tgt_seg = es_unit['seg']
            if np.sum(src_seg) == 0 or np.sum(tgt_seg) == 0:
                continue
            # for old version, need to #
            if len(np.unique(src_seg)) is not self.getDatasetName(c_no) or len(
                    np.unique(tgt_seg)) is not self.getDatasetName(c_no):
                continue
            src_img = normalize(ed_unit['img'])
            tgt_img = normalize(es_unit['img'])

            inimg_size = src_img.shape
            center = (inimg_size[1] // 2, inimg_size[0] // 2)

            src_img = crop(src_img, center, self.outimg_size)
            tgt_img = crop(tgt_img, center, self.outimg_size)
            
            
            src_seg = crop(src_seg, center, self.outimg_size)
            tgt_seg = crop(tgt_seg, center, self.outimg_size)

            src_img = torch.tensor(src_img).unsqueeze(0).unsqueeze(0)
            tgt_img = torch.tensor(tgt_img).unsqueeze(0).unsqueeze(0)
            src_seg = torch.tensor(src_seg).unsqueeze(0).unsqueeze(0)
            tgt_seg = torch.tensor(tgt_seg).unsqueeze(0).unsqueeze(0)

            if c_no not in self.case_list:
                self.packed_case_pair[c_no] = {
                    's_list': [],
                    'src_img': [],
                    'tgt_img': [],
                    'src_seg': [],
                    'tgt_seg': [],
                    'resolution': resolutions[c_no - 1][0]
                }
                self.case_list.append(c_no)
            self.packed_case_pair[c_no]['s_list'].append(s)
            self.packed_case_pair[c_no]['src_img'].append(src_img)
            self.packed_case_pair[c_no]['tgt_img'].append(tgt_img)
            self.packed_case_pair[c_no]['src_seg'].append(src_seg)
            self.packed_case_pair[c_no]['tgt_seg'].append(tgt_seg)
        self.case_list.sort()

    def getDatasetName(self, case_no):
        if case_no <= 33:
            return 3
        elif case_no > 33 and case_no <= 78:
            return 2
        else:
            return 4

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        c_no = self.case_list[idx]
        return self.getByCaseNo(c_no)

    def getByCaseNo(self, c_no):
        packed_src_img = torch.cat(self.packed_case_pair[c_no]['src_img'], 0)
        packed_tgt_img = torch.cat(self.packed_case_pair[c_no]['tgt_img'], 0)
        packed_src_seg = torch.cat(self.packed_case_pair[c_no]['src_seg'], 0)
        packed_tgt_seg = torch.cat(self.packed_case_pair[c_no]['tgt_seg'], 0)
        resolution = self.packed_case_pair[c_no]['resolution']

        return {
            'src': packed_src_img,
            'tgt': packed_tgt_img,
            'src_seg': packed_src_seg,
            'tgt_seg': packed_tgt_seg,
            'case_no': c_no,
            'slice': self.packed_case_pair[c_no]['s_list'],
            'resolution': resolution
        }

    