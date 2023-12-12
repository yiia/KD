import os
from .image_folder import make_dataset
import numpy as np
import torch
from .Utils import normalize

class refHist():
    def __init__(self):

        self.values,self.quantiles=self.get_ref_Hist()
    def get_ref_Hist(self):

        ref_paths=self.getPath_hist('/home/Dk/ref/')
        values_ref,quantiles_ref=[],[]
        for classes_index in range(2):
            img_time =[]
            #print("ref_paths[0][0]",ref_paths)
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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pair_list_path, pair_dir, transform=None):
        self.pair_list = np.loadtxt(pair_list_path).astype(np.int)
        self.transform = transform
        self.pair_dir = pair_dir
        self.preload()     
        self.hist = refHist()


    def preload(self):
        self.dataset = {}
        self.patent_list =[]
        self.pair_list_pro =[]
        total_s = 0
        # The dimension of ACDC dataset is 4 from the physical perspective,
        # and the dimension of ACDC is separated into 3D cardiac data and 1D time,
        # in the same time, ACDC dataset has multiply patients. Moreover,
        # an 3D cardiac data consists of 28~40 frames(also means 28~40 slices),
        # variable 's' means the serial number of slice. Generally, we select 'ed'
        # slice and 'es' slice in the same 's'. As we know, the cardiac of a patient
        # is changing by time, variable 'ed' and 'es' means the start time and end
        # time at one cardiac cycle. So, 'c_no' is the number of patient, 's' is
        # the serial number of slice of this patient, 'ed' and 'es' is the start
        # time and end time at one cardiac cycle of this patient.
        #print("pair_list",self.pair_list)
        for c_no, s, ed, es in self.pair_list:
            ed_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, ed, s)))
            es_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, es, s)))
            if c_no not in self.dataset:
                self.dataset[c_no] = {}

            data_type = 'ACDC'  #'mnms  ACDC  miccai'
            if data_type == 'ACDC':
                if c_no not in self.patent_list:
                    if c_no > 78 and c_no<=228 :
                        self.patent_list.append(c_no)   
                if c_no>78 and c_no<=228 :        
                    total_s = total_s + 1
                    self.pair_list_pro.append([c_no, s, ed, es])
            elif  data_type == 'miccai':
                if c_no not in self.patent_list:
                    if c_no > 33 and c_no<=78 :
                        self.patent_list.append(c_no)   
                if c_no > 33 and c_no<=78 :        
                    total_s = total_s + 1
                    self.pair_list_pro.append([c_no, s, ed, es])
            else:
                if c_no not in self.patent_list:
                    if c_no >= 0 :
                        self.patent_list.append(c_no)   
                if c_no >=0:        
                    total_s = total_s + 1
                    self.pair_list_pro.append([c_no, s, ed, es])

            self.dataset[c_no][s] = {
                ed: {
                    'img': ed_unit['img'].astype(np.float),
                    'seg': ed_unit['seg'].astype(np.float)
                },
                es: {
                    'img': es_unit['img'].astype(np.float),            
                    'seg': es_unit['seg'].astype(np.float)
                },
            }
        self.pair_list = np.array(self.pair_list_pro)


        print("Total images:",total_s)
        print("Total Num:",len(self.patent_list))

    def __len__(self):
        return self.pair_list.shape[0]

    def __getitem__(self, idx):
        c_no, s, ed, es = self.pair_list[idx]

        output = {
            'src': self.dataset[c_no][s][ed],
            'tgt': self.dataset[c_no][s][es]
        }
        if type(output['src']['img'] ) != torch.Tensor:
            output['src']['img'] =  torch.tensor(normalize(output['src']['img']).astype(np.float)).unsqueeze(0).unsqueeze(0).float()
            output['tgt']['img'] =  torch.tensor(normalize(output['tgt']['img']).astype(np.float)).unsqueeze(0).unsqueeze(0).float()

            output['src']['seg'] = torch.tensor(output['src']['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float()
            output['tgt']['seg'] = torch.tensor(output['tgt']['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float()


        if self.transform:
            output = self.transform(output)

        return output

class Dataset_validation_in_training(torch.utils.data.Dataset):
    def __init__(self, pair_list_path, pair_dir, transform=None):
        self.pair_list = np.loadtxt(pair_list_path).astype(np.int)
        self.transform = transform
        self.pair_dir = pair_dir
        self.preload()

    def preload(self):
        self.dataset = {}
        # The dimension of ACDC dataset is 4 from the physical perspective,
        # and the dimension of ACDC is separated into 3D cardiac data and 1D time,
        # in the same time, ACDC dataset has multiply patients. Moreover,
        # an 3D cardiac data consists of 28~40 frames(also means 28~40 slices),
        # variable 's' means the serial number of slice. Generally, we select 'ed'
        # slice and 'es' slice in the same 's'. As we know, the cardiac of a patient
        # is changing by time, variable 'ed' and 'es' means the start time and end
        # time at one cardiac cycle. So, 'c_no' is the number of patient, 's' is
        # the serial number of slice of this patient, 'ed' and 'es' is the start
        # time and end time at one cardiac cycle of this patient.
        count = -1
        count1 = 0
        count2 = 0
        for c_no, s, ed, es in self.pair_list:
            count = count + 1
            ed_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, ed, s)))
            es_unit = np.load(
                os.path.join(self.pair_dir, '%d-%d-%d.npz' % (c_no, es, s)))
            src_seg = torch.tensor(ed_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float()
            tgt_seg = torch.tensor(es_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float()
            if torch.sum(src_seg) == 0 or torch.sum(tgt_seg) == 0:
                self.pair_list = np.delete(self.pair_list, count, axis=0)
                count = count - 1
                count1 = count1 + 1
                continue
            if len(torch.unique(src_seg)) is not self.getDatasetName(c_no) or len(torch.unique(tgt_seg)) is not self.getDatasetName(c_no):
                self.pair_list = np.delete(self.pair_list, count, axis=0)
                count = count - 1
                count2 = count2 + 1
                continue
            if c_no not in self.dataset:
                self.dataset[c_no] = {}
            self.dataset[c_no][s] = {
                ed: {
                    'img': torch.tensor(normalize(ed_unit['img'].astype(np.float))).unsqueeze(0).unsqueeze(0).float(),
                    'seg': torch.tensor(ed_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float(),
                },
                es: {
                    'img': torch.tensor(normalize(es_unit['img'].astype(np.float))).unsqueeze(0).unsqueeze(0).float(),
                    'seg': torch.tensor(es_unit['seg'].astype(np.float)).unsqueeze(0).unsqueeze(0).float()
                },
            }
        print(count1)
        print(count2)

    def __len__(self):
        return self.pair_list.shape[0]

    def __getitem__(self, idx):
        c_no, s, ed, es = self.pair_list[idx]

        output = {
            'src': self.dataset[c_no][s][ed],
            'tgt': self.dataset[c_no][s][es]
        }

        if self.transform:
            output = self.transform(output)

        return output

    def getDatasetName(self, case_no):
        if case_no <= 33:
            return 3
        elif case_no > 33 and case_no <= 78:
            return 2
        else:
            return 4


class Collate(object):
    def __call__(self, batch):
        output = {
            'src': {
                'img': torch.cat([d['src']['img'] for d in batch], 0),
                'seg': torch.cat([d['src']['seg'] for d in batch], 0),
            },
            'tgt': {
                'img': torch.cat([d['tgt']['img'] for d in batch], 0),
                'seg': torch.cat([d['tgt']['seg'] for d in batch], 0),
            }
        }

        return output


class CollateByDevice(object):
    def __init__(self, device, transforms=None):
        self.transforms = transforms
        self.device = device

    def collate(self, batch):
        output = {
            'src': {
                'img': torch.cat([d['src']['img'] for d in batch], 0).to(self.device),
                'seg': torch.cat([d['src']['seg'] for d in batch], 0).to(self.device),
            },
            'tgt': {
                'img': torch.cat([d['tgt']['img'] for d in batch], 0).to(self.device),
                'seg': torch.cat([d['tgt']['seg'] for d in batch], 0).to(self.device),
            }
        }
        return output

    def __call__(self, batch): 
        batch = self.collate(batch)
        if self.transforms:
            batch = self.transforms(batch)
        return batch
