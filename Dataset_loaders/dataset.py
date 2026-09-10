import torch
import os
import numpy as np
import glob
import random
from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image
from einops import rearrange


class DataLoader(Dataset):

    def __init__(self, maps_path, datatype, transforms=None, data_aug=False):
        st_map_path = []
        for file in maps_path:
            st_map_path.extend(glob.glob(file+"/*"))
        self.st_map_path = st_map_path
        self.transforms=transforms
        self.data_aug=data_aug
        # sampling rate of facial video
        self.sampling_rate = 30
        # Low frequency of physiological signal bandwidth
        self.low_freq = 0.66
        # High frequency of physiological signal bandwidth
        self.high_freq = 3.0
        self.datatype = datatype

    def __len__(self):
        return len(self.st_map_path)

    def __getitem__(self, item):
        root_path = self.st_map_path[item]
        path = root_path.split('/')[-2]+'-'+root_path.split('/')[-1]
        rgb_img_path = os.path.join(root_path, "img_rgb.png")
        yuv_img_path = os.path.join(root_path, "img_yuv.png")
        map1 = Image.open(rgb_img_path).convert("RGB")
        map2 = Image.open(yuv_img_path).convert("RGB")
        if self.transforms:
            feature_map = np.concatenate((np.array(map1), np.array(map2)), axis=2)
            feature_map = self.transforms(feature_map)
        
        hr = loadmat(os.path.join(root_path, "hr.mat"))["hr"].squeeze(0)
        if self.datatype == "MMPD":
            bvp = loadmat(os.path.join(root_path, "bvp.mat"))["bvp"].squeeze(1)
        else:
            bvp = loadmat(os.path.join(root_path, "bvp.mat"))["bvp"].squeeze(0)
        bvp = (bvp-np.mean(bvp))/np.std(bvp)
        hr = torch.tensor(hr, dtype=torch.float32)
        bvp = torch.tensor(bvp, dtype=torch.float32)

        bvp_aug = torch.zeros((300), dtype=torch.float32)
        feature_map_aug = torch.zeros((6,63,300), dtype=torch.float32)

        # subject = root_path.split('/')[-2].split('_')[-2]
        if self.data_aug == True:
            p = random.random()
            if p < 0.5 and hr < 80:
                hr_aug = 2 * hr
                bvp = torch.cat((bvp, bvp),dim=-1)
                feature_map = torch.cat((feature_map,feature_map),dim=-1)
                
                for x in range(300):
                    bvp_aug[x] = bvp[x*2]
                    feature_map_aug[:,:,x] = feature_map[:,:,2*x]
                feature_map = feature_map_aug
                bvp = bvp_aug
                hr = hr_aug
            elif p < 0.5 and hr > 80:
                hr_aug = hr/2
                for t in range(300):
                    if t%2==0:
                        bvp_aug[t] = bvp[t//2]
                        feature_map_aug[:,:,t] = feature_map[:,:,t//2]
                    else:
                        bvp_aug[t] = bvp[t//2]/2+bvp[t//2+1]/2
                        feature_map_aug[:,:,t] = feature_map[:,:,t//2]/2+feature_map[:,:,t//2+1]/2
                feature_map = feature_map_aug
                bvp = bvp_aug
                hr = hr_aug
        
        diff_feature_map = torch.zeros((6,63,300), dtype=torch.float32)
        diff_feature_map[:,:,:299] = torch.diff(feature_map,dim=-1)
        diff_feature_map = diff_feature_map / torch.std(diff_feature_map)
        
        temp_list = []
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                x = feature_map[i,j,:]
                xf = torch.fft.rfft(x)
                rfreqs = torch.fft.rfftfreq(feature_map.shape[-1], 1/self.sampling_rate)
                pass_f = (torch.abs(rfreqs) >= self.low_freq) & (torch.abs(rfreqs) <= self.high_freq)
                rx = torch.fft.irfft(xf * pass_f)
                temp_list.append(rx)
        physio_bandpass_feature = torch.stack(temp_list)
        physio_bandpass_feature = rearrange(physio_bandpass_feature, '(C N) T -> C N T', C=6)

        feature_map = torch.cat((feature_map,diff_feature_map),dim=0)

        return {"st_maps": feature_map, "phys_maps": physio_bandpass_feature, "gt_HR": hr, "bvp": bvp, "path": path}


if __name__ == "__main__":
    pass