import torch
import os
from einops import rearrange
from tqdm import tqdm
import config
import matplotlib
from scipy.interpolate import interp1d
matplotlib.use("Agg")
from matplotlib import pyplot as plt
plt.switch_backend( 'agg')
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from Utils import post_process



def train_fn(model, data_loader, optimizer, lossfunc_pearson, lossfunc_mse):
    model.train()
    loss_per_batch = []
    target_bvp_per_batch = []
    target_hr_per_batch = []
    predicted_bvp_per_batch = []
    
    print("Training Model...")
    tk_iterator = tqdm(data_loader, total=len(data_loader))
    for data in tk_iterator:
        maps = data['st_maps'].cuda()
        phys_maps = data['phys_maps'].cuda()
        gt_HR = data['gt_HR'].cuda()
        bvp = data['bvp'].cuda()
        optimizer.zero_grad()

        ecg = model(maps, phys_maps, bvp)
        ecg = (ecg-torch.mean(ecg)) /torch.std(ecg)
        ecg_fft = torch.fft.rfft(ecg, dim=-1, norm='ortho')
        bvp_fft = torch.fft.rfft(bvp, dim=-1, norm='ortho')

        loss_ecg = lossfunc_pearson(ecg, bvp)
        loss_ecg_fft = lossfunc_mse(torch.real(ecg_fft), torch.real(bvp_fft))+lossfunc_mse(torch.imag(ecg_fft), torch.imag(bvp_fft))

        loss_total = loss_ecg + loss_ecg_fft
        if torch.isnan(loss_total).any():
            print(loss_total)
            print(data['path'])

        loss_total.backward()
        optimizer.step()
        
        target_hr_per_batch.extend(gt_HR.detach().cpu().numpy())
        target_bvp_per_batch.extend(bvp.detach().cpu().numpy())
        predicted_bvp_per_batch.extend(ecg.detach().cpu().numpy())
        loss_per_batch.append(loss_total.item())
        
    return target_hr_per_batch, target_bvp_per_batch, predicted_bvp_per_batch, loss_per_batch


def val_fn(model_eval_temp, data_loader, lossfunc_pearson, lossfunc_mse, dataset):
    model_eval_temp.eval()
    loss_per_batch = []
    target_bvp_per_batch = []
    target_hr_per_batch = []
    predicted_bvp_per_batch = []
    predicted_hr_per_batch = []
    path_per_batch = []

    with torch.no_grad():
        print("Validating Model...")
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for data in tk_iterator:
            maps = data['st_maps'].cuda()
            phys_maps = data['phys_maps'].cuda()
            gt_HR = data['gt_HR'].cuda()
            bvp = data['bvp'].cuda()
            path = data['path']

            ecg = model_eval_temp(maps, phys_maps, bvp)
            ecg = (ecg-torch.mean(ecg)) /torch.std(ecg)

            ecg_fft = torch.fft.rfft(ecg, dim=-1, norm='ortho')
            bvp_fft = torch.fft.rfft(bvp, dim=-1, norm='ortho')

            loss_ecg = lossfunc_pearson(ecg, bvp)
            loss_ecg_fft = lossfunc_mse(torch.real(ecg_fft), torch.real(bvp_fft))+lossfunc_mse(torch.imag(ecg_fft), torch.imag(bvp_fft))
            loss_total = loss_ecg + loss_ecg_fft

            cache_path = f'/home/qianwei/dataset/FreqPhys{dataset}_val_predict_data'
            if not os.path.exists(cache_path):  
                os.makedirs(cache_path)

            with parallel_backend("loky"):
                Parallel(n_jobs=-1)(delayed(process_rppg)(ecg.detach().cpu().numpy(), subject, cache_path)
                                   for subject in range(ecg.shape[0]))
            for subject in range(ecg.shape[0]):
                file_path = os.path.join(cache_path, f"{subject}.npy")
                predicted_hr_per_batch.append(np.load(file_path).item())
                target_hr_per_batch.append(gt_HR.squeeze(1)[subject].item())

            target_bvp_per_batch.extend(bvp.detach().cpu().numpy())
            predicted_bvp_per_batch.extend(ecg.detach().cpu().numpy())
            loss_per_batch.append(loss_total.item())
            path_per_batch.extend(path)

    return target_hr_per_batch, predicted_hr_per_batch, target_bvp_per_batch, predicted_bvp_per_batch, loss_per_batch, path_per_batch


def test_fn(model_eval_temp, data_loader, lossfunc_pearson, lossfunc_mse, dataset):
    model_eval_temp.eval()
    loss_per_batch = []
    target_bvp_per_batch = []
    target_hr_per_batch = []
    predicted_bvp_per_batch = []
    predicted_hr_per_batch = []
    path_per_batch = []

    with torch.no_grad():
        print("Testing Model...")
        tk_iterator = tqdm(data_loader, total=len(data_loader))
        for data in tk_iterator:
            maps = data['st_maps'].cuda()
            phys_maps = data['phys_maps'].cuda()
            gt_HR = data['gt_HR'].cuda()
            bvp = data['bvp'].cuda()
            path = data['path']

            ecg = model_eval_temp(maps, phys_maps, bvp)
            ecg = (ecg-torch.mean(ecg)) /torch.std(ecg)

            ecg_fft = torch.fft.rfft(ecg, dim=-1, norm='ortho')
            bvp_fft = torch.fft.rfft(bvp, dim=-1, norm='ortho')

            loss_ecg = lossfunc_pearson(ecg, bvp)
            loss_ecg_fft = lossfunc_mse(torch.real(ecg_fft), torch.real(bvp_fft))+lossfunc_mse(torch.imag(ecg_fft), torch.imag(bvp_fft))
            loss_total = loss_ecg + loss_ecg_fft

            cache_path = f'/home/qianwei/dataset/{dataset}_test_predict_data'
            if not os.path.exists(cache_path):  
                os.makedirs(cache_path)

            with parallel_backend("loky"):
                Parallel(n_jobs=-1)(delayed(process_rppg)(ecg.detach().cpu().numpy(), subject, cache_path)
                                   for subject in range(ecg.shape[0]))
            for subject in range(ecg.shape[0]):
                file_path = os.path.join(cache_path, f"{subject}.npy")
                predicted_hr_per_batch.append(np.load(file_path).item())
                target_hr_per_batch.append(gt_HR.squeeze(1)[subject].item())

            target_bvp_per_batch.extend(bvp.detach().cpu().numpy())
            predicted_bvp_per_batch.extend(ecg.detach().cpu().numpy())
            loss_per_batch.append(loss_total.item())
            path_per_batch.extend(path)

    return target_hr_per_batch, predicted_hr_per_batch, target_bvp_per_batch, predicted_bvp_per_batch, loss_per_batch, path_per_batch


def process_rppg(predicted_per_batch, subject, cache_path):
    predict_hr = post_process.calculate_metric_per_video(predicted_per_batch[subject])
    np.save(os.path.join(cache_path,f"{subject}.npy"), predict_hr)





