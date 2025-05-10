import math
import torch.nn
import torch.optim
import torchvision
import numpy as np
from models.model import Model_hiding, Model_authen, Model_lock
from config import cfg
from datasets import test_dataset
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from torch.nn import functional as F
import time
import lpips
import os


def load(name, net1, net2, net3):
    state_dicts = torch.load(name)
    network_state_dict1 = {k: v for k, v in state_dicts['net1'].items() if 'tmp_var' not in k}
    network_state_dict2 = {k: v for k, v in state_dicts['net2'].items() if 'tmp_var' not in k}
    network_state_dict3 = {k: v for k, v in state_dicts['net3'].items() if 'tmp_var' not in k}  
    net1.load_state_dict(network_state_dict1)
    net2.load_state_dict(network_state_dict2)
    net3.load_state_dict(network_state_dict3)


def computePSNR(origin, pred):
    if origin.shape != pred.shape:
        raise ValueError("Input images must have the same dimensions.")
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def computeSSIM(origin, pred):
    if origin.shape != pred.shape:
        raise ValueError("Input images must have the same dimensions.")
    if len(origin.shape) == 2:  # Grayscale image
        ssim_value, _ = compare_ssim(origin, pred, full=True)
    elif len(origin.shape) == 3:  # Color image
        origin = origin.astype(np.uint8)
        pred = pred.astype(np.uint8)
        if origin.shape[0] == 3:
            ssim_value = compare_ssim(origin, pred, channel_axis=0)
        else:
            raise ValueError("Color images must have 3 channels (RGB).")
    else:
        raise ValueError("Unsupported image shape.")
    
    return ssim_value


def downsample_2x(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)


os.makedirs(f"{cfg.image_save_path}secret", exist_ok=True)
os.makedirs(f"{cfg.image_save_path}reveal", exist_ok=True)
os.makedirs(f"{cfg.image_save_path}cover", exist_ok=True)
os.makedirs(f"{cfg.image_save_path}stego", exist_ok=True)
device = torch.device("cuda", 0)
net1 = Model_hiding(device, secret_num=cfg.num_hiding)
net2 = Model_authen(device, cond_num=3)
net3 = Model_lock(device)
net1.cuda()
net2.cuda()
net3.cuda()
model_path = f"{cfg.test_path}{cfg.suffix_test}_{cfg.dataset_test_mode}_{cfg.num_hiding}.pt"
save_path = cfg.image_save_path
load(model_path, net1, net2, net3)

testloader = DataLoader(
    test_dataset,
    batch_size=cfg.batchsize_test,
    pin_memory=True,
    num_workers=1,
    drop_last=True,
    shuffle=False
)

with torch.no_grad():
    net1.eval()
    net2.eval()
    net3.eval()
    psnr_s_list = []
    psnr_c_list = []
    ssim_s_list = []
    ssim_c_list = []
    lpips_s_list = []
    lpips_c_list = []
    times = []
    lpips_model = lpips.LPIPS(net="alex", version="0.1").cuda()
    for num, x in enumerate(testloader):
        start = time.time()
        x = x.to(device)
        batch = x.shape[0]
        portion = cfg.num_hiding + 1
        secrets = []
        locks = []
        cover = x[:batch // portion]
        cover_lock = net3(cover, mode='cover_lock')
        for i in range(cfg.num_hiding):
            secret = x[(i + 1) * batch // portion: (i + 2) * batch // portion]
            secret_lock = net3(downsample_2x(secret), mode='secret_lock')
            secrets.append(secret)
            locks.append(0.5 * cover_lock + 0.5 * secret_lock)

        secrets_low = []
        for secret, lock in zip(secrets, locks):
            low, high = net2(secret, lock)
            secrets_low.append(low)
        stego, _, _, r_o = net1(cover, secrets_low)

        reveals = []
        cover_key = net3(stego, mode='cover_key')
        reveals_low, r_p = net1(stego, cfg.num_hiding, rev=True)
        for reveal_low in reveals_low:
            secret_key = net3(reveal_low, mode='secret_key')
            key = 0.5 * cover_key + 0.5 * secret_key
            reveal = net2(reveal_low, key, rev=True)
            reveals.append(reveal)
            
        end = time.time()
        times.append(end-start)
        
        lpips_c_list.append(lpips_model(cover, stego).cpu())
        
        reveals_255 = []
        secrets_255 = []
        for count, (s, r) in enumerate(zip(secrets, reveals)):
            lpips_s_list.append(lpips_model(s, r).cpu())
            reveals_255.append(r.cpu().numpy().squeeze() * 255)
            secrets_255.append(s.cpu().numpy().squeeze() * 255)
            torchvision.utils.save_image(s, f'{save_path}secret/{num}_{count}.png')
            torchvision.utils.save_image(r, f'{save_path}reveal/{num}_{count}.png')

        cover_255 = cover.cpu().numpy().squeeze() * 255
        stego_255 = stego.cpu().numpy().squeeze() * 255
        
        torchvision.utils.save_image(cover, f'{save_path}cover/{num}.png')
        torchvision.utils.save_image(stego, f'{save_path}stego/{num}.png')

        for i, (s, r) in enumerate(zip(secrets_255, reveals_255)):
            psnr_s_list.append(computePSNR(s, r))
            ssim_s_list.append(computeSSIM(s, r))

        psnr_c_list.append(computePSNR(cover_255, stego_255))
        ssim_c_list.append(computeSSIM(cover_255, stego_255))
    
    psnr_s = np.mean(np.array(psnr_s_list))
    psnr_c = np.mean(np.array(psnr_c_list))
    ssim_s = np.mean(np.array(ssim_s_list))
    ssim_c = np.mean(np.array(ssim_c_list))
    lpips_s = np.mean(np.array(lpips_s_list))
    lpips_c = np.mean(np.array(lpips_c_list))
    t = np.mean(np.array(times))

    print(f"PSNR_cover_stego:{psnr_c} SSIM_cover_stego:{ssim_c} LPIPS_cover_stego:{lpips_c} PSNR_secret_reveal:{psnr_s} SSIM_secret_reveal:{ssim_s} LPIPS_secret_reveal:{lpips_s}")
    print(f"time:{t}")
