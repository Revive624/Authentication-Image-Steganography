import torch.distributed as dist
import torch.nn as nn
import torch.optim
import math
import numpy as np
from models.model import Model_hiding, Model_authen, Model_lock, init_model
from config import cfg
from datasets import train_dataset, val_dataset
from torch.utils.data import DataLoader
import warnings
import os
import torch.multiprocessing as mp
from torch.nn import functional as F

warnings.filterwarnings("ignore")


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)

    if mse < 1.0e-10:
        return 100

    if mse > 1.0e15:
        return -100

    return 10 * math.log10(255.0 ** 2 / mse)


def js_divergence_loss(pred, target):
    M = 0.5 * (pred + target)
    kl_p_m = F.kl_div(
        F.log_softmax(pred, dim=1),
        F.softmax(M, dim=1),
        reduction='mean'
    )
    kl_q_m = F.kl_div(
        F.log_softmax(target, dim=1),
        F.softmax(M, dim=1),
        reduction='mean'
    )
    return 0.5 * (kl_p_m + kl_q_m)


def triplet_loss(anchor, positive, negative, margin=1):
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    losses = F.relu(pos_dist - neg_dist + margin)
    return losses.mean()


def downsample_2x(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def l1_loss(target, original):
    loss_fn = torch.nn.L1Loss(reduction='mean')
    loss = loss_fn(target, original)
    return loss


def l2_loss(target, original):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(target, original)
    return loss


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.L1Loss(reduction='mean')
    loss = loss_fn(ll_input, gt_input)
    return loss


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net1, net2, net3, optim1=None, optim2=None, optim3=None):
    state_dicts = torch.load(name)
    network_state_dict1 = {k: v for k, v in state_dicts['net1'].items() if 'tmp_var' not in k}
    network_state_dict2 = {k: v for k, v in state_dicts['net2'].items() if 'tmp_var' not in k}
    network_state_dict3 = {k: v for k, v in state_dicts['net3'].items() if 'tmp_var' not in k}  
    net1.module.load_state_dict(network_state_dict1)
    net2.module.load_state_dict(network_state_dict2)
    net3.module.load_state_dict(network_state_dict3)
    if optim1 is not None:
        optim1.load_state_dict(state_dicts['opt1'])
    if optim2 is not None:
        optim2.load_state_dict(state_dicts['opt2'])
    if optim3 is not None:
        optim3.load_state_dict(state_dicts['opt3'])


def load_pretrain(path, net1, net2, net3):
    state_dicts = torch.load(path)
    network_state_dict1 = {k: v for k, v in state_dicts['net1'].items() if 'tmp_var' not in k}
    network_state_dict2 = {k: v for k, v in state_dicts['net2'].items() if 'tmp_var' not in k} 
    network_state_dict3 = {k: v for k, v in state_dicts['net3'].items() if 'tmp_var' not in k} 
    net1.module.load_state_dict(network_state_dict1)
    net2.module.load_state_dict(network_state_dict2)
    net3.module.load_state_dict(network_state_dict3)
        

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = cfg.port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device("cuda", rank)
    net1 = Model_hiding(device, secret_num=cfg.num_hiding)
    net2 = Model_authen(device, cond_num=3)
    net3 = Model_lock(device)
    net1.to(device)
    net2.to(device)
    net3.to(device)
    init_model(net1)
    net1 = nn.parallel.DistributedDataParallel(net1, device_ids=[torch.cuda.current_device()])
    net2 = nn.parallel.DistributedDataParallel(net2, device_ids=[torch.cuda.current_device()])
    net3 = nn.parallel.DistributedDataParallel(net3, device_ids=[torch.cuda.current_device()])
    para1 = get_parameter_number(net1)
    para2 = get_parameter_number(net2)
    para3 = get_parameter_number(net3)
    if rank == 0:
        print(para1)
        print(para2)
        print(para3)
    params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
    params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
    params_trainable3 = (list(filter(lambda p: p.requires_grad, net3.parameters())))
    optim1 = torch.optim.Adam(params_trainable1, lr=cfg.lr_hiding, betas=cfg.betas, weight_decay=cfg.weight_decay)
    optim2 = torch.optim.Adam(params_trainable2, lr=cfg.lr_authen, betas=cfg.betas, weight_decay=cfg.weight_decay)
    optim3 = torch.optim.Adam(params_trainable3, lr=cfg.lr_gen, betas=cfg.betas, weight_decay=cfg.weight_decay)
    weight_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=cfg.epochs)
    weight_scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=cfg.epochs)
    weight_scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optim3, T_max=cfg.epochs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=cfg.shuffle_val)

    trainloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
        sampler=train_sampler
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=cfg.batchsize_val,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
        sampler=val_sampler
    )

    if cfg.train_next:
        load(f"{cfg.model_path}{cfg.suffix_load}_{cfg.num_hiding}.pt", net1, net2, net3, optim1, optim2, optim3)

    if cfg.pretrain:
        load_pretrain(f"{cfg.pretrain_path}{cfg.suffix_pretrain}.pt", net1, net2, net3)

    for i_epoch in range(cfg.trained_epoch+1, cfg.epochs+1):
        trainloader.sampler.set_epoch(i_epoch)
        loss_total = []
        net1.train()
        net2.train()
        net3.train()
        for i_batch, data in enumerate(trainloader):
            data = data.to(device)
            batch = data.shape[0]
            portion = cfg.num_hiding + 1
            secrets = []
            secrets_down = []
            locks = []
            secrets_low = []
            cover = data[:batch // portion]
            cover_lock = net3(cover, mode='cover_lock')
            for i in range(cfg.num_hiding):
                secret = data[(i+1) * batch // portion: (i+2) * batch // portion]
                secret_down = downsample_2x(secret)
                secret_lock = net3(secret_down, mode='secret_lock')
                secrets.append(secret)
                secrets_down.append(secret_down)
                lock = 0.5 * cover_lock + 0.5 * secret_lock
                locks.append(lock)
            for secret, lock in zip(secrets, locks):
                low, high = net2(secret, lock)
                secrets_low.append(low)
            stego, cover_low, stego_low, r_o = net1(cover, secrets_low)
            
            reveals = []
            keys = []
            fake_reveals = []
            cover_key = net3(stego, mode='cover_key')
            reveals_low, r_p = net1(stego, cfg.num_hiding, rev=True)
            for reveal_low in reveals_low:
                secret_key = net3(reveal_low, mode='secret_key')
                key = 0.5 * cover_key + 0.5 * secret_key
                keys.append(key)
                reveal = net2(reveal_low, key, rev=True)
                reveals.append(reveal)
            for i in range(cfg.num_hiding):
                fake_key = keys[(i+1) % cfg.num_hiding]
                fake_reveal = net2(reveals_low[i], fake_key, rev=True)
                fake_reveals.append(fake_reveal)

            reveal_loss = 0
            js_loss = 0
            reveal_low_loss = 0
            key_loss = 0
            trip_loss = 0
            stego_low_loss = low_frequency_loss(stego_low, cover_low)
            stego_loss = l2_loss(stego, cover)
            for s, r in zip(secrets, reveals):
                reveal_loss += l2_loss(r, s)
                js_loss += js_divergence_loss(r, s)
            for sl, rl in zip(secrets_low, reveals_low):
                reveal_low_loss += l1_loss(rl, sl)
            for l, k in zip(locks, keys):
                key_loss += l1_loss(k, l)
            redunt_loss = l2_loss(r_p, r_o)
            for s, r, fr in zip(secrets, reveals, fake_reveals):
                trip_loss += triplet_loss(s, r, fr)
            total_loss = 2 * stego_loss + 3 * reveal_loss + redunt_loss + 4 * reveal_low_loss + 2 * stego_low_loss + key_loss + trip_loss + js_loss
            total_loss.backward()

            optim1.step()
            optim2.step()
            optim3.step()

            optim1.zero_grad()
            optim2.zero_grad()
            optim3.zero_grad()
            loss_total.append(total_loss.item())

        epoch_losses = np.mean(np.array(loss_total))
        epoch_losses = torch.tensor([epoch_losses.item()]).cuda(rank)
        dist.all_reduce(epoch_losses)
        mean_epoch_loss = epoch_losses.item() / dist.get_world_size()
        if rank == 0:
            print(f"[{i_epoch}/{cfg.epochs}], total: {mean_epoch_loss}")

        if i_epoch % cfg.val_freq == 0:
            with torch.no_grad():
                psnr_s = [[] for _ in range(cfg.num_hiding)]
                psnr_c = []
                psnr_fake = [[] for _ in range(cfg.num_hiding)]
                net1.eval()
                net2.eval()
                net3.eval()
                for x in valloader:
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
                    fake_reveals = []
                    cover_key = net3(stego, mode='cover_key')
                    reveals_low, r_p = net1(stego, cfg.num_hiding, rev=True)
                    for reveal_low in reveals_low:
                        secret_key = net3(reveal_low, mode='secret_key')
                        key = 0.5 * cover_key + 0.5 * secret_key
                        reveal = net2(reveal_low, key, rev=True)
                        reveals.append(reveal)
                        if i_epoch % 3 == 0:
                            fake_key = torch.randn_like(key).to(device)
                        elif i_epoch % 3 == 1:
                            fake_key = torch.zeros_like(key).to(device)
                        else:
                            fake_key = torch.ones_like(key).to(device)
                        fake_reveal = net2(reveal_low, fake_key, rev=True)
                        fake_reveals.append(fake_reveal)

                    reveals_255 = []
                    secrets_255 = []
                    fake_reveals_255 = []
                    for s, r, fr in zip(secrets, reveals, fake_reveals):
                        reveals_255.append(r.cpu().numpy().squeeze() * 255)
                        secrets_255.append(s.cpu().numpy().squeeze() * 255)
                        fake_reveals_255.append(fr.cpu().numpy().squeeze() * 255)

                    cover_255 = cover.cpu().numpy().squeeze() * 255
                    stego_255 = stego.cpu().numpy().squeeze() * 255

                    for i, (s, r) in enumerate(zip(secrets_255, reveals_255)):
                        psnr_temp_s = computePSNR(r, s)
                        psnr_s[i].append(psnr_temp_s)

                    for i, (s, f) in enumerate(zip(secrets_255, fake_reveals_255)):
                        psnr_temp_fake = computePSNR(f, s)
                        psnr_fake[i].append(psnr_temp_fake)

                    psnr_temp_c = computePSNR(stego_255, cover_255)
                    psnr_c.append(psnr_temp_c)

                psnr_s_all = []
                psnr_fake_all = []
                for i in range(cfg.num_hiding):
                    psnr_s_tmp = torch.tensor([np.mean(np.array(psnr_s[i])).item()]).cuda(rank)
                    dist.all_reduce(psnr_s_tmp)
                    mean_s = psnr_s_tmp.item() / dist.get_world_size()
                    psnr_s_all.append(mean_s)

                for i in range(cfg.num_hiding):
                    psnr_fake_tmp = torch.tensor([np.mean(np.array(psnr_fake[i])).item()]).cuda(rank)
                    dist.all_reduce(psnr_fake_tmp)
                    mean_fake = psnr_fake_tmp.item() / dist.get_world_size()
                    psnr_fake_all.append(mean_fake)

                psnr_c = torch.tensor([np.mean(np.array(psnr_c)).item()]).cuda(rank)
                dist.all_reduce(psnr_c)
                mean_c = psnr_c.item() / dist.get_world_size()
                if rank == 0:
                    log_text = f"[validation] psnr_c: {mean_c} "
                    for i in range(cfg.num_hiding):
                        log_text += f"psnr_s{i+1}: {psnr_s_all[i]} "
                    log_text += "\n"
                    for i in range(cfg.num_hiding):
                        log_text += f"psnr_fake{i+1}: {psnr_fake_all[i]} "
                    print(log_text)

        if i_epoch > 0 and (i_epoch % cfg.save_freq) == 0 and rank == 0:
            torch.save({'opt1': optim1.state_dict(),
                        'net1': net1.module.state_dict(),
                        'opt2': optim2.state_dict(),
                        'net2': net2.module.state_dict(),
                        'opt3': optim3.state_dict(),
                        'net3': net3.module.state_dict()}, f"{cfg.model_path}checkpoint_{cfg.dataset_train_mode}_{i_epoch:05d}_{cfg.num_hiding}.pt")
        weight_scheduler1.step()
        weight_scheduler2.step()
        weight_scheduler3.step()
    if dist.get_rank() == 0:
        torch.save({'net1': net1.module.state_dict(),
                    'net2': net2.module.state_dict(),
                    'net3': net3.module.state_dict()}, f"{cfg.model_path}model_{cfg.dataset_train_mode}_{cfg.num_hiding}.pt")

    dist.barrier()
    cleanup()


if __name__ == "__main__":
    world_size = cfg.num_gpus
    os.makedirs(cfg.model_path, exist_ok=True)
    mp.spawn(main_worker, args=(world_size, ), nprocs=world_size, join=True)
