# -*- coding:utf-8 -*-
import matplotlib
import argparse
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
from matplotlib.pyplot import savefig
import shutil
from tqdm import tqdm
import socket
import random
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
np.seterr(divide='ignore', invalid='ignore')
from glob import glob
import torch.nn as nn
from model.deeplabv3.model.deeplabv3 import DeepLabV3
# from data_process.utils import RemoteSensingDataset, transform_train, transform_val, add_weight_decay
from data_process.cut_img_train import cut_image_train
from data_process.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
SEED = 2020
if SEED >= 0:
    # seed_everything
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train():
    """ train model in an epoch """
    train_data = tqdm(train_data_loader, position=0, leave=True)
    net.train()
    history = NNF1History()
    for data in train_data:
        images = data[0].to(device)
        labels = data[1].to(device)
        if len(images) <= 1:
            continue
        # forward
        optimizer.zero_grad()
        outputs = net(images)  # [bs,2,w,h]
        labels = labels.view(-1)
        outputs = torch.stack((outputs[:, 0, :, :].reshape(-1), outputs[:, 1, :, :].reshape(-1)), dim=1)  # [bs*w*h,2]
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        history.append(loss, preds, labels)
        loss.backward()
        optimizer.step()

        # visualize_preds(images, labels, preds, image_fns, comment='train')
        p, r, f1 = history.avg_prf1()
        iou = history.avg_iou()
        train_data.set_postfix({'loss': loss.item(), 'f2': f1, 'iou': iou, 'acc': history.avg_accuracy()})

    return history

def evaluate():
    val_data = tqdm(val_data_loader, position=0, leave=True)
    net.eval()
    with torch.no_grad():
        history = NNF1History()
        for data in val_data:
            images = data[0].cuda()
            labels = data[1].cuda()
            outputs = net(images)  # [bs,2,w,h]
            labels = labels.view(-1)
            outputs = torch.stack((outputs[:, 0, :, :].reshape(-1), outputs[:, 1, :, :].reshape(-1)), dim=1)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            history.append(loss, preds, labels)

            # visualize_preds(images, labels, preds, image_fns, comment='val')
            p, r, f1 = history.avg_prf1()
            iou = history.avg_iou()
            val_data.set_postfix({'loss': loss.item(), 'f2': f1, 'iou': iou, 'acc': history.avg_accuracy()})

    return history


def log(msg, end='\n'):
    # import-friendly (True for import)
    is_print = True

    if is_print:
        print(msg, end=end)
    with open(f'./logs/Log.log', 'a+') as f_:
        f_.write(msg)
        f_.write(end)


def log_and_save():
    global best_val_loss, best_val_iou, best_val_f1, best_epoch_loss, best_epoch_iou, best_epoch_f1

    train_loss, val_loss = train_history.avg_loss(), val_history.avg_loss()
    train_iou = train_history.avg_iou()
    val_iou = val_history.avg_iou()
    train_p, train_r, train_f1 = train_history.avg_prf1()
    val_p, val_r, val_f1 = val_history.avg_prf1()
    train_acc = train_history.avg_accuracy()
    val_acc = val_history.avg_accuracy()

    # ===Print& Log
    log(
        f"Train:\tLoss={train_loss:.3f}, IoU={train_iou:.3f}, F/P/R={train_f1:.3f}/{train_p:.3f}/{train_r:.3f}, Acc={train_acc:.3f}")
    if val_iou > val_iou:
        log("*", end='')
    log(f"Val:\tLoss={val_loss:.3f}, IoU={val_iou:.3f}, F/P/R={val_f1:.3f}/{val_p:.3f}/{val_r:.3f}, Acc={val_acc:.3f}")

    # torch.save(net.state_dict(), './model/DeeplabV3.pth')
    # ===Save
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch_f1 = epoch
        # torch.save(net.state_dict(), './model/DeeplabV3_best_f1.pth')
        torch.save(net.state_dict(), f'./model/{model_fullname} -f2 {val_f1:.3f} -iou {val_iou:.3f} -loss {val_loss:.3f} -ep {epoch:02d}.pth')
        for path in sorted(glob(f'./model/{model_fullname} -f2*.pth'))[:-1]:
            os.remove(path)

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_epoch_iou = epoch
        # torch.save(net.state_dict(), './model/DeeplabV3_best_iou.pth')
        # torch.save(net.state_dict(), f'./model/{model_fullname} -iou {val_iou:.3f} -f2 {val_f1:.3f} -loss {val_loss:.3f} -ep {epoch:02d}.pth')
        # for path in sorted(glob(f'./model/{model_fullname} -iou*.pth'))[:-1]:
        #     os.remove(path)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch_loss = epoch
        # torch.save(net.state_dict(), './model/DeeplabV3_best_loss.pth')
        # torch.save(net.state_dict(), f'./model/{model_fullname} -loss {val_loss:.3f} -f2 {val_f1:.3f} -iou {val_iou:.3f} -ep {epoch:02d}.pth')
        # for path in sorted(glob(f'./model/{model_fullname} -loss*.pth'))[1:]:
        #     os.remove(path)

    # ===Writer
    if epoch == 1:
        fwriter.write('Epoch, Train Loss, Train IoU, Train F2, Valid Loss, Valid IoU, Val F2, lr\n')
    fwriter.write(f'{epoch}, {train_loss}, {train_iou}, {train_f1}, {val_loss}, {val_iou}, {val_f1}, {lr_}\n')

    # swriter.add_scalar('Train/Loss', train_loss, epoch)
    # swriter.add_scalar('Train/IoU', train_iou, epoch)
    # swriter.add_scalar('Train/F2', train_f1, epoch)
    # swriter.add_scalar('Valid/Loss', val_loss, epoch)
    # swriter.add_scalar('Valid/IoU', val_iou, epoch)
    # swriter.add_scalar('Valid/F2', val_f1, epoch)

    # ===Print result, Close writer
    if epoch == args.epochs:  # the last epoch
        log(f'\n*Best Valid: Loss={best_val_loss:.3f} at epoch {best_epoch_loss}. IoU={best_val_iou:.3f} at epoch {best_epoch_iou}. F2={best_val_f1:.3f} at epoch {best_epoch_f1}')
        fwriter.write('\n#Time cost, ' + get_elapsed_time(start_time))
        fwriter.close()
        # swriter.close()

def get_args():
    parser = argparse.ArgumentParser('DL Project')
    parser.add_argument('-c', '--city', type=str, default='BJ')
    parser.add_argument('-m', '--resnet', type=str, default='ResNet18_OS8')
    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-b', '--batch_size', type=int, default=8)   # depends on the GPU memory, the larger the better
    parser.add_argument('-l', '--lr', type=float, default=2e-4)
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('--verbose', type=int, default=2, help='verbose level, > 0 means True')
    parser.add_argument('--fp16', action='store_true', help="whether to use 16-bit float")
    # about fp16: https://zhpmatrix.github.io/2019/07/01/model-mix-precision-acceleration/
    args_ = parser.parse_args()

    return args_

if __name__=='__main__':
    # ========================================================================================== Params & Model
    args = get_args()
    path = './'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # cut image for training
    Image.MAX_IMAGE_PIXELS = int(5000000000)
    cut_image_train(args.city, path=path)

    # load data
    train_dataset = RemoteSensingDataset(True, transform_train, path)  # with data augmentation
    # train_dataset = RemoteSensingDataset(True, transform_val, path)  # no data augmentation
    val_dataset = RemoteSensingDataset(False, transform_val, path)
    train_data_loader = data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    val_data_loader = data.DataLoader(val_dataset, args.batch_size, num_workers=4)

    # model
    id = 0
    model_fullname = f'DeeplabV3_{args.resnet}_{args.city}_{id} -lr {args.lr} -bs {args.batch_size}'
    net = DeepLabV3(args.resnet, path)
    net = net.to(device)
    # ========================================================================================== Train

    # weighted CrossEntropyLoss
    if args.city == 'BJ':
        weight = torch.tensor([0.18, 0.82]).to(device)  # BJ
    if args.city == 'SZ':
        weight = torch.tensor([0.12, 0.88]).to(device)  # SZ
    if args.city == 'LZ':
        weight = torch.tensor([0.077, 0.923]).to(device)  # LZ
    if args.city == 'Hybrid':
        weight = torch.tensor([0.14, 0.86]).to(device)  # Hybrid

    criterion = nn.CrossEntropyLoss(weight=weight)  # with weighted loss
    # criterion = nn.CrossEntropyLoss()  # no weighted loss

    # optimizer and scheduler
    params = add_weight_decay(net, l2_value=0.0001)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
        mode='min',
        factor=0.5,
        patience=3,  # with scheduler
        # patience=args.epochs,  # no scheduler
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-6,
        eps=1e-08)

    # logger initialization
    dt_now = datetime.now().strftime('%m%d-%H%M')
    fwriter = open(f'./logs/csv/{model_fullname} [{dt_now}].csv', 'a+')
    # swriter = SummaryWriter(log_dir=f'./logs/runs/{model_fullname} [{dt_now}] {socket.gethostname()}')
    log(f'[{model_fullname}]')
    start_time = time.time()
    best_val_loss, best_val_iou, best_val_f1 = 1e6, 0, 0
    best_epoch_loss, best_epoch_iou, best_epoch_f1 = -1, -1, -1

    # train & eval, log & save
    try:
        for epoch in range(1, args.epochs + 1):
            lr_ = optimizer.param_groups[0]['lr']
            log(f"=== Epoch: {epoch}/{args.epochs}\t(lr: {lr_})")
            train_history = train()
            val_history = evaluate()
            valid_loss = val_history.avg_loss()
            log_and_save()
            scheduler.step(valid_loss)
    except KeyboardInterrupt:
        if epoch > 1:  # at least finished the first epoch
            print('\n=== KeyboardInterrupt, training early stop at epoch: {}/{}\n'.format(epoch, args.epochs))
            epoch = args.epochs
            log_and_save()
        else:
            print('\n=== KeyboardInterrupt, exit\n')

    log('\n#Time cost: ' + get_elapsed_time(start_time))
    

