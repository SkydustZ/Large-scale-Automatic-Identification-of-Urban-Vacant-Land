# -*- coding:utf-8 -*-


import os
import time
import logging
from datetime import datetime
import matplotlib
from PIL import Image
import numpy as np
import random
import platform
from sklearn.metrics import precision_recall_fscore_support, classification_report

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler

matplotlib.use('Agg')
np.set_printoptions(threshold=1e100)

def read_images(train=True, path='./'):
    txt_fname = path + 'data_process/new_dataset/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(path + 'data_process/new_dataset/'+'image', i+'.jpg') for i in images]
    label = [os.path.join(path + 'data_process/new_dataset/'+'label', i+'.jpg') for i in images]
    return data, label


class RemoteSensingDataset(data.Dataset):

    def __init__(self, train, transforms, path):
        self.transforms = transforms
        self.data_list, self.label_list = read_images(train=train, path=path)
        print('Read ' + str(len(self.data_list)) + ' images')
        
    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        image_fn = os.path.split(self.data_list[idx])[1]
        img = Image.open(img)
        label = Image.open(label)
        img, label = self.transforms(img, label)
        label[label > 0.5] = 1
        label[label <= 0.5] = 0
        label = label.long().squeeze()
        return img, label, image_fn
    
    def __len__(self):
        return len(self.data_list)

    def get_label_info(self, idx):
        image_fn = os.path.split(self.data_list[idx])[1]
        # image = Image.open(self.image_paths[idx])
        label = Image.open(self.label_list[idx])
        return image_fn, np.array(label).max() < 100  # gen label_info.csv


def get_data_loader(img_transforms, path='./', batch_size=4):
    train_dataset = RemoteSensingDataset(True, img_transforms, path)
    val_dataset = RemoteSensingDataset(False, img_transforms, path)

    # ref: http://spytensor.com/index.php/archives/45/, https://discuss.pytorch.org/t/using-weightedrandomsampler-for-an-imbalanced-classes/74571
    train_sampler = WeightedRandomSampler(get_label_weights(img_transforms), len(train_dataset))

    num_workers = 1 if 'Windows' in platform.platform() else 4
    train_data_loader = data.DataLoader(train_dataset,  batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    val_data_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data_loader, val_data_loader


def transform_train(image, label, output_size=(224,224)):
    """performs identical transform in both image & label
        ref: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7?u=zhou-yucheng
    """
    # Random Resized Crop
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.75, 1.0), ratio=(0.75, 1.33))
    image = TF.resized_crop(image, i, j, h, w, output_size)
    label = TF.resized_crop(label, i, j, h, w, output_size)

    # # Random crop
    # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=output_size)
    # image = TF.crop(image, i, j, h, w)
    # label = TF.crop(label, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)

    # Transform to tensor
    image = TF.to_tensor(image)
    label = TF.to_tensor(label)

    # ImageNet: mean, std: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image, label

def transform_val(image, label, output_size=(224,224)):
    """performs identical transform in both image & label
        ref: https://discuss.pytorch.org/t/torchvision-transfors-how-to-perform-identical-transform-on-both-image-and-target/10606/7?u=zhou-yucheng
    """
    # Resize
    image = TF.center_crop(image, output_size)
    label = TF.center_crop(label, output_size)

    # Transform to tensor
    image = TF.to_tensor(image)
    label = TF.to_tensor(label)

    image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    return image, label

def get_label_weights(img_transforms, data_dir_='./', w1=0.1):
    label_weights_path = os.path.join(data_dir_+'data_process\\new_dataset', 'label_weights.npy')
    if os.path.exists(label_weights_path):
        weights = np.load(label_weights_path)
        return weights

    dataset = RemoteSensingDataset(True, img_transforms, data_dir_)
    N = len(dataset)
    weights = np.ones(N)
    for i in range(N):
        image_fn, is_all_background = dataset.get_label_info(i)
        if is_all_background:
            weights[i] = w1
    np.save(label_weights_path, weights)

    return weights


def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


class NNHistory:
    def __init__(self):
        """包含 loss, acc, n_iter的dict，用于记录NN train/val 过程中的数据
        n_iter等于batch_size，在计算loss/acc加权平均时使用；默认为全1，即不考虑加权
        """
        self.data = {'loss': [], 'acc': [], 'n_iter': []}

    def append(self, loss, acc, n_iter=1):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if isinstance(acc, torch.Tensor):
            acc = acc.item()

        self.data['loss'].append(loss)
        self.data['acc'].append(acc)
        self.data['n_iter'].append(n_iter)

    def avg(self):
        loss = np.array(self.data['loss'])
        acc = np.array(self.data['acc'])
        n_iter = np.array(self.data['n_iter'])

        n_sum = float(np.sum(n_iter))

        loss_avg = np.sum(loss * n_iter) / n_sum  # dot product, 点乘
        acc_avg = np.sum(acc * n_iter) / n_sum

        return loss_avg, acc_avg

    def last(self):
        return self.data['loss'][-1], self.data['acc'][-1]


class NNFullHistory:
    def __init__(self):
        """储存所有pred & label记录，用于后续复杂计算，例如precision/recall/f1，confusion matrix"""
        self.predss = []  # list of nd-array (of int): [[],[],...]
        self.labelss = []

        self.losses = []
        self.counts = []

    def all_preds(self):
        _all_preds = []
        for preds in self.predss:
            _all_preds.extend(preds)
        return np.array(_all_preds)

    def all_labels(self):
        _all_labels = []
        for labels in self.labelss:
            _all_labels.extend(labels)
        return np.array(_all_labels)

    def append(self, loss, preds, labels):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if isinstance(preds, torch.Tensor):
            preds = preds.view(-1).detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.view(-1).detach().cpu().numpy()

        assert preds.size == labels.size
        count = len(labels)

        self.predss.append(preds)
        self.labelss.append(labels)

        self.losses.append(loss)
        self.counts.append(count)

    def avg_accuracy(self):
        all_preds = self.all_preds()
        all_labels = self.all_labels()
        return np.sum(all_preds == all_labels) / all_preds.size

    def avg_loss(self):
        losses = np.array(self.losses)
        counts = np.array(self.counts)

        return (losses * counts).sum() / counts.sum()  # dot-product for weighted

    def avg_prf1_binary(self, neg_label=0):
        """preds和labels中int的种类为分类的类数，通常大于2；
        可以指定其中一种类别为negative，其他全部算positive，从而得到precision_recall_f1
        ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html"""
        all_preds = self.all_preds()
        all_labels = self.all_labels()

        pos_label = neg_label + 1
        all_preds[all_preds != neg_label] = pos_label
        all_labels[all_labels != neg_label] = pos_label

        if all(all_labels == neg_label):  # all negative label, avoid warning
            return -1, -1, -1

        p, r, f1, s = precision_recall_fscore_support(all_labels, all_preds, beta=2, pos_label=pos_label, average='binary')
        return p, r, f1

    def avg_prf1_all(self, output_dict=True, label_tags=None):
        """ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html"""
        all_preds = self.all_preds()
        all_labels = self.all_labels()
        return classification_report(all_labels, all_preds, target_names=label_tags, output_dict=output_dict)

    def avg_prf1_weight(self):
        dw = self.avg_prf1_all(True)['weighted avg']

        return dw['precision'], dw['recall'], dw['f1-score']


class NNF1History:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.counts = []
        self.losses = []

    def append(self, loss, preds, labels):
        """preds, labels are Tensor contains 0/1 only, represent negative/positive"""
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        self.losses.append(loss)

        labels = labels.view(-1)
        preds = preds.view(-1)
        assert preds.shape == labels.shape and len(preds) == len(labels)
        count = len(labels)
        self.counts.append(count)

        x = preds + 2 * labels
        # true/false negative/positive
        tp = torch.sum(x == 3).item()
        tn = torch.sum(x == 0).item()
        fp = torch.sum(x == 1).item()
        fn = torch.sum(x == 2).item()
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

        assert count == tp + tn + fp + fn

    def avg_accuracy(self):
        return (self.tp + self.tn) / sum(self.counts)

    def avg_loss(self):
        losses = np.array(self.losses)
        counts = np.array(self.counts)

        return (losses * counts).sum() / counts.sum()  # dot-product for weighted

    def avg_prf1(self, beta=2):
        """use default F2-score, beta = 2
            ref: https://en.wikipedia.org/wiki/F1_score
                https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9"""
        if self.tp == 0 or self.tn == 0:
            return -1, -1, -1

        p = self.tp / (self.tp + self.fp)
        r = self.tp / (self.tp + self.fn)

        b2 = beta ** 2
        f1 = (1 + b2) * p * r / (b2 * p + r)

        return p, r, f1

    def avg_iou(self):
        """ref: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/"""
        if self.tp == 0:
            return 0
        return self.tp / (self.tp + self.fp + self.fn)

    # def avg_prf1_binary(self, beta=1):
    #     """for compatibility?"""
    #     return self.avg_prf1(beta)


class Logger:
    def __init__(self, file_dir='./logs/', file_name=None, init_mode='a+', print_log_level=1):
        if not file_name:
            file_name = f"log_{datetime.now().strftime('%m-%d')}.log"
        self.file_path = os.path.join(file_dir, file_name)
        self.print_log_level = print_log_level

        with open(self.file_path, init_mode) as f:
            if 'a' in init_mode:
                f.write('\n')
            f.write('[{}]\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def log(self, msg, end='\n', level=1, print_log=None, use_warning=False):
        """
        :param msg:
        :param end:
        :param level:       higher (importance) level, more likely to be print
        :param print_log:   None means auto, set True/False to control print
        :param use_warning: use warning instead of print
        """
        if print_log is not None:
            if print_log:  # True
                level = self.print_log_level + 10
            else:  # False
                level = self.print_log_level - 10

        if level >= self.print_log_level:  # print
            if use_warning:
                # warnings.formatwarning = custom_formatwarning
                # warnings.warn(msg)
                logging.warning(msg)
            else:
                print(msg, end=end)

        self._write_log(f'{msg}{end}')  # log file

    def _write_log(self, msg):
        with open(self.file_path, 'a+') as f:
            f.write(msg)


def get_elapsed_time(start_time):
    dt = time.time() - start_time
    if dt < 1:
        str_ = '{:.4f} s'.format(dt)
    elif dt < 60:
        str_ = '{:.2f} s'.format(dt)
    else:
        str_ = '{:.1f} min'.format(dt / 60.0)
    return str_
    


    