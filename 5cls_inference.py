import argparse
import warnings
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
import datetime
import utliz
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


def cal_auc_multi_class(label, pred):
    # label of size: N
    # pred of size: NxC
    if type(label) is not np.ndarray:
        label = label.detach().cpu().numpy()
    if type(pred) is not np.ndarray:
        pred = pred.detach().cpu().numpy()

    cate = np.unique(label)
    auc_all = []
    for i in range(len(cate)):
        label_binary_i = np.zeros_like(label)
        pred_binary_i = np.zeros([pred.shape[0]])
        label_binary_i[label==cate[i]] = 1
        pred_binary_i = pred[:, i]
        auc_i = utliz.cal_auc(label_binary_i, pred_binary_i, pos_label=1)
        auc_all.append(auc_i)
    return auc_all


def train_one_epoch(epoch, model, loader, optimizer, dev, writer):
    model = model.train()
    XE = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 10, 10, 20]).to(dev))
    patch_label_gt = torch.zeros([loader.dataset.__len__()]).long()
    pred_prob_all = torch.zeros([loader.dataset.__len__(), 5])
    # for iter, (data, label, selected) in enumerate(tqdm(loader, ascii=True, desc='training epoch_{}'.format(epoch))):
    for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Training Epoch {}'.format(epoch))):
        optimizer.zero_grad()
        niter = epoch * len(loader) + iter
        patch_label_gt[selected] = label
        data = data.to(dev)
        final = model(data)
        final_prob = torch.softmax(final, dim=1)
        loss = XE(final, label.to(dev))
        loss.backward()
        optimizer.step()
        # log
        pred_prob_all[selected, :] = final_prob.detach().cpu()
        writer.add_scalar('train_loss', loss.item(), niter)

    pred_logit = pred_prob_all.argmax(1)
    train_acc = torch.sum(pred_logit == patch_label_gt)/patch_label_gt.shape[0]
    writer.add_scalar('train_patch_acc', train_acc, epoch)
    print('Epoch:{} train_patch_acc: {}'.format(epoch, train_acc))

    auc_all_cate = cal_auc_multi_class(patch_label_gt, pred_prob_all)
    for class_id, auc_class_i in enumerate(auc_all_cate):
        writer.add_scalar('train_patch_auc_class{}'.format(class_id), auc_class_i, epoch)

    return 0


def eval(model, loader, dev):
    model = model.eval()
    patch_label_gt = torch.zeros([loader.dataset.__len__()]).long()
    pred_prob_all = torch.zeros([loader.dataset.__len__(), 5])
    for iter, (data, label, selected) in enumerate(loader):
        patch_label_gt[selected] = label
        data = data.to(dev)
        with torch.no_grad():
            final = model(data)
            final_prob = torch.softmax(final, dim=1)
        pred_prob_all[selected, :] = final_prob.detach().cpu()

    pred_logit = pred_prob_all.argmax(1)

    np.save("/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1/TCGA_kirc/TCGA_kirc_pred.npy",
    pred_logit.detach().cpu().numpy())
    csv_path = "/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1/TCGA_kirc/TCGA_kirc_pred.csv" 
    df = pd.DataFrame(pred_logit.detach().cpu().numpy(), columns=['pred_label'])
    df.to_csv(csv_path, index=False)
    
    print(f"Predictions saved to {csv_path} (numpy) and {csv_path} (CSV)")
    return pred_logit


def save_ckpt(model, epoch, save_name):
    state = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    torch.save(state, save_name)
    return 0


def load_ckpt(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['model'])
    epoch = state['epoch']
    return model, epoch


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=200, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64',choices=['f64','f32'], type=str, help='SK-algo dtype (default: f64)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=6, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='ExternalTest_addDropOut', type=str, help='name for tensorboardX')
    parser.add_argument('--resume', default='', type=str, help='resume ckpt path')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--downsample', default=0.1, type=float, help='downsample the whole dataset')

    return parser.parse_args()


class prediction_head(nn.Module):
    def __init__(self):
        super(prediction_head, self).__init__()
        self.fcs = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(1536, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(2048, 5),
        )

    def forward(self, x):
        return self.fcs(x)


class Region_5Classes_Feat_filterNan(torch.utils.data.Dataset):
    def __init__(self):

        # center_dir = '/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1/TCGA_kirp'
        # center_dir = '/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1/TCGA_kich'
        center_dir = '/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1/TCGA_kirc'
        all_patches_file = np.load(os.path.join(center_dir, "all_patch_feat.npy"))
        all_patches_fileName = np.load(os.path.join(center_dir, "all_patch_fileName.npy"))

        t_ = all_patches_file.max(axis=1)
        idx_nan = np.isnan(t_)
        print("remove {} nan feat vector".format(idx_nan.sum()))
        all_patches_file = all_patches_file[~idx_nan]
        all_patches_fileName = all_patches_fileName[~idx_nan]

        self.all_patches_file = all_patches_file
        self.all_patches_fileName = all_patches_fileName

    def __getitem__(self, index):
        img = self.all_patches_file[index]
        label = -1
        return img, label, index

    def __len__(self):
        return len(self.all_patches_file)


if __name__ == '__main__':
    args = get_parser()

    Val_ds = Region_5Classes_Feat_filterNan()
    Val_loader = torch.utils.data.DataLoader(Val_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

    dev = 'cuda:0'
    model = prediction_head()
    model.to(dev)

    start_epoch = 0
    if args.resume != '':
        model, start_epoch = load_ckpt(model, args.resume)
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))

    eval(model=model, loader=Val_loader, dev=dev)
