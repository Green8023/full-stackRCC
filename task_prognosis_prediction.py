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
from dataset import get_train_test_ds_MultiCenter_region_trainwithTCGA, TumorRegion_Prognosis_Feat_yuHouFenZu
import datetime
import utliz
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from util.losses import *
from util.utils import *
import math
from sksurv.metrics import concordance_index_censored
import torchtuples as tt
from collections import defaultdict

def statistics_dataset(labels):
    # labels of sihape N
    num_samples = labels.shape[0]
    all_cate = np.unique(labels)
    for i in range(all_cate.shape[0]):
        num_samples_cls_i = np.sum(labels == all_cate[i])
        print("class {}: {}/{} samples, ratio:{:.4f}".format(all_cate[i], num_samples_cls_i, num_samples, num_samples_cls_i/num_samples))
    return 0

def cal_auc_multi_class(label, pred):
    # label of size: N
    # pred of size: NxC
    if type(label) is not np.ndarray:
        label = label.detach().cpu().numpy()
    if type(pred) is not np.ndarray:
        pred = pred.detach().cpu().numpy()

    num_classes = pred.shape[1]
    auc_all = []
    for i in range(num_classes):
        label_binary_i = np.zeros_like(label)
        pred_binary_i = np.zeros([pred.shape[0]])
        label_binary_i[label == i] = 1
        pred_binary_i = pred[:, i]
        if label_binary_i.sum() != 0:
            auc_i = utliz.cal_auc(label_binary_i, pred_binary_i, pos_label=1)
            if np.isnan(auc_i):
                auc_i = 0
        else:
            auc_i = 0
        auc_all.append(auc_i)
    return auc_all

def train_one_epoch(epoch, model, loader, optimizer, dev, writer):
    model = model.train()
    XE = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).to(dev))
    slide_label_gt = torch.zeros([loader.dataset.__len__()]).long()
    pred_prob_all = torch.zeros([loader.dataset.__len__(), 2])
    save_info = []
    for iter, (data, label, label_patho, label_isup, label_size, label_tnm, slide_yuHouFenZu, selected) in enumerate(tqdm(loader, desc='Training Epoch {}'.format(epoch))):
        if data.shape[1] == 1:
            continue
        optimizer.zero_grad()
        niter = epoch * len(loader) + iter
        slide_label_gt[selected] = slide_yuHouFenZu
        data = data.to(dev)
        final = model(data)
        final_prob = torch.softmax(final, dim=1)
        loss = XE(final, slide_yuHouFenZu.to(dev))
        loss.backward()
        optimizer.step()
        # log
        pred_prob_all[selected, :] = final_prob.detach().cpu()
        writer.add_scalar('train_loss', loss.item(), niter)
        slide_names_str = ';'.join(loader.dataset.patient_slide_names[selected])
        zhuyuanhao = loader.dataset.patient_zhuyuanhao[selected]
        save_info.append([zhuyuanhao,
                          label.squeeze()[2].item(), label.squeeze()[3].item(),
                          label.squeeze()[4].item(), label.squeeze()[5].item(),
                          final_prob[0, 1].item(),
                          label_patho.item(), label_isup.item(), label_size.item(), label_tnm.item(),
                          slide_yuHouFenZu.item(), slide_names_str])

    pred_logit = pred_prob_all.argmax(1)
    train_acc = torch.sum(pred_logit == slide_label_gt)/slide_label_gt.shape[0]
    auc = utliz.cal_auc(slide_label_gt, pred_prob_all[:, 1])
    print('Epoch:{} train_bag_acc: {}'.format(epoch, train_acc))
    print('Epoch:{} train_bag_auc: {}'.format(epoch, auc))
    writer.add_scalar('train_bag_acc', train_acc, epoch)
    writer.add_scalar('train_bag_auc', auc, epoch)

    save_info = np.stack(save_info)

    rfs_e = np.array(save_info[:, 1], dtype=float)
    rfs_t = np.array(save_info[:, 2], dtype=float)
    dss_e = np.array(save_info[:, 3], dtype=float)
    dss_t = np.array(save_info[:, 4], dtype=float)
    pred_ai = np.array(save_info[:, 5], dtype=float)
    isup = np.array(save_info[:, 7], dtype=float)
    yuHouFenZu = np.array(save_info[:, 10], dtype=float)
    idx_available = np.where(np.logical_and(rfs_e != -1, dss_e != -1))[0]
    rfs_e = rfs_e[idx_available]
    rfs_t = rfs_t[idx_available]
    dss_e = dss_e[idx_available]
    dss_t = dss_t[idx_available]
    pred_ai = pred_ai[idx_available]
    isup = isup[idx_available]
    yuHouFenZu = yuHouFenZu[idx_available]
    rfs_C_index = concordance_index_censored(rfs_e.astype(np.bool_), rfs_t, pred_ai)[0]
    dss_C_index = concordance_index_censored(dss_e.astype(np.bool_), dss_t, pred_ai)[0]
    rfs_C_index_isup = concordance_index_censored(rfs_e.astype(np.bool_), rfs_t, isup)[0]
    dss_C_index_isup = concordance_index_censored(dss_e.astype(np.bool_), dss_t, isup)[0]
    rfs_C_index_yuHouFenZu = concordance_index_censored(rfs_e.astype(np.bool_), rfs_t, yuHouFenZu)[0]
    dss_C_index_yuHouFenZu = concordance_index_censored(dss_e.astype(np.bool_), dss_t, yuHouFenZu)[0]
    print('Epoch:{} train_bag_rfs_C_index: {}'.format(epoch, rfs_C_index))
    print('Epoch:{} train_bag_dss_C_index: {}'.format(epoch, dss_C_index))
    print('Epoch:{} train_bag_rfs_C_index_isup: {}'.format(epoch, rfs_C_index_isup))
    print('Epoch:{} train_bag_dss_C_index_isup: {}'.format(epoch, dss_C_index_isup))
    print('Epoch:{} train_bag_rfs_C_index_yuHouFenZu: {}'.format(epoch, rfs_C_index_yuHouFenZu))
    print('Epoch:{} train_bag_dss_C_index_yuHouFenZu: {}'.format(epoch, dss_C_index_yuHouFenZu))
    writer.add_scalar('train_rfs_C_index', rfs_C_index, epoch)
    writer.add_scalar('train_dss_C_index', dss_C_index, epoch)

    save_dir = os.path.join("./results_yuHouFenZu", writer.logdir.split('/')[-1])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = "InternalTrain_Epoch{}_AUC{:.4f}_rfs{:.4f}_dss{:.4f}.csv".format(epoch, auc, rfs_C_index, dss_C_index)
    pd.DataFrame(save_info).to_csv(os.path.join(save_dir, save_name))
    return 0

def eval(epoch, model, loader, dev, writer):
    for loader_i, prefix in zip(loader, ["InternalTest", "ExternalTest", "HuaDong"]):
        model = model.eval()
        slide_label_gt = torch.zeros([loader_i.dataset.__len__()]).long()
        pred_prob_all = torch.zeros([loader_i.dataset.__len__(), 2])
        save_info = []
        for iter, (data, label, label_patho, label_isup, label_size, label_tnm, slide_yuHouFenZu, selected) in enumerate(tqdm(loader_i, desc='Testing Epoch {}'.format(epoch))):
            slide_label_gt[selected] = slide_yuHouFenZu
            data = data.to(dev)
            with torch.no_grad():
                final = model(data)
                final_prob = torch.softmax(final, dim=1)
            # log
            pred_prob_all[selected, :] = final_prob.detach().cpu()
            zhuyuanhao = loader_i.dataset.patient_zhuyuanhao[selected]
            slide_names_str = ';'.join(loader_i.dataset.patient_slide_names[selected])
            save_info.append([zhuyuanhao,
                              label.squeeze()[2].item(), label.squeeze()[3].item(),
                              label.squeeze()[4].item(), label.squeeze()[5].item(),
                              final_prob[0, 1].item(),
                              label_patho.item(), label_isup.item(), label_size.item(), label_tnm.item(),
                              slide_yuHouFenZu.item(), slide_names_str])

        pred_logit = pred_prob_all.argmax(1)
        acc = torch.sum(pred_logit == slide_label_gt)/slide_label_gt.shape[0]
        print('[{}] Epoch:{} test_bag_acc: {}'.format(prefix, epoch, acc))

        auc = utliz.cal_auc(slide_label_gt, pred_prob_all[:, 1])
        print('[{}] Epoch:{} test_bag_auc: {}'.format(prefix, epoch, auc))

        writer.add_scalar('{}_bag_acc'.format(prefix), acc, epoch)
        writer.add_scalar('{}_bag_auc'.format(prefix), auc, epoch)

        save_info = np.stack(save_info)

        rfs_e = np.array(save_info[:, 1], dtype=float)
        rfs_t = np.array(save_info[:, 2], dtype=float)
        dss_e = np.array(save_info[:, 3], dtype=float)
        dss_t = np.array(save_info[:, 4], dtype=float)
        pred_ai = np.array(save_info[:, 5], dtype=float)
        isup = np.array(save_info[:, 7], dtype=float)
        yuHouFenZu = np.array(save_info[:, 10], dtype=float)
        idx_available = np.where(np.logical_and(rfs_e != -1, dss_e != -1))[0]
        rfs_e = rfs_e[idx_available]
        rfs_t = rfs_t[idx_available]
        dss_e = dss_e[idx_available]
        dss_t = dss_t[idx_available]
        pred_ai = pred_ai[idx_available]
        isup = isup[idx_available]
        yuHouFenZu = yuHouFenZu[idx_available]
        rfs_C_index = concordance_index_censored(rfs_e.astype(np.bool_), rfs_t, pred_ai)[0]
        dss_C_index = concordance_index_censored(dss_e.astype(np.bool_), dss_t, pred_ai)[0]
        rfs_C_index_isup = concordance_index_censored(rfs_e.astype(np.bool_), rfs_t, isup)[0]
        dss_C_index_isup = concordance_index_censored(dss_e.astype(np.bool_), dss_t, isup)[0]
        rfs_C_index_yuHouFenZu = concordance_index_censored(rfs_e.astype(np.bool_), rfs_t, yuHouFenZu)[0]
        dss_C_index_yuHouFenZu = concordance_index_censored(dss_e.astype(np.bool_), dss_t, yuHouFenZu)[0]
        print('[{}] Epoch:{} test_bag_rfs_C_index: {}'.format(prefix, epoch, rfs_C_index))
        print('[{}] Epoch:{} test_bag_dss_C_index: {}'.format(prefix, epoch, dss_C_index))
        print('[{}] Epoch:{} test_bag_rfs_C_index_isup: {}'.format(prefix, epoch, rfs_C_index_isup))
        print('[{}] Epoch:{} test_bag_dss_C_index_isup: {}'.format(prefix, epoch, dss_C_index_isup))
        print('[{}] Epoch:{} test_bag_rfs_C_index_yuHouFenZu: {}'.format(prefix, epoch, rfs_C_index_yuHouFenZu))
        print('[{}] Epoch:{} test_bag_dss_C_index_yuHouFenZu: {}'.format(prefix, epoch, dss_C_index_yuHouFenZu))
        
        writer.add_scalar('{}_rfs_C_index'.format(prefix), rfs_C_index, epoch)
        writer.add_scalar('{}_dss_C_index'.format(prefix), dss_C_index, epoch)

        save_dir = os.path.join("./results_yuHouFenZu_pycox", writer.logdir.split('/')[-1])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = "{}_Epoch{}_AUC{:.4f}_rfs{:.4f}_dss{:.4f}.csv".format(prefix, epoch, auc, rfs_C_index, dss_C_index)
        pd.DataFrame(save_info).to_csv(os.path.join(save_dir, save_name))
        
        save_ckpt_dir = os.path.join("./results_yuHouFenZu_pycox/ckpt", writer.logdir.split('/')[-1])
        if not os.path.exists(save_ckpt_dir):
            os.mkdir(save_ckpt_dir)
        save_ckpt(model, epoch, save_name=os.path.join(save_ckpt_dir, "{}_Epoch_{}_AUC_{:.4f}_rfs_C_index_{:.4f}_dss_C_index_{:.4f}".format(
            prefix, epoch, auc, rfs_C_index, dss_C_index)))
        print("[model saved]")
    return 0


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

class Attention_MIL(nn.Module):
    def __init__(self, num_classes, init=True):
        super(Attention_MIL, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1536, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
        )
        self.L = 512
        self.D = 128
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.BatchNorm1d(self.D),
            nn.ReLU(),
            nn.Linear(self.D, self.K)
        )

        self.top_layer = nn.Sequential(nn.Linear(1536*1, num_classes))

        if init:
            self._initialize_weights()

    def forward(self, x):
        x = x.squeeze(0)
        x = self.classifier(x)

        # Attention module
        A_raw = self.attention(x)  # NxK
        A_raw = torch.transpose(A_raw, 1, 0)  # KxN
        A = F.softmax(A_raw, dim=1)  # softmax over N
        x = torch.mm(A, x)

        x = self.top_layer(x)

        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    # optimizer
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=200, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64', choices=['f64', 'f32'], type=str, help='SK-algo dtype (default: f64)')

    # 
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=0, type=int, help='number workers (default: 6)')
    parser.add_argument('--comment', default='ExternalTest_addDropOut', type=str, help='name for tensorboardX')
    # parser.add_argument('--resume', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/results_yuHouFenZu_pycox/ckpt/20250904_110853_Bs1_lr0.001_downsample1_seed42_rate4/InternalTest_Epoch_5_AUC_0.7543_rfs_C_index_0.7523_dss_C_index_0.7719', type=str, help='resume ckpt path')
    parser.add_argument('--resume', default='', type=str, help='resume ckpt path')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--turn', default=10, type=int, help='iter time')
    parser.add_argument('--rate', default=4, type=int, help='rate')
    parser.add_argument('--downsample', default=1, type=float, help='downsample the whole dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) 
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    InternalTrain, InternalTest, ExternalTest, HuadongTest = get_train_test_ds_MultiCenter_region_trainwithTCGA(downsample=args.downsample)

    print("Num of slides in InternalTrain/InternalTest/ExternalTest/HuaDongTest: {} / {} / {} / {}".format(
        len(InternalTrain[0]), len(InternalTest[0]), len(ExternalTest[0]), len(HuadongTest[0])))

    train_ds = TumorRegion_Prognosis_Feat_yuHouFenZu(InternalTrain, filter_yuHouFenZu_score2=True)
    InternalVal_ds = TumorRegion_Prognosis_Feat_yuHouFenZu(InternalTest)
    ExternalVal_ds = TumorRegion_Prognosis_Feat_yuHouFenZu(ExternalTest)
    HuadongVal_ds = TumorRegion_Prognosis_Feat_yuHouFenZu(HuadongTest)

    print("Num of slide in InternalTrain/InternalTest/ExternalTest/HuaDongTest: {} / {} / {} / {}".format(
        len(train_ds), len(InternalVal_ds), len(ExternalVal_ds), len(HuadongVal_ds)))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=False)
    InternalVal_loader = torch.utils.data.DataLoader(InternalVal_ds, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=args.workers, drop_last=False,
                                                     pin_memory=False)
    ExternalVal_loader = torch.utils.data.DataLoader(ExternalVal_ds, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=args.workers, drop_last=False,
                                                     pin_memory=False)
    HuadongVal_loader = torch.utils.data.DataLoader(HuadongVal_ds, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=args.workers, drop_last=False,
                                                    pin_memory=False)
    print("[Data] {} training samples".format(len(train_loader.dataset)))
    print("[Data] {} Internal evaluating samples".format(len(InternalVal_loader.dataset)))
    print("[Data] {} External evaluating samples".format(len(ExternalVal_loader.dataset)))
    print("[Data] {} Huadong evaluating samples".format(len(HuadongVal_loader.dataset)))

    dev = 'cuda:0'
    model = Attention_MIL(num_classes=2)
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    start_epoch = 0
    if args.resume != '':
        model, start_epoch = load_ckpt(model, args.resume)
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Bs{}_lr{}_downsample{}_seed{}_rate{}".format(
        args.batch_size, args.lr, args.downsample, args.seed, args.rate)
    writer = SummaryWriter('./runs_yuHouFenZu_pycox/%s' % name)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(epoch=epoch, model=model, loader=train_loader, optimizer=optimizer, dev=dev, writer=writer)
        eval(epoch=epoch, model=model, loader=[InternalVal_loader, ExternalVal_loader, HuadongVal_loader], dev=dev, writer=writer)

