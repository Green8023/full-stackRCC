import argparse
import warnings
import os
import time
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import math
from tensorboardX import SummaryWriter
from dataset import get_train_ds_ZSCenter_forNuclearLevel, TumorRegion_NuclearLevel_Feat
import datetime
import utliz
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_recall_curve, auc


def multiClass_BCE_loss(label, pred, weight=None, weight_class=None):
    # label of size: NxC
    # pred of size: NXC
    # weight: the weight of Pos/Neg in each class
    # weight_class: the weight of each class
    if weight is None:
        weight = [0.5, 0.5, 0.5, 0.5]
    if weight_class is None:
        weight_class = [1.0, 1.0, 1.0, 1.0]
    N, C = label.shape
    loss_all = 0
    for class_i in range(C):
        loss_i = -torch.mean(weight[class_i] * (1-label[:, class_i]) * torch.log(1-pred[:, class_i] + 1e-5) +
                             (1-weight[class_i]) * label[:, class_i] * torch.log(pred[:, class_i] + 1e-5))
        loss_all = loss_all + loss_i * weight_class[class_i]
    return loss_all

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

def aggregate_bag_label(instance_predictions):
    if type(instance_predictions) is torch.Tensor:
        instance_predictions = instance_predictions.detach().cpu().numpy()
    instance_predictions_logit = instance_predictions.argmax(1)
    pred_classes = np.unique(instance_predictions_logit)
    ratio_pred = []
    for class_i in pred_classes:
        num_pred_class_i = len(np.where(instance_predictions_logit == class_i)[0])
        ratio_pred.append(num_pred_class_i/instance_predictions_logit.shape[0])

    pred_bag_label = pred_classes[np.argmax(np.array(ratio_pred))]

    pred_bag_label_prob = instance_predictions[np.where(instance_predictions_logit == pred_bag_label)[0], :].mean(0)
    # pred_bag_label_prob = instance_predictions[:, :].mean(0)
    return pred_bag_label, pred_bag_label_prob


def sort_into_bags(instance_labels, instance_preds, slide_corresponding_zhuyuanhao):
    slide_corresponding_zhuyuanhao = np.array(slide_corresponding_zhuyuanhao)
    unique_zhuyuanhao = np.unique(slide_corresponding_zhuyuanhao)
    bag_labels = []
    bag_predictions_prob = []
    bag_predictions_logit = []
    for zhuyuanhao_i in unique_zhuyuanhao:
        idx_from_slide_i = np.where(slide_corresponding_zhuyuanhao == zhuyuanhao_i)[0]
        bag_label_i = instance_labels[idx_from_slide_i]
        bag_pred_logit_i, bag_pred_prob_i = aggregate_bag_label(instance_preds[idx_from_slide_i])
        if bag_label_i.max() != bag_label_i.min():
             raise ValueError(f"Bag {zhuyuanhao_i} has inconsistent labels: {bag_label_i}")
        bag_labels.append(bag_label_i.max())
        bag_predictions_prob.append(bag_pred_prob_i)
        bag_predictions_logit.append(bag_pred_logit_i)
    return np.array(bag_labels), np.array(bag_predictions_prob), np.array(bag_predictions_logit)


def train_one_epoch(epoch, model, loader, optimizer, dev, writer, num_classes=3, class_weight=[0.5, 0.5, 0.5]):
    model = model.train()
    # XE = torch.nn.CrossEntropyLoss()
    XE = torch.nn.CrossEntropyLoss()
    slide_label_gt = torch.zeros([loader.dataset.__len__()]).long()
    pred_prob_all = torch.zeros([loader.dataset.__len__(), num_classes])
    # for iter, (data, label, selected) in enumerate(tqdm(loader, ascii=True, desc='training epoch_{}'.format(epoch))):
    for iter, (data, label, selected) in enumerate(tqdm(loader, desc='Training Epoch {}'.format(epoch))):
        if len(torch.squeeze(data).shape) == 1:
            continue
        optimizer.zero_grad()
        niter = epoch * len(loader) + iter
        slide_label_gt[selected] = label
        data = data.to(dev)
        final, _ = model(data)
        final_prob = torch.softmax(final, dim=1)
        loss = multiClass_BCE_loss(torch.eye(num_classes)[label].to(dev), final_prob, weight_class=class_weight)

        loss.backward()
        optimizer.step()
        # log
        pred_prob_all[selected, :] = final_prob.detach().cpu()
        writer.add_scalar('train_loss', loss.item(), niter)

    pred_logit = pred_prob_all.argmax(1)
    train_acc = torch.sum(pred_logit == slide_label_gt)/slide_label_gt.shape[0]
    kappa = cohen_kappa_score(slide_label_gt, pred_logit, weights='linear')
    writer.add_scalar('train_bag_acc', train_acc, epoch)
    print('Epoch:{} train_bag_acc: {}'.format(epoch, train_acc))

    auc_all_cate = cal_auc_multi_class(slide_label_gt, pred_prob_all)
    writer.add_scalar('train_bag_kappa', kappa, epoch)
    for class_id, auc_class_i in enumerate(auc_all_cate):
        writer.add_scalar('train_bag_auc_class{}'.format(class_id), auc_class_i, epoch)

    return 0

def eval(epoch, model, loader, dev, writer, num_classes=4):
        save_info = []
        model = model.eval()
        slide_label_gt = torch.zeros([loader.dataset.__len__()]).long()
        pred_prob_all = torch.zeros([loader.dataset.__len__(), num_classes])
        for iter, (data, label, selected) in enumerate(loader):
            if len(torch.squeeze(data).shape) == 1:
                continue
            slide_label_gt[selected] = label
            data = data.to(dev)
            with torch.no_grad():
                final, _ = model(data)
                final_prob = torch.softmax(final, dim=1)
            # log
            pred_prob_all[selected, :] = final_prob.detach().cpu()

        pred_logit = pred_prob_all.argmax(1)
        for i in range(loader.dataset.__len__()):
            save_info.append([
                loader.dataset.patient_zhuyuanhao[i],
                float(pred_prob_all[i, 1]),  
                int(loader.dataset.patient_patho_labels[i]),
                int(loader.dataset.patient_nuclearLevel_label[i]),
                ';'.join(loader.dataset.patient_slide_names[i]),
                int(pred_logit[i].item())  
            ])
        eval_acc = torch.sum(pred_logit == slide_label_gt)/slide_label_gt.shape[0]
        print('Epoch:{} test_bag_acc: {}'.format(epoch, eval_acc))

        auc_all_cate = cal_auc_multi_class(slide_label_gt, pred_prob_all)
        kappa = cohen_kappa_score(slide_label_gt, pred_logit, weights='linear')
        print('Epoch:{} test_bag_auc: {}'.format(epoch, auc_all_cate))
        print('Epoch:{} test_kappa: {}'.format(epoch, kappa))
        print(confusion_matrix(slide_label_gt, pred_logit))

        writer.add_scalar('test_bag_acc', eval_acc, epoch)
        writer.add_scalar('test_bag_kappa', kappa, epoch)
        for class_id, auc_class_i in enumerate(auc_all_cate):
            writer.add_scalar('test_bag_auc_class{}'.format(class_id), auc_class_i, epoch)
    
        save_ckpt(model, epoch, save_name="./results_nuclearlevel/Epoch_{}_AUC_{:.4f}_kappa{:.4f}".format(epoch, auc_all_cate[0], kappa))
        columns = ['zhuyuanhao', 'nuclear_level_pred', 'patho_label', 'nuclear_label', 'slide_names', 'pred_logit']
        df = pd.DataFrame(save_info, columns=columns)
        df.to_excel("test_eval_results.xlsx", index=False)
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

class Attention_MIL(nn.Module):
    def __init__(self, num_classes, init=True):
        super(Attention_MIL, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(1536, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(2048, 1536),
        )
        self.L = 1536
        self.D = 256
        self.K = 1

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.top_layer = nn.Sequential(nn.Linear(1536*1, num_classes))

        if init:
            self._initialize_weights()

    def forward(self, x):
        x = x.squeeze(0)
        x = self.classifier(x)

        # Attention module
        A_ = self.attention(x)  # NxK
        A_ = torch.transpose(A_, 1, 0)  # KxN
        A = F.softmax(A_, dim=1)  # softmax over N

        x = torch.mm(A, x)  # KxL

        x = self.top_layer(x)

        return x, A_

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
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=200, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64',choices=['f64','f32'], type=str, help='SK-algo dtype (default: f64)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='ExternalTest_addDropOut', type=str, help='name for tensorboardX')
    parser.add_argument('--resume', default='', type=str, help='resume ckpt path')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=10, type=int, help='random seed')

    parser.add_argument('--downsample', default=1, type=float, help='downsample the whole dataset')
    parser.add_argument('--pathoClassSetting', default=0, type=int, help='patho class, 0 for all pathoClass, 1 for pathoClass 1.1 and 2.1, 2 for pathClass 1.1, 3 for pathClass 2.1')
    parser.add_argument('--numClass', default=3, type=int, help='4:0/1/2/3, 3:12/3/4, 2:12/34')
    parser.add_argument('--ClassWeight',nargs='+', type=float, help='weight of multiple teacher, like: 1.0 1.0', required=True)
    
    parser.add_argument('--all_region', default=True, type=bool, help='threshold for 5 class classification model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    InternalTrain, InternalTest = get_train_ds_ZSCenter_forNuclearLevel(downsample=args.downsample)
    print("Num of slides in InternalTrain/InternalTest: {} / {}".format(
        len(InternalTrain[0]), len(InternalTest[0])))

    if args.pathoClassSetting == 0:
        patho_filter = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif args.pathoClassSetting == 1:
        patho_filter = [0, 1]
    elif args.pathoClassSetting == 2:
        patho_filter = [0]
    elif args.pathoClassSetting == 3:
        patho_filter = [1]
    else:
        raise ValueError

    train_ds = TumorRegion_NuclearLevel_Feat(InternalTrain, return_bag=True, patho_filter=patho_filter, numClass=args.numClass, allRegion=args.all_region)
    InternalVal_ds =TumorRegion_NuclearLevel_Feat(InternalTest, return_bag=True, patho_filter=patho_filter, numClass=args.numClass, allRegion=args.all_region)
    
    print("Num of patches in InternalTrain/InternalTest {} / {} ".format(
        len(train_ds), len(InternalVal_ds)))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=True)
    InternalVal_loader = torch.utils.data.DataLoader(InternalVal_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

    print("[Data] {} training samples".format(len(train_loader.dataset)))
    print("[Data] {} Internal evaluating samples".format(len(InternalVal_loader.dataset)))

    dev = 'cuda:0'
    model = Attention_MIL(num_classes=args.numClass)
    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    start_epoch = 0
    if args.resume != '':
        model, start_epoch = load_ckpt(model, args.resume)
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Bs{}_lr{}_downsample{}_numClasses{}_allRegion{}_patch_threshold{}_region{}".format(
        args.batch_size, args.lr, args.downsample, args.numClass, args.all_region, args.patch_cls_threshold, args.region_threshold)
    writer = SummaryWriter('./runs_NuclearLevel_feat/%s' % name)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(epoch=epoch, model=model, loader=train_loader, optimizer=optimizer, dev=dev, writer=writer, num_classes=args.numClass, class_weight=args.ClassWeight)
        eval(epoch=epoch, model=model, loader=InternalVal_loader, dev=dev, writer=writer, num_classes=args.numClass)