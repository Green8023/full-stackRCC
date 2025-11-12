import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter
from dataset import get_train_test_ds_MultiCenter_region_trainwithTCGA, TumorRegion_PathologyType_Feat
import datetime
import utliz
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

def find_best_threshold_youden(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    if len(fpr) == 0 or len(tpr) == 0 or len((tpr - fpr)) == 0:
        if len(fpr) != len(tpr) or len(tpr) != len(thresholds):
            return None, None, None
        if len(np.unique(thresholds)) == 1:
            if len(np.unique(y_true)) < 2:
                return None, None, None
            else:
                return 0.5, None, None
        return None, None, None

    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)

    best_threshold = thresholds[optimal_idx]
    best_tpr = tpr[optimal_idx]
    best_fpr = fpr[optimal_idx]

    return best_threshold, best_tpr, best_fpr

def find_best_thresholds_youden_multiclass(y_true, y_pred_proba, class_names):
    thresholds_dict = {}
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    for i, class_name in enumerate(class_names):
        y_true_class_i = y_true_bin[:, i]
        y_scores_class_i = y_pred_proba[:, i]
        threshold, tpr, fpr = find_best_threshold_youden(y_true_class_i, y_scores_class_i)
        thresholds_dict[class_name] = {
            'threshold': threshold,
            'TPR': tpr,
            'FPR': fpr
        }
        print(f"[{class_name}] Best Youden Threshold = {threshold:.3f} (TPR={tpr:.3f}, FPR={fpr:.3f})")
    
    return thresholds_dict

def predict_with_thresholds(prob_matrix, threshold_dict, class_names):
    pred_labels = []
    for probs in prob_matrix:
        passed = []
        for i, class_name in enumerate(class_names):
            threshold = threshold_dict[class_name]['threshold']
            if probs[i] >= threshold:
                passed.append(i)

        if len(passed) == 1:
            pred_labels.append(passed[0])
        elif len(passed) > 1:
            best_i = passed[np.argmax([probs[i] for i in passed])]
            pred_labels.append(best_i)
        else:
            pred_labels.append(np.argmax(probs))

    return np.array(pred_labels)

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


def sort_into_bags(instance_labels, instance_preds, instance_corresponding_slideName):
    unique_slide_name = np.unique(instance_corresponding_slideName)
    bag_labels = []
    bag_predictions_prob = []
    bag_predictions_logit = []
    slide_names = []
    patch_preds_per_slide = []
    for slide_name_i in unique_slide_name:
        idx_from_slide_i = np.where(instance_corresponding_slideName == slide_name_i)[0]
        bag_label_i = instance_labels[idx_from_slide_i]
        bag_pred_logit_i, bag_pred_prob_i = aggregate_bag_label(instance_preds[idx_from_slide_i])
        if bag_label_i.max() != bag_label_i.min():
            raise
        bag_labels.append(bag_label_i.max())
        bag_predictions_prob.append(bag_pred_prob_i)
        bag_predictions_logit.append(bag_pred_logit_i)
        slide_names.append(slide_name_i)
        patch_preds = instance_preds[idx_from_slide_i].argmax(1)
        patch_preds_per_slide.append(patch_preds.tolist())
    return np.array(bag_labels), np.array(bag_predictions_prob), np.array(bag_predictions_logit), np.array(slide_names), patch_preds_per_slide

def train_one_epoch(epoch, model, loader, optimizer, dev, writer):
    class_names = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8']
    model = model.train()
    # XE = torch.nn.CrossEntropyLoss()
    XE = torch.nn.CrossEntropyLoss(weight=torch.tensor([
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0
    ]).to(dev))
    patch_label_gt = torch.zeros([loader.dataset.__len__()]).long()
    pred_prob_all = torch.zeros([loader.dataset.__len__(), 9])
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

    # obtain bag prediction
    bag_labels, bag_preds, bag_preds_logit, _, _ = sort_into_bags(patch_label_gt, pred_prob_all, loader.dataset.all_patch_slideName)
    thresholds_dict = find_best_thresholds_youden_multiclass(
            y_true=bag_labels, 
            y_pred_proba=bag_preds, 
            class_names=class_names
    )
    auc_all_cate = cal_auc_multi_class(bag_labels, bag_preds)
    for class_id, auc_class_i in enumerate(auc_all_cate):
        writer.add_scalar('train_bag_auc_class{}'.format(class_id), auc_class_i, epoch)
    return thresholds_dict

def eval(epoch, model, loader, dev, writer, threshold_dict):
    bag_labels_list = []
    bag_preds_list = []
    dataset_names = []
    bag_acc_all = {}
    roc_auc_all = {}
    class_names = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8']
    for loader_i, prefix in zip(loader, ["InternalTest", "ExternalTest", "HuaDong"]):
        model = model.eval()
        patch_label_gt = torch.zeros([loader_i.dataset.__len__()]).long()
        pred_prob_all = torch.zeros([loader_i.dataset.__len__(), 9])
        # for iter, (data, label, selected) in enumerate(tqdm(loader, ascii=True, desc='testing epoch_{}'.format(epoch))):
        for iter, (data, label, selected) in enumerate(tqdm(loader_i, ascii=True, desc='eval')):
            patch_label_gt[selected] = label
            data = data.to(dev)
            with torch.no_grad():
                final = model(data)
                final_prob = torch.softmax(final, dim=1)
            # log
            pred_prob_all[selected, :] = final_prob.detach().cpu()

        pred_logit = pred_prob_all.argmax(1)
        train_acc = torch.sum(pred_logit == patch_label_gt)/patch_label_gt.shape[0]
        print('[{}] Epoch:{} test_patch_acc: {}'.format(prefix, epoch, train_acc))

        auc_all_cate = cal_auc_multi_class(patch_label_gt, pred_prob_all)
        print('[{}] Epoch:{} test_patch_auc: {}'.format(prefix, epoch, auc_all_cate))
        print('[{}] Epoch:{} test_kappa: {}'.format(prefix, epoch, cohen_kappa_score(patch_label_gt, pred_logit)))
        print(confusion_matrix(patch_label_gt, pred_logit))

        writer.add_scalar('{}_patch_acc'.format(prefix), train_acc, epoch)
        for class_id, auc_class_i in enumerate(auc_all_cate):
            writer.add_scalar('{}_patch_auc_class{}'.format(prefix, class_id), auc_class_i, epoch)
    
        bag_labels, bag_preds, bag_preds_logit, slide_names, patch_preds_per_slide = sort_into_bags(patch_label_gt, pred_prob_all, loader_i.dataset.all_patch_slideName)
        bag_preds_logit = predict_with_thresholds(
            prob_matrix=bag_preds,
            threshold_dict=threshold_dict,
            class_names=class_names
        )
        bag_labels_list.append(bag_labels)
        bag_preds_list.append(bag_preds)
        dataset_names.append(prefix)
        auc_all_cate = cal_auc_multi_class(bag_labels, bag_preds)
        for class_id, auc_class_i in enumerate(auc_all_cate):
            writer.add_scalar('{}_bag_auc_class{}'.format(prefix, class_id), auc_class_i, epoch)
        bag_acc = (bag_preds_logit == bag_labels).sum() / len(bag_labels)
        print(f"[{prefix}] Epoch:{epoch} slide_level_acc: {bag_acc:.4f}")
        writer.add_scalar(f'{prefix}_slide_acc', bag_acc, epoch)
        bag_acc_all[prefix] = bag_acc
        save_path = os.path.join("./results_ckpt_patho",f"{prefix}_Epoch{epoch}_AUC_{auc_all_cate[0]:.4f}")
        save_ckpt(model, epoch, save_path, threshold_dict)
        print("[model saved]")

        print('[{}] Epoch:{} test_slide_auc: {}'.format(prefix, epoch, auc_all_cate))
        print('[{}] Epoch:{} test_slide_kappa: {}'.format(prefix, epoch, cohen_kappa_score(bag_labels, bag_preds_logit)))
        print(confusion_matrix(bag_labels, bag_preds_logit))
        
    return 0

def save_ckpt(model, epoch, save_name, threshold_dict=None):
    state = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    if threshold_dict is not None:  
        state['threshold_dict'] = threshold_dict
    torch.save(state, save_name)
    return 0

def load_ckpt(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['model'])
    epoch = state['epoch']
    threshold_dict = state.get('threshold_dict', None) 
    return model, epoch, threshold_dict


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
    parser.add_argument('--epochs', default=5, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size (default: 256)')
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
    parser.add_argument('--resume', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/results_ckpt_patho/InternalTest_Epoch3_AUC_0.9902', type=str, help='resume ckpt path')
    parser.add_argument('--log-intv', default=1, type=int, help='save stuff every x epochs (default: 1)')
    parser.add_argument('--log_iter', default=200, type=int, help='log every x-th batch (default: 200)')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument('--downsample', default=1, type=float, help='downsample the whole dataset')

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
            nn.Linear(2048, 9),
        )
    def forward(self, x):
        return self.fcs(x)


if __name__ == '__main__':
    args = get_parser()

    InternalTrain, InternalTest, ExternalTest, huadong = get_train_test_ds_MultiCenter_region_trainwithTCGA(downsample=args.downsample)

    print("Num of slides in InternalTrain/InternalTest/ExternalTest: {} / {} / {} / {}".format(
        len(InternalTrain[0]), len(InternalTest[0]), len(ExternalTest[0]), len(huadong[0])))

    train_ds = TumorRegion_PathologyType_Feat(InternalTrain, return_bag=False)
    InternalVal_ds =TumorRegion_PathologyType_Feat(InternalTest, return_bag=False)
    ExternalVal_ds =TumorRegion_PathologyType_Feat(ExternalTest, return_bag=False)
    ExternalHDVal_ds =TumorRegion_PathologyType_Feat(huadong, return_bag=False)

    print("Num of patches in InternalTrain/InternalTest/ExternalTest: {} / {} / {} / {}".format(
        len(train_ds), len(InternalVal_ds), len(ExternalVal_ds), len(ExternalHDVal_ds)))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, drop_last=False, pin_memory=True)
    InternalVal_loader = torch.utils.data.DataLoader(InternalVal_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
    ExternalVal_loader = torch.utils.data.DataLoader(ExternalVal_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)
    ExternalHDVal_loader = torch.utils.data.DataLoader(ExternalHDVal_ds, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

    print("[Data] {} training samples".format(len(train_loader.dataset)))
    print("[Data] {} Internal evaluating samples".format(len(InternalVal_loader.dataset)))
    print("[Data] {} External evaluating samples".format(len(ExternalVal_loader.dataset)))
    print("[Data] {} HuaDong evaluating samples".format(len(ExternalHDVal_loader.dataset)))

    dev = 'cuda:0'
    model = prediction_head()
    model.to(dev)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    start_epoch = 0
    if args.resume != '':
        model, start_epoch, threshold_dict = load_ckpt(model, args.resume)
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Bs{}_lr{}_downsample{}".format(
        args.batch_size, args.lr, args.downsample)
    writer = SummaryWriter('./runs_RegionPathology_feat/%s' % name)

    for epoch in range(start_epoch, args.epochs):
        thresholds_dict = train_one_epoch(epoch=epoch, model=model, loader=train_loader, optimizer=optimizer, dev=dev, writer=writer)
        eval(epoch=epoch, model=model, loader=[InternalVal_loader, ExternalVal_loader, ExternalHDVal_loader], dev=dev, writer=writer, threshold_dict=threshold_dict)
