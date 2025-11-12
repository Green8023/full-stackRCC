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
from dataset import get_train_test_ds_MultiCenter_region_trainwithTCGA, TumorRegion_PathologyType_Feat
import datetime
import utliz
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, random_state=42, **kwargs):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    stats = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n) 
        if len(np.unique(y_true[indices])) < 2: 
            continue
        try:
            score = metric_fn(y_true[indices], y_pred[indices], **kwargs)
        except Exception:
            score = np.nan
        stats.append(score)
    stats = np.array(stats)
    stats = stats[~np.isnan(stats)]
    mean = np.mean(stats)
    ci = np.percentile(stats, [2.5, 97.5])
    return mean, tuple(ci)


def bootstrap_ci_per_class(y_true, y_pred, num_classes=5, n_bootstraps=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    results = {i: {'sensitivity': [], 'specificity': [], 'f1': []} for i in range(num_classes)}

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        cm = confusion_matrix(y_t, y_p, labels=range(num_classes))

        for i in range(num_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP
            TN = cm.sum() - (TP + FN + FP)
            sensitivity = TP / (TP + FN + 1e-8)
            specificity = TN / (TN + FP + 1e-8)
            f1 = f1_score(y_t, y_p, labels=[i], average='macro', zero_division=0)

            results[i]['sensitivity'].append(sensitivity)
            results[i]['specificity'].append(specificity)
            results[i]['f1'].append(f1)

    ci_summary = {}
    for i in range(num_classes):
        ci_summary[i] = {
            'sensitivity_mean': np.nanmean(results[i]['sensitivity']),
            'sensitivity_CI': np.nanpercentile(results[i]['sensitivity'], [2.5, 97.5]),
            'specificity_mean': np.nanmean(results[i]['specificity']),
            'specificity_CI': np.nanpercentile(results[i]['specificity'], [2.5, 97.5]),
            'f1_mean': np.nanmean(results[i]['f1']),
            'f1_CI': np.nanpercentile(results[i]['f1'], [2.5, 97.5])
        }
    return ci_summary

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

def eval(epoch, model, loader, dev, writer, threshold_dict):
    bag_acc_all = {}
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
        y_true_patch = patch_label_gt.numpy()
        y_pred_patch = pred_logit.numpy()

        # ===== 全局指标CI =====
        acc_mean, acc_ci = bootstrap_ci(y_true_patch, y_pred_patch, accuracy_score)
        f1_macro_mean, f1_macro_ci = bootstrap_ci(y_true_patch, y_pred_patch, f1_score, average='macro')
        prec_mean, prec_ci = bootstrap_ci(y_true_patch, y_pred_patch, precision_score, average='macro')
        recall_mean, recall_ci = bootstrap_ci(y_true_patch, y_pred_patch, recall_score, average='macro')

        print(f"\n[{prefix}] Patch-level Metrics with 95% CI")
        print(f"Accuracy: {acc_mean:.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
        print(f"F1-macro: {f1_macro_mean:.3f} [{f1_macro_ci[0]:.3f}, {f1_macro_ci[1]:.3f}]")
        print(f"Precision-macro: {prec_mean:.3f} [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]")
        print(f"Recall-macro: {recall_mean:.3f} [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")

        # ===== 各类别CI =====
        ci_patch_class = bootstrap_ci_per_class(y_true_patch, y_pred_patch, num_classes=len(class_names))
        print(f"\nPer-class Patch-level metrics (mean [95%CI]):")
        for i, cls in enumerate(class_names):
            v = ci_patch_class[i]
            print(f"{cls}: Sens {v['sensitivity_mean']:.3f}[{v['sensitivity_CI'][0]:.3f},{v['sensitivity_CI'][1]:.3f}] | "
                  f"Spec {v['specificity_mean']:.3f}[{v['specificity_CI'][0]:.3f},{v['specificity_CI'][1]:.3f}] | "
                  f"F1 {v['f1_mean']:.3f}[{v['f1_CI'][0]:.3f},{v['f1_CI'][1]:.3f}]")
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
        auc_all_cate = cal_auc_multi_class(bag_labels, bag_preds)
        for class_id, auc_class_i in enumerate(auc_all_cate):
            writer.add_scalar('{}_bag_auc_class{}'.format(prefix, class_id), auc_class_i, epoch)
        bag_acc = (bag_preds_logit == bag_labels).sum() / len(bag_labels)
        print(f"[{prefix}] Epoch:{epoch} slide_level_acc: {bag_acc:.4f}")
        writer.add_scalar(f'{prefix}_slide_acc', bag_acc, epoch)
        bag_acc_all[prefix] = bag_acc

        print('[{}] Epoch:{} test_slide_auc: {}'.format(prefix, epoch, auc_all_cate))
        print('[{}] Epoch:{} test_slide_kappa: {}'.format(prefix, epoch, cohen_kappa_score(bag_labels, bag_preds_logit)))
        y_true_slide = np.array(bag_labels)
        y_pred_slide = np.array(bag_preds_logit)

        # ===== 全局指标CI =====
        acc_mean, acc_ci = bootstrap_ci(y_true_slide, y_pred_slide, accuracy_score)
        f1_macro_mean, f1_macro_ci = bootstrap_ci(y_true_slide, y_pred_slide, f1_score, average='macro')
        prec_mean, prec_ci = bootstrap_ci(y_true_slide, y_pred_slide, precision_score, average='macro')
        recall_mean, recall_ci = bootstrap_ci(y_true_slide, y_pred_slide, recall_score, average='macro')

        print(f"\n[{prefix}] Slide-level Metrics with 95% CI")
        print(f"Accuracy: {acc_mean:.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]")
        print(f"F1-macro: {f1_macro_mean:.3f} [{f1_macro_ci[0]:.3f}, {f1_macro_ci[1]:.3f}]")
        print(f"Precision-macro: {prec_mean:.3f} [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]")
        print(f"Recall-macro: {recall_mean:.3f} [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")

        # ===== 各类别CI =====
        ci_slide_class = bootstrap_ci_per_class(y_true_slide, y_pred_slide, num_classes=len(class_names))
        print(f"\nPer-class Slide-level metrics (mean [95%CI]):")
        for i, cls in enumerate(class_names):
            v = ci_slide_class[i]
            print(f"{cls}: Sens {v['sensitivity_mean']:.3f}[{v['sensitivity_CI'][0]:.3f},{v['sensitivity_CI'][1]:.3f}] | "
                  f"Spec {v['specificity_mean']:.3f}[{v['specificity_CI'][0]:.3f},{v['specificity_CI'][1]:.3f}] | "
                  f"F1 {v['f1_mean']:.3f}[{v['f1_CI'][0]:.3f},{v['f1_CI'][1]:.3f}]")
        print(confusion_matrix(bag_labels, bag_preds_logit))

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

def rebalance_dataset(train_patches, train_labels, sample_ratio=[1.0, 1.0, 10]):
    if type(train_patches) is not np.ndarray:
        train_patches = np.array(train_patches)

    class_id = np.unique(train_labels)
    class_ratio = [np.sum(train_labels == i)/train_labels.shape[0] for i in class_id]
    for i, j in enumerate(class_ratio):
        print("Class {} ratio in training dataset: {:.4f}".format(i, j))
    if sample_ratio is None:
        blance_class_ratio = 1/len(class_ratio)
        sample_ratio = [blance_class_ratio/i for i in class_ratio]
    idx_class_i_new_all = []
    for i, class_i in enumerate(class_id):
        idx_class_i = np.where(train_labels == class_i)[0]
        resample_idx = resample(len(idx_class_i), sample_ratio[i])
        idx_class_i_new = idx_class_i[resample_idx]
        idx_class_i_new_all.append(idx_class_i_new)
    idx_class_i_new_all = np.concatenate(idx_class_i_new_all)
    train_patches = train_patches[idx_class_i_new_all]
    train_labels = train_labels[idx_class_i_new_all]

    class_id = np.unique(train_labels)
    class_ratio = [np.sum(train_labels == i)/train_labels.shape[0] for i in class_id]
    for i, j in enumerate(class_ratio):
        print("New class {} ratio in training dataset: {:.4f}".format(i, j))
    # train_patches = train_patches.tolist()
    return train_patches, train_labels


def resample(length_arr, ratio):
    if ratio <= 1:
        idx_choice = np.random.choice(length_arr, int(length_arr*ratio), replace=False)
    else:
        ratio = int(ratio)
        idx_choice = np.repeat(np.arange(length_arr), ratio)
    return idx_choice

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

    # train_ds.all_patches_feat, train_ds.all_patches_label = rebalance_dataset(train_ds.all_patches_feat, train_ds.all_patches_label, sample_ratio=[1, 1, 1, 1, 10])

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
        eval(epoch=epoch, model=model, loader=[InternalVal_loader, ExternalVal_loader, ExternalHDVal_loader], dev=dev, writer=writer, threshold_dict=threshold_dict)
