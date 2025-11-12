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
from dataset import Region_5Classes_Feat, get_train_test_ds_MultiCenter_region_5Cls
import datetime
import utliz
import random
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, recall_score, confusion_matrix, precision_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- 字体和 SVG 输出设置 ----
mpl.rcParams['font.family'] = 'DejaVu Sans' 
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, random_state=42, **kwargs):
    """计算95%置信区间（Bootstrap法）"""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    stats = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)  # 有放回采样
        if len(np.unique(y_true[indices])) < 2:  # 避免只采到单一类别
            continue
        score = metric_fn(y_true[indices], y_pred[indices], **kwargs)
        stats.append(score)
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    mean = np.mean(stats)
    return mean, (lower, upper)

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

            try:
                f1 = f1_score(y_t, y_p, labels=[i], average='macro', zero_division=0)
            except:
                f1 = np.nan

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

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None, normalize=True):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))

    # 设置颜色范围固定为0-1，强调对比
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1 if normalize else None)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()

def plot_multiclass_roc(y_true, y_score, num_classes, save_path, class_names=None):
    """
    绘制多分类 ROC 曲线（一个图里画所有类别），美化样式适配论文级别展示。
    """

    # ---- One-hot 编码 ----
    y_true_onehot = np.eye(num_classes)[y_true]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ---- 开始绘图 ----
    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))

    legend_handles = []
    legend_labels = []

    for i in range(num_classes):
        name = class_names[i] if class_names else f"Class {i}"
        line, = plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                         label=f'{name} (AUC = {roc_auc[i]:.3f})')
        legend_handles.append(line)
        legend_labels.append(f'{name} (AUC = {roc_auc[i]:.3f})')

    # 添加随机猜测线
    line_random, = plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Guess (AUC = 0.5)')
    legend_handles.append(line_random)
    legend_labels.append('Random Guess (AUC = 0.5)')

    # 图形美化
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Multiclass ROC Curve', fontsize=14)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    # 设置 legend 到图右侧
    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
              fontsize=10, frameon=True, edgecolor='black', framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"✅ 已保存 ROC 图到: {save_path}")

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

def eval(epoch, model, loader, dev, writer, threshold):
    for loader_i, prefix in zip(loader, ["InternalTest", "ExternalTest", "HuaDongTest"]):
        model = model.eval()
        patch_label_gt = torch.zeros([loader_i.dataset.__len__()]).long()
        pred_prob_all = torch.zeros([loader_i.dataset.__len__(), 5])
        # for iter, (data, label, selected) in enumerate(tqdm(loader, ascii=True, desc='testing epoch_{}'.format(epoch))):
        for iter, (data, label, selected) in enumerate(loader_i):
            patch_label_gt[selected] = label
            data = data.to(dev)
            with torch.no_grad():
                final = model(data)
                final_prob = torch.softmax(final, dim=1)
            # log
            pred_prob_all[selected, :] = final_prob.detach().cpu()

        pred_logit = pred_prob_all.argmax(1)
        pred_logit_modified = pred_logit.clone() if torch.is_tensor(pred_logit) else pred_logit.copy()
        condition_mask = pred_prob_all[:, 4] > threshold
        pred_logit_modified[condition_mask] = 4
        pred_logit = pred_logit_modified
        train_acc = torch.sum(pred_logit == patch_label_gt)/patch_label_gt.shape[0]
        print('[{}] Epoch:{} test_patch_acc: {}'.format(prefix, epoch, train_acc))

        auc_all_cate = cal_auc_multi_class(patch_label_gt, pred_prob_all)
        print('[{}] Epoch:{} test_patch_auc: {}'.format(prefix, epoch, auc_all_cate))
        print('[{}] Epoch:{} test_kappa: {}'.format(prefix, epoch, cohen_kappa_score(patch_label_gt, pred_logit)))
        print(confusion_matrix(patch_label_gt, pred_logit))
        
        y_true_np = patch_label_gt.numpy()
        y_pred_np = pred_logit.numpy()
        ci_results = bootstrap_ci_per_class(y_true_np, y_pred_np, num_classes=5, n_bootstraps=1000)

        print(f"\n[{prefix}] Epoch:{epoch} --- Per-class metrics with 95% CI ---")
        for i in range(5):
            sen_mean = ci_results[i]['sensitivity_mean']
            sen_ci = ci_results[i]['sensitivity_CI']
            spe_mean = ci_results[i]['specificity_mean']
            spe_ci = ci_results[i]['specificity_CI']
            f1_mean = ci_results[i]['f1_mean']
            f1_ci = ci_results[i]['f1_CI']

            print(f"Class {i}:")
            print(f"  Sensitivity: {sen_mean:.3f} ({sen_ci[0]:.3f}-{sen_ci[1]:.3f})")
            print(f"  Specificity: {spe_mean:.3f} ({spe_ci[0]:.3f}-{spe_ci[1]:.3f})")
            print(f"  F1-score:    {f1_mean:.3f} ({f1_ci[0]:.3f}-{f1_ci[1]:.3f})")

        # 加权平均指标
        weighted_f1 = f1_score(y_true_np, y_pred_np, average='macro')
        weighted_recall = recall_score(y_true_np, y_pred_np, average='macro')
        weighted_precision = precision_score(y_true_np, y_pred_np, average='macro')
        mean_f1, ci_f1 = bootstrap_ci(y_true_np, y_pred_np, f1_score, average='macro')
        mean_recall, ci_recall = bootstrap_ci(y_true_np, y_pred_np, recall_score, average='macro')
        mean_precision, ci_precision = bootstrap_ci(y_true_np, y_pred_np, precision_score, average='macro')
        mean_acc, ci_acc = bootstrap_ci(y_true_np, y_pred_np, accuracy_score)
        print(f"[{prefix}] Weighted F1: {mean_f1:.4f} (95% CI: {ci_f1[0]:.4f}-{ci_f1[1]:.4f})")
        print(f"[{prefix}] Weighted Recall: {mean_recall:.4f} (95% CI: {ci_recall[0]:.4f}-{ci_recall[1]:.4f})")
        print(f"[{prefix}] Weighted Precision: {mean_precision:.4f} (95% CI: {ci_precision[0]:.4f}-{ci_precision[1]:.4f})")
        print(f"[{prefix}] Accuracy: {mean_acc:.4f} (95% CI: {ci_acc[0]:.4f}-{ci_acc[1]:.4f})")

#         writer.add_scalar('{}_patch_acc'.format(prefix), train_acc, epoch)
#         for class_id, auc_class_i in enumerate(auc_all_cate):
#             writer.add_scalar('{}_patch_auc_class{}'.format(prefix, class_id), auc_class_i, epoch)

        save_dir = os.path.join("./results_ckpt_region", writer.logdir.split('/')[-1])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
#         y_true_np = patch_label_gt.numpy()
#         y_score_np = pred_prob_all.numpy()

#         class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  # 可替换为真实名称
#         roc_save_path = os.path.join("./results_ckpt_region", f"{prefix}_Epoch{epoch}_ROC.svg")

#         plot_multiclass_roc(
#             y_true=y_true_np,
#             y_score=y_score_np,
#             num_classes=5,
#             save_path=roc_save_path,
#             class_names=class_names
#         )
        
#         cm_save_path = os.path.join("./results_ckpt_region", f"{prefix}_Epoch{epoch}_ConfusionMatrix.svg")
#         plot_confusion_matrix(
#             y_true=patch_label_gt.numpy(),
#             y_pred=pred_logit.numpy(),
#             classes=class_names,
#             save_path=cm_save_path,
#             normalize=True  # 或 True，根据需要
#         )
        if auc_all_cate[0] > 0.90:
            save_ckpt(model, epoch, save_name=os.path.join(save_dir, "Epoch_{}_AUC_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}".format(
                epoch, auc_all_cate[0], auc_all_cate[1], auc_all_cate[2], auc_all_cate[3], auc_all_cate[4])), threshold4=threshold)
            print("[model saved]")
    return 0


def save_ckpt(model, epoch, save_name, threshold4=None):
    state = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    if threshold4 is not None:  # 只有在需要的时候才保存
        state['threshold4'] = threshold4
    torch.save(state, save_name)
    return 0


def load_ckpt(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['model'])
    epoch = state['epoch']
    threshold4 = state.get('threshold4', None)  # 没有就返回 None
    return model, epoch, threshold4


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
    parser.add_argument('--epochs', default=4, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size (default: 256)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--lrdrop', default=200, type=int, help='multiply LR by 0.5 every (default: 150 epochs)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: (-5)')
    parser.add_argument('--dtype', default='f64',choices=['f64','f32'], type=str, help='SK-algo dtype (default: f64)')

    # housekeeping
    parser.add_argument('--device', default='0', type=str, help='GPU devices to use for storage and model')
    parser.add_argument('--modeldevice', default='0', type=str, help='GPU numbers on which the CNN runs')
    parser.add_argument('--exp', default='self-label-default', help='path to experiment directory')
    parser.add_argument('--workers', default=6, type=int,help='number workers (default: 6)')
    parser.add_argument('--comment', default='ExternalTest_addDropOut', type=str, help='name for tensorboardX')
    parser.add_argument('--resume', default='/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/results_ckpt_region/20250903_213042_Bs512_lr0.0001_downsample1/Epoch_2_AUC_0.9900_0.9761_0.9904_0.9868_0.9786', type=str, help='resume ckpt path')
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
            nn.Linear(2048, 5),
        )

    def forward(self, x):
        return self.fcs(x)


if __name__ == '__main__':
    args = get_parser()

    InternalTrain, InternalTest, ExternalTest, huadong = get_train_test_ds_MultiCenter_region_5Cls(downsample=args.downsample)

    train_patches, train_labels, train_names_slide, train_names_patch = InternalTrain[0:4]
    InternalTest_patches, InternalTest_labels, InternalTest_names_slide, InternalTest_names_patch = InternalTest[0:4]
    test_patches, test_labels, test_names_slide, test_names_patch = ExternalTest[0:4]
    HDtest_patches, HDtest_labels, HDtest_names_slide, HDtest_names_patch = huadong[0:4]

    print("Num of slides in InternalTrain/InternalTest/ExternalTest: {} / {} / {}/ {}".format(
        len(train_patches), len(InternalTest_patches), len(test_patches), len(HDtest_patches)))

    train_ds = Region_5Classes_Feat(train_patches, train_labels)
    InternalVal_ds =Region_5Classes_Feat(InternalTest_patches, InternalTest_labels)
    ExternalVal_ds =Region_5Classes_Feat(test_patches, test_labels)
    ExternalHDVal_ds =Region_5Classes_Feat(HDtest_patches, HDtest_labels)

    print("Num of patches in InternalTrain/InternalTest/ExternalTest/HuaDongTest: {} / {} / {} / {}".format(
        len(train_ds), len(InternalVal_ds), len(ExternalVal_ds), len(ExternalHDVal_ds)))

    train_ds.all_patches_info, train_ds.all_patches_label = rebalance_dataset(np.array(train_ds.all_patches_info), np.array(train_ds.all_patches_label), sample_ratio=[1, 1, 5, 5, 10])

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

    dev = 'cuda:0'
    model = prediction_head()
    model.to(dev)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    start_epoch = 0
    if args.resume != '':
        model, start_epoch, threshold = load_ckpt(model, args.resume)
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))

    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Bs{}_lr{}_downsample{}".format(
        args.batch_size, args.lr, args.downsample)
    writer = SummaryWriter('./runs_Region5Classes_feat/%s' % name)

    for epoch in range(start_epoch, args.epochs):
        eval(epoch=epoch, model=model, loader=[InternalVal_loader, ExternalVal_loader, ExternalHDVal_loader], dev=dev, writer=writer, threshold=threshold)

    
