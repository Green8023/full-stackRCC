from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
def plot_auc_curves(y_true, y_score, classes, title, save_path):

    plt.figure(figsize=(8, 6))
    
    for i, cls in enumerate(classes):
        y_true_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
    
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve', fontsize=16)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, frameon=True, edgecolor='black', framealpha=0.8)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_multiclass_roc(y_true, y_score, num_classes, save_path, class_names=None):

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

def plot_per_class_roc_across_datasets(y_true_list, y_score_list, dataset_names, class_names, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    n_classes = len(class_names)

    for i in range(n_classes):
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        for y_true, y_score, name in zip(y_true_list, y_score_list, dataset_names):
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            if not np.any(y_true_bin[:, i]): 
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.6, label='Random Guess')
        ax.set_aspect('equal', adjustable='box')
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title(f'ROC for Class: {class_names[i]}', fontsize=14)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=12)
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, frameon=True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        save_path = os.path.join(save_dir, f'Class_{i}_{class_names[i]}.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"save")