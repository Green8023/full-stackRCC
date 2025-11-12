import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import random
from new_dataset_classification import get_train_test_ds_MultiCenter_region_TCGA_huadong_tsne, get_train_ds_ZSCenter_tsne
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

class Region_5Classes_Feat(torch.utils.data.Dataset):
    def __init__(self, ds_data, ds_label, return_bag=False):
        self.all_slide_info = ds_data
        self.all_slide_label = ds_label
        self.return_bag = return_bag
        
        self.feature_arrays = {}
        for slide in self.all_slide_info:
            for patch in slide:
                path = patch['feature_path']
                if path not in self.feature_arrays:
                    self.feature_arrays[path] = np.load(path, mmap_mode='r')

        if not return_bag:
            self.all_patches_info = sum(self.all_slide_info, [])
            self.all_patches_label = np.concatenate(self.all_slide_label)
            filtered_info, filtered_label = [], []

            for info, label in zip(self.all_patches_info, self.all_patches_label):
                if label != -1:
                    filtered_info.append(info)
                    filtered_label.append(label)

            self.all_patches_info = filtered_info
            self.all_patches_label = np.array(filtered_label)

    def __getitem__(self, index):
        patch_info = self.all_patches_info[index]
        feature = self.feature_arrays[patch_info['feature_path']][patch_info['feature_index']]
        label = self.all_patches_label[index]
        return torch.tensor(feature), torch.tensor(label, dtype=torch.long), index

    def __len__(self):
        return len(self.all_patches_info)

def sample_from_dataset_per_class(dataset, num_per_class=5000, batch_size=256, num_classes=5):
    all_indices = list(range(len(dataset)))
    class_to_indices = {i: [] for i in range(num_classes)}
    for idx in all_indices:
        _, label, _ = dataset[idx]
        class_to_indices[label.item()].append(idx)

    sampled_indices = []
    for class_idx, indices in class_to_indices.items():
        if len(indices) < num_per_class:
            print(f"{class_idx} sample not enough，only {len(indices)}")
        sampled = random.sample(indices, min(num_per_class, len(indices)))
        sampled_indices.extend(sampled)

    subset = Subset(dataset, sampled_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    all_features, all_labels = [], []
    for X_tensor, Y_tensor, _ in tqdm(loader, desc="Loading features (per class)", leave=False):
        all_features.append(X_tensor.numpy())
        all_labels.append(Y_tensor.numpy())

    X_out = np.concatenate(all_features, axis=0)
    Y_out = np.concatenate(all_labels, axis=0)
    return X_out, Y_out

def run_tsne_and_plot(X, Y, dataset_name, save_prefix='tsne_output'):
    class_names = ['Normal', 'Tumor', 'Capsule', 'Necrosis', 'Sarcomatoid']
    colors = {
        0: '#1f77b4',  # Normal
        1: '#ff7f0e',  # Tumor
        2: '#2ca02c',  # Capsule
        3: '#d62728',  # Necrosis
        4: '#9467bd',  # Sarcomatoid
    }

    print(f"➡️ Running t-SNE for {dataset_name} ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    for class_idx in range(5):
        indices = (Y == class_idx)
        plt.scatter(
            X_embedded[indices, 0],
            X_embedded[indices, 1],
            s=3, label=class_names[class_idx],
            alpha=0.6, color=colors[class_idx]
        )

    plt.legend(fontsize=12)
    plt.title(f"t-SNE of {dataset_name} Feature Distribution", fontsize=10)
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_{dataset_name}.png", dpi=300)
    plt.close()
    print(f"✅ Saved: {save_prefix}_{dataset_name}.png")

if __name__ == '__main__':
    ZS, _ = get_train_ds_ZSCenter_tsne(downsample=0.3)
    TCGA, huadong = get_train_test_ds_MultiCenter_region_TCGA_huadong_tsne(downsample=0.5)

    train_patches, train_labels = ZS[0], ZS[1]
    test_patches, test_labels = TCGA[0], TCGA[1]
    HDtest_patches, HDtest_labels = huadong[0], huadong[1]

    InternalTrain_ds = Region_5Classes_Feat(train_patches, train_labels)
    ExternalTest_ds = Region_5Classes_Feat(test_patches, test_labels)
    HuaDong_ds = Region_5Classes_Feat(HDtest_patches, HDtest_labels)

    X_train, Y_train = sample_from_dataset_per_class(InternalTrain_ds, num_per_class=5000)
    run_tsne_and_plot(X_train, Y_train, dataset_name='Internal', save_prefix='tsne_center')

    X_tcga, Y_tcga = sample_from_dataset_per_class(ExternalTest_ds, num_per_class=5000)
    run_tsne_and_plot(X_tcga, Y_tcga, dataset_name='TCGA', save_prefix='tsne_center')

    X_hd, Y_hd = sample_from_dataset_per_class(HuaDong_ds, num_per_class=5000)
    run_tsne_and_plot(X_hd, Y_hd, dataset_name='HuaDong', save_prefix='tsne_center')
