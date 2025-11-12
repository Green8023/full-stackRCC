import argparse
import glob
import warnings
import os
import time
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
# from dataset_nuclear import TCGA_nuclear
import datetime
import utliz
import random
import torchvision.transforms as transforms
import re
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colorbar import ColorbarBase

color_map = {
    0: [200, 54, 54],    
    1: [70, 114, 178],    
    2: [87, 160, 96],     
    3: [230, 145, 56],    
    4: [139, 102, 160],   
    5: [64, 170, 173],    
    6: [232, 150, 190],   
    7: [158, 188, 80],    
    8: [240, 236, 120]    
}
def color_mask(color, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[:] = color
        return mask

def lerp_color(color1, color2, t):
    return tuple([int((1-t)*c1 + t*c2) for c1, c2 in zip(color1, color2)])

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

def eval_a_slide_9classes(model_5cls, model_pathocls, patches_feat_path, patches_filename_path, slide_patches_path,
                          threshold4, targetcls, dev='cuda:0'):
    slide_name = slide_patches_path.split("/")[-2]
    patch_feats = np.load(patches_feat_path)         
    patch_feats = torch.from_numpy(patch_feats).float().to(dev)
    patch_names = np.load(patches_filename_path)     
    patch_all, patch_loc_all = [], []
    pred_prob_all = torch.zeros([len(patch_feats), 5])
    pred_prob_all_patho = torch.zeros([len(patch_feats), 9])
    for patch_name in patch_names:
        patch_name = str(patch_name).strip()
        base_name = patch_name.split("/")[-1]
        img_path = os.path.join(slide_patches_path, base_name)
        img = Image.open(img_path).convert('RGB')
        patch_all.append(np.array(img))
        basename = os.path.basename(patch_name)
        m = re.search(r'pos_(\d+)_(\d+)_label', basename)
        if m:
            x = int(m.group(1)) // 4   
            y = int(m.group(2)) // 4
        else:
            x, y = 0, 0 
        patch_loc_all.append([x, y])
        
    patch_all = np.array(patch_all)      
    patch_loc_all = np.array(patch_loc_all)  

    model_5cls = model_5cls.eval()
    for i in range(0, len(patch_feats), 512):
        batch = patch_feats[i:i+512]
        batch = torch.tensor(batch, device=dev)
        with torch.no_grad():
            final = model_5cls(batch)
            final_prob = torch.softmax(final, dim=1)
            # log
        pred_prob_all[i:i+batch.shape[0]] = final_prob.detach().cpu()

    pred_logit_5cls = pred_prob_all.argmax(1)
    pred_logit_modified = pred_logit_5cls.clone() if torch.is_tensor(pred_logit_5cls) else pred_logit_5cls.copy()
    condition_mask = pred_prob_all[:, 4] > threshold4
    pred_logit_modified[condition_mask] = 4
    pred_logit_5cls = pred_logit_modified
    
    pred_patho_all_classes = np.zeros(len(patch_all), dtype=int)
    model_pathocls = model_pathocls.eval()
    for i in range(0, len(patch_feats), 512):
        batch = patch_feats[i:i+512]
        batch = torch.tensor(batch, device=dev)
        with torch.no_grad():
            final = model_pathocls(batch)
            final_prob = torch.softmax(final, dim=1)
        # log
        pred_prob_all_patho[i:i+batch.shape[0]] = final_prob.detach().cpu()        
    pred_logit_patho = pred_prob_all_patho.argmax(1)
    
    for i in range(len(patch_feats)):
        if pred_logit_5cls[i] == 0:
            pred_prob_all_patho[i, :] = -1
            
    for i in range(len(patch_feats)):
        if pred_logit_5cls[i] != 0:
            pred_patho_all_classes[i] = pred_logit_patho[i]
        else:
            pred_patho_all_classes[i] = -1
    print("All patches now have patho labels.")
    patch_h, patch_w = patch_all.shape[1], patch_all.shape[2]
    patch_loc_all = np.concatenate([patch_loc_all[:, 1:2], patch_loc_all[:, 0:1]], axis=1)
    patch_loc_all[:, 0] = patch_loc_all[:, 0].max() - patch_loc_all[:, 0]

    x_min, y_min = patch_loc_all.min(axis=0)
    x_max, y_max = patch_loc_all.max(axis=0)
    whole_slide_h = x_max - x_min + patch_h
    whole_slide_w = y_max - y_min + patch_w

    pred_slide_color = Image.new("RGB", (whole_slide_w, whole_slide_h), (255, 255, 255))  
    pred_slide_blend = Image.new("RGB", (whole_slide_w, whole_slide_h), (255, 255, 255)) 
    heat_slide = Image.new("RGB", (whole_slide_w, whole_slide_h), (255, 255, 255))  

    for i in range(len(patch_all)):
        x_rel = patch_loc_all[i, 0] - x_min
        y_rel = patch_loc_all[i, 1] - y_min
        cls_id = pred_patho_all_classes[i]

        patch_img = Image.fromarray(patch_all[i])
        if cls_id == -1:
            color = (128, 128, 128) 
        else:
            color = tuple(map(int, color_map[cls_id]))
        color_img = Image.new("RGB", (patch_w, patch_h), color)
        blended = Image.blend(patch_img, color_img, alpha=0.4)

        pred_slide_color.paste(color_img, (x_rel, y_rel))
        pred_slide_blend.paste(blended, (x_rel, y_rel))
      
    for i in range(len(patch_all)):
        x_rel = patch_loc_all[i, 0] - x_min
        y_rel = patch_loc_all[i, 1] - y_min
        prob = float(pred_prob_all_patho[i, targetcls]) 
        if prob == -1:
            color = (128, 128, 128) 
        else:
            color = lerp_color((255, 255, 255), color_map[targetcls], prob)
        color_img = Image.new("RGB", (patch_w, patch_h), color)
        heat_slide.paste(color_img, (x_rel, y_rel))

    save_dir = './figures_patho_cls_final'
    os.makedirs(save_dir, exist_ok=True)

    legend_info = {
        'color_map': [color_map[i] for i in range(len(color_map))], 
        'class_names': ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8"]
    }
    show_img(np.array(pred_slide_color), save_file_name=os.path.join(save_dir, f"{slide_name}_mask.svg"), format='svg', legend_info=legend_info)
    show_img(np.array(pred_slide_blend), save_file_name=os.path.join(save_dir, f"{slide_name}_fused.svg"), format='svg', legend_info=legend_info)
    show_heat_map(np.array(heat_slide), save_file_name=os.path.join(save_dir, f"{slide_name}_heatmap_cls.svg"), format='svg', target_color=color_map[targetcls])
    print(f"Saved 3 images for {slide_name}")

    return 0

def show_img(img, save_file_name='', format='svg', dpi=600, title='', legend_info=None):
    if type(img) == torch.Tensor:
        img = img.cpu().detach().numpy()
    if len(img.shape) == 3:  
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title(title)

    if legend_info is not None:
        color_map = legend_info['color_map']
        class_names = legend_info['class_names']
        num_classes = len(class_names)
        rect_w, rect_h, gap = 70, 70, 20
        legend_height = num_classes * rect_h + (num_classes + 1) * gap
        legend_img = np.ones((legend_height, rect_w + 80, 3), dtype=np.uint8) * 255
        for i in range(num_classes):
            y0 = gap + i * (rect_h + gap)
            color = color_map[i]
            legend_img[y0:y0+rect_h, gap:gap+rect_w, :] = color
        ax_legend = fig.add_axes([0.88, 0.2, 0.03, 0.35]) 
        ax_legend.imshow(legend_img)
        ax_legend.axis('off')
        for i in range(num_classes):
            y0 = gap + i * (rect_h + gap) + rect_h // 2
            ax_legend.text(gap + rect_w + 10, y0, class_names[i], va='center', ha='left', fontsize=8, color='black', fontweight='bold')

    if save_file_name != '':
        plt.savefig(save_file_name, format=format, dpi=dpi, pad_inches=0.0, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return

def show_heat_map(img, save_file_name='', format='svg', dpi=600, title='', target_color=None):
    if type(img) == torch.Tensor:
        img = img.cpu().detach().numpy()
    if len(img.shape) == 3:  # HxWx3 or 3xHxW, treat as RGB image
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    ax.set_title(title)

    if target_color is not None:
        cmap = LinearSegmentedColormap.from_list(
            "prob_cmap", [(0, (1,1,1)), (1, tuple(np.array(target_color)/255.0))]
        )
        norm = Normalize(vmin=0, vmax=1)
        cbar_ax = fig.add_axes([0.88, 0.2, 0.015, 0.18])  
        cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cbar_ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        cbar_ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])

        for spine in cbar_ax.spines.values():
            spine.set_visible(False)

    if save_file_name != '':
        plt.savefig(save_file_name, format=format, dpi=dpi, pad_inches=0.0, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return

def load_ckpt_5cls(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['model'])
    epoch = state['epoch']
    threshold4 = state.get('threshold4', None)  
    return model, epoch, threshold4


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
    
class prediction_head_5cls(nn.Module):
    def __init__(self):
        super(prediction_head_5cls, self).__init__()
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

class prediction_head_pathocls(nn.Module):
    def __init__(self):
        super(prediction_head_pathocls, self).__init__()
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

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')

    parser.add_argument('--workers', default=0, type=int,help='number workers (default: 6)')
    parser.add_argument('--resume_5cls', default='', type=str, help='resume ckpt path')
    parser.add_argument('--resume', default='', type=str, help='resume ckpt path')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    patches_path = ""
    patches_feat_path = ""
    patches_filename_path = ""
    dev = 'cuda:0'
    model_5cls = prediction_head_5cls()
    model_pathocls = prediction_head_pathocls()
    model_5cls.to(dev)
    model_pathocls.to(dev)

    model_5cls, start_epoch, threshold4 = load_ckpt_5cls(model_5cls, args.resume_5cls)
    model_pathocls, start_epoch, threshold_dict = load_ckpt(model_pathocls, args.resume)
    print("Load from {} and {}".format(args.resume_5cls, args.resume))

    eval_a_slide_9classes(model_5cls, model_pathocls, patches_feat_path, patches_filename_path, patches_path, threshold4=threshold4, targetcls=6, dev=dev)
