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
import datetime
import utliz
import random
import torchvision.transforms as transforms
import re
from PIL import Image, ImageFilter
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image

color_map = {
        0: [181, 46, 87],  # red
        1: [204, 164, 79],    # green
        2: [85, 155, 93],  # yellow
        3: [96, 110, 173],  # blue
        4: [134, 106, 163],    # purple
}

def color_mask(color, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[:] = color
        return mask

def eval_a_slide_5classes(model, patches_feat_path, patches_filename_path, slide_patches_path, dev='cuda:0', threshold=0.9):
    slide_name = slide_patches_path.split("/")[-2]
    patch_feats = np.load(patches_feat_path)         
    patch_names = np.load(patches_filename_path)    
    patch_all = []
    patch_loc_all = []
    patch_pred_all = []
    pred_prob_all = torch.zeros([len(patch_names), 5])

    for patch_name in patch_names:
        patch_name = str(patch_name).strip()
        base_name = patch_name.split("/")[-1]
        img_path = os.path.join(slide_patches_path, base_name)
        img = Image.open(img_path).convert('RGB')
        patch_all.append(img)
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

    with torch.no_grad():
        patch_feats = torch.from_numpy(patch_feats).float().to(dev)
        final = model(patch_feats)
        final_prob = torch.softmax(final, dim=1)
    pred_prob_all[:] = final_prob.detach().cpu()
    pred_logit = pred_prob_all.argmax(1)
        
    pred_logit_modified = pred_logit.clone() if torch.is_tensor(pred_logit) else pred_logit.copy()
    condition_mask = pred_prob_all[:, 4] > threshold
    pred_logit_modified[condition_mask] = 4
    pred_logit = pred_logit_modified

    patch_pred_all = pred_logit

    patch_loc_all = np.concatenate([patch_loc_all[:, 1:2], patch_loc_all[:, 0:1]], axis=1)
    patch_loc_all[:, 0] = patch_loc_all[:, 0].max() - patch_loc_all[:, 0]

    patch_h, patch_w = patch_all.shape[1], patch_all.shape[2]
    x_min, y_min = patch_loc_all.min(axis=0)
    x_max, y_max = patch_loc_all.max(axis=0)

    whole_slide_h = x_max - x_min + patch_h
    whole_slide_w = y_max - y_min + patch_w

    whole_slide = np.ones((whole_slide_h, whole_slide_w, 3), dtype=np.uint8) * 255

    for i in range(len(patch_all)):
        x_rel = patch_loc_all[i, 0] - x_min
        y_rel = patch_loc_all[i, 1] - y_min
        whole_slide[x_rel:x_rel+patch_h, y_rel:y_rel+patch_w] = patch_all[i]
    whole_slide = sharpen_image(whole_slide)

    mask_img = np.ones((whole_slide_h, whole_slide_w, 3), dtype=np.uint8) * 255

    for i in range(len(patch_all)):
        if pred_prob_all[i, pred_logit[i]] > 0:
            x_rel = patch_loc_all[i, 0] - x_min
            y_rel = patch_loc_all[i, 1] - y_min
            color = color_map[int(patch_pred_all[i])]
            mask_patch = color_mask(color, (patch_h, patch_w, 3))
            mask_img[x_rel:x_rel+patch_h, y_rel:y_rel+patch_w] = mask_patch

    save_path_1 = os.path.join("./figures_5cls", slide_name + "mask.svg")
    show_img(mask_img, save_file_name=save_path_1, format='svg')
    save_path_2 = os.path.join("./figures_5cls", slide_name + "original.svg")
    show_img(whole_slide, save_file_name=save_path_2, format='svg')
    return 0

def sharpen_image(img):
    pil_img = Image.fromarray(img)
    pil_img = pil_img.filter(ImageFilter.SHARPEN) 
    return np.array(pil_img)


def show_img(img, save_file_name='',format='svg', dpi=600, title=''):
    if type(img) == torch.Tensor:
        img = img.cpu().detach().numpy()
    if len(img.shape) == 3:  # HxWx3 or 3xHxW, treat as RGB image
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
    fig = plt.figure()
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.title(title)
    if save_file_name != '':
        plt.savefig(save_file_name, format=format, dpi=dpi, pad_inches=0.0, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.colorbar()
        plt.show()
    return


def load_ckpt(model, ckpt_path):
    state = torch.load(ckpt_path)
    model.load_state_dict(state['model'])
    epoch = state['epoch']
    threshold4 = state.get('threshold4', None)
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

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')
    parser.add_argument('--resume', default='', type=str, help='resume ckpt path')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    patches_path = ""
    patches_feat_path = ""
    patches_filename_path = ""
    dev = 'cuda:0'
    model = prediction_head()
    model.to(dev)

    start_epoch = 0
    if args.resume != '':
        model, start_epoch, threshold = load_ckpt(model, args.resume)
        start_epoch = start_epoch + 1
        print("Load from {}".format(args.resume))

    eval_a_slide_5classes(model, patches_feat_path, patches_filename_path, patches_path, dev=dev, threshold=threshold)
