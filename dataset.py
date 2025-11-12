import os, sys
import random

import numpy as np
import pandas as pd
import torch
import glob
from skimage import io
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
from random import sample
from collections import defaultdict
from collections import Counter
from sklearn.model_selection import train_test_split

def count_patho_types(slide_patho_label):
    counter = Counter(slide_patho_label)
    print("各病理类型slide数量统计：")
    for k, v in sorted(counter.items()):
        print(f"类型 {k}: {v} 张slide")
    return counter

def course_match(index_bing_li_hao, name_i):
    # new_index = np.zeros_like(index_bing_li_hao)
    # for i in range(len(index_bing_li_hao)):
    #     new_name_i = index_bing_li_hao[i].replace('S2013', '13')
    #     new_name_i = new_name_i.replace('-0', '-')
    #     new_index[i] = new_name_i
    # idx_match = np.where(new_index == name_i)[0]

    name_components = name_i.split('-')
    if len(name_components) == 1:
        return []

    if name_components[0] in ['10', '11', '12', '13', '14']:
        name_components[0] = name_components[0].replace('10', 'S2010')
        name_components[0] = name_components[0].replace('11', 'S2011')
        name_components[0] = name_components[0].replace('12', 'S2012')
        name_components[0] = name_components[0].replace('13', 'S2013')
        name_components[0] = name_components[0].replace('14', 'S2014')
    elif name_components[0] in ['2010', '2011', '2012', '2013', '2014']:
        name_components[0] = name_components[0].replace('2010', 'S2010')
        name_components[0] = name_components[0].replace('2011', 'S2011')
        name_components[0] = name_components[0].replace('2012', 'S2012')
        name_components[0] = name_components[0].replace('2013', 'S2013')
        name_components[0] = name_components[0].replace('2014', 'S2014')
    elif name_components[0] in ['S11']:
        name_components[0] = name_components[0].replace('S11', 'S2011')

    if name_components[1] in ['001', '005', '1', '2']:
        new_name_i = name_components[0]
    elif len(name_components[1]) <= 5:
        name_components[1] = "{:0>5}".format(name_components[1])
        new_name_i = "-".join(name_components)
    else:
        print("[Unexpected Name] {}".format(name_i))
        raise

    idx_match = np.where(index_bing_li_hao == new_name_i)[0]

    # re-match
    if len(idx_match) == 0:
        name_components = name_i.split('-')
        if name_components[0] in ['10', '11', '12', '13', '14']:
            name_components[0] = name_components[0].replace('10', '10S')
            name_components[0] = name_components[0].replace('11', '11S')
            name_components[0] = name_components[0].replace('12', '12S')
            name_components[0] = name_components[0].replace('13', '13S')
            name_components[0] = name_components[0].replace('14', '14S')
        elif name_components[0] in ['2010', '2011', '2012', '2013', '2014']:
            name_components[0] = name_components[0].replace('2010', '10S')
            name_components[0] = name_components[0].replace('2011', '11S')
            name_components[0] = name_components[0].replace('2012', '12S')
            name_components[0] = name_components[0].replace('2013', '13S')
            name_components[0] = name_components[0].replace('2014', '14S')

        if name_components[1] in ['001', '005', '1', '2']:
            new_name_i = name_components[0]
        elif len(name_components[1]) <= 5:
            name_components[1] = "{:0>5}".format(name_components[1])
            new_name_i = "".join(name_components)
        else:
            print("[Unexpected Name] {}".format(name_i))
            raise

        idx_match = np.where(index_bing_li_hao == new_name_i)[0]

    return idx_match


def match_pathology_label_ZS(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_yi_ji_hao = slide_info['医技号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    slide_tnm_all = []
    slide_necrosis_all = [] # huaisi and rouliu
    slide_isup_all = []
    slide_size_all = []
    slide_yuHouFenZu_all = []
    slide_zhuyuanhao_all=[]
    for slide_name_i_raw in slide_name:
        slide_name_i = slide_name_i_raw.split(' ')[0]
        slide_name_i = slide_name_i.split('(')[0]
        slide_name_i = slide_name_i.replace('s', 'S')
        if "_" in slide_name_i:
            slide_name_i = slide_name_i.split('_')[0]
            # print("[Replace Name] {} --> {}".format(slide_name_i_raw, slide_name_i))
        if len(slide_name_i.split('-')) >= 3:
            slide_name_i = slide_name_i.split('-')[0]
            # print("[Replace Name] {} --> {}".format(slide_name_i_raw, slide_name_i))

        if '-' in slide_name_i:
            if slide_name_i.split('-')[1] in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6','a10', 'a15', 'a16', 'a17',
                                              '1', '2', '3', '4', 'N1']:
                slide_name_i = slide_name_i.split('-')[0]

        idx_match_0 = np.where(index_zhu_yuan_hao == slide_name_i)[0]
        idx_match_1 = np.where(index_yi_ji_hao == slide_name_i)[0]
        idx_match_2 = np.where(index_bing_li_hao == slide_name_i)[0]

        if len(idx_match_0) + len(idx_match_1) + len(idx_match_2) == 0:
            # print("[Course Matching Slide] {}".format(slide_name_i))
            idx_match_3 = course_match(index_bing_li_hao, slide_name_i)
            if len(idx_match_3) != 1 :
                print("[Slide Not Found or Found Twice] {}".format(slide_name_i))
                course_match(index_bing_li_hao, slide_name_i)
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
                slide_tnm = -1
                slide_necrosis = -1  # huaisi and rouliu
                slide_isup = -1
                slide_size = -1
                slide_yuHouFenZu = -1
                slide_zhuyuanhao = ""
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_3[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_3[0]]
                slide_prognosis = slide_info.to_numpy()[:, 30:36][idx_match_3[0]]
                slide_tnm = slide_info.to_numpy()[:, 19][idx_match_3[0]]
                slide_necrosis = slide_info.to_numpy()[:, [22, 25]][idx_match_3[0]]  # huaisi and rouliu
                slide_isup = slide_info.to_numpy()[:, 13][idx_match_3[0]]
                slide_size = slide_info.to_numpy()[:, 15][idx_match_3[0]]
                slide_yuHouFenZu = slide_info.to_numpy()[:, 55][idx_match_3[0]]
                slide_zhuyuanhao = index_zhu_yuan_hao[idx_match_3[0]]
        else:
            idx_match_final = np.unique(np.concatenate([idx_match_0, idx_match_1, idx_match_2]))
            if len(idx_match_final) != 1:
                print("[Slide Found Twice] {}".format(slide_name_i))
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
                slide_tnm = -1
                slide_necrosis = -1  # huaisi and rouliu
                slide_isup = -1
                slide_size = -1
                slide_yuHouFenZu = -1
                slide_zhuyuanhao = ""
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_final[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_final[0]]
                slide_prognosis = slide_info.to_numpy()[:, 30:36][idx_match_final[0]]
                slide_tnm = slide_info.to_numpy()[:, 19][idx_match_final[0]]
                slide_necrosis = slide_info.to_numpy()[:, [22, 25]][idx_match_final[0]]  # huaisi and rouliu
                slide_isup = slide_info.to_numpy()[:, 13][idx_match_final[0]]
                slide_size = slide_info.to_numpy()[:, 15][idx_match_final[0]]
                slide_yuHouFenZu = slide_info.to_numpy()[:, 55][idx_match_final[0]]
                slide_zhuyuanhao = index_zhu_yuan_hao[idx_match_final[0]]

        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
        slide_isup_all.append(slide_isup)
        slide_necrosis_all.append(slide_necrosis)
        slide_tnm_all.append(slide_tnm)
        slide_size_all.append(slide_size)
        slide_yuHouFenZu_all.append(slide_yuHouFenZu)
        slide_zhuyuanhao_all.append(slide_zhuyuanhao)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all, slide_isup_all, slide_necrosis_all, slide_tnm_all, slide_size_all, slide_yuHouFenZu_all, slide_zhuyuanhao_all


def match_pathology_label_ZS_touming(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_yi_ji_hao = slide_info['医技号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    slide_tnm_all = []
    slide_necrosis_all = [] # huaisi and rouliu
    slide_isup_all = []
    slide_size_all = []
    slide_yuHouFenZu_all = []
    slide_zhuyuanhao_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = slide_name_i_raw.split(' ')[0]
        slide_name_i = slide_name_i.split('(')[0]
        slide_name_i = slide_name_i.replace('s', 'S')
        slide_name_i = slide_name_i.replace('10-', 'S2010-')
        slide_name_i = slide_name_i.replace('S11-', 'S2011-')
        if "_" in slide_name_i:
            slide_name_i = slide_name_i.split('_')[0]
            # print("[Replace Name] {} --> {}".format(slide_name_i_raw, slide_name_i))
        if len(slide_name_i.split('-')) >= 3:
            slide_name_i = slide_name_i.split('-')[0]
            # print("[Replace Name] {} --> {}".format(slide_name_i_raw, slide_name_i))

        if '-' in slide_name_i:
            if slide_name_i.split('-')[1] in ['a1', 'a2', 'a3', 'a4', 'a5', 'a10', 'a15', 'a16', 'a17',
                                              '1', '2', '3', '4', 'N1']:
                slide_name_i = slide_name_i.split('-')[0]

        idx_match_0 = np.where(index_zhu_yuan_hao == slide_name_i)[0]
        idx_match_1 = np.where(index_yi_ji_hao == slide_name_i)[0]
        idx_match_2 = np.where(index_bing_li_hao == slide_name_i)[0]

        if len(idx_match_0) + len(idx_match_1) + len(idx_match_2) == 0:
            # print("[Course Matching Slide] {}".format(slide_name_i))
            idx_match_3 = course_match(index_bing_li_hao, slide_name_i)
            if len(idx_match_3) != 1 :
                print("[Slide Not Found or Found Twice] {}".format(slide_name_i))
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
                slide_tnm = -1
                slide_necrosis = -1  # huaisi and rouliu
                slide_isup = -1
                slide_size = -1
                slide_yuHouFenZu = -1
                slide_zhuyuanhao = ""
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_3[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_3[0]]
                slide_prognosis = slide_info.to_numpy()[:, 30:36][idx_match_3[0]]
                slide_tnm = slide_info.to_numpy()[:, 19][idx_match_3[0]]
                slide_necrosis = slide_info.to_numpy()[:, [22, 25]][idx_match_3[0]]  # huaisi and rouliu
                slide_isup = slide_info.to_numpy()[:, 13][idx_match_3[0]]
                slide_size = slide_info.to_numpy()[:, 15][idx_match_3[0]]
                slide_yuHouFenZu = slide_info.to_numpy()[:, 55][idx_match_3[0]]
                slide_zhuyuanhao = index_zhu_yuan_hao[idx_match_3[0]]
        else:
            idx_match_final = np.unique(np.concatenate([idx_match_0, idx_match_1, idx_match_2]))
            if len(idx_match_final) != 1:
                print("[Slide Found Twice] {}".format(slide_name_i))
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
                slide_tnm = -1
                slide_necrosis = -1  # huaisi and rouliu
                slide_isup = -1
                slide_size = -1
                slide_yuHouFenZu = -1
                slide_zhuyuanhao = ""
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_final[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_final[0]]
                slide_prognosis = slide_info.to_numpy()[:, 30:36][idx_match_final[0]]
                slide_tnm = slide_info.to_numpy()[:, 19][idx_match_final[0]]
                slide_necrosis = slide_info.to_numpy()[:, [22, 25]][idx_match_final[0]]  # huaisi and rouliu
                slide_isup = slide_info.to_numpy()[:, 13][idx_match_final[0]]
                slide_size = slide_info.to_numpy()[:, 15][idx_match_final[0]]
                slide_yuHouFenZu = slide_info.to_numpy()[:, 55][idx_match_final[0]]
                slide_zhuyuanhao = index_zhu_yuan_hao[idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
        slide_isup_all.append(slide_isup)
        slide_necrosis_all.append(slide_necrosis)
        slide_tnm_all.append(slide_tnm)
        slide_size_all.append(slide_size)
        slide_yuHouFenZu_all.append(slide_yuHouFenZu)
        slide_zhuyuanhao_all.append(slide_zhuyuanhao)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all, slide_isup_all, slide_necrosis_all, slide_tnm_all, slide_size_all, slide_yuHouFenZu_all, slide_zhuyuanhao_all


def match_pathology_label_TCGA(slide_info, slide_name):
    index_bianhao = slide_info['住院号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    slide_tnm_all = []
    slide_necrosis_all = [] # huaisi and rouliu
    slide_isup_all = []
    slide_size_all = []
    slide_yuHouFenZu_all = []
    slide_zhuyuanhao_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = "-".join(slide_name_i_raw.split('-')[3:6])
        idx_match_final = np.where(index_bianhao == slide_name_i)[0]

        if len(idx_match_final) != 1:
            print("[Slide Found Twice] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
            slide_tnm = -1
            slide_necrosis = -1  # huaisi and rouliu
            slide_isup = -1
            slide_size = -1
            slide_yuHouFenZu = -1
            slide_zhuyuanhao = ""
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 9][idx_match_final[0]]
            slide_nuclearLevel_label = slide_info.to_numpy()[:, 11][idx_match_final[0]]
            slide_prognosis = slide_info.to_numpy()[:, 29:35][idx_match_final[0]]
            slide_tnm = slide_info.to_numpy()[:, 18][idx_match_final[0]]
            slide_necrosis = slide_info.to_numpy()[:, [21, 24]][idx_match_final[0]]  # huaisi and rouliu
            slide_isup = slide_info.to_numpy()[:, 12][idx_match_final[0]]
            slide_size = slide_info.to_numpy()[:, 14][idx_match_final[0]]
            slide_yuHouFenZu = slide_info.to_numpy()[:, 35][idx_match_final[0]]
            slide_zhuyuanhao = index_bianhao[idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
        slide_isup_all.append(slide_isup)
        slide_necrosis_all.append(slide_necrosis)
        slide_tnm_all.append(slide_tnm)
        slide_size_all.append(slide_size)
        slide_yuHouFenZu_all.append(slide_yuHouFenZu)
        slide_zhuyuanhao_all.append(slide_zhuyuanhao)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all, slide_isup_all, slide_necrosis_all, slide_tnm_all, slide_size_all, slide_yuHouFenZu_all, slide_zhuyuanhao_all


def match_pathology_label_CCRCC(slide_info, slide_name):
    index_bianhao = slide_info['住院号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    slide_tnm_all = []
    slide_necrosis_all = []  # huaisi and rouliu
    slide_isup_all = []
    slide_size_all = []
    slide_yuHouFenZu_all = []
    slide_zhuyuanhao_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = slide_name_i_raw.rsplit('-', 1)[0]  # 从右分割1次后取前半部分
        idx_match_final = np.where(index_bianhao == slide_name_i)[0]
        if len(idx_match_final) != 1:
            print("[Slide Found Twice] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
            slide_tnm = -1
            slide_necrosis = -1  # huaisi and rouliu
            slide_isup = -1
            slide_size = -1
            slide_yuHouFenZu = -1
            slide_zhuyuanhao = ""
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_final[0]]
            slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_final[0]]
            slide_prognosis = slide_info.to_numpy()[:, 30:36][idx_match_final[0]]
            slide_tnm = slide_info.to_numpy()[:, 19][idx_match_final[0]]
            slide_necrosis = slide_info.to_numpy()[:, [22, 25]][idx_match_final[0]]  # huaisi and rouliu
            slide_isup = slide_info.to_numpy()[:, 13][idx_match_final[0]]
            slide_size = slide_info.to_numpy()[:, 15][idx_match_final[0]]
            slide_yuHouFenZu = slide_info.to_numpy()[:, 55][idx_match_final[0]]
            slide_zhuyuanhao = index_bianhao[idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
        slide_isup_all.append(slide_isup)
        slide_necrosis_all.append(slide_necrosis)
        slide_tnm_all.append(slide_tnm)
        slide_size_all.append(slide_size)
        slide_yuHouFenZu_all.append(slide_yuHouFenZu)
        slide_zhuyuanhao_all.append(slide_zhuyuanhao)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all, slide_isup_all, slide_necrosis_all, slide_tnm_all, slide_size_all, slide_yuHouFenZu_all, slide_zhuyuanhao_all


def match_pathology_label_xiamen(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    slide_tnm_all = []
    slide_necrosis_all = []  # huaisi and rouliu
    slide_isup_all = []
    slide_size_all = []
    slide_yuHouFenZu_all = []
    slide_zhuyuanhao_all = []
    for slide_name_i_raw in slide_name:
        if '-' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split('-')[0]
        elif ' ' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split(' ')[0]
        else:
            slide_name_i = slide_name_i_raw

        idx_match_0 = np.where(index_zhu_yuan_hao == slide_name_i)[0]
        idx_match_1 = np.where(index_bing_li_hao == slide_name_i)[0]
        idx_match_final = np.unique(np.concatenate([idx_match_0, idx_match_1]))

        if len(idx_match_final) != 1:
            print("[Slide Found Twice or not Found] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
            slide_tnm = -1
            slide_necrosis = -1  # huaisi and rouliu
            slide_isup = -1
            slide_size = -1
            slide_yuHouFenZu = -1
            slide_zhuyuanhao = ""
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_final[0]]
            slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_final[0]]
            slide_prognosis = slide_info.to_numpy()[:, 30:36][idx_match_final[0]]
            slide_tnm = slide_info.to_numpy()[:, 19][idx_match_final[0]]
            slide_necrosis = slide_info.to_numpy()[:, [22, 25]][idx_match_final[0]]  # huaisi and rouliu
            slide_isup = slide_info.to_numpy()[:, 13][idx_match_final[0]]
            slide_size = slide_info.to_numpy()[:, 15][idx_match_final[0]]
            slide_yuHouFenZu = slide_info.to_numpy()[:, 55][idx_match_final[0]]
            slide_zhuyuanhao = index_zhu_yuan_hao[idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
        slide_isup_all.append(slide_isup)
        slide_necrosis_all.append(slide_necrosis)
        slide_tnm_all.append(slide_tnm)
        slide_size_all.append(slide_size)
        slide_yuHouFenZu_all.append(slide_yuHouFenZu)
        slide_zhuyuanhao_all.append(slide_zhuyuanhao)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all, slide_isup_all, slide_necrosis_all, slide_tnm_all, slide_size_all, slide_yuHouFenZu_all, slide_zhuyuanhao_all


def match_pathology_label_zhangye(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    slide_tnm_all = []
    slide_necrosis_all = []  # huaisi and rouliu
    slide_isup_all = []
    slide_size_all = []
    slide_yuHouFenZu_all = []
    slide_zhuyuanhao_all = []
    for slide_name_i_raw in slide_name:
        if '-' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split('-')[0]
        elif ' ' in slide_name_i_raw:
            slide_name_i = slide_name_i_raw.split(' ')[0]
        else:
            slide_name_i = slide_name_i_raw

        idx_match_0 = np.where(index_zhu_yuan_hao == slide_name_i)[0]
        idx_match_1 = np.where(index_bing_li_hao == slide_name_i)[0]
        idx_match_final = np.unique(np.concatenate([idx_match_0, idx_match_1]))

        if len(idx_match_final) != 1:
            print("[Slide Found Twice] {}".format(slide_name_i))
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
            slide_tnm = -1
            slide_necrosis = -1  # huaisi and rouliu
            slide_isup = -1
            slide_size = -1
            slide_yuHouFenZu = -1
            slide_zhuyuanhao = ""
        else:
            slide_pathology_label = slide_info.to_numpy()[:, 10][idx_match_final[0]]
            slide_nuclearLevel_label = slide_info.to_numpy()[:, 12][idx_match_final[0]]
            slide_prognosis = slide_info.to_numpy()[:, 30:36][idx_match_final[0]]
            slide_tnm = slide_info.to_numpy()[:, 19][idx_match_final[0]]
            slide_necrosis = slide_info.to_numpy()[:, [22, 25]][idx_match_final[0]]  # huaisi and rouliu
            slide_isup = slide_info.to_numpy()[:, 13][idx_match_final[0]]
            slide_size = slide_info.to_numpy()[:, 15][idx_match_final[0]]
            slide_yuHouFenZu = slide_info.to_numpy()[:, 55][idx_match_final[0]]
            slide_zhuyuanhao = index_zhu_yuan_hao[idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
        slide_isup_all.append(slide_isup)
        slide_necrosis_all.append(slide_necrosis)
        slide_tnm_all.append(slide_tnm)
        slide_size_all.append(slide_size)
        slide_yuHouFenZu_all.append(slide_yuHouFenZu)
        slide_zhuyuanhao_all.append(slide_zhuyuanhao)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all, slide_isup_all, slide_necrosis_all, slide_tnm_all, slide_size_all, slide_yuHouFenZu_all, slide_zhuyuanhao_all


# don't have isup data
def match_pathology_label_huadong(slide_info, slide_name):
    index_zhu_yuan_hao = slide_info['住院号'].to_numpy().astype(str)
    index_bing_li_hao = slide_info['病理号'].to_numpy().astype(str)
    slide_pathology_label_all = []
    slide_nuclearLevel_label_all = []
    slide_prognosis_all = []
    slide_tnm_all = []
    slide_necrosis_all = []  # huaisi and rouliu
    slide_isup_all = []
    slide_size_all = []
    slide_yuHouFenZu_all = []
    slide_zhuyuanhao_all = []
    for slide_name_i_raw in slide_name:
        slide_name_i = slide_name_i_raw.split(' ')[0]
        slide_name_i = slide_name_i.replace('_', '')
        if len(slide_name_i.split('-')[1]) < 5:
            slide_name_i = slide_name_i.split('-')[0] + '-' + slide_name_i.split('-')[1].zfill(5)

        idx_match = np.where(index_bing_li_hao == slide_name_i)[0]

        if len(idx_match) == 0:
            print("[Slide Not Found] {}".format(slide_name_i))
            course_match(index_bing_li_hao, slide_name_i)
            slide_pathology_label = -1
            slide_nuclearLevel_label = -1
            slide_prognosis = -1
            slide_tnm = -1
            slide_necrosis = -1  # huaisi and rouliu
            slide_isup = -1
            slide_size = -1
            slide_yuHouFenZu = -1
            slide_zhuyuanhao = ""
        else:
            idx_match_final = np.unique(np.concatenate([idx_match]))
            if len(idx_match_final) != 1:
                print("[Slide Found Twice] {}".format(slide_name_i))
                slide_pathology_label = -1
                slide_nuclearLevel_label = -1
                slide_prognosis = -1
                slide_tnm = -1
                slide_necrosis = -1  # huaisi and rouliu
                slide_isup = -1
                slide_size = -1
                slide_yuHouFenZu = -1
                slide_zhuyuanhao = ""
            else:
                slide_pathology_label = slide_info.to_numpy()[:, 9][idx_match_final[0]]
                slide_nuclearLevel_label = slide_info.to_numpy()[:, 11][idx_match_final[0]]
                slide_prognosis = slide_info.to_numpy()[:, 29:35][idx_match_final[0]]
                slide_tnm = slide_info.to_numpy()[:, 18][idx_match_final[0]]
                slide_necrosis = slide_info.to_numpy()[:, [21, 24]][idx_match_final[0]]  # huaisi and rouliu
                slide_isup = slide_info.to_numpy()[:, 12][idx_match_final[0]]
                slide_size = slide_info.to_numpy()[:, 14][idx_match_final[0]]
                slide_yuHouFenZu = slide_info.to_numpy()[:, 35][idx_match_final[0]]
                slide_zhuyuanhao = index_zhu_yuan_hao[idx_match_final[0]]
        slide_pathology_label_all.append(slide_pathology_label)
        slide_nuclearLevel_label_all.append(slide_nuclearLevel_label)
        slide_prognosis_all.append(slide_prognosis)
        slide_isup_all.append(slide_isup)
        slide_size_all.append(slide_size)
        slide_necrosis_all.append(slide_necrosis)
        slide_tnm_all.append(slide_tnm)
        slide_yuHouFenZu_all.append(slide_yuHouFenZu)
        slide_zhuyuanhao_all.append(slide_zhuyuanhao)
    return slide_pathology_label_all, slide_nuclearLevel_label_all, slide_prognosis_all, slide_isup_all, slide_size_all, slide_necrosis_all, slide_tnm_all, slide_yuHouFenZu_all, slide_zhuyuanhao_all

def statistics_dataset(labels):
    # labels of sihape N
    num_samples = labels.shape[0]
    all_cate = np.unique(labels)
    for i in range(all_cate.shape[0]):
        num_samples_cls_i = np.sum(labels == all_cate[i])
        print("class {}: {}/{} samples, ratio:{:.4f}".format(all_cate[i], num_samples_cls_i, num_samples, num_samples_cls_i/num_samples))
    return 0


def parse_fileName(fileName):
    if type(fileName) is np.ndarray:
        label = []
        for i in range(len(fileName)):
            label_str = fileName[i].split('/')[-1].split('_')[-1][5:-4]
            if label_str == '-1':
                label.append(-1)
            else:
                label.append(int(label_str) - 1)
        label = np.array(label)
    else:
        label_str = fileName.split('/')[-1].split('_')[-1][5:-4]
        if label_str == '-1':
            label = -1
        else:
            label = int(label_str) - 1
    return label


def convert_necrosis_label(raw_list):
    new_list = []
    for i in range(len(raw_list)):
        if type(raw_list[i]) is int:
            if raw_list[i] == -1:
                new_list.append(np.array([-1, -1]).astype(np.float32))
        else:
            new_list.append(np.nan_to_num(raw_list[i].astype(np.float32), nan=-1))
    return new_list


def convert_patho_label(raw_list):
    map_dict = {
        '-1': -1,
        '1.1': 0, '1.2': -1,
        '2.1': 1, 
        '3.1': 2, '3.2': 3, '3.3': -1,
        '4.1': 4, 
        '5.1': 5, '5.2': 6, '5.3': -1, '5.4': -1, '5.5': -1, '5.6': 7,
        '6.1': 8, '6.2': -1, '6.3': -1, '6.4': -1, '6.5': -1,
        '7.1': -1, '7.2': -1, '7.3': -1,
        'b': -1,
        'c': -1,
        'd1': -1, 'd2': -1, 'd3': -1, 'd4': -1, 'd5': -1, 'd6': -1, 'd7': -1, 'd8': -1, 'd9': -1,
        'e1': -1, 'e2': -1,
        'f1': -1, 'f2': -1,
        'g': -1,
        'h1': -1, 'h2': -1, 'h3': -1, 'h4': -1,
        'i': -1,
        'j': -1,
        'k': -1,
        'l': -1,
        'z': -1, 'nan':-1
    }
    new_list = np.zeros_like(raw_list, dtype=int)
    for i in range(len(raw_list)):
        new_list[i] = map_dict[str(raw_list[i])]
    return new_list


def convert_nuclearLevel_label(raw_list):
    map_dict = {
        '-1': -1,
        '1': 1,   '2': 2,   '3': 3,   '4': 4,
        '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4,
        
        '1.2': 2, '1.3': 3, '1.4': 4, '2.3': 3, '2.4': 4, '3.4': 4,
        
        '1,2': 2, '2,3': 3, '2,4': 4, '3,4': 4,
        '1,2,4': 4, '2,3,4': 4,

        'nan': -1, 'NA': -1, '无': -1
    }
    new_list = np.zeros_like(raw_list, dtype=int)
    for i in range(len(raw_list)):
        new_list[i] = map_dict[str(raw_list[i])]
    return new_list


def convert_isup_label(raw_list):
    map_dict = {
        '-1': -1, '0': -1, '1.5':-1, '无':-1,
        '1': 1,   '2': 2,   '3': 3,   '4': 4,
        '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4,

        '1,2': 2, '2,3': 3, '2,4': 4, '3,4': 4,
        '1,2,4': 4, '2,3,4': 4,
        '10': -1, '3a':-1,
        'nan': -1, 'NA': -1
    }
    new_list = np.zeros_like(raw_list, dtype=int)
    for i in range(len(raw_list)):
        new_list[i] = map_dict[str(raw_list[i])]
    return new_list


def convert_prognosis_label(raw_list):
    new_list = []
    for i in range(len(raw_list)):
        if type(raw_list[i]) is int:
            if raw_list[i] == -1:
                new_list.append(np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32))
        elif type(raw_list[i]) is np.ndarray:
            new_list.append(np.nan_to_num(raw_list[i].astype(np.float32), nan=-1))
    return new_list


def convert_size_label(raw_list):
    map_dict = {
        '-1': -1, '0': -1, '1.5': -1,
        '1': 1, '2': 2, '3': 3, '4': 4,
        '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4,

        '1,2': 2, '2,3': 3, '2,4': 4, '3,4': 4,
        '1,2,4': 4, '2,3,4': 4,
        '10': -1, '3a': -1,
        'nan': -1, 'NA': -1, '补充': -1
    }
    new_list = []
    for i in range(len(raw_list)):
        if not (type(raw_list[i]) is float or type(raw_list[i]) is int):
            continue
        new_list.append(np.nan_to_num(raw_list[i], nan=-1))
    return new_list


def convert_tnm_label(raw_list):
    map_dict = {
        '-1': -1, '0': -1, '1.5': -1, '无': -1,
        '1': 1,   '2': 2,   '3': 3,   '4': 4,
        '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4,
        'nan': -1, 'NA': -1, '': -1
    }
    new_list = []
    for item in raw_list:
        val = map_dict.get(str(item).strip(), -1)
        new_list.append(val)
    return new_list


def convert_yuHouFenZu_label(raw_list):
    map_dict = {
        '-1': -1,
        '1': 1,   '2': 2,   '3': 3,   '4': 4,
        '1.0': 1, '2.0': 2, '3.0': 3, '4.0': 4,
        'nan': -1, 'NA': -1, '无': -1, '0.0': -1
    }
    new_list = np.zeros_like(raw_list, dtype=int)
    for i in range(len(raw_list)):
        new_list[i] = map_dict[str(raw_list[i])]
    return new_list.tolist()


def get_sort_one_center(center_dir, downsample=1.0, shuffle=True, slide_patho_anno_path=None, center='ZS'):
    # 1. 只加载patch路径，不加载特征内容
    feat_file_path = os.path.join(center_dir, "all_patch_feat.npy")
    file_name_path = os.path.join(center_dir, "all_patch_fileName.npy")
    all_patches_fileName = np.load(file_name_path)  # shape (N,)

    # 1.1 只用memmap方式打开特征，检查NaN，避免加载全部特征进内存
    feat_memmap = np.load(feat_file_path, mmap_mode='r')
    t_ = feat_memmap.max(axis=1)
    idx_nan = np.isnan(t_)
    print("remove {} nan feat vector".format(idx_nan.sum()))
    valid_indices = np.where(~idx_nan)[0]
    all_patches_fileName = all_patches_fileName[valid_indices]

    # 2. 按slide分组
    all_patche_corresponding_slideName = all_patches_fileName.tolist()
    for i in range(len(all_patche_corresponding_slideName)):
        parts = all_patche_corresponding_slideName[i].split('/')
        slide_name = parts[-3]  # 1195657-a1-G3
        sub_slide = parts[-2]   # 0
        if sub_slide == '0':
            all_patche_corresponding_slideName[i] = slide_name
        else:
            all_patche_corresponding_slideName[i] = f"{slide_name}-{sub_slide}"
    all_patche_corresponding_slideName = np.array(all_patche_corresponding_slideName)
    unique_slideName = np.unique(all_patche_corresponding_slideName)

    if downsample < 1.0:
        unique_slideName = np.random.choice(unique_slideName, int(len(unique_slideName)*downsample), replace=False)
    if shuffle:
        unique_slideName = np.random.permutation(unique_slideName)

    # 3. 只保存特征文件路径和索引，不加载特征内容
    slide_patch_info = []
    slide_patch_label = []
    slide_patch_fileName = []
    for slide_i in unique_slideName:
        idx_from_slide_i = np.where(all_patche_corresponding_slideName == slide_i)[0]
        indices = valid_indices[idx_from_slide_i]
        # 只保存特征文件路径和索引
        slide_patch_info.append([
            {'feature_path': feat_file_path, 'feature_index': int(idx)} for idx in indices
        ])
        # patch标签仍然可以用parse_fileName解析
        slide_patch_label.append(parse_fileName(all_patches_fileName[idx_from_slide_i]))
        slide_patch_fileName.append(all_patches_fileName[idx_from_slide_i])

    # 4. 加载病理信息
    slide_info = pd.read_excel(slide_patho_anno_path)
    slide_names_for_patho = [slide_patch_fileName[i][0].split('/')[7] for i in range(len(slide_patch_fileName))]
    if center == 'ZS':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis, slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_ZS(slide_info, slide_names_for_patho)
    elif center == 'TCGA':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis, slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_TCGA(slide_info, slide_names_for_patho)
    elif center == 'CCRCC':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis, slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_CCRCC(slide_info, slide_names_for_patho)
    elif center == 'xiamen':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis, slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_xiamen(slide_info, slide_names_for_patho)
    elif center == 'zhangye':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis , slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_zhangye(slide_info, slide_names_for_patho)
    elif center == 'huadong':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis, slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_huadong(slide_info, slide_names_for_patho)
    else:
        raise

    slide_patho_label = convert_patho_label(slide_patho_label).tolist()
    slide_nuclearLevel_label = convert_nuclearLevel_label(slide_nuclearLevel_label).tolist()
    slide_isup = convert_isup_label(slide_isup).tolist()
    slide_tnm = convert_tnm_label(slide_tnm)
    slide_size = convert_size_label(slide_size)
    if slide_yuHouFenZu is not None:
        slide_yuHouFenZu = convert_yuHouFenZu_label(slide_yuHouFenZu)
    else:
        slide_yuHouFenZu = [-1 for _ in range(len(slide_patho_label))]

    if slide_prognosis is None:
        slide_prognosis_label = [np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32) for _ in range(len(slide_patho_label))]
    else:
        slide_prognosis_label = convert_prognosis_label(slide_prognosis)

    return (slide_patch_info, slide_patch_label,
            unique_slideName.tolist(), slide_patch_fileName,
            slide_patho_label, slide_nuclearLevel_label, slide_prognosis_label,
            slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao)


def get_sort_pred_label(center_dir, downsample=1.0, shuffle=True, slide_patho_anno_path=None, center='ZS'):
    # 1. 只加载patch路径，不加载特征内容
    feat_file_path = os.path.join(center_dir, "all_patch_feat.npy")
    file_name_path = os.path.join(center_dir, "all_patch_fileName.npy")
    all_patch_label = np.load(os.path.join(center_dir, "patch_pred_label.npy"))
    all_patches_fileName = np.load(file_name_path)  # shape (N,)

    # 1.1 只用memmap方式打开特征，检查NaN，避免加载全部特征进内存
    feat_memmap = np.load(feat_file_path, mmap_mode='r')    
    t_ = feat_memmap.max(axis=1)
    idx_nan = np.isnan(t_)
    print("remove {} nan feat vector".format(idx_nan.sum()))
    valid_indices = np.where(~idx_nan)[0]
    all_patches_fileName = all_patches_fileName[valid_indices]
    all_patch_label = all_patch_label[valid_indices]

    # 2. 按slide分组
    all_patche_corresponding_slideName = all_patches_fileName.tolist()
    for i in range(len(all_patche_corresponding_slideName)):
        parts = all_patche_corresponding_slideName[i].split('/')
        slide_name = parts[-3]  
        sub_slide = parts[-2]   
        if sub_slide == '0':
            all_patche_corresponding_slideName[i] = slide_name
        else:
            all_patche_corresponding_slideName[i] = f"{slide_name}-{sub_slide}"
    all_patche_corresponding_slideName = np.array(all_patche_corresponding_slideName)
    unique_slideName = np.unique(all_patche_corresponding_slideName)

    if downsample < 1.0:
        unique_slideName = np.random.choice(unique_slideName, int(len(unique_slideName)*downsample), replace=False)
    if shuffle:
        unique_slideName = np.random.permutation(unique_slideName)

    # 3. 只保存特征文件路径和索引，不加载特征内容
    slide_patch_info = []
    slide_patch_label = []
    slide_patch_fileName = []
    for slide_i in unique_slideName:
        idx_from_slide_i = np.where(all_patche_corresponding_slideName == slide_i)[0]
        indices = valid_indices[idx_from_slide_i]
        # 只保存特征文件路径和索引
        slide_patch_info.append([
            {'feature_path': feat_file_path, 'feature_index': int(idx)} for idx in indices
        ])
        slide_patch_label.append(all_patch_label[indices])
        slide_patch_fileName.append(all_patches_fileName[idx_from_slide_i])

    # 4. 加载病理信息
    slide_info = pd.read_excel(slide_patho_anno_path)
    slide_names_for_patho = [slide_patch_fileName[i][0].split('/')[7] for i in range(len(slide_patch_fileName))]
    if center == 'ZS':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis, slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_ZS(slide_info, slide_names_for_patho)
    elif center == 'TCGA':
        slide_patho_label, slide_nuclearLevel_label, slide_prognosis, slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao = match_pathology_label_TCGA(slide_info, slide_names_for_patho)

    slide_patho_label = convert_patho_label(slide_patho_label).tolist()
    slide_nuclearLevel_label = convert_nuclearLevel_label(slide_nuclearLevel_label).tolist()
    slide_isup = convert_isup_label(slide_isup).tolist()
    slide_tnm = convert_isup_label(slide_tnm).tolist()
    slide_size = convert_size_label(slide_size)
    if slide_yuHouFenZu is not None:
        slide_yuHouFenZu = convert_yuHouFenZu_label(slide_yuHouFenZu)
    else:
        slide_yuHouFenZu = [-1 for _ in range(len(slide_patho_label))]

    if slide_prognosis is None:
        slide_prognosis_label = [np.array([-1, -1, -1, -1, -1, -1]).astype(np.float32) for _ in range(len(slide_patho_label))]
    else:
        slide_prognosis_label = convert_prognosis_label(slide_prognosis)

    return (slide_patch_info, slide_patch_label,
            unique_slideName.tolist(), slide_patch_fileName,
            slide_patho_label, slide_nuclearLevel_label, slide_prognosis_label,
            slide_isup, slide_necrosis, slide_tnm, slide_size, slide_yuHouFenZu, slide_zhuyuanhao)


def get_train_test_ds_MultiCenter_region_5Cls(
         data_root='/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1',
        downsample=1.0):

    slide_patch_feat_XIAMEN, slide_patch_label_XIAMEN, slide_fileName_XIAMEN, slide_patch_fileName_XIAMEN, slide_patho_label_XIAMEN, slide_nuclearLevel_label_XIAMEN, slide_prognosis_XIAMEN, slide_isup_XIAMEN, slide_necrosis_XIAMEN, slide_tnm_XIAMEN, slide_size_XIAMEN, slide_yuHouFenZu_XIAMEN, slide_zhuyuanhao_XIAMEN = get_sort_one_center(os.path.join(data_root, "XIAMEN"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='xiamen')
    print("xia_men Load")

    slide_patch_feat_ZHANGYE, slide_patch_label_ZHANGYE, slide_fileName_ZHANGYE, slide_patch_fileName_ZHANGYE, slide_patho_label_ZHANGYE, slide_nuclearLevel_label_ZHANGYE, slide_prognosis_ZHANGYE, slide_isup_ZHANGYE, slide_necrosis_ZHANGYE, slide_tnm_ZHANGYE, slide_size_ZHANGYE, slide_yuHouFenZu_ZHANGYE, slide_zhuyuanhao_ZHANGYE = get_sort_one_center(os.path.join(data_root, "ZHANGYE"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='zhangye')
    print("zhang_ye Load")

    slide_patch_feat_HUADONG, slide_patch_label_HUADONG, slide_fileName_HUADONG, slide_patch_fileName_HUADONG, slide_patho_label_HUADONG, slide_nuclearLevel_label_HUADONG, slide_prognosis_HUADONG, slide_isup_HUADONG, slide_necrosis_HUADONG, slide_tnm_HUADONG, slide_size_HUADONG, slide_yuHouFenZu_HUADONG, slide_zhuyuanhao_HUADONG = get_sort_one_center(os.path.join(data_root, "HUADONG"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='huadong')
    print("hua_dong Load")

    slide_patch_feat_TCGAKICH, slide_patch_label_TCGAKICH, slide_fileName_TCGAKICH, slide_patch_fileName_TCGAKICH, slide_patho_label_TCGAKICH, slide_nuclearLevel_label_TCGAKICH, slide_prognosis_TCGAKICH, slide_isup_TCGAKICH, slide_necrosis_TCGAKICH, slide_tnm_TCGAKICH, slide_size_TCGAKICH, slide_yuHouFenZu_TCGAKICH, slide_zhuyuanhao_TCGAKICH = get_sort_one_center(os.path.join(data_root, "TCGA_kich"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kich Load")
    
    slide_patch_feat_TCGAKIRC, slide_patch_label_TCGAKIRC, slide_fileName_TCGAKIRC, slide_patch_fileName_TCGAKIRC, slide_patho_label_TCGAKIRC, slide_nuclearLevel_label_TCGAKIRC, slide_prognosis_TCGAKIRC, slide_isup_TCGAKIRC, slide_necrosis_TCGAKIRC, slide_tnm_TCGAKIRC, slide_size_TCGAKIRC, slide_yuHouFenZu_TCGAKIRC, slide_zhuyuanhao_TCGAKIRC = get_sort_one_center(os.path.join(data_root, "TCGA_kirc"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kirc Load")

    slide_patch_feat_TCGAKIRP, slide_patch_label_TCGAKIRP, slide_fileName_TCGAKIRP, slide_patch_fileName_TCGAKIRP, slide_patho_label_TCGAKIRP, slide_nuclearLevel_label_TCGAKIRP, slide_prognosis_TCGAKIRP, slide_isup_TCGAKIRP, slide_necrosis_TCGAKIRP, slide_tnm_TCGAKIRP, slide_size_TCGAKIRP, slide_yuHouFenZu_TCGAKIRP, slide_zhuyuanhao_TCGAKIRP = get_sort_one_center(os.path.join(data_root, "TCGA_kirp"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kirp Load")

    slide_patch_feat_CCRCC, slide_patch_label_CCRCC, slide_fileName_CCRCC, slide_patch_fileName_CCRCC, slide_patho_label_CCRCC, slide_nuclearLevel_label_CCRCC, slide_prognosis_CCRCC, slide_isup_CCRCC, slide_necrosis_CCRCC, slide_tnm_CCRCC, slide_size_CCRCC, slide_yuHouFenZu_CCRCC, slide_zhuyuanhao_CCRCC = get_sort_one_center(os.path.join(data_root, "CCRCC"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='CCRCC')
    print("CCRCC Load")

    slide_patch_feat_ZSRUTOU, slide_patch_label_ZSRUTOU, slide_fileName_ZSRUTOU, slide_patch_fileName_ZSRUTOU, slide_patho_label_ZSRUTOU, slide_nuclearLevel_label_ZSRUTOU, slide_prognosis_ZSRUTOU, slide_isup_ZSRUTOU, slide_necrosis_ZSRUTOU, slide_tnm_ZSRUTOU, slide_size_ZSRUTOU, slide_yuHouFenZu_ZSRUTOU, slide_zhuyuanhao_ZSRUTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_RUTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_rutou Load")

    slide_patch_feat_ZSXIANSE, slide_patch_label_ZSXIANSE, slide_fileName_ZSXIANSE, slide_patch_fileName_ZSXIANSE, slide_patho_label_ZSXIANSE, slide_nuclearLevel_label_ZSXIANSE, slide_prognosis_ZSXIANSE, slide_isup_ZSXIANSE, slide_necrosis_ZSXIANSE, slide_tnm_ZSXIANSE, slide_size_ZSXIANSE, slide_yuHouFenZu_ZSXIANSE, slide_zhuyuanhao_ZSXIANSE = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XIANSE"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xianse Load")
    #
    slide_patch_feat_ZSXINTOU, slide_patch_label_ZSXINTOU, slide_fileName_ZSXINTOU, slide_patch_fileName_ZSXINTOU, slide_patho_label_ZSXINTOU, slide_nuclearLevel_label_ZSXINTOU, slide_prognosis_ZSXINTOU, slide_isup_ZSXINTOU, slide_necrosis_ZSXINTOU, slide_tnm_ZSXINTOU, slide_size_ZSXINTOU, slide_yuHouFenZu_ZSXINTOU, slide_zhuyuanhao_ZSXINTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XINTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xintou Load")

    slide_patch_feat_ZSBUCHONG, slide_patch_label_ZSBUCHONG, slide_fileName_ZSBUCHONG, slide_patch_fileName_ZSBUCHONG, slide_patho_label_ZSBUCHONG, slide_nuclearLevel_label_ZSBUCHONG, slide_prognosis_ZSBUCHONG, slide_isup_ZSBUCHONG, slide_necrosis_ZSBUCHONG, slide_tnm_ZSBUCHONG, slide_size_ZSBUCHONG, slide_yuHouFenZu_ZSBUCHONG, slide_zhuyuanhao_ZSBUCHONG = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_BUCHONG"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_buchong Load")

    slide_patch_feat_ZSQITA, slide_patch_label_ZSQITA, slide_fileName_ZSQITA, slide_patch_fileName_ZSQITA, slide_patho_label_ZSQITA, slide_nuclearLevel_label_ZSQITA, slide_prognosis_ZSQITA, slide_isup_ZSQITA, slide_necrosis_ZSQITA, slide_tnm_ZSQITA, slide_size_ZSQITA, slide_yuHouFenZu_ZSQITA, slide_zhuyuanhao_ZSQITA = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_QITA"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_qita Load")
    

    train_data =               slide_patch_feat_ZSRUTOU +         slide_patch_feat_ZSXIANSE +         slide_patch_feat_ZSQITA +         slide_patch_feat_ZSXINTOU +         slide_patch_feat_ZSBUCHONG +         slide_patch_feat_XIAMEN +         slide_patch_feat_ZHANGYE +         slide_patch_feat_CCRCC     +slide_patch_feat_TCGAKIRC +         slide_patch_feat_TCGAKICH +         slide_patch_feat_TCGAKIRP
    train_label =              slide_patch_label_ZSRUTOU +        slide_patch_label_ZSXIANSE +        slide_patch_label_ZSQITA +        slide_patch_label_ZSXINTOU +        slide_patch_label_ZSBUCHONG +        slide_patch_label_XIAMEN +        slide_patch_label_ZHANGYE +        slide_patch_label_CCRCC   +slide_patch_label_TCGAKIRC +        slide_patch_label_TCGAKICH +        slide_patch_label_TCGAKIRP
    train_name_slide =         slide_fileName_ZSRUTOU +           slide_fileName_ZSXIANSE+            slide_fileName_ZSQITA +           slide_fileName_ZSXINTOU +           slide_fileName_ZSBUCHONG +           slide_fileName_XIAMEN +           slide_fileName_ZHANGYE +           slide_fileName_CCRCC         +slide_fileName_TCGAKIRC +           slide_fileName_TCGAKICH +           slide_fileName_TCGAKIRP
    train_name_patch =         slide_patch_fileName_ZSRUTOU +     slide_patch_fileName_ZSXIANSE +     slide_patch_fileName_ZSQITA +     slide_patch_fileName_ZSXINTOU +     slide_patch_fileName_ZSBUCHONG +     slide_patch_fileName_XIAMEN +     slide_patch_fileName_ZHANGYE +     slide_patch_fileName_CCRCC  +slide_patch_fileName_TCGAKIRC +     slide_patch_fileName_TCGAKICH +     slide_patch_fileName_TCGAKIRP
    train_patho_label =        slide_patho_label_ZSRUTOU +        slide_patho_label_ZSXIANSE +        slide_patho_label_ZSQITA +        slide_patho_label_ZSXINTOU +        slide_patho_label_ZSBUCHONG +        slide_patho_label_XIAMEN +        slide_patho_label_ZHANGYE +        slide_patho_label_CCRCC   +slide_patho_label_TCGAKIRC +        slide_patho_label_TCGAKICH +        slide_patho_label_TCGAKIRP
    train_nuclearLevel_label = slide_nuclearLevel_label_ZSRUTOU + slide_nuclearLevel_label_ZSXIANSE + slide_nuclearLevel_label_ZSQITA + slide_nuclearLevel_label_ZSXINTOU + slide_nuclearLevel_label_ZSBUCHONG + slide_nuclearLevel_label_XIAMEN + slide_nuclearLevel_label_ZHANGYE + slide_nuclearLevel_label_CCRCC +slide_nuclearLevel_label_TCGAKIRC + slide_nuclearLevel_label_TCGAKICH + slide_nuclearLevel_label_TCGAKIRP
    train_prognosis =          slide_prognosis_ZSRUTOU +          slide_prognosis_ZSXIANSE +          slide_prognosis_ZSQITA +          slide_prognosis_ZSXINTOU +          slide_prognosis_ZSBUCHONG +          slide_prognosis_XIAMEN +          slide_prognosis_ZHANGYE +          slide_prognosis_CCRCC  +slide_prognosis_TCGAKIRC +          slide_prognosis_TCGAKICH +          slide_prognosis_TCGAKIRP
    train_isup =               slide_isup_ZSRUTOU +               slide_isup_ZSXIANSE +               slide_isup_ZSQITA +               slide_isup_ZSXINTOU +               slide_isup_ZSBUCHONG +               slide_isup_XIAMEN +               slide_isup_ZHANGYE +               slide_isup_CCRCC   +slide_isup_TCGAKIRC +               slide_isup_TCGAKICH +               slide_isup_TCGAKIRP
    train_size =               slide_size_ZSRUTOU +               slide_size_ZSXIANSE +               slide_size_ZSQITA +               slide_size_ZSXINTOU +               slide_size_ZSBUCHONG +               slide_size_XIAMEN +               slide_size_ZHANGYE +               slide_size_CCRCC   +slide_size_TCGAKIRC +               slide_size_TCGAKICH +               slide_size_TCGAKIRP
    train_necrosis =           slide_necrosis_ZSRUTOU +           slide_necrosis_ZSXIANSE +           slide_necrosis_ZSQITA +           slide_necrosis_ZSXINTOU +           slide_necrosis_ZSBUCHONG +           slide_necrosis_XIAMEN +           slide_necrosis_ZHANGYE +           slide_necrosis_CCRCC   +slide_necrosis_TCGAKIRC +           slide_necrosis_TCGAKICH +           slide_necrosis_TCGAKIRP
    train_tnm =                slide_tnm_ZSRUTOU +                slide_tnm_ZSXIANSE +                slide_tnm_ZSQITA +                slide_tnm_ZSXINTOU +                slide_tnm_ZSBUCHONG +                slide_tnm_XIAMEN +                slide_tnm_ZHANGYE +                slide_tnm_CCRCC  +slide_tnm_TCGAKIRC +                slide_tnm_TCGAKICH +                slide_tnm_TCGAKIRP
    train_yuHouFenZu =         slide_yuHouFenZu_ZSRUTOU +         slide_yuHouFenZu_ZSXIANSE +         slide_yuHouFenZu_ZSQITA +         slide_yuHouFenZu_ZSXINTOU +         slide_yuHouFenZu_ZSBUCHONG +         slide_yuHouFenZu_XIAMEN +         slide_yuHouFenZu_ZHANGYE +         slide_yuHouFenZu_CCRCC   +slide_yuHouFenZu_TCGAKIRC +         slide_yuHouFenZu_TCGAKICH +         slide_yuHouFenZu_TCGAKIRP
    train_zhuyuanhao =         slide_zhuyuanhao_ZSRUTOU +         slide_zhuyuanhao_ZSXIANSE +         slide_zhuyuanhao_ZSQITA +         slide_zhuyuanhao_ZSXINTOU +         slide_zhuyuanhao_ZSBUCHONG +         slide_zhuyuanhao_XIAMEN +         slide_zhuyuanhao_ZHANGYE +         slide_zhuyuanhao_CCRCC   +slide_zhuyuanhao_TCGAKIRC +         slide_zhuyuanhao_TCGAKICH +         slide_zhuyuanhao_TCGAKIRP
  
    # shuffle
    train_all = list(zip(train_data, train_label, train_name_slide, train_name_patch, train_patho_label, train_nuclearLevel_label, train_prognosis, train_isup, train_size, train_necrosis, train_tnm, train_yuHouFenZu, train_zhuyuanhao))
    train_data[:], train_label[:], train_name_slide[:], train_name_patch[:], train_patho_label[:], train_nuclearLevel_label[:], train_prognosis[:], train_isup[:], train_size[:], train_necrosis[:], train_tnm[:], train_yuHouFenZu[:], train_zhuyuanhao[:] = zip(*train_all)

    huadong_data = slide_patch_feat_HUADONG
    huadong_label = slide_patch_label_HUADONG
    huadong_name_slide = slide_fileName_HUADONG
    huadong_name_patch = slide_patch_fileName_HUADONG
    huadong_patho_label = slide_patho_label_HUADONG
    huadong_nuclearLevel_label = slide_nuclearLevel_label_HUADONG
    huadong_prognosis_label = slide_prognosis_HUADONG
    huadong_isup_label = slide_isup_HUADONG
    huadong_size_label = slide_size_HUADONG
    huadong_necrosis_label = slide_necrosis_HUADONG
    huadong_tnm_label = slide_tnm_HUADONG
    huadong_yuHouFenZu = slide_yuHouFenZu_HUADONG
    huadong_zhuyuanhao = slide_zhuyuanhao_HUADONG
    huadong_all = list(zip(huadong_data, huadong_label, huadong_name_slide, huadong_name_patch, huadong_patho_label, huadong_nuclearLevel_label, huadong_prognosis_label, huadong_isup_label, huadong_size_label, huadong_necrosis_label, huadong_tnm_label, huadong_yuHouFenZu, huadong_zhuyuanhao))
    huadong_data[:], huadong_label[:], huadong_name_slide[:], huadong_name_patch[:], huadong_patho_label[:], huadong_nuclearLevel_label[:], huadong_prognosis_label[:], huadong_isup_label[:], huadong_size_label[:], huadong_necrosis_label[:], huadong_tnm_label[:], huadong_yuHouFenZu[:], huadong_zhuyuanhao[:] = zip(*huadong_all)

    patient2indices = defaultdict(list)
    for idx, zhuyuanhao in enumerate(train_zhuyuanhao):
        patient2indices[zhuyuanhao].append(idx)
    
    internal_train_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split.xlsx", sheet_name = "internaltrain")
    internal_test_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split.xlsx", sheet_name = "internaltest")

    internal_train_ids = internal_train_df['住院号'].astype(str).tolist()
    internal_test_ids = internal_test_df['住院号'].astype(str).tolist()
    
    train_patient_ids = []
    test_patient_ids = []

    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_ids:
            train_patient_ids.append(zhuyuanhao)
        if zhuyuanhao in internal_test_ids:
            test_patient_ids.append(zhuyuanhao)

    internal_train_set = set(internal_train_ids)
    internal_test_set = set(internal_test_ids)

    overlap = internal_train_set & internal_test_set
    if overlap:
        print(f"警告: 住院号 {overlap} 同时出现在训练和测试集中!")

    train_patient_ids = []
    test_patient_ids = []
    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_set:
            train_patient_ids.append(zhuyuanhao)
        elif zhuyuanhao in internal_test_set:  # 使用elif避免重复
            test_patient_ids.append(zhuyuanhao)
    
    train_patient_ids = list(set(train_patient_ids))
    test_patient_ids = list(set(test_patient_ids))
    # 收集索引
    train_indices = []
    test_indices = []
    for pid in train_patient_ids:
        train_indices.extend(patient2indices[pid])
    for pid in test_patient_ids:
        test_indices.extend(patient2indices[pid])
    
    def select_by_indices(data_list, indices):
        return [data_list[i] for i in indices]

    InternalTrain_data = select_by_indices(train_data, train_indices)
    InternalTrain_label = select_by_indices(train_label, train_indices)
    InternalTrain_name_slide = select_by_indices(train_name_slide, train_indices)
    InternalTrain_name_patch = select_by_indices(train_name_patch, train_indices)
    InternalTrain_patho_label = select_by_indices(train_patho_label, train_indices)
    InternalTrain_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, train_indices)
    InternalTrain_prognosis_label = select_by_indices(train_prognosis, train_indices)
    InternalTrain_isup_label = select_by_indices(train_isup, train_indices)
    InternalTrain_size_label = select_by_indices(train_size, train_indices)
    InternalTrain_necrosis_label = select_by_indices(train_necrosis, train_indices)
    InternalTrain_tnm_label = select_by_indices(train_tnm, train_indices)
    InternalTrain_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, train_indices)
    InternalTrain_zhuyuanhao = select_by_indices(train_zhuyuanhao, train_indices)

    InternalTest_data = select_by_indices(train_data, test_indices)
    InternalTest_label = select_by_indices(train_label, test_indices)
    InternalTest_name_slide = select_by_indices(train_name_slide, test_indices)
    InternalTest_name_patch = select_by_indices(train_name_patch, test_indices)
    InternalTest_patho_label = select_by_indices(train_patho_label, test_indices)
    InternalTest_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, test_indices)
    InternalTest_prognosis_label = select_by_indices(train_prognosis, test_indices)
    InternalTest_isup_label = select_by_indices(train_isup, test_indices)
    InternalTest_size_label = select_by_indices(train_size, test_indices)
    InternalTest_necrosis_label = select_by_indices(train_necrosis, test_indices)
    InternalTest_tnm_label = select_by_indices(train_tnm, test_indices)
    InternalTest_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, test_indices)
    InternalTest_zhuyuanhao = select_by_indices(train_zhuyuanhao, test_indices)

    # Split some external test into train
    def is_tcga_slide(slide_name):
        return any(tcga_key in slide_name for tcga_key in ["TCGA", "TCGAKIRC", "TCGAKICH", "TCGAKIRP"])

    # 在 InternalTest 中筛选 TCGA 样本索引
    external_test_indices = [i for i, name in enumerate(InternalTest_name_slide) if is_tcga_slide(name)]

    # 只用这些索引从 InternalTest 里提取 TCGA 作为 ExternalTest
    ExternalTest_data = select_by_indices(InternalTest_data, external_test_indices)
    ExternalTest_label = select_by_indices(InternalTest_label, external_test_indices)
    ExternalTest_name_slide = select_by_indices(InternalTest_name_slide, external_test_indices)
    ExternalTest_name_patch = select_by_indices(InternalTest_name_patch, external_test_indices)
    ExternalTest_patho_label = select_by_indices(InternalTest_patho_label, external_test_indices)
    ExternalTest_nuclearLevel_label = select_by_indices(InternalTest_nuclearLevel_label, external_test_indices)
    ExternalTest_prognosis_label = select_by_indices(InternalTest_prognosis_label, external_test_indices)
    ExternalTest_isup_label = select_by_indices(InternalTest_isup_label, external_test_indices)
    ExternalTest_size_label = select_by_indices(InternalTest_size_label, external_test_indices)
    ExternalTest_necrosis_label = select_by_indices(InternalTest_necrosis_label, external_test_indices)
    ExternalTest_tnm_label = select_by_indices(InternalTest_tnm_label, external_test_indices)
    ExternalTest_yuHouFenZu_label = select_by_indices(InternalTest_yuHouFenZu_label, external_test_indices)
    ExternalTest_zhuyuanhao = select_by_indices(InternalTest_zhuyuanhao, external_test_indices)
    
    def export_to_excel(all_datasets, sheet_names, filename):
        with pd.ExcelWriter(filename) as writer:
            for dataset, sheet_name in zip(all_datasets, sheet_names):
                df = pd.DataFrame({
                    'feat_dim': [len(x) if hasattr(x, '__len__') else 'unknown' for x in dataset[0]],  # 特征维度
                    'label': [",".join(map(str, lbl)) if isinstance(lbl, (list, np.ndarray)) else str(lbl) for lbl in dataset[1]],
                    'slide_name': dataset[2],
                    'patch_name': dataset[3],
                    'patho_label': dataset[4],
                    'nuclear_level_label': dataset[5],
                    'prognosis_label': dataset[6],
                    'isup_label': dataset[7],
                    'size_label': dataset[8],
                    'tnm_label': dataset[9],
                    'necrosis_label': dataset[10],
                    'yuHouFenZu_label': dataset[11],
                    'zhuyuanhao': dataset[12],
                })
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"导出成功：{filename}")
    
    export_to_excel(
        all_datasets=[
            [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label, InternalTrain_isup_label, InternalTrain_size_label, InternalTrain_tnm_label, InternalTrain_necrosis_label, InternalTrain_yuHouFenZu_label, InternalTrain_zhuyuanhao],
            [InternalTest_data,  InternalTest_label,  InternalTest_name_slide,  InternalTest_name_patch,  InternalTest_patho_label,  InternalTest_nuclearLevel_label,  InternalTest_prognosis_label,  InternalTest_isup_label,  InternalTest_size_label,  InternalTest_tnm_label,  InternalTest_necrosis_label, InternalTest_yuHouFenZu_label, InternalTest_zhuyuanhao],
            [ExternalTest_data,  ExternalTest_label,  ExternalTest_name_slide,  ExternalTest_name_patch,  ExternalTest_patho_label,  ExternalTest_nuclearLevel_label,  ExternalTest_prognosis_label,  ExternalTest_isup_label,  ExternalTest_size_label,  ExternalTest_tnm_label,  ExternalTest_necrosis_label, ExternalTest_yuHouFenZu_label, ExternalTest_zhuyuanhao],
            [huadong_data,       huadong_label,       huadong_name_slide,       huadong_name_patch,       huadong_patho_label,       huadong_nuclearLevel_label,       huadong_prognosis_label,       huadong_isup_label,       huadong_size_label,       huadong_tnm_label,       huadong_necrosis_label, huadong_yuHouFenZu, huadong_zhuyuanhao],
        ],
        sheet_names=["InternalTrain", "InternalTest", "ExternalTest", "HuadongTest"],
        filename="MultiCenter_Split_Info_5cls.xlsx"
    )

    print("ALL FEAT LOADED")
    return (
        [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label, InternalTrain_isup_label, InternalTrain_size_label, InternalTrain_tnm_label, InternalTrain_necrosis_label, InternalTrain_yuHouFenZu_label, InternalTrain_zhuyuanhao],
        [InternalTest_data,  InternalTest_label,  InternalTest_name_slide,  InternalTest_name_patch,  InternalTest_patho_label,  InternalTest_nuclearLevel_label,  InternalTest_prognosis_label,  InternalTest_isup_label,  InternalTest_size_label,  InternalTest_tnm_label,  InternalTest_necrosis_label, InternalTest_yuHouFenZu_label, InternalTest_zhuyuanhao],
        [ExternalTest_data,  ExternalTest_label,  ExternalTest_name_slide,  ExternalTest_name_patch,  ExternalTest_patho_label,  ExternalTest_nuclearLevel_label,  ExternalTest_prognosis_label,  ExternalTest_isup_label,  ExternalTest_size_label,  ExternalTest_tnm_label,  ExternalTest_necrosis_label, ExternalTest_yuHouFenZu_label, ExternalTest_zhuyuanhao],
        [huadong_data,       huadong_label,       huadong_name_slide,       huadong_name_patch,       huadong_patho_label,       huadong_nuclearLevel_label,       huadong_prognosis_label,       huadong_isup_label,       huadong_size_label,       huadong_tnm_label,       huadong_necrosis_label, huadong_yuHouFenZu, huadong_zhuyuanhao],
    )

def get_train_test_ds_MultiCenter_region_trainwithTCGA(
         data_root='/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1',
        downsample=1.0):
    
    slide_patch_feat_ZSTOUMING, slide_patch_label_ZSTOUMING, slide_fileName_ZSTOUMING, slide_patch_fileName_ZSTOUMING, slide_patho_label_ZSTOUMING, slide_nuclearLevel_label_ZSTOUMING, slide_prognosis_ZSTOUMING, slide_isup_ZSTOUMING, slide_necrosis_ZSTOUMING, slide_tnm_ZSTOUMING, slide_size_ZSTOUMING, slide_yuHouFenZu_ZSTOUMING, slide_zhuyuanhao_ZSTOUMING = get_sort_pred_label(os.path.join(data_root, "ZHONGSHAN_TOUMING"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_touming Load")

    slide_patch_feat_XIAMEN, slide_patch_label_XIAMEN, slide_fileName_XIAMEN, slide_patch_fileName_XIAMEN, slide_patho_label_XIAMEN, slide_nuclearLevel_label_XIAMEN, slide_prognosis_XIAMEN, slide_isup_XIAMEN, slide_necrosis_XIAMEN, slide_tnm_XIAMEN, slide_size_XIAMEN, slide_yuHouFenZu_XIAMEN, slide_zhuyuanhao_XIAMEN = get_sort_one_center(os.path.join(data_root, "XIAMEN"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='xiamen')
    print("xia_men Load")

    slide_patch_feat_ZHANGYE, slide_patch_label_ZHANGYE, slide_fileName_ZHANGYE, slide_patch_fileName_ZHANGYE, slide_patho_label_ZHANGYE, slide_nuclearLevel_label_ZHANGYE, slide_prognosis_ZHANGYE, slide_isup_ZHANGYE, slide_necrosis_ZHANGYE, slide_tnm_ZHANGYE, slide_size_ZHANGYE, slide_yuHouFenZu_ZHANGYE, slide_zhuyuanhao_ZHANGYE = get_sort_one_center(os.path.join(data_root, "ZHANGYE"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='zhangye')
    print("zhang_ye Load")

    slide_patch_feat_HUADONG, slide_patch_label_HUADONG, slide_fileName_HUADONG, slide_patch_fileName_HUADONG, slide_patho_label_HUADONG, slide_nuclearLevel_label_HUADONG, slide_prognosis_HUADONG, slide_isup_HUADONG, slide_necrosis_HUADONG, slide_tnm_HUADONG, slide_size_HUADONG, slide_yuHouFenZu_HUADONG, slide_zhuyuanhao_HUADONG = get_sort_one_center(os.path.join(data_root, "HUADONG"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='huadong')
    print("hua_dong Load")

    slide_patch_feat_TCGAKICH, slide_patch_label_TCGAKICH, slide_fileName_TCGAKICH, slide_patch_fileName_TCGAKICH, slide_patho_label_TCGAKICH, slide_nuclearLevel_label_TCGAKICH, slide_prognosis_TCGAKICH, slide_isup_TCGAKICH, slide_necrosis_TCGAKICH, slide_tnm_TCGAKICH, slide_size_TCGAKICH, slide_yuHouFenZu_TCGAKICH, slide_zhuyuanhao_TCGAKICH = get_sort_one_center(os.path.join(data_root, "TCGA_kich"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kich Load")
    
    slide_patch_feat_TCGAKIRC, slide_patch_label_TCGAKIRC, slide_fileName_TCGAKIRC, slide_patch_fileName_TCGAKIRC, slide_patho_label_TCGAKIRC, slide_nuclearLevel_label_TCGAKIRC, slide_prognosis_TCGAKIRC, slide_isup_TCGAKIRC, slide_necrosis_TCGAKIRC, slide_tnm_TCGAKIRC, slide_size_TCGAKIRC, slide_yuHouFenZu_TCGAKIRC, slide_zhuyuanhao_TCGAKIRC = get_sort_one_center(os.path.join(data_root, "TCGA_kirc"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kirc Load")

    slide_patch_feat_TCGAKIRP, slide_patch_label_TCGAKIRP, slide_fileName_TCGAKIRP, slide_patch_fileName_TCGAKIRP, slide_patho_label_TCGAKIRP, slide_nuclearLevel_label_TCGAKIRP, slide_prognosis_TCGAKIRP, slide_isup_TCGAKIRP, slide_necrosis_TCGAKIRP, slide_tnm_TCGAKIRP, slide_size_TCGAKIRP, slide_yuHouFenZu_TCGAKIRP, slide_zhuyuanhao_TCGAKIRP = get_sort_one_center(os.path.join(data_root, "TCGA_kirp"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kirp Load")

    slide_patch_feat_CCRCC, slide_patch_label_CCRCC, slide_fileName_CCRCC, slide_patch_fileName_CCRCC, slide_patho_label_CCRCC, slide_nuclearLevel_label_CCRCC, slide_prognosis_CCRCC, slide_isup_CCRCC, slide_necrosis_CCRCC, slide_tnm_CCRCC, slide_size_CCRCC, slide_yuHouFenZu_CCRCC, slide_zhuyuanhao_CCRCC = get_sort_one_center(os.path.join(data_root, "CCRCC"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='CCRCC')
    print("CCRCC Load")

    slide_patch_feat_ZSRUTOU, slide_patch_label_ZSRUTOU, slide_fileName_ZSRUTOU, slide_patch_fileName_ZSRUTOU, slide_patho_label_ZSRUTOU, slide_nuclearLevel_label_ZSRUTOU, slide_prognosis_ZSRUTOU, slide_isup_ZSRUTOU, slide_necrosis_ZSRUTOU, slide_tnm_ZSRUTOU, slide_size_ZSRUTOU, slide_yuHouFenZu_ZSRUTOU, slide_zhuyuanhao_ZSRUTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_RUTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_rutou Load")

    slide_patch_feat_ZSXIANSE, slide_patch_label_ZSXIANSE, slide_fileName_ZSXIANSE, slide_patch_fileName_ZSXIANSE, slide_patho_label_ZSXIANSE, slide_nuclearLevel_label_ZSXIANSE, slide_prognosis_ZSXIANSE, slide_isup_ZSXIANSE, slide_necrosis_ZSXIANSE, slide_tnm_ZSXIANSE, slide_size_ZSXIANSE, slide_yuHouFenZu_ZSXIANSE, slide_zhuyuanhao_ZSXIANSE = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XIANSE"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xianse Load")
    #
    slide_patch_feat_ZSXINTOU, slide_patch_label_ZSXINTOU, slide_fileName_ZSXINTOU, slide_patch_fileName_ZSXINTOU, slide_patho_label_ZSXINTOU, slide_nuclearLevel_label_ZSXINTOU, slide_prognosis_ZSXINTOU, slide_isup_ZSXINTOU, slide_necrosis_ZSXINTOU, slide_tnm_ZSXINTOU, slide_size_ZSXINTOU, slide_yuHouFenZu_ZSXINTOU, slide_zhuyuanhao_ZSXINTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XINTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xintou Load")

    slide_patch_feat_ZSBUCHONG, slide_patch_label_ZSBUCHONG, slide_fileName_ZSBUCHONG, slide_patch_fileName_ZSBUCHONG, slide_patho_label_ZSBUCHONG, slide_nuclearLevel_label_ZSBUCHONG, slide_prognosis_ZSBUCHONG, slide_isup_ZSBUCHONG, slide_necrosis_ZSBUCHONG, slide_tnm_ZSBUCHONG, slide_size_ZSBUCHONG, slide_yuHouFenZu_ZSBUCHONG, slide_zhuyuanhao_ZSBUCHONG = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_BUCHONG"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_buchong Load")

    slide_patch_feat_ZSQITA, slide_patch_label_ZSQITA, slide_fileName_ZSQITA, slide_patch_fileName_ZSQITA, slide_patho_label_ZSQITA, slide_nuclearLevel_label_ZSQITA, slide_prognosis_ZSQITA, slide_isup_ZSQITA, slide_necrosis_ZSQITA, slide_tnm_ZSQITA, slide_size_ZSQITA, slide_yuHouFenZu_ZSQITA, slide_zhuyuanhao_ZSQITA = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_QITA"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_qita Load")

    train_data =               slide_patch_feat_ZSTOUMING + slide_patch_feat_ZSRUTOU +         slide_patch_feat_ZSXIANSE +         slide_patch_feat_ZSQITA +         slide_patch_feat_ZSXINTOU +         slide_patch_feat_ZSBUCHONG +         slide_patch_feat_XIAMEN +         slide_patch_feat_ZHANGYE +         slide_patch_feat_CCRCC     +slide_patch_feat_TCGAKIRC +         slide_patch_feat_TCGAKICH +         slide_patch_feat_TCGAKIRP
    train_label =              slide_patch_label_ZSTOUMING + slide_patch_label_ZSRUTOU +        slide_patch_label_ZSXIANSE +        slide_patch_label_ZSQITA +        slide_patch_label_ZSXINTOU +        slide_patch_label_ZSBUCHONG +        slide_patch_label_XIAMEN +        slide_patch_label_ZHANGYE +        slide_patch_label_CCRCC   +slide_patch_label_TCGAKIRC +        slide_patch_label_TCGAKICH +        slide_patch_label_TCGAKIRP
    train_name_slide =         slide_fileName_ZSTOUMING + slide_fileName_ZSRUTOU +           slide_fileName_ZSXIANSE+            slide_fileName_ZSQITA +           slide_fileName_ZSXINTOU +           slide_fileName_ZSBUCHONG +           slide_fileName_XIAMEN +           slide_fileName_ZHANGYE +           slide_fileName_CCRCC         +slide_fileName_TCGAKIRC +           slide_fileName_TCGAKICH +           slide_fileName_TCGAKIRP
    train_name_patch =         slide_patch_fileName_ZSTOUMING + slide_patch_fileName_ZSRUTOU +     slide_patch_fileName_ZSXIANSE +     slide_patch_fileName_ZSQITA +     slide_patch_fileName_ZSXINTOU +     slide_patch_fileName_ZSBUCHONG +     slide_patch_fileName_XIAMEN +     slide_patch_fileName_ZHANGYE +     slide_patch_fileName_CCRCC  +slide_patch_fileName_TCGAKIRC +     slide_patch_fileName_TCGAKICH +     slide_patch_fileName_TCGAKIRP
    train_patho_label =        slide_patho_label_ZSTOUMING + slide_patho_label_ZSRUTOU +        slide_patho_label_ZSXIANSE +        slide_patho_label_ZSQITA +        slide_patho_label_ZSXINTOU +        slide_patho_label_ZSBUCHONG +        slide_patho_label_XIAMEN +        slide_patho_label_ZHANGYE +        slide_patho_label_CCRCC   +slide_patho_label_TCGAKIRC +        slide_patho_label_TCGAKICH +        slide_patho_label_TCGAKIRP
    train_nuclearLevel_label = slide_nuclearLevel_label_ZSTOUMING + slide_nuclearLevel_label_ZSRUTOU + slide_nuclearLevel_label_ZSXIANSE + slide_nuclearLevel_label_ZSQITA + slide_nuclearLevel_label_ZSXINTOU + slide_nuclearLevel_label_ZSBUCHONG + slide_nuclearLevel_label_XIAMEN + slide_nuclearLevel_label_ZHANGYE + slide_nuclearLevel_label_CCRCC +slide_nuclearLevel_label_TCGAKIRC + slide_nuclearLevel_label_TCGAKICH + slide_nuclearLevel_label_TCGAKIRP
    train_prognosis =          slide_prognosis_ZSTOUMING + slide_prognosis_ZSRUTOU +          slide_prognosis_ZSXIANSE +          slide_prognosis_ZSQITA +          slide_prognosis_ZSXINTOU +          slide_prognosis_ZSBUCHONG +          slide_prognosis_XIAMEN +          slide_prognosis_ZHANGYE +          slide_prognosis_CCRCC  +slide_prognosis_TCGAKIRC +          slide_prognosis_TCGAKICH +          slide_prognosis_TCGAKIRP
    train_isup =               slide_isup_ZSTOUMING + slide_isup_ZSRUTOU +               slide_isup_ZSXIANSE +               slide_isup_ZSQITA +               slide_isup_ZSXINTOU +               slide_isup_ZSBUCHONG +               slide_isup_XIAMEN +               slide_isup_ZHANGYE +               slide_isup_CCRCC   +slide_isup_TCGAKIRC +               slide_isup_TCGAKICH +               slide_isup_TCGAKIRP
    train_size =               slide_size_ZSTOUMING + slide_size_ZSRUTOU +               slide_size_ZSXIANSE +               slide_size_ZSQITA +               slide_size_ZSXINTOU +               slide_size_ZSBUCHONG +               slide_size_XIAMEN +               slide_size_ZHANGYE +               slide_size_CCRCC   +slide_size_TCGAKIRC +               slide_size_TCGAKICH +               slide_size_TCGAKIRP
    train_necrosis =           slide_necrosis_ZSTOUMING + slide_necrosis_ZSRUTOU +           slide_necrosis_ZSXIANSE +           slide_necrosis_ZSQITA +           slide_necrosis_ZSXINTOU +           slide_necrosis_ZSBUCHONG +           slide_necrosis_XIAMEN +           slide_necrosis_ZHANGYE +           slide_necrosis_CCRCC   +slide_necrosis_TCGAKIRC +           slide_necrosis_TCGAKICH +           slide_necrosis_TCGAKIRP
    train_tnm =                slide_tnm_ZSTOUMING + slide_tnm_ZSRUTOU +                slide_tnm_ZSXIANSE +                slide_tnm_ZSQITA +                slide_tnm_ZSXINTOU +                slide_tnm_ZSBUCHONG +                slide_tnm_XIAMEN +                slide_tnm_ZHANGYE +                slide_tnm_CCRCC  +slide_tnm_TCGAKIRC +                slide_tnm_TCGAKICH +                slide_tnm_TCGAKIRP
    train_yuHouFenZu =         slide_yuHouFenZu_ZSTOUMING + slide_yuHouFenZu_ZSRUTOU +         slide_yuHouFenZu_ZSXIANSE +         slide_yuHouFenZu_ZSQITA +         slide_yuHouFenZu_ZSXINTOU +         slide_yuHouFenZu_ZSBUCHONG +         slide_yuHouFenZu_XIAMEN +         slide_yuHouFenZu_ZHANGYE +         slide_yuHouFenZu_CCRCC   +slide_yuHouFenZu_TCGAKIRC +         slide_yuHouFenZu_TCGAKICH +         slide_yuHouFenZu_TCGAKIRP
    train_zhuyuanhao =         slide_zhuyuanhao_ZSTOUMING + slide_zhuyuanhao_ZSRUTOU +         slide_zhuyuanhao_ZSXIANSE +         slide_zhuyuanhao_ZSQITA +         slide_zhuyuanhao_ZSXINTOU +         slide_zhuyuanhao_ZSBUCHONG +         slide_zhuyuanhao_XIAMEN +         slide_zhuyuanhao_ZHANGYE +         slide_zhuyuanhao_CCRCC   +slide_zhuyuanhao_TCGAKIRC +         slide_zhuyuanhao_TCGAKICH +         slide_zhuyuanhao_TCGAKIRP
    
    train_all = list(zip(train_data, train_label, train_name_slide, train_name_patch, train_patho_label, train_nuclearLevel_label, train_prognosis, train_isup, train_size, train_necrosis, train_tnm, train_yuHouFenZu, train_zhuyuanhao))
    train_data[:], train_label[:], train_name_slide[:], train_name_patch[:], train_patho_label[:], train_nuclearLevel_label[:], train_prognosis[:], train_isup[:], train_size[:], train_necrosis[:], train_tnm[:], train_yuHouFenZu[:], train_zhuyuanhao[:] = zip(*train_all)

    huadong_data = slide_patch_feat_HUADONG
    huadong_label = slide_patch_label_HUADONG
    huadong_name_slide = slide_fileName_HUADONG
    huadong_name_patch = slide_patch_fileName_HUADONG
    huadong_patho_label = slide_patho_label_HUADONG
    huadong_nuclearLevel_label = slide_nuclearLevel_label_HUADONG
    huadong_prognosis_label = slide_prognosis_HUADONG
    huadong_isup_label = slide_isup_HUADONG
    huadong_size_label = slide_size_HUADONG
    huadong_necrosis_label = slide_necrosis_HUADONG
    huadong_tnm_label = slide_tnm_HUADONG
    huadong_yuHouFenZu = slide_yuHouFenZu_HUADONG
    huadong_zhuyuanhao = slide_zhuyuanhao_HUADONG
    huadong_all = list(zip(huadong_data, huadong_label, huadong_name_slide, huadong_name_patch, huadong_patho_label, huadong_nuclearLevel_label, huadong_prognosis_label, huadong_isup_label, huadong_size_label, huadong_necrosis_label, huadong_tnm_label, huadong_yuHouFenZu, huadong_zhuyuanhao))
    huadong_data[:], huadong_label[:], huadong_name_slide[:], huadong_name_patch[:], huadong_patho_label[:], huadong_nuclearLevel_label[:], huadong_prognosis_label[:], huadong_isup_label[:], huadong_size_label[:], huadong_necrosis_label[:], huadong_tnm_label[:], huadong_yuHouFenZu[:], huadong_zhuyuanhao[:] = zip(*huadong_all)

    patient2indices = defaultdict(list)
    for idx, zhuyuanhao in enumerate(train_zhuyuanhao):
        patient2indices[zhuyuanhao].append(idx)
    
    internal_train_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split.xlsx", sheet_name = "internaltrain")
    internal_test_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split.xlsx", sheet_name = "internaltest")

    internal_train_ids = internal_train_df['住院号'].astype(str).tolist()
    internal_test_ids = internal_test_df['住院号'].astype(str).tolist()
    
    train_patient_ids = []
    test_patient_ids = []

    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_ids:
            train_patient_ids.append(zhuyuanhao)
        if zhuyuanhao in internal_test_ids:
            test_patient_ids.append(zhuyuanhao)

    internal_train_set = set(internal_train_ids)
    internal_test_set = set(internal_test_ids)

    overlap = internal_train_set & internal_test_set
    if overlap:
        print(f"警告: 住院号 {overlap} 同时出现在训练和测试集中!")

    train_patient_ids = []
    test_patient_ids = []
    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_set:
            train_patient_ids.append(zhuyuanhao)
        elif zhuyuanhao in internal_test_set:  # 使用elif避免重复
            test_patient_ids.append(zhuyuanhao)
    
    train_patient_ids = list(set(train_patient_ids))
    test_patient_ids = list(set(test_patient_ids))
            
    # 收集索引
    train_indices = []
    test_indices = []
    for pid in train_patient_ids:
        train_indices.extend(patient2indices[pid])
    for pid in test_patient_ids:
        test_indices.extend(patient2indices[pid])
    
    def select_by_indices(data_list, indices):
        return [data_list[i] for i in indices]

    InternalTrain_data = select_by_indices(train_data, train_indices)
    InternalTrain_label = select_by_indices(train_label, train_indices)
    InternalTrain_name_slide = select_by_indices(train_name_slide, train_indices)
    InternalTrain_name_patch = select_by_indices(train_name_patch, train_indices)
    InternalTrain_patho_label = select_by_indices(train_patho_label, train_indices)
    InternalTrain_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, train_indices)
    InternalTrain_prognosis_label = select_by_indices(train_prognosis, train_indices)
    InternalTrain_isup_label = select_by_indices(train_isup, train_indices)
    InternalTrain_size_label = select_by_indices(train_size, train_indices)
    InternalTrain_necrosis_label = select_by_indices(train_necrosis, train_indices)
    InternalTrain_tnm_label = select_by_indices(train_tnm, train_indices)
    InternalTrain_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, train_indices)
    InternalTrain_zhuyuanhao = select_by_indices(train_zhuyuanhao, train_indices)

    InternalTest_data = select_by_indices(train_data, test_indices)
    InternalTest_label = select_by_indices(train_label, test_indices)
    InternalTest_name_slide = select_by_indices(train_name_slide, test_indices)
    InternalTest_name_patch = select_by_indices(train_name_patch, test_indices)
    InternalTest_patho_label = select_by_indices(train_patho_label, test_indices)
    InternalTest_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, test_indices)
    InternalTest_prognosis_label = select_by_indices(train_prognosis, test_indices)
    InternalTest_isup_label = select_by_indices(train_isup, test_indices)
    InternalTest_size_label = select_by_indices(train_size, test_indices)
    InternalTest_necrosis_label = select_by_indices(train_necrosis, test_indices)
    InternalTest_tnm_label = select_by_indices(train_tnm, test_indices)
    InternalTest_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, test_indices)
    InternalTest_zhuyuanhao = select_by_indices(train_zhuyuanhao, test_indices)

    # Split some external test into train
    def is_tcga_slide(slide_name):
        return any(tcga_key in slide_name for tcga_key in ["TCGA", "TCGAKIRC", "TCGAKICH", "TCGAKIRP"])

    # 在 InternalTest 中筛选 TCGA 样本索引
    external_test_indices = [i for i, name in enumerate(InternalTest_name_slide) if is_tcga_slide(name)]

    # 只用这些索引从 InternalTest 里提取 TCGA 作为 ExternalTest
    ExternalTest_data = select_by_indices(InternalTest_data, external_test_indices)
    ExternalTest_label = select_by_indices(InternalTest_label, external_test_indices)
    ExternalTest_name_slide = select_by_indices(InternalTest_name_slide, external_test_indices)
    ExternalTest_name_patch = select_by_indices(InternalTest_name_patch, external_test_indices)
    ExternalTest_patho_label = select_by_indices(InternalTest_patho_label, external_test_indices)
    ExternalTest_nuclearLevel_label = select_by_indices(InternalTest_nuclearLevel_label, external_test_indices)
    ExternalTest_prognosis_label = select_by_indices(InternalTest_prognosis_label, external_test_indices)
    ExternalTest_isup_label = select_by_indices(InternalTest_isup_label, external_test_indices)
    ExternalTest_size_label = select_by_indices(InternalTest_size_label, external_test_indices)
    ExternalTest_necrosis_label = select_by_indices(InternalTest_necrosis_label, external_test_indices)
    ExternalTest_tnm_label = select_by_indices(InternalTest_tnm_label, external_test_indices)
    ExternalTest_yuHouFenZu_label = select_by_indices(InternalTest_yuHouFenZu_label, external_test_indices)
    ExternalTest_zhuyuanhao = select_by_indices(InternalTest_zhuyuanhao, external_test_indices)
    
    def export_to_excel(all_datasets, sheet_names, filename):
        with pd.ExcelWriter(filename) as writer:
            for dataset, sheet_name in zip(all_datasets, sheet_names):
                df = pd.DataFrame({
                    'feat_dim': [len(x) if hasattr(x, '__len__') else 'unknown' for x in dataset[0]],  # 特征维度
                    'label': [",".join(map(str, lbl)) if isinstance(lbl, (list, np.ndarray)) else str(lbl) for lbl in dataset[1]],
                    'slide_name': dataset[2],
                    'patch_name': dataset[3],
                    'patho_label': dataset[4],
                    'nuclear_level_label': dataset[5],
                    'prognosis_label': dataset[6],
                    'isup_label': dataset[7],
                    'size_label': dataset[8],
                    'tnm_label': dataset[9],
                    'necrosis_label': dataset[10],
                    'yuHouFenZu_label': dataset[11],
                    'zhuyuanhao': dataset[12],
                })
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"导出成功：{filename}")
    
    export_to_excel(
        all_datasets=[
            [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label, InternalTrain_isup_label, InternalTrain_size_label, InternalTrain_tnm_label, InternalTrain_necrosis_label, InternalTrain_yuHouFenZu_label, InternalTrain_zhuyuanhao],
            [InternalTest_data,  InternalTest_label,  InternalTest_name_slide,  InternalTest_name_patch,  InternalTest_patho_label,  InternalTest_nuclearLevel_label,  InternalTest_prognosis_label,  InternalTest_isup_label,  InternalTest_size_label,  InternalTest_tnm_label,  InternalTest_necrosis_label, InternalTest_yuHouFenZu_label, InternalTest_zhuyuanhao],
            [ExternalTest_data,  ExternalTest_label,  ExternalTest_name_slide,  ExternalTest_name_patch,  ExternalTest_patho_label,  ExternalTest_nuclearLevel_label,  ExternalTest_prognosis_label,  ExternalTest_isup_label,  ExternalTest_size_label,  ExternalTest_tnm_label,  ExternalTest_necrosis_label, ExternalTest_yuHouFenZu_label, ExternalTest_zhuyuanhao],
            [huadong_data,       huadong_label,       huadong_name_slide,       huadong_name_patch,       huadong_patho_label,       huadong_nuclearLevel_label,       huadong_prognosis_label,       huadong_isup_label,       huadong_size_label,       huadong_tnm_label,       huadong_necrosis_label, huadong_yuHouFenZu, huadong_zhuyuanhao],
        ],
        sheet_names=["InternalTrain", "InternalTest", "ExternalTest", "HuadongTest"],
        filename="MultiCenter_Split_Info.xlsx"
    )

    print("ALL FEAT LOADED")
    return (
        [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label, InternalTrain_isup_label, InternalTrain_size_label, InternalTrain_tnm_label, InternalTrain_necrosis_label, InternalTrain_yuHouFenZu_label, InternalTrain_zhuyuanhao],
        [InternalTest_data,  InternalTest_label,  InternalTest_name_slide,  InternalTest_name_patch,  InternalTest_patho_label,  InternalTest_nuclearLevel_label,  InternalTest_prognosis_label,  InternalTest_isup_label,  InternalTest_size_label,  InternalTest_tnm_label,  InternalTest_necrosis_label, InternalTest_yuHouFenZu_label, InternalTest_zhuyuanhao],
        [ExternalTest_data,  ExternalTest_label,  ExternalTest_name_slide,  ExternalTest_name_patch,  ExternalTest_patho_label,  ExternalTest_nuclearLevel_label,  ExternalTest_prognosis_label,  ExternalTest_isup_label,  ExternalTest_size_label,  ExternalTest_tnm_label,  ExternalTest_necrosis_label, ExternalTest_yuHouFenZu_label, ExternalTest_zhuyuanhao],
        [huadong_data,       huadong_label,       huadong_name_slide,       huadong_name_patch,       huadong_patho_label,       huadong_nuclearLevel_label,       huadong_prognosis_label,       huadong_isup_label,       huadong_size_label,       huadong_tnm_label,       huadong_necrosis_label, huadong_yuHouFenZu, huadong_zhuyuanhao],
    )

def get_train_ds_ZSCenter_forNuclearLevel(
         data_root='/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1',
        downsample=1.0):
    
    slide_patch_feat_ZSTOUMING, slide_patch_label_ZSTOUMING, slide_fileName_ZSTOUMING, slide_patch_fileName_ZSTOUMING, slide_patho_label_ZSTOUMING, slide_nuclearLevel_label_ZSTOUMING, slide_prognosis_ZSTOUMING, slide_isup_ZSTOUMING, slide_necrosis_ZSTOUMING, slide_tnm_ZSTOUMING, slide_size_ZSTOUMING, slide_yuHouFenZu_ZSTOUMING, slide_zhuyuanhao_ZSTOUMING = get_sort_pred_label(os.path.join(data_root, "ZHONGSHAN_TOUMING"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_touming Load")

    slide_patch_feat_ZSRUTOU, slide_patch_label_ZSRUTOU, slide_fileName_ZSRUTOU, slide_patch_fileName_ZSRUTOU, slide_patho_label_ZSRUTOU, slide_nuclearLevel_label_ZSRUTOU, slide_prognosis_ZSRUTOU, slide_isup_ZSRUTOU, slide_necrosis_ZSRUTOU, slide_tnm_ZSRUTOU, slide_size_ZSRUTOU, slide_yuHouFenZu_ZSRUTOU, slide_zhuyuanhao_ZSRUTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_RUTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_rutou Load")

    slide_patch_feat_ZSXIANSE, slide_patch_label_ZSXIANSE, slide_fileName_ZSXIANSE, slide_patch_fileName_ZSXIANSE, slide_patho_label_ZSXIANSE, slide_nuclearLevel_label_ZSXIANSE, slide_prognosis_ZSXIANSE, slide_isup_ZSXIANSE, slide_necrosis_ZSXIANSE, slide_tnm_ZSXIANSE, slide_size_ZSXIANSE, slide_yuHouFenZu_ZSXIANSE, slide_zhuyuanhao_ZSXIANSE = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XIANSE"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xianse Load")
    #
    slide_patch_feat_ZSXINTOU, slide_patch_label_ZSXINTOU, slide_fileName_ZSXINTOU, slide_patch_fileName_ZSXINTOU, slide_patho_label_ZSXINTOU, slide_nuclearLevel_label_ZSXINTOU, slide_prognosis_ZSXINTOU, slide_isup_ZSXINTOU, slide_necrosis_ZSXINTOU, slide_tnm_ZSXINTOU, slide_size_ZSXINTOU, slide_yuHouFenZu_ZSXINTOU, slide_zhuyuanhao_ZSXINTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XINTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xintou Load")

    slide_patch_feat_ZSBUCHONG, slide_patch_label_ZSBUCHONG, slide_fileName_ZSBUCHONG, slide_patch_fileName_ZSBUCHONG, slide_patho_label_ZSBUCHONG, slide_nuclearLevel_label_ZSBUCHONG, slide_prognosis_ZSBUCHONG, slide_isup_ZSBUCHONG, slide_necrosis_ZSBUCHONG, slide_tnm_ZSBUCHONG, slide_size_ZSBUCHONG, slide_yuHouFenZu_ZSBUCHONG, slide_zhuyuanhao_ZSBUCHONG = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_BUCHONG"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_buchong Load")

    slide_patch_feat_ZSQITA, slide_patch_label_ZSQITA, slide_fileName_ZSQITA, slide_patch_fileName_ZSQITA, slide_patho_label_ZSQITA, slide_nuclearLevel_label_ZSQITA, slide_prognosis_ZSQITA, slide_isup_ZSQITA, slide_necrosis_ZSQITA, slide_tnm_ZSQITA, slide_size_ZSQITA, slide_yuHouFenZu_ZSQITA, slide_zhuyuanhao_ZSQITA = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_QITA"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_qita Load")

    train_data =               slide_patch_feat_ZSTOUMING + slide_patch_feat_ZSRUTOU +         slide_patch_feat_ZSXIANSE +         slide_patch_feat_ZSQITA +         slide_patch_feat_ZSXINTOU +         slide_patch_feat_ZSBUCHONG
    train_label =              slide_patch_label_ZSTOUMING + slide_patch_label_ZSRUTOU +        slide_patch_label_ZSXIANSE +        slide_patch_label_ZSQITA +        slide_patch_label_ZSXINTOU +        slide_patch_label_ZSBUCHONG 
    train_name_slide =         slide_fileName_ZSTOUMING + slide_fileName_ZSRUTOU +           slide_fileName_ZSXIANSE+            slide_fileName_ZSQITA +           slide_fileName_ZSXINTOU +           slide_fileName_ZSBUCHONG 
    train_name_patch =         slide_patch_fileName_ZSTOUMING + slide_patch_fileName_ZSRUTOU +     slide_patch_fileName_ZSXIANSE +     slide_patch_fileName_ZSQITA +     slide_patch_fileName_ZSXINTOU +     slide_patch_fileName_ZSBUCHONG
    train_patho_label =        slide_patho_label_ZSTOUMING + slide_patho_label_ZSRUTOU +        slide_patho_label_ZSXIANSE +        slide_patho_label_ZSQITA +        slide_patho_label_ZSXINTOU +        slide_patho_label_ZSBUCHONG 
    train_nuclearLevel_label = slide_nuclearLevel_label_ZSTOUMING + slide_nuclearLevel_label_ZSRUTOU + slide_nuclearLevel_label_ZSXIANSE + slide_nuclearLevel_label_ZSQITA + slide_nuclearLevel_label_ZSXINTOU + slide_nuclearLevel_label_ZSBUCHONG
    train_prognosis =          slide_prognosis_ZSTOUMING + slide_prognosis_ZSRUTOU +          slide_prognosis_ZSXIANSE +          slide_prognosis_ZSQITA +          slide_prognosis_ZSXINTOU +          slide_prognosis_ZSBUCHONG
    train_isup =               slide_isup_ZSTOUMING + slide_isup_ZSRUTOU +               slide_isup_ZSXIANSE +               slide_isup_ZSQITA +               slide_isup_ZSXINTOU +               slide_isup_ZSBUCHONG
    train_size =               slide_size_ZSTOUMING + slide_size_ZSRUTOU +               slide_size_ZSXIANSE +               slide_size_ZSQITA +               slide_size_ZSXINTOU +               slide_size_ZSBUCHONG
    train_necrosis =           slide_necrosis_ZSTOUMING + slide_necrosis_ZSRUTOU +           slide_necrosis_ZSXIANSE +           slide_necrosis_ZSQITA +           slide_necrosis_ZSXINTOU +           slide_necrosis_ZSBUCHONG
    train_tnm =                slide_tnm_ZSTOUMING + slide_tnm_ZSRUTOU +                slide_tnm_ZSXIANSE +                slide_tnm_ZSQITA +                slide_tnm_ZSXINTOU +                slide_tnm_ZSBUCHONG
    train_yuHouFenZu =         slide_yuHouFenZu_ZSTOUMING + slide_yuHouFenZu_ZSRUTOU +         slide_yuHouFenZu_ZSXIANSE +         slide_yuHouFenZu_ZSQITA +         slide_yuHouFenZu_ZSXINTOU +         slide_yuHouFenZu_ZSBUCHONG
    train_zhuyuanhao =         slide_zhuyuanhao_ZSTOUMING + slide_zhuyuanhao_ZSRUTOU +         slide_zhuyuanhao_ZSXIANSE +         slide_zhuyuanhao_ZSQITA +         slide_zhuyuanhao_ZSXINTOU +         slide_zhuyuanhao_ZSBUCHONG
    
    train_all = list(zip(train_data, train_label, train_name_slide, train_name_patch, train_patho_label, train_nuclearLevel_label, train_prognosis, train_isup, train_size, train_necrosis, train_tnm, train_yuHouFenZu, train_zhuyuanhao))
    train_data[:], train_label[:], train_name_slide[:], train_name_patch[:], train_patho_label[:], train_nuclearLevel_label[:], train_prognosis[:], train_isup[:], train_size[:], train_necrosis[:], train_tnm[:], train_yuHouFenZu[:], train_zhuyuanhao[:] = zip(*train_all)

    patient2indices = defaultdict(list)
    for idx, zhuyuanhao in enumerate(train_zhuyuanhao):
        patient2indices[zhuyuanhao].append(idx)
    
    internal_train_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split_nuclear.xlsx", sheet_name = "internaltrain")
    internal_test_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split_nuclear.xlsx", sheet_name = "internaltest")

    internal_train_ids = internal_train_df['住院号'].astype(str).tolist()
    internal_test_ids = internal_test_df['住院号'].astype(str).tolist()
    
    train_patient_ids = []
    test_patient_ids = []

    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_ids:
            train_patient_ids.append(zhuyuanhao)
        if zhuyuanhao in internal_test_ids:
            test_patient_ids.append(zhuyuanhao)

    internal_train_set = set(internal_train_ids)
    internal_test_set = set(internal_test_ids)

    overlap = internal_train_set & internal_test_set
    if overlap:
        print(f"警告: 住院号 {overlap} 同时出现在训练和测试集中!")

    train_patient_ids = []
    test_patient_ids = []
    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_set:
            train_patient_ids.append(zhuyuanhao)
        elif zhuyuanhao in internal_test_set:  # 使用elif避免重复
            test_patient_ids.append(zhuyuanhao)
    
    train_patient_ids = list(set(train_patient_ids))
    test_patient_ids = list(set(test_patient_ids))
            
    # 收集索引
    train_indices = []
    test_indices = []
    for pid in train_patient_ids:
        train_indices.extend(patient2indices[pid])
    for pid in test_patient_ids:
        test_indices.extend(patient2indices[pid])
    
    def select_by_indices(data_list, indices):
        return [data_list[i] for i in indices]

    InternalTrain_data = select_by_indices(train_data, train_indices)
    InternalTrain_label = select_by_indices(train_label, train_indices)
    InternalTrain_name_slide = select_by_indices(train_name_slide, train_indices)
    InternalTrain_name_patch = select_by_indices(train_name_patch, train_indices)
    InternalTrain_patho_label = select_by_indices(train_patho_label, train_indices)
    InternalTrain_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, train_indices)
    InternalTrain_prognosis_label = select_by_indices(train_prognosis, train_indices)
    InternalTrain_isup_label = select_by_indices(train_isup, train_indices)
    InternalTrain_size_label = select_by_indices(train_size, train_indices)
    InternalTrain_necrosis_label = select_by_indices(train_necrosis, train_indices)
    InternalTrain_tnm_label = select_by_indices(train_tnm, train_indices)
    InternalTrain_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, train_indices)
    InternalTrain_zhuyuanhao = select_by_indices(train_zhuyuanhao, train_indices)

    InternalTest_data = select_by_indices(train_data, test_indices)
    InternalTest_label = select_by_indices(train_label, test_indices)
    InternalTest_name_slide = select_by_indices(train_name_slide, test_indices)
    InternalTest_name_patch = select_by_indices(train_name_patch, test_indices)
    InternalTest_patho_label = select_by_indices(train_patho_label, test_indices)
    InternalTest_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, test_indices)
    InternalTest_prognosis_label = select_by_indices(train_prognosis, test_indices)
    InternalTest_isup_label = select_by_indices(train_isup, test_indices)
    InternalTest_size_label = select_by_indices(train_size, test_indices)
    InternalTest_necrosis_label = select_by_indices(train_necrosis, test_indices)
    InternalTest_tnm_label = select_by_indices(train_tnm, test_indices)
    InternalTest_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, test_indices)
    InternalTest_zhuyuanhao = select_by_indices(train_zhuyuanhao, test_indices)


    print("ALL FEAT LOADED")
    return (
        [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label, InternalTrain_isup_label, InternalTrain_size_label, InternalTrain_tnm_label, InternalTrain_necrosis_label, InternalTrain_yuHouFenZu_label, InternalTrain_zhuyuanhao],
        [InternalTest_data,  InternalTest_label,  InternalTest_name_slide,  InternalTest_name_patch,  InternalTest_patho_label,  InternalTest_nuclearLevel_label,  InternalTest_prognosis_label,  InternalTest_isup_label,  InternalTest_size_label,  InternalTest_tnm_label,  InternalTest_necrosis_label, InternalTest_yuHouFenZu_label, InternalTest_zhuyuanhao]
    )

def get_train_ds_ZSCenter_tsne(
         data_root='/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1',
        downsample=1.0):
    
    slide_patch_feat_ZSTOUMING, slide_patch_label_ZSTOUMING, slide_fileName_ZSTOUMING, slide_patch_fileName_ZSTOUMING, slide_patho_label_ZSTOUMING, slide_nuclearLevel_label_ZSTOUMING, slide_prognosis_ZSTOUMING, slide_isup_ZSTOUMING, slide_necrosis_ZSTOUMING, slide_tnm_ZSTOUMING, slide_size_ZSTOUMING, slide_yuHouFenZu_ZSTOUMING, slide_zhuyuanhao_ZSTOUMING = get_sort_pred_label(os.path.join(data_root, "ZHONGSHAN_TOUMING"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_touming Load")

    slide_patch_feat_ZSRUTOU, slide_patch_label_ZSRUTOU, slide_fileName_ZSRUTOU, slide_patch_fileName_ZSRUTOU, slide_patho_label_ZSRUTOU, slide_nuclearLevel_label_ZSRUTOU, slide_prognosis_ZSRUTOU, slide_isup_ZSRUTOU, slide_necrosis_ZSRUTOU, slide_tnm_ZSRUTOU, slide_size_ZSRUTOU, slide_yuHouFenZu_ZSRUTOU, slide_zhuyuanhao_ZSRUTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_RUTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_rutou Load")

    slide_patch_feat_ZSXIANSE, slide_patch_label_ZSXIANSE, slide_fileName_ZSXIANSE, slide_patch_fileName_ZSXIANSE, slide_patho_label_ZSXIANSE, slide_nuclearLevel_label_ZSXIANSE, slide_prognosis_ZSXIANSE, slide_isup_ZSXIANSE, slide_necrosis_ZSXIANSE, slide_tnm_ZSXIANSE, slide_size_ZSXIANSE, slide_yuHouFenZu_ZSXIANSE, slide_zhuyuanhao_ZSXIANSE = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XIANSE"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xianse Load")
    #
    slide_patch_feat_ZSXINTOU, slide_patch_label_ZSXINTOU, slide_fileName_ZSXINTOU, slide_patch_fileName_ZSXINTOU, slide_patho_label_ZSXINTOU, slide_nuclearLevel_label_ZSXINTOU, slide_prognosis_ZSXINTOU, slide_isup_ZSXINTOU, slide_necrosis_ZSXINTOU, slide_tnm_ZSXINTOU, slide_size_ZSXINTOU, slide_yuHouFenZu_ZSXINTOU, slide_zhuyuanhao_ZSXINTOU = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_XINTOU"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_xintou Load")

    slide_patch_feat_ZSBUCHONG, slide_patch_label_ZSBUCHONG, slide_fileName_ZSBUCHONG, slide_patch_fileName_ZSBUCHONG, slide_patho_label_ZSBUCHONG, slide_nuclearLevel_label_ZSBUCHONG, slide_prognosis_ZSBUCHONG, slide_isup_ZSBUCHONG, slide_necrosis_ZSBUCHONG, slide_tnm_ZSBUCHONG, slide_size_ZSBUCHONG, slide_yuHouFenZu_ZSBUCHONG, slide_zhuyuanhao_ZSBUCHONG = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_BUCHONG"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_buchong Load")

    slide_patch_feat_ZSQITA, slide_patch_label_ZSQITA, slide_fileName_ZSQITA, slide_patch_fileName_ZSQITA, slide_patho_label_ZSQITA, slide_nuclearLevel_label_ZSQITA, slide_prognosis_ZSQITA, slide_isup_ZSQITA, slide_necrosis_ZSQITA, slide_tnm_ZSQITA, slide_size_ZSQITA, slide_yuHouFenZu_ZSQITA, slide_zhuyuanhao_ZSQITA = get_sort_one_center(os.path.join(data_root, "ZHONGSHAN_QITA"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/中厦张C-629.xlsx", center='ZS')
    print("zhongshan_qita Load")

    train_data =               slide_patch_feat_ZSTOUMING + slide_patch_feat_ZSRUTOU +         slide_patch_feat_ZSXIANSE +         slide_patch_feat_ZSQITA +         slide_patch_feat_ZSXINTOU +         slide_patch_feat_ZSBUCHONG
    train_label =              slide_patch_label_ZSTOUMING + slide_patch_label_ZSRUTOU +        slide_patch_label_ZSXIANSE +        slide_patch_label_ZSQITA +        slide_patch_label_ZSXINTOU +        slide_patch_label_ZSBUCHONG 
    train_name_slide =         slide_fileName_ZSTOUMING + slide_fileName_ZSRUTOU +           slide_fileName_ZSXIANSE+            slide_fileName_ZSQITA +           slide_fileName_ZSXINTOU +           slide_fileName_ZSBUCHONG 
    train_name_patch =         slide_patch_fileName_ZSTOUMING + slide_patch_fileName_ZSRUTOU +     slide_patch_fileName_ZSXIANSE +     slide_patch_fileName_ZSQITA +     slide_patch_fileName_ZSXINTOU +     slide_patch_fileName_ZSBUCHONG
    train_patho_label =        slide_patho_label_ZSTOUMING + slide_patho_label_ZSRUTOU +        slide_patho_label_ZSXIANSE +        slide_patho_label_ZSQITA +        slide_patho_label_ZSXINTOU +        slide_patho_label_ZSBUCHONG 
    train_nuclearLevel_label = slide_nuclearLevel_label_ZSTOUMING + slide_nuclearLevel_label_ZSRUTOU + slide_nuclearLevel_label_ZSXIANSE + slide_nuclearLevel_label_ZSQITA + slide_nuclearLevel_label_ZSXINTOU + slide_nuclearLevel_label_ZSBUCHONG
    train_prognosis =          slide_prognosis_ZSTOUMING + slide_prognosis_ZSRUTOU +          slide_prognosis_ZSXIANSE +          slide_prognosis_ZSQITA +          slide_prognosis_ZSXINTOU +          slide_prognosis_ZSBUCHONG
    train_isup =               slide_isup_ZSTOUMING + slide_isup_ZSRUTOU +               slide_isup_ZSXIANSE +               slide_isup_ZSQITA +               slide_isup_ZSXINTOU +               slide_isup_ZSBUCHONG
    train_size =               slide_size_ZSTOUMING + slide_size_ZSRUTOU +               slide_size_ZSXIANSE +               slide_size_ZSQITA +               slide_size_ZSXINTOU +               slide_size_ZSBUCHONG
    train_necrosis =           slide_necrosis_ZSTOUMING + slide_necrosis_ZSRUTOU +           slide_necrosis_ZSXIANSE +           slide_necrosis_ZSQITA +           slide_necrosis_ZSXINTOU +           slide_necrosis_ZSBUCHONG
    train_tnm =                slide_tnm_ZSTOUMING + slide_tnm_ZSRUTOU +                slide_tnm_ZSXIANSE +                slide_tnm_ZSQITA +                slide_tnm_ZSXINTOU +                slide_tnm_ZSBUCHONG
    train_yuHouFenZu =         slide_yuHouFenZu_ZSTOUMING + slide_yuHouFenZu_ZSRUTOU +         slide_yuHouFenZu_ZSXIANSE +         slide_yuHouFenZu_ZSQITA +         slide_yuHouFenZu_ZSXINTOU +         slide_yuHouFenZu_ZSBUCHONG
    train_zhuyuanhao =         slide_zhuyuanhao_ZSTOUMING + slide_zhuyuanhao_ZSRUTOU +         slide_zhuyuanhao_ZSXIANSE +         slide_zhuyuanhao_ZSQITA +         slide_zhuyuanhao_ZSXINTOU +         slide_zhuyuanhao_ZSBUCHONG
    
    train_all = list(zip(train_data, train_label, train_name_slide, train_name_patch, train_patho_label, train_nuclearLevel_label, train_prognosis, train_isup, train_size, train_necrosis, train_tnm, train_yuHouFenZu, train_zhuyuanhao))
    train_data[:], train_label[:], train_name_slide[:], train_name_patch[:], train_patho_label[:], train_nuclearLevel_label[:], train_prognosis[:], train_isup[:], train_size[:], train_necrosis[:], train_tnm[:], train_yuHouFenZu[:], train_zhuyuanhao[:] = zip(*train_all)

    patient2indices = defaultdict(list)
    for idx, zhuyuanhao in enumerate(train_zhuyuanhao):
        patient2indices[zhuyuanhao].append(idx)
    
    internal_train_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split_nuclear.xlsx", sheet_name = "internaltrain")
    internal_test_df = pd.read_excel("/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/split_nuclear.xlsx", sheet_name = "internaltest")

    internal_train_ids = internal_train_df['住院号'].astype(str).tolist()
    internal_test_ids = internal_test_df['住院号'].astype(str).tolist()
    
    train_patient_ids = []
    test_patient_ids = []

    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_ids:
            train_patient_ids.append(zhuyuanhao)
        if zhuyuanhao in internal_test_ids:
            test_patient_ids.append(zhuyuanhao)

    internal_train_set = set(internal_train_ids)
    internal_test_set = set(internal_test_ids)

    overlap = internal_train_set & internal_test_set
    if overlap:
        print(f"警告: 住院号 {overlap} 同时出现在训练和测试集中!")

    train_patient_ids = []
    test_patient_ids = []
    for zhuyuanhao in train_zhuyuanhao:
        if zhuyuanhao in internal_train_set:
            train_patient_ids.append(zhuyuanhao)
        elif zhuyuanhao in internal_test_set:  # 使用elif避免重复
            test_patient_ids.append(zhuyuanhao)
    
    train_patient_ids = list(set(train_patient_ids))
    test_patient_ids = list(set(test_patient_ids))
            
    # 收集索引
    train_indices = []
    test_indices = []
    for pid in train_patient_ids:
        train_indices.extend(patient2indices[pid])
    for pid in test_patient_ids:
        test_indices.extend(patient2indices[pid])
    
    def select_by_indices(data_list, indices):
        return [data_list[i] for i in indices]

    InternalTrain_data = select_by_indices(train_data, train_indices)
    InternalTrain_label = select_by_indices(train_label, train_indices)
    InternalTrain_name_slide = select_by_indices(train_name_slide, train_indices)
    InternalTrain_name_patch = select_by_indices(train_name_patch, train_indices)
    InternalTrain_patho_label = select_by_indices(train_patho_label, train_indices)
    InternalTrain_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, train_indices)
    InternalTrain_prognosis_label = select_by_indices(train_prognosis, train_indices)
    InternalTrain_isup_label = select_by_indices(train_isup, train_indices)
    InternalTrain_size_label = select_by_indices(train_size, train_indices)
    InternalTrain_necrosis_label = select_by_indices(train_necrosis, train_indices)
    InternalTrain_tnm_label = select_by_indices(train_tnm, train_indices)
    InternalTrain_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, train_indices)
    InternalTrain_zhuyuanhao = select_by_indices(train_zhuyuanhao, train_indices)

    InternalTest_data = select_by_indices(train_data, test_indices)
    InternalTest_label = select_by_indices(train_label, test_indices)
    InternalTest_name_slide = select_by_indices(train_name_slide, test_indices)
    InternalTest_name_patch = select_by_indices(train_name_patch, test_indices)
    InternalTest_patho_label = select_by_indices(train_patho_label, test_indices)
    InternalTest_nuclearLevel_label = select_by_indices(train_nuclearLevel_label, test_indices)
    InternalTest_prognosis_label = select_by_indices(train_prognosis, test_indices)
    InternalTest_isup_label = select_by_indices(train_isup, test_indices)
    InternalTest_size_label = select_by_indices(train_size, test_indices)
    InternalTest_necrosis_label = select_by_indices(train_necrosis, test_indices)
    InternalTest_tnm_label = select_by_indices(train_tnm, test_indices)
    InternalTest_yuHouFenZu_label = select_by_indices(train_yuHouFenZu, test_indices)
    InternalTest_zhuyuanhao = select_by_indices(train_zhuyuanhao, test_indices)


    print("ALL FEAT LOADED")
    return (
        [InternalTrain_data, InternalTrain_label, InternalTrain_name_slide, InternalTrain_name_patch, InternalTrain_patho_label, InternalTrain_nuclearLevel_label, InternalTrain_prognosis_label, InternalTrain_isup_label, InternalTrain_size_label, InternalTrain_tnm_label, InternalTrain_necrosis_label, InternalTrain_yuHouFenZu_label, InternalTrain_zhuyuanhao],
        [InternalTest_data,  InternalTest_label,  InternalTest_name_slide,  InternalTest_name_patch,  InternalTest_patho_label,  InternalTest_nuclearLevel_label,  InternalTest_prognosis_label,  InternalTest_isup_label,  InternalTest_size_label,  InternalTest_tnm_label,  InternalTest_necrosis_label, InternalTest_yuHouFenZu_label, InternalTest_zhuyuanhao]
    )

def get_train_test_ds_MultiCenter_region_TCGA_huadong_tsne(
         data_root='/cpfs01/projects-SSD/cfff-bb5d866c17c2_SSD/public/Pathology/gigapath_feat_mpp1',
        downsample=1.0):

    slide_patch_feat_HUADONG, slide_patch_label_HUADONG, slide_fileName_HUADONG, slide_patch_fileName_HUADONG, slide_patho_label_HUADONG, slide_nuclearLevel_label_HUADONG, slide_prognosis_HUADONG, slide_isup_HUADONG, slide_necrosis_HUADONG, slide_tnm_HUADONG, slide_size_HUADONG, slide_yuHouFenZu_HUADONG, slide_zhuyuanhao_HUADONG = get_sort_one_center(os.path.join(data_root, "HUADONG"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='huadong')
    print("hua_dong Load")

    slide_patch_feat_TCGAKICH, slide_patch_label_TCGAKICH, slide_fileName_TCGAKICH, slide_patch_fileName_TCGAKICH, slide_patho_label_TCGAKICH, slide_nuclearLevel_label_TCGAKICH, slide_prognosis_TCGAKICH, slide_isup_TCGAKICH, slide_necrosis_TCGAKICH, slide_tnm_TCGAKICH, slide_size_TCGAKICH, slide_yuHouFenZu_TCGAKICH, slide_zhuyuanhao_TCGAKICH = get_sort_one_center(os.path.join(data_root, "TCGA_kich"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kich Load")
    
    slide_patch_feat_TCGAKIRC, slide_patch_label_TCGAKIRC, slide_fileName_TCGAKIRC, slide_patch_fileName_TCGAKIRC, slide_patho_label_TCGAKIRC, slide_nuclearLevel_label_TCGAKIRC, slide_prognosis_TCGAKIRC, slide_isup_TCGAKIRC, slide_necrosis_TCGAKIRC, slide_tnm_TCGAKIRC, slide_size_TCGAKIRC, slide_yuHouFenZu_TCGAKIRC, slide_zhuyuanhao_TCGAKIRC = get_sort_one_center(os.path.join(data_root, "TCGA_kirc"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kirc Load")

    slide_patch_feat_TCGAKIRP, slide_patch_label_TCGAKIRP, slide_fileName_TCGAKIRP, slide_patch_fileName_TCGAKIRP, slide_patho_label_TCGAKIRP, slide_nuclearLevel_label_TCGAKIRP, slide_prognosis_TCGAKIRP, slide_isup_TCGAKIRP, slide_necrosis_TCGAKIRP, slide_tnm_TCGAKIRP, slide_size_TCGAKIRP, slide_yuHouFenZu_TCGAKIRP, slide_zhuyuanhao_TCGAKIRP = get_sort_one_center(os.path.join(data_root, "TCGA_kirp"), downsample=downsample, slide_patho_anno_path="/cpfs01/projects-HDD/cfff-bb5d866c17c2_HDD/lxy_19111010030/LXY_BACKUP0428/RENAL/dataset/华东T-324.xlsx", center='TCGA')
    print("TCGA_kirp Load")
    
    test_data =               slide_patch_feat_TCGAKIRC +         slide_patch_feat_TCGAKICH +         slide_patch_feat_TCGAKIRP
    test_label =              slide_patch_label_TCGAKIRC +        slide_patch_label_TCGAKICH +        slide_patch_label_TCGAKIRP
    test_name_slide =         slide_fileName_TCGAKIRC +           slide_fileName_TCGAKICH +           slide_fileName_TCGAKIRP
    test_name_patch =         slide_patch_fileName_TCGAKIRC +     slide_patch_fileName_TCGAKICH +     slide_patch_fileName_TCGAKIRP
    test_patho_label =        slide_patho_label_TCGAKIRC +        slide_patho_label_TCGAKICH +        slide_patho_label_TCGAKIRP
    test_nuclearLevel_label = slide_nuclearLevel_label_TCGAKIRC + slide_nuclearLevel_label_TCGAKICH + slide_nuclearLevel_label_TCGAKIRP
    test_prognosis =          slide_prognosis_TCGAKIRC +          slide_prognosis_TCGAKICH +          slide_prognosis_TCGAKIRP
    test_isup =               slide_isup_TCGAKIRC +               slide_isup_TCGAKICH +               slide_isup_TCGAKIRP
    test_size =               slide_size_TCGAKIRC +               slide_size_TCGAKICH +               slide_size_TCGAKIRP
    test_necrosis =           slide_necrosis_TCGAKIRC +           slide_necrosis_TCGAKICH +           slide_necrosis_TCGAKIRP
    test_tnm =                slide_tnm_TCGAKIRC +                slide_tnm_TCGAKICH +                slide_tnm_TCGAKIRP
    test_zhuyuanhao =         slide_zhuyuanhao_TCGAKIRC +         slide_zhuyuanhao_TCGAKICH +         slide_zhuyuanhao_TCGAKIRP
    # shuffle
    test_all = list(zip(test_data, test_label, test_name_slide, test_name_patch, test_patho_label, test_nuclearLevel_label, test_prognosis, test_isup, test_size, test_necrosis, test_tnm, test_zhuyuanhao))
    test_data[:], test_label[:], test_name_slide[:], test_name_patch[:], test_patho_label[:], test_nuclearLevel_label[:], test_prognosis[:], test_isup[:], test_size[:], test_necrosis[:], test_tnm[:], test_zhuyuanhao[:] = zip(*test_all)


    huadong_data = slide_patch_feat_HUADONG
    huadong_label = slide_patch_label_HUADONG
    huadong_name_slide = slide_fileName_HUADONG
    huadong_name_patch = slide_patch_fileName_HUADONG
    huadong_patho_label = slide_patho_label_HUADONG
    huadong_nuclearLevel_label = slide_nuclearLevel_label_HUADONG
    huadong_prognosis_label = slide_prognosis_HUADONG
    huadong_isup_label = slide_isup_HUADONG
    huadong_size_label = slide_size_HUADONG
    huadong_necrosis_label = slide_necrosis_HUADONG
    huadong_tnm_label = slide_tnm_HUADONG
    huadong_yuHouFenZu = slide_yuHouFenZu_HUADONG
    huadong_zhuyuanhao = slide_zhuyuanhao_HUADONG
    huadong_all = list(zip(huadong_data, huadong_label, huadong_name_slide, huadong_name_patch, huadong_patho_label, huadong_nuclearLevel_label, huadong_prognosis_label, huadong_isup_label, huadong_size_label, huadong_necrosis_label, huadong_tnm_label, huadong_yuHouFenZu, huadong_zhuyuanhao))
    huadong_data[:], huadong_label[:], huadong_name_slide[:], huadong_name_patch[:], huadong_patho_label[:], huadong_nuclearLevel_label[:], huadong_prognosis_label[:], huadong_isup_label[:], huadong_size_label[:], huadong_necrosis_label[:], huadong_tnm_label[:], huadong_yuHouFenZu[:], huadong_zhuyuanhao[:] = zip(*huadong_all)
    
    print("ALL FEAT LOADED")
    return (
        [test_data,  test_label,  test_name_slide,  test_name_patch,  test_patho_label,  test_nuclearLevel_label,  test_prognosis,  test_isup,  test_size,  test_tnm,  test_necrosis, test_zhuyuanhao],
        [huadong_data,       huadong_label,       huadong_name_slide,       huadong_name_patch,       huadong_patho_label,       huadong_nuclearLevel_label,       huadong_prognosis_label,       huadong_isup_label,       huadong_size_label,       huadong_tnm_label,       huadong_necrosis_label, huadong_yuHouFenZu, huadong_zhuyuanhao],
    )

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

        # 过滤掉 label 为 -1 的样本
        self.all_patches_info = sum(self.all_slide_info, [])
        self.all_patches_label = np.concatenate(self.all_slide_label)
        
        # 筛选出 label != -1 的样本
        filtered_info = []
        filtered_label = []
        
        for info, label in zip(self.all_patches_info, self.all_patches_label):
            if label != -1:  # 只保留 label 不为 -1 的样本
                filtered_info.append(info)
                filtered_label.append(label)

        # 更新为过滤后的数据
        self.all_patches_info = filtered_info
        self.all_patches_label = np.array(filtered_label)
        
        print('============= Instance Labels =============')
        statistics_dataset(self.all_patches_label)

    def __getitem__(self, index):
        if self.return_bag:
            patch_infos = self.all_slide_info[index]
            patch_features = np.stack([
                self.feature_arrays[p['feature_path']][p['feature_index']]
                for p in patch_infos
            ])
            label = self.all_slide_label[index]
            return torch.tensor(patch_features), torch.tensor(label, dtype=torch.long), index
        else:
            patch_info = self.all_patches_info[index]
            feature = self.feature_arrays[patch_info['feature_path']][patch_info['feature_index']]
            label = self.all_patches_label[index]
            return torch.tensor(feature), torch.tensor(label, dtype=torch.long), index

    def __len__(self):
        if self.return_bag:
            return len(self.all_slide_info)
        else:
            return len(self.all_patches_info)
        

class TumorRegion_PathologyType_Feat(torch.utils.data.Dataset):
    def __init__(self, ds_data_all, return_bag=False):
        self.all_slide_patch_info = ds_data_all[0]
        self.all_slide_patch_label = ds_data_all[1]
        self.all_slide_name = ds_data_all[2]
        self.all_slide_patho_label = ds_data_all[4]
        self.all_slide_nuclearLevel_label = ds_data_all[5]
        self.all_slide_zhuyuanhao = ds_data_all[12]
        self.return_bag = return_bag

        new_slide_patch_info = []
        new_slide_patch_label = []
        new_slide_patho_label = []
        new_slide_nuclearLevel_label = []
        new_slide_patch_patho_label = []
        new_slide_patch_nuclearLevel_label = []
        new_slide_name = []
        new_slide_patch_corresponding_slideName = []

        for i in tqdm(range(len(self.all_slide_patch_info)), desc="Filtering tumor patches"):
            tumor_indices = np.where(np.isin(self.all_slide_patch_label[i], [1, 2, 3, 4]))[0]
            slide_name = self.all_slide_name[i]
            patho_label = self.all_slide_patho_label[i]
            reason = None  # 过滤原因

            if len(tumor_indices) == 0:
                reason = "No tumor patch"
            elif patho_label == -1:
                reason = "Invalid pathology label (-1)"

            if reason is not None:
                # filtered_slides_info.append({
                #     "slide_name": slide_name,
                #     "pathology_label": patho_label,
                #     "num_patches": len(self.all_slide_patch_info[i]),
                #     "reason": reason
                # })
                continue  # 直接跳过，进入下一个 slide
            tumor_patch_info = [self.all_slide_patch_info[i][idx] for idx in tumor_indices]
            new_slide_patch_info.append(tumor_patch_info)
            new_slide_patch_label.append(self.all_slide_patch_label[i][tumor_indices])
            new_slide_patho_label.append(self.all_slide_patho_label[i])
            new_slide_nuclearLevel_label.append(self.all_slide_nuclearLevel_label[i])
            new_slide_patch_patho_label.append(self.all_slide_patho_label[i] * np.ones_like(tumor_indices))
            new_slide_patch_nuclearLevel_label.append(self.all_slide_nuclearLevel_label[i] * np.ones_like(tumor_indices))
            new_slide_name.append(self.all_slide_name[i])
            new_slide_patch_corresponding_slideName.append(np.repeat(self.all_slide_name[i], len(tumor_indices)))
        # if filtered_slides_info:
        #     filtered_df = pd.DataFrame(filtered_slides_info)
        #     filtered_df.to_excel("filtered_slides.xlsx", index=False)
        #     print(f"已保存过滤掉的 {len(filtered_slides_info)} 个 slide 信息到 'filtered_slides.xlsx'")
        # else:
        #     print("没有 slide 被过滤掉。")
        self.all_slide_patch_info = new_slide_patch_info
        self.all_slide_patch_label = new_slide_patch_label
        self.all_slide_patho_label = new_slide_patho_label
        self.all_slide_nuclear_label = new_slide_nuclearLevel_label
        self.all_slide_patch_patho_label = new_slide_patch_patho_label
        self.all_slide_patch_nuclear_label = new_slide_patch_nuclearLevel_label
        self.all_slide_name = new_slide_name
        self.all_slide_patch_corresponding_slideName = new_slide_patch_corresponding_slideName

        self.feature_arrays = {}
        for slide in self.all_slide_patch_info:
            for patch in slide:
                path = patch['feature_path']
                if path not in self.feature_arrays:
                    self.feature_arrays[path] = np.load(path, mmap_mode='r')

        # 打印统计
        print('============= Bag Pathology Labels =============')
        statistics_dataset(np.array(self.all_slide_patho_label))

        if not return_bag:
            self.all_patch_info = sum(self.all_slide_patch_info, [])
            self.all_patch_patho_label = np.concatenate(self.all_slide_patch_patho_label)
            self.all_patch_nuclear_label = np.concatenate(self.all_slide_patch_nuclear_label)
            self.all_patch_slideName = np.concatenate(self.all_slide_patch_corresponding_slideName)
            print('============= Instance Pathology Labels =============')
            statistics_dataset(self.all_patch_patho_label)

    def __len__(self):
        return len(self.all_slide_patch_info) if self.return_bag else len(self.all_patch_info)

    def __getitem__(self, index):
        if self.return_bag:
            patch_infos = self.all_slide_patch_info[index]
            patch_features = np.stack([
                self.feature_arrays[p['feature_path']][p['feature_index']] for p in patch_infos
            ])
            label = self.all_slide_patho_label[index]
            return torch.tensor(patch_features), torch.tensor(label, dtype=torch.long), index
        else:
            patch_info = self.all_patch_info[index]
            feature = self.feature_arrays[patch_info['feature_path']][patch_info['feature_index']]
            label = self.all_patch_patho_label[index]
            return torch.tensor(feature), torch.tensor(label, dtype=torch.long), index

    
class TumorRegion_NuclearLevel_Feat(torch.utils.data.Dataset):
    def __init__(self, ds_data_all, return_bag=False, patho_filter=[-1,0,1,2,3,4,5,6,7,8,9], numClass=4, allRegion=False):
        self.all_slide_patch_info = ds_data_all[0]
        self.all_slide_patch_label = ds_data_all[1]
        self.all_slide_name = ds_data_all[2]
        self.all_slide_patho_label = ds_data_all[4]
        self.all_slide_nuclearLevel_label = ds_data_all[7]
        self.all_slide_zhuyuanhao = ds_data_all[12]
        self.allRegion = allRegion

        for i in range(len(self.all_slide_nuclearLevel_label)):
            if self.all_slide_nuclearLevel_label[i] != -1:
                self.all_slide_nuclearLevel_label[i] = self.all_slide_nuclearLevel_label[i] - 1

        self.return_bag = return_bag
        self.patho_filter = patho_filter
        self.numClass = numClass

        # 1. filter out non-tumor patches
        num_slide = len(self.all_slide_patch_info)
        new_slide_patch_info = []
        new_slide_patch_label = []
        new_slide_patho_label = []
        new_slide_nuclearLevel_label = []
        new_slide_patch_patho_label = []
        new_slide_patch_nuclearLevel_label = []
        new_slide_name = []
        new_slide_patch_corresponding_slideName = []
        new_slide_corresponding_zhuyuanhao = []
        for slide_idx_i in tqdm(range(num_slide), desc='filtering non-tumor region'):
            tumor_patch_idx = np.where(np.isin(self.all_slide_patch_label[slide_idx_i], [1, 2, 3, 4]))[0]
            if (len(tumor_patch_idx) != 0) and (self.all_slide_nuclearLevel_label[slide_idx_i] != -1) and (self.all_slide_patho_label[slide_idx_i] in self.patho_filter):
                tumor_patch_info = [self.all_slide_patch_info[slide_idx_i][idx] for idx in tumor_patch_idx]
                new_slide_patch_info.append(tumor_patch_info)
                new_slide_patch_label.append(self.all_slide_patch_label[slide_idx_i][tumor_patch_idx])
                new_slide_patho_label.append(self.all_slide_patho_label[slide_idx_i])
                new_slide_nuclearLevel_label.append(self.all_slide_nuclearLevel_label[slide_idx_i])
                new_slide_patch_patho_label.append(self.all_slide_patho_label[slide_idx_i] * np.ones_like(tumor_patch_idx))
                new_slide_patch_nuclearLevel_label.append(self.all_slide_nuclearLevel_label[slide_idx_i] * np.ones_like(tumor_patch_idx))
                new_slide_name.append(self.all_slide_name[slide_idx_i])
                new_slide_patch_corresponding_slideName.append(np.repeat(self.all_slide_name[slide_idx_i], len(tumor_patch_idx)))
                new_slide_corresponding_zhuyuanhao.append(self.all_slide_zhuyuanhao[slide_idx_i])
        self.all_slide_patch_info = new_slide_patch_info
        self.all_slide_patch_label = new_slide_patch_label
        self.all_slide_patho_label = new_slide_patho_label
        self.all_slide_nuclearLevel_label = new_slide_nuclearLevel_label
        self.all_slide_patch_patho_label = new_slide_patch_patho_label
        self.all_slide_patch_nuclearLevel_label = new_slide_patch_nuclearLevel_label
        self.all_slide_name = new_slide_name
        self.all_slide_patch_corresponding_slideName = new_slide_patch_corresponding_slideName
        self.all_slide_corresponding_zhuyuanhao = new_slide_corresponding_zhuyuanhao 
        self.all_slide_nuclearLevel_label = np.array(self.all_slide_nuclearLevel_label)
        num_slide = len(self.all_slide_patch_info)
        kept_slide_names = set(new_slide_name)  # 前面你保留下来的 slide 名字
        all_slide_names = set(self.all_slide_name)

        if self.numClass == 3:
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==0] = 0
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==1] = 0
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==2] = 1
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==3] = 2
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==0] = 0
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==1] = 0
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==2] = 1
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==3] = 2
        elif self.numClass == 2:
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==0] = 0
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==1] = 0
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==2] = 1
            self.all_slide_nuclearLevel_label[self.all_slide_nuclearLevel_label==3] = 1
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==0] = 0
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==1] = 0
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==2] = 1
            # self.all_slide_patch_nuclearLevel_label[self.all_slide_patch_nuclearLevel_label==3] = 1
        self.all_slide_nuclearLevel_label = self.all_slide_nuclearLevel_label.tolist()

        patient2slide_idx = defaultdict(list)
        for idx, zhuyuanhao in enumerate(self.all_slide_corresponding_zhuyuanhao):
            patient2slide_idx[zhuyuanhao].append(idx)

        self.patient_bags_info = []
        self.patient_patho_labels = []
        self.patient_slide_names = []
        self.patient_nuclearLevel_label = []
        self.patient_zhuyuanhao = []
        patient_stats = []

        for zhuyuanhao, slide_indices in patient2slide_idx.items():
            if zhuyuanhao is None or str(zhuyuanhao).strip() == "":
                continue 
            # 合并所有slide的patch
            patches = []
            for idx in slide_indices:
                patches.extend(self.all_slide_patch_info[idx])
            self.patient_bags_info.append(patches)
            self.patient_patho_labels.append(self.all_slide_patho_label[slide_indices[0]])
            self.patient_nuclearLevel_label.append(self.all_slide_nuclearLevel_label[slide_indices[0]])
            self.patient_slide_names.append([self.all_slide_name[idx] for idx in slide_indices])
            self.patient_zhuyuanhao.append(zhuyuanhao)

        self.feature_arrays = {}
        for slide in self.patient_bags_info:
            for patch in slide:
                path = patch['feature_path']
                if path not in self.feature_arrays:
                    self.feature_arrays[path] = np.load(path, mmap_mode='r')

        print('============= Bag Nuclear Labels =============')
        statistics_dataset(np.array(self.patient_nuclearLevel_label))

        if not return_bag:
            self.all_patches_info = sum(self.all_slide_patch_info, [])
            self.all_slide_patch_patho_label = np.concatenate(self.all_slide_patch_patho_label)
            self.all_slide_patch_nuclearLevel_label = np.concatenate(self.all_slide_patch_nuclearLevel_label)
            # self.all_slide_name = np.array(self.all_slide_name)
            self.all_patch_slideName = np.concatenate(self.all_slide_patch_corresponding_slideName)
            # print('============= Instance Pathology Labels =============')
            # statistics_dataset(self.all_slide_patch_patho_label)

    def __getitem__(self, index):
        if self.return_bag:
            bag_infos = self.patient_bags_info[index]
            bag_features = np.stack([
                self.feature_arrays[p['feature_path']][p['feature_index']] for p in bag_infos
            ])
            label = self.patient_nuclearLevel_label[index]
            return torch.tensor(bag_features), torch.tensor(label, dtype=torch.long), index
        else:
            patch_info = self.all_patch_info[index]
            feature = self.feature_arrays[patch_info['feature_path']][patch_info['feature_index']]
            label = self.all_slide_patch_nuclearLevel_label[index]
            return torch.tensor(feature), torch.tensor(label, dtype=torch.long), index

    def __len__(self):
        return len(self.patient_bags_info) if self.return_bag else len(self.all_slide_patches_info)
    
class TumorRegion_Prognosis_Feat_yuHouFenZu(torch.utils.data.Dataset):
    def __init__(self, ds_data_all, return_bag=True, filter_yuHouFenZu_score2=False):
        self.all_slide_patch_info = ds_data_all[0]
        self.all_slide_patch_label = ds_data_all[1]
        self.all_slide_name = ds_data_all[2]
        self.all_slide_patho_label = ds_data_all[4]
        self.all_slide_prognosis_label = ds_data_all[6]
        self.all_slide_isup_label = ds_data_all[7]
        self.all_slide_size_label = ds_data_all[8]
        self.all_slide_tnm_label = ds_data_all[9]
        self.all_slide_huaisi_label = ds_data_all[10]
        self.all_slide_yuHouFenZu_label = ds_data_all[11]
        self.all_slide_zhuyuanhao = ds_data_all[12]

        if (self.all_slide_yuHouFenZu_label is None) or (len(self.all_slide_yuHouFenZu_label) == 0):
            self.all_slide_yuHouFenZu_label = [-1 for i in range(len(self.all_slide_feat))]

        # 1. filter out non-tumor patches and slides
        num_slide = len(self.all_slide_patch_info)
        new_slide_patch_info = []
        new_slide_patch_label = []
        new_slide_patho_label = []
        new_slide_prognosis_label = []
        new_slide_isup_label = []
        new_slide_size_label = []
        new_slide_tnm_label = []
        new_slide_huaisi_label = []
        new_slide_name = []
        new_slide_patch_corresponding_slideName = []
        new_slide_yuHouFenZu_label = []
        new_slide_patch_patho_label = []
        new_slide_corresponding_zhuyuanhao = []
        filtered_slides_info = []
        for slide_idx_i in tqdm(range(num_slide), desc='1. filtering non-tumor region in each slide; '
                                                       '2. filtering slides without tumor region or without yuHouFenZu slideLabel'):
            patch_label = np.array(self.all_slide_patch_label[slide_idx_i], dtype=np.int64)
            slide_name = self.all_slide_name[slide_idx_i]
            patho_label = self.all_slide_patho_label[slide_idx_i]
            prognosis_label = self.all_slide_prognosis_label[slide_idx_i]
            reason = None  # 过滤原因
            tumor_patch_idx = np.where(patch_label >= 1)[0]
            if filter_yuHouFenZu_score2:
                if self.all_slide_yuHouFenZu_label[slide_idx_i] == 2:
                    reason = "yuHouFenZu label = 2"
                elif self.all_slide_yuHouFenZu_label[slide_idx_i] == -1:
                    reason = "Invalid yuHouFenZu label(-1)"
            if len(tumor_patch_idx) == 0:
                reason = "No tumor patch"
            if np.all(np.array(prognosis_label[-4:]) == -1):
                reason = "no prognosis info"
            if reason is not None:
                filtered_slides_info.append({
                    "slide_name": slide_name,
                    "pathology_label": patho_label,
                    "num_patches": len(self.all_slide_patch_info[slide_idx_i]),
                    "reason": reason
                })
                continue  # 直接跳过，进入下一个 slide
            tumor_patch_info = [self.all_slide_patch_info[slide_idx_i][idx] for idx in tumor_patch_idx]
            new_slide_patch_info.append(tumor_patch_info)
            new_slide_patch_label.append(self.all_slide_patch_label[slide_idx_i][tumor_patch_idx])
            new_slide_name.append(self.all_slide_name[slide_idx_i])
            new_slide_patho_label.append(self.all_slide_patho_label[slide_idx_i])
            new_slide_prognosis_label.append(self.all_slide_prognosis_label[slide_idx_i])
            new_slide_isup_label.append(self.all_slide_isup_label[slide_idx_i])
            new_slide_size_label.append(self.all_slide_size_label[slide_idx_i])
            new_slide_tnm_label.append(self.all_slide_tnm_label[slide_idx_i])
            new_slide_huaisi_label.append(self.all_slide_huaisi_label[slide_idx_i])
            new_slide_yuHouFenZu_label.append(self.all_slide_yuHouFenZu_label[slide_idx_i])
            new_slide_patch_patho_label.append(self.all_slide_patho_label[slide_idx_i] * np.ones_like(self.all_slide_patch_label[slide_idx_i][tumor_patch_idx]))
            new_slide_patch_corresponding_slideName.append(np.repeat(self.all_slide_name[slide_idx_i], len(tumor_patch_idx)))
            new_slide_corresponding_zhuyuanhao.append(self.all_slide_zhuyuanhao[slide_idx_i])
        self.filtered_slides_info = filtered_slides_info
        self.all_slide_patch_info = new_slide_patch_info
        self.all_slide_patch_label = new_slide_patch_label
        self.all_slide_name = new_slide_name
        self.all_slide_patho_label = new_slide_patho_label
        self.all_slide_prognosis_label = new_slide_prognosis_label
        self.all_slide_isup_label = new_slide_isup_label
        self.all_slide_size_label = new_slide_size_label
        self.all_slide_tnm_label = new_slide_tnm_label
        self.all_slide_huaisi_label = new_slide_huaisi_label
        self.all_slide_yuHouFenZu_label = new_slide_yuHouFenZu_label
        self.all_slide_patch_patho_label = new_slide_patch_patho_label
        self.all_slide_patch_corresponding_slideName = new_slide_patch_corresponding_slideName
        self.all_slide_corresponding_zhuyuanhao = new_slide_corresponding_zhuyuanhao
        
        patient2slide_idx = defaultdict(list)
        for idx, zhuyuanhao in enumerate(self.all_slide_corresponding_zhuyuanhao):
            patient2slide_idx[zhuyuanhao].append(idx)

        self.patient_bags_info = []
        self.patient_prognosis_labels = []
        self.patient_patho_labels = []
        self.patient_isup_labels = []
        self.patient_size_labels = []
        self.patient_tnm_labels = []
        self.patient_huaisi_labels = []
        self.patient_yuHouFenZu_labels = []
        self.patient_slide_names = []
        self.patient_zhuyuanhao = []
        

        for zhuyuanhao, slide_indices in patient2slide_idx.items():
            if zhuyuanhao is None or str(zhuyuanhao).strip() == "":
                continue 
            patches = []
            for idx in slide_indices:
                patches.extend(self.all_slide_patch_info[idx])
            self.patient_bags_info.append(patches)
            self.patient_prognosis_labels.append(self.all_slide_prognosis_label[slide_indices[0]])
            self.patient_patho_labels.append(self.all_slide_patho_label[slide_indices[0]])
            self.patient_isup_labels.append(self.all_slide_isup_label[slide_indices[0]])
            self.patient_size_labels.append(self.all_slide_size_label[slide_indices[0]])
            self.patient_tnm_labels.append(self.all_slide_tnm_label[slide_indices[0]])
            self.patient_huaisi_labels.append(self.all_slide_huaisi_label[slide_indices[0]])
            self.patient_yuHouFenZu_labels.append(self.all_slide_yuHouFenZu_label[slide_indices[0]])
            self.patient_slide_names.append([self.all_slide_name[idx] for idx in slide_indices])
            self.patient_zhuyuanhao.append(zhuyuanhao)

        self.feature_arrays = {}
        for bag in self.patient_bags_info:
            for patch in bag:
                path = patch['feature_path']
                if path not in self.feature_arrays:
                    self.feature_arrays[path] = np.load(path, mmap_mode='r')

        print('============= Bag Pathology Labels =============')
        statistics_dataset(np.array(self.all_slide_patho_label))
    def __getitem__(self, index):
        bag_infos = self.patient_bags_info[index]
        bag_features = np.stack([
            self.feature_arrays[p['feature_path']][p['feature_index']] for p in bag_infos
        ])
        prognosis_labels = self.patient_prognosis_labels[index]
        patho_labels = self.patient_patho_labels[index]
        isup_labels = self.patient_isup_labels[index]
        size_labels = self.patient_size_labels[index]
        tnm_labels = self.patient_tnm_labels[index]
        yuHouFenZu_labels = self.patient_yuHouFenZu_labels[index]
        if yuHouFenZu_labels == 1:
            yuHouFenZu_labels = 0
        elif yuHouFenZu_labels == 3:
            yuHouFenZu_labels = 1
        elif yuHouFenZu_labels == -1:
            pass
        elif yuHouFenZu_labels == 2:
            pass
        else:
            raise
        return (
            torch.tensor(bag_features),
            torch.tensor(prognosis_labels, dtype=torch.float32),
            torch.tensor(patho_labels, dtype=torch.long),
            torch.tensor(isup_labels, dtype=torch.long),
            torch.tensor(size_labels, dtype=torch.long),
            torch.tensor(tnm_labels, dtype=torch.long),
            torch.tensor(yuHouFenZu_labels, dtype=torch.long),
            index
        )
    def __len__(self):
        return len(self.patient_bags_info)

