import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_metric_learning import losses

import math
import skimage.morphology as morph
from skimage import measure
import numpy as np
import cv2
import unet3
import time
import random

# 指定目录路径
directory = "/home/xuxinan/mmCode2/unetans4"

# 定义目标尺寸
target_size = (522, 775)  # (height, width)

# 初始化列表来保存图像数组
def find_two_arrays(directory,target_size):
    gt_arrays = []
    pred_arrays = []

    # 遍历目录下的所有文件，首先收集所有的'gt.png'和'pred.png'文件路径
    gt_files = sorted([f for f in os.listdir(directory) if f.endswith("gt.png")])
    pred_files = sorted([f for f in os.listdir(directory) if f.endswith("pred.png")])

    # 检查两个列表长度是否一致，确保每个gt图像都有对应的pred图像
    if len(gt_files) != len(pred_files):
        print("警告：gt图像和pred图像的数量不匹配！")
    else:
        for gt_filename, pred_filename in zip(gt_files, pred_files):
            # 提取文件名（不含扩展名）以验证对应关系
            gt_prefix = os.path.splitext(gt_filename)[0].replace("gt", "")
            pred_prefix = os.path.splitext(pred_filename)[0].replace("pred", "")
            
            # 确保gt和pred文件名前缀相同（除了后缀）
            if gt_prefix != pred_prefix:
                print(f"警告：未找到{gt_filename}的对应pred图像")
                continue
            
            # 构造完整的文件路径
            gt_path = os.path.join(directory, gt_filename)
            pred_path = os.path.join(directory, pred_filename)
            
            # 使用cv2读取并调整大小
            img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            img_pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
            
            if img_gt is not None and img_pred is not None:
                # 调整图像大小
                img_gt_resized = cv2.resize(img_gt, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                img_pred_resized = cv2.resize(img_pred, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
                
                # 将调整大小后的图像转换为NumPy数组（实际上是已经是）
                img_gt_array = np.array(img_gt_resized)
                img_pred_array = np.array(img_pred_resized)
                
                img_gt_array = 1 * (img_gt_array > 0)
                img_pred_array = 1 * (img_pred_array > 0)
                
                # 将数组添加到列表中
                gt_arrays.append(img_gt_array)
                pred_arrays.append(img_pred_array)
                
                # 这里可以对img_gt_array和img_pred_array进行操作
                # 例如，打印调整后的数组形状
                # print(f"{gt_filename} 的调整后数组形状是: {img_gt_array.shape}")
                # print(f"{pred_filename} 的调整后数组形状是: {img_pred_array.shape}")
            else:
                print(f"无法读取图像文件 {gt_path} 或 {pred_path}")
    return gt_arrays,pred_arrays


# print(len(gt_arrays))

# def count_values(array):
#     greater_than_zero_count = 0
#     exactly_255_count = 0
    
#     for row in array:
#         for item in row:
#             if item > 0:
#                 greater_than_zero_count += 1
#             if item == 255:
#                 exactly_255_count += 1
                
#     return greater_than_zero_count, exactly_255_count

# # 获取第一个二维数组
# first_2d_array = pred_arrays[3]

# # 调用函数并打印结果
# greater_than_zero, exactly_255 = count_values(first_2d_array)
# print(f"在第一个二维数组中，大于0的元素个数为: {greater_than_zero}")
# print(f"值为255的元素个数为: {exactly_255}")

#计算gt和pred图片中局部所有腺体分别对应的面积 
def segment_level_loss(gt, pred,rate=60 ,op='xor', out_size=(160, 160)):

    gt = cv2.resize(gt, out_size[::-1], interpolation=cv2.INTER_NEAREST)
    pred = cv2.resize(pred, out_size[::-1], interpolation=cv2.INTER_NEAREST)
    count_over_conn = 0
    count_under_conn = 0
    if op == 'none':
        return np.zeros(gt.shape, dtype=np.uint8)
    # pred[gt==255] = 0
    pred = morph.remove_small_objects(pred == 1, connectivity=1)
    gt = morph.remove_small_objects(gt == 1, connectivity=1)

    pred_labeled, pred_num = measure.label(pred, return_num=True, connectivity=1)
    gt_labeled, gt_num = measure.label(gt, return_num=True, connectivity=1)

    results = []
    #pred
    res = np.zeros(gt.shape, dtype=np.uint8)
    ans = np.zeros(gt.shape, dtype=np.uint8)
    process_list=[]
    
    # 记录过欠连接
    pred_process_list = []
    pred_betti_over_num = []
    for i in range(0, pred_num):
        i += 1
        pred_labeled_i = pred_labeled == i
        mask = (pred_labeled_i) & (gt != 0)
        if len(gt_labeled[mask]) == 0:
            res[pred_labeled_i] = 1
            ans[pred_labeled_i] = 1
            count_over_conn += 1
            continue
        if gt_labeled[mask].min() != gt_labeled[mask].max():
            res[pred_labeled_i] = 1
            gt_unique_values=np.unique(gt_labeled[mask])
            pred_betti_over_num.append(len(gt_unique_values)-1)
            new_pic = np.zeros(gt.shape, dtype=np.uint8)            
            for value in gt_unique_values:
                new_pic[gt_labeled == value] = 1
            w = unet3.weight_add_np(new_pic, rate)
            w = w / w.max() 
            process_list.append(w)
            pred_process_list.append(w)
            count_over_conn += 1

        else:
            # corresponding gt gland area is less than 50%
            if mask.sum() / pred_labeled_i.sum() < 0.5:
                #res[pred_labeled == i] = 1
                res[pred_labeled_i] = 1
                ans[pred_labeled_i] = 1
                count_over_conn += 1

    results.append(res)

    res = np.zeros(gt.shape, dtype=np.uint8)
    gt_process_list = []
    gt_betti_over_num = []
    for i in range(0, gt_num):
        i += 1
        gt_labeled_i = gt_labeled == i
        #mask = (gt_labeled == i) & (pred != 0)
        mask = gt_labeled_i & (pred != 0)

        if len(pred_labeled[mask]) == 0:
            res[gt_labeled_i] = 1
            ans[gt_labeled_i] = 1
            count_under_conn += 1
            continue

        if pred_labeled[mask].min() != pred_labeled[mask].max():
            res[gt_labeled_i] = 1
            count_under_conn += 1
            # 处理pred
            pred_unique_values=np.unique(pred_labeled[mask])
            gt_betti_over_num.append(len(pred_unique_values) - 1)
            new_pic = np.zeros(pred.shape, dtype=np.uint8)
            
            for value in pred_unique_values:
                new_pic[pred_labeled == value] = 1
                
            w = unet3.weight_add_np(new_pic, rate)
            w = w / w.max() 
            process_list.append(w)
            gt_process_list.append(w)

        else:
            if mask.sum() / gt_labeled_i.sum() < 0.5:
                res[gt_labeled_i] = 1
                ans[gt_labeled_i] = 1
                count_under_conn += 1

    results.append(res)


    res = cv2.bitwise_or(results[0], results[1])

    if op == 'or':
        return res

    elif op == 'xor':

        #cc = res.copy()
        gt_res = np.zeros(gt.shape, dtype=np.uint8)
        for i in range(0, gt_num):
            i += 1
            if res[gt_labeled == i].max() != 0:
                gt_res[gt_labeled == i] = 1

        pred_res = np.zeros(gt.shape, dtype=np.uint8)
        for i in range(0, pred_num):
            i += 1
            if res[pred_labeled == i].max() != 0:
                pred_res[pred_labeled == i] = 1

        res = cv2.bitwise_xor(pred_res, gt_res)
        for index, pic in enumerate(process_list):
            pic_xor = pic * res
            ans[pic_xor >= 0.3]=1
        # 记录最大的
        # max_1_ans = np.zeros(gt.shape, dtype=np.uint8)
        # for index, pic in enumerate(pred_process_list):
        #     pic = unet3.return_n_betti(pic,pred_betti_over_num[index])
        #     pic_xor = pic * res
        #     max_1_ans[pic_xor!=0]=1
        # for index, pic in enumerate(gt_process_list):
        #     pic = unet3.return_n_betti(pic,gt_betti_over_num[index])
        #     pic_xor = pic * res
        #     max_1_ans[pic_xor!=0]=1
        return np.sum(ans),np.sum(res)
    else:
        raise ValueError('operation not suportted')

#计算gt和pred图片中局部1个腺体分别对应的面积 
def compute_two_area(gt,pred,rate):
    gt_arr = cv2.bitwise_or(gt, pred)
    narrow_arr = unet3.weight_add_np(pred, rate)
    narrow_arr = np.where(narrow_arr > 0.3, 1, 0)
    return np.sum(narrow_arr), np.sum(gt_arr)

def adjust_rate(current_r, area_ratio,threshold,left_area_ratio):
    diff = math.floor(abs(threshold - area_ratio) * 10) 
    
    if diff == 1:
        diff = 2
    else :
        diff = diff **2    
    
    if area_ratio > left_area_ratio:
        return  current_r + diff
    else :
        return  current_r - diff
        
    
    
    
def learn_rate(gt, pred,  max_iterations = 10, r_min = 20, r_max = 60,
               left_area_ratio = 0.5, right_area_ratio = 0.5,threshold=0.6):
    current_r = random.randrange(r_min, r_max + 1, 5)
    
    for iteration in range(max_iterations):

        predS, gtS = compute_two_area(gt, pred,current_r)
        if gtS == 0 or predS == 0:
            return 0.3
        area_ratio = predS / gtS
        if left_area_ratio <= area_ratio <= right_area_ratio:
            break  
        current_r = adjust_rate(current_r, area_ratio,threshold,left_area_ratio)
          
    
    return current_r

if __name__ == "__main__":
    gt_arrays = []
    pred_arrays = []
    gt_arrays,pred_arrays = find_two_arrays(directory,target_size)
    print(pred_arrays[1].shape)
    for i in range(len(gt_arrays)):
        gt_element = gt_arrays[i]
        pred_element = pred_arrays[i]
        print(learn_rate(gt_element,pred_element))
        # rate = 29
        # s1,s2 = segment_level_loss(gt_element, pred_element,rate)
        # print(f"rate为{rate}")
        # print(s1,s2)
        # print(s1/s2)
        exit(0)