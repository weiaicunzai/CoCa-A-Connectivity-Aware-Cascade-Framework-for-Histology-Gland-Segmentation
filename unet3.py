from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.io import imsave
from skimage import measure
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) #分别对应通道 R G B
def weight_add(path,rate):
    gt = io.imread(path)
    # print(type(gt))
    gt = 1 * (gt > 0)
    print(gt.shape)
    print((gt==1).sum())
    # 【1】计算细胞和背景的像素频率
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())
    # 【2】归一化
    c_weights /= c_weights.max()
    # 【3】得到c_w字典
    c_weights.tolist()
    cw = {}
    for i in range(len(c_weights)):
        cw[i]=c_weights[i]
    weightMap_ = UnetWeightMap(gt,rate, cw)
    return weightMap_

def weight_add_np(gt,rate):
    # print(type(gt))
    gt = 1 * (gt > 0)
    # 【1】计算细胞和背景的像素频率
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())
    # 【2】归一化
    c_weights /= c_weights.max()
    # 【3】得到c_w字典
    c_weights.tolist()
    cw = {}
    for i in range(len(c_weights)):
        cw[i]=c_weights[i]
    weightMap_ = UnetWeightMap(gt,rate, cw)
    return weightMap_
def UnetWeightMap(mask,rate, wc=None, w0=10, sigma=5):
 
    mask_with_labels = label(mask)
    no_label_parts = mask_with_labels == 0
    label_ids = np.unique(mask_with_labels)[1:]
    # print(label_ids)
    if len(label_ids) > 1:
        distances = np.zeros((mask.shape[0], mask.shape[1], len(label_ids)))
        for i, label_id in enumerate(label_ids):
            # cv2.imwrite('{}.jpg'.format(label_id), (mask_with_labels != label_id) * 255)
            # 不等于label_id的是True，等于label_id的是False，计算True到False的距离
            distances[:, :, i] = distance_transform_edt(mask_with_labels != label_id)
            # cv2.imwrite('{}_distance.jpg'.format(label_id), distances[:, :, 1] / distances[:, :, 1].max() * 255)
        distances = np.sort(distances, axis=2)
        d1 = distances[:, :, 0]
        d2 = distances[:, :, 1]
        
        # d1 = d1 / d1.max()  * rate
        # d2 = d2 / d2.max() * rate
        # print(d1.mean(), d1.std(), d2.mean(), d2.std())
        # print((d1 + d2).max())
        sum = (d1 + d2) / (d1 + d2).max() * rate

        # weight_map = w0 * np.exp(-1/2 * ((d1+d2)/sigma) ** 2) * no_label_parts
        weight_map = w0 * np.exp(-1/2 * (sum / sigma) ** 2) * no_label_parts
        # weight_map = weight_map + np.ones_like(weight_map)

 
        # if wc is not None:
        #     class_weights = np.zeros_like(mask)
        #     for k, v in wc.items():
        #         class_weights[mask == k] = v
        #     weight_map = weight_map + class_weights
 
    else:
        weight_map = np.zeros_like(mask)
    return weight_map

def get_gt(path):
    gt = io.imread(path)
    gt = 1 * (gt > 0)
    return gt
    



# exit(0)

# w1 = get_gt(image_x_path)
# ans = w * w1
# ans = ans / ans.max() * 255.0
# ans = ans.astype(np.uint8)
# imsave(f'{rate}unet_xor.png', ans) 
# print("ok")



def process_and_label(ans, threshold=0.3, connectivity=1):
    """
    对 ans 数组进行阈值处理，并标记连通区域。

    :param ans: 输入的 NumPy 数组
    :param threshold: 阈值，默认为 0.3
    :param connectivity: 连通性类型，1 表示 4 连通，2 表示 8 连通
    :return: 标记后的数组和连通组件的数量
    """
    # 将大于阈值的元素设置为 1，其余的设置为 0
    binary_ans = ans.copy()
    binary_ans[ans > threshold] = 1
    binary_ans[ans <= threshold] = 0
    
    # 标记连通区域
    pred_labeled, pred_num = measure.label(binary_ans, return_num=True, connectivity=connectivity)
    
    # 计算每个连通区域的面积并排序
        # 获取每个连通区域的属性
    regions = measure.regionprops(pred_labeled)
    areas = [(region.area, region) for region in regions]
    areas.sort(key=lambda x: x[0], reverse=True)  # 按面积从大到小排序
    
    # 获取前5个最大面积的连通组件（如果没有那么多，则全部列出）
    top_areas = [area[0] for area in areas[:5]]
    
    return pred_labeled, pred_num, top_areas


# print(np.unique(w), w.max(), w.min())
# w1 = get_gt(image_path)
# print(f"w: {w}, type(w): {type(w)}")
# print(f"w1: {w1}, type(w1): {type(w1)}")
# ans = w * w1
# print(ans.max(), 111)
# print(w.max(), w.min())
# print(np.unique(w))
# np.savetxt('unet3w.txt', w, fmt='%1.3f')
# print(np.unique(w))  # 在这里之前是挺多值的
# w = w / w.max() * 255.0
# ans = ans / ans.max() * 255.0
# print(w)
# print(np.unique(w))
# np.savetxt('unet3w1.txt', w, fmt='%1.3f')
# ans = ans.astype(np.uint8)
# w = w.astype(np.uint8)
# np.savetxt('unet3.txt', ans, fmt='%1.3f')
# imsave('unet3.png', ans) 
# imsave('unet3.png', w) 


def return_n_betti(ans, n=1, threshold=0.35, connectivity=1):
    """
    根据给定的阈值和连通性，返回满足条件的前n个最大区域。
    如果最大区域的面积超过最小区域面积的5倍，则减少返回的区域数，直到满足条件。
    """
    # 将大于阈值的元素设置为 1，其余的设置为 0
    binary_ans = ans.copy()
    binary_ans[ans > threshold] = 1
    binary_ans[ans <= threshold] = 0
    
    pred_labeled, pred_num = measure.label(binary_ans, return_num=True, connectivity=connectivity)
    
    regions = measure.regionprops(pred_labeled)
    areas = [(region.area, region) for region in regions]
    areas.sort(key=lambda x: x[0], reverse=True)  # 按面积从大到小排序
    
    selected_areas = []
    for i in range(min(n, len(areas))):
        if i == 0 or (selected_areas and areas[i][0] * 3 >= selected_areas[-1][0]):
            selected_areas.append(areas[i])
        else:
            break
    
    # 准备返回的数据：创建一个新的numpy数组，仅包含所选区域，并将这些区域标记为1
    result_array = np.zeros_like(binary_ans)
    for area, region in selected_areas:
        coords = region.coords
        result_array[coords[:, 0], coords[:, 1]] = 1
    
    # 返回结果：标注的图像、总连通组件的数量以及选定的区域面积
    return result_array

if __name__ == "__main__":
    # image_path = '/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/data/Warwick QU Dataset (Released 2016_07_08)/testA_25_anno.bmp' 
    image_x_path = '/home/xuxinan/mmCode2/evalAns/4eval_xor.png'
    # image_path = '/home/xuxinan/mmCode2/evalAns/4eval_pred.png'
    image_path = '/home/xuxinan/mmCode2/3square.png'
    # for rate in range(10, 101, 5):
    #     w = weight_add(image_path,rate)
    #     w = w / w.max() * 255.0
    #     w = w.astype(np.uint8)
    #     imsave(f'unet/{rate}unet_pred.png', w) 
    #     w1 = get_gt(image_x_path)
    #     ans = w * w1
    #     ans = ans / ans.max() * 255.0
    #     ans = ans.astype(np.uint8)
    #     imsave(f'unet/{rate}unet_xor.png', ans) 
    rate=30
    w = weight_add(image_path,rate)
    print(w.max())
    w = w / w.max() * 255.0
    w = w.astype(np.uint8)
    imsave(f'3sq{rate}unet_pred.png', w) 

    arr = return_n_betti(w)
    print(arr.shape)
    # print(top)