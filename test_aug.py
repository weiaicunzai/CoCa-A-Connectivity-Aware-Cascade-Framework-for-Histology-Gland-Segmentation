
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2





def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def slide_inference(img, ori_shape, model, crop_size, stride, rescale, num_classes):
    """Inference by sliding-window with overlap.
    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """

    # print(stride)
    #h_stride, w_stride = self.test_cfg.stride
    h_stride, w_stride = stride
    #h_crop, w_crop = self.test_cfg.crop_size
    h_crop, w_crop = crop_size

    img = img.unsqueeze(0)
    batch_size, _, h_img, w_img = img.size()
    # out_channels = self.out_channels
    out_channels = num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    # img = img.cuda()
    preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            # crop_seg_logit = self.encode_decode(crop_img, img_meta)
            crop_seg_logit = model(crop_img.cuda()).cpu()
            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    #if torch.onnx.is_in_onnx_export():
    #    # cast count_mat to constant while exporting to ONNX
    #    count_mat = torch.from_numpy(
    #        count_mat.cpu().detach().numpy()).to(device=img.device)

    preds = preds / count_mat
    # if rescale:
        # remove padding area
    # resize_shape = img_meta[0]['img_shape'][:2]
    # resize_shape = img_meta[0]['img_shape'][:2]
        # preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
    #print(preds.shape)

    #################################
    #gland_preds = preds[:, :2, :, :]
    #cnt_preds = preds[:, 2:, :, :]
    #gland_preds = resize(gland_preds,
    #        size=ori_shape,
    #        mode='bilinear',
    #        align_corners=True,
    #        warning=False)
    #cnt_preds = resize(cnt_preds,
    #        size=ori_shape,
    #        mode='bilinear',
    #        align_corners=True,
    #        warning=False)
    #preds = torch.cat([gland_preds, cnt_preds], dim=1)
    #################################


    if rescale:
        preds = resize(preds,
            size=ori_shape,
            mode='bilinear',
            align_corners=True,
            warning=False)

    return preds


# def whole_inference(img, ori_shape, )

# def whole_inference(self, img, img_meta, rescale):
def whole_inference(img, ori_shape, model, rescale):
        """Inference with full image."""

        # seg_logit = self.encode_decode(img, img_meta)
        img = img.unsqueeze(0).cuda()
        seg_logit = model(img)
        # if rescale:
            # support dynamic shape for onnx
            # if torch.onnx.is_in_onnx_export():
                # size = img.shape[2:]
            # else:
                # remove padding area
                # resize_shape = img_meta[0]['img_shape'][:2]
                # seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                # size = img_meta[0]['ori_shape'][:2]

        # print(seg_logit.shape, ori_shape)
        # if rescale:
        #     seg_logit = resize(
        #         seg_logit,
        #         # size=size,
        #         size=ori_shape,
        #         mode='bilinear',
        #         align_corners=True,
        #         warning=False)

        return seg_logit


def inference(img, ori_shape, flip_direction, mode, model, num_classes, crop_size=None, stride=None, rescale=True):
    """Inference a single image with slide/whole style.
    Args:
        img (Tensor): The input image of shape (N, 3, H, W).
        img_meta (dict): Image info dict where each dict has: 'img_shape',
            'scale_factor', 'flip', and may also contain
            'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            For details on the values of these keys see
            `mmseg/datasets/pipelines/formatting.py:Collect`.
        rescale (bool): Whether rescale back to original shape.
    Returns:
        Tensor: The output segmentation map.
    """

    # assert self.test_cfg.mode in ['slide', 'whole']
    assert mode in ['slide', 'whole']
    # ori_shape = img_meta[0]['ori_shape']
    # assert all(_['ori_shape'] == ori_shape for _ in img_meta)
    # if self.test_cfg.mode == 'slide':
    # print(img.shape)
    if mode == 'slide':
        seg_logit = slide_inference(
            img=img,
            ori_shape=ori_shape,
            # flip=flip,
            crop_size=crop_size,
            stride=stride,
            model=model,
            rescale=rescale,
            num_classes=num_classes)
    else:
        # seg_logit = self.whole_inference(img, img_meta, rescale)
        seg_logit = whole_inference(
            img=img,
            ori_shape=ori_shape,
            rescale=rescale,
            model=model
        )

    # print('after', img.shape, seg_logit.shape, ori_shape, rescale)

    # if self.out_channels == 1:
    # if num_classes == 1:
    #      output = F.sigmoid(seg_logit)
    # else:
    #     #gland_output = seg_logit[:, :2, :, :]
    #     #cnt_output = seg_logit[:, 2:, :, :]
    #     #gland_output = F.softmax(gland_output, dim=1)
    #     #cnt_output = F.softmax(cnt_output, dim=1)
    #     #output = torch.cat([gland_output, cnt_output], dim=1)
    #     output = F.softmax(seg_logit, dim=1)

    # if rescale:
    #     seg_logit = resize(
    #         seg_logit,
    #         # size=size,
    #         size=ori_shape,
    #         mode='bilinear',
    #         align_corners=True,
    #         warning=False)

    # print(seg_logit.shape, output.shape)
    output = seg_logit
    # if flip:
        # flip_direction = img_meta[0]['flip_direction']
        # assert flip_direction in ['horizontal', 'vertical']
    #if flip_direction == 'horizontal':
    if flip_direction == 'h':
            output = output.flip(dims=(3, ))
        # elif flip_direction == 'vertical':
    if flip_direction == 'v':
            output = output.flip(dims=(2, ))

    if flip_direction == 'vh':
            output = output.flip(dims=(3, ))
            output = output.flip(dims=(2, ))
        #output = output.flip(dims=)

    if flip_direction == 'hv':
            output = output.flip(dims=(2, ))
            output = output.flip(dims=(3, ))

    if flip_direction == 'r90':
            output = torch.rot90(output, k=1, dims=[2, 3])

    if flip_direction == 'r90h':
            output = output.flip(dims=(3, ))
            output = torch.rot90(output, k=1, dims=[2, 3])

    if flip_direction == 'r90v':
            output = output.flip(dims=(2, ))
            output = torch.rot90(output, k=1, dims=[2, 3])

    if flip_direction == 'r90hv':
            output = output.flip(dims=(2, ))
            output = output.flip(dims=(3, ))
            output = torch.rot90(output, k=1, dims=[2, 3])

    if rescale:
        # seg_logit = resize(
        output = resize(
            # seg_logit,
            output,
            # size=size,
            size=ori_shape,
            mode='bilinear',
            align_corners=True,
            warning=False)

    if num_classes == 1:
        output = F.sigmoid(output)
    else:
        #gland_output = seg_logit[:, :2, :, :]
        #cnt_output = seg_logit[:, 2:, :, :]
        #gland_output = F.softmax(gland_output, dim=1)
        #cnt_output = F.softmax(cnt_output, dim=1)
        #output = torch.cat([gland_output, cnt_output], dim=1)
        #output = F.softmax(seg_logit, dim=1)
        output = F.softmax(output, dim=1)

    # if rescale:
    #     # seg_logit = resize(
    #     output = resize(
    #         # seg_logit,
    #         output,
    #         # size=size,
    #         size=ori_shape,
    #         mode='bilinear',
    #         align_corners=True,
    #         warning=False)

    # print('img shape', img.shape, 'seg_logit', seg_logit.shape, 'output:', output.shape, 'ori_shape', ori_shape, 'rescale', rescale, 'flip_direction', flip_direction)

    return output

def aug_test(imgs, flip_direction, ori_shape, model, num_classes, mode, threshold=0.5, crop_size=None, stride=None, rescale=True):
    """aug_test:  test time augmentation with different img_ratios"""
    """aug_data: list of (img, gt_seg) pairs with img_ratios"""
    # print(stride)


    seg_logit = 0
    count = 0
    for img, flip_direction in zip(imgs, flip_direction):
        #print(img.shape)
        # if idx == 0:
            # continue
        # img, gt_seg = data
        # import cv2
        # print('aa{}.jpg'.format(direction))
        #print('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/tmp/img_{}.jpg'.format(flip_direction))
        # print(type(img))
        # s = cv2.imwrite('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/tmp/img_{}.jpg'.format(flip_direction), img)
        # print(s)
        cur_seg_logit = inference(
            img=img,
            # flip=flip,
            flip_direction=flip_direction,
            ori_shape=ori_shape,
            model=model,
            mode=mode,
            crop_size=crop_size,
            stride=stride,
            rescale=rescale,
            num_classes=num_classes)

        # if seg_logit is None:
            # seg_logit = cur_seg_logit

        #output = cur_seg_logit.argmax(dim=1).squeeze(0)
        #cv2.imwrite('/data/hdd1/by/House-Prices-Advanced-Regression-Techniques/tmp/after_{}.jpg'.format(flip_direction), output.cpu().numpy() * 255)


        seg_logit += cur_seg_logit

    seg_logit /= len(imgs)

    if num_classes == 1:
        seg_pred = (seg_logit > threshold).to(seg_logit).squeeze(1)
    else:
        seg_pred = seg_logit.argmax(dim=1)

    #gland_pred = seg_logit[:, :2, :, :].argmax(dim=1)
    #cnt_pred = seg_logit[:, 2:, :, :].argmax(dim=1)
    #gland_pred[cnt_pred==1] = 0
    #seg_pred = gland_pred


    seg_pred = seg_pred.cpu().numpy()
    seg_pred = seg_pred.squeeze()

    # import sys; sys.exit()
    # unravel batch dim
    # seg_pred = list(seg_pred)
    return seg_pred, seg_logit
