from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot
from mmdet.apis import inference_detector_huge_image

import mmcv

import glob
import os
import numpy as np
import BboxToolkit as bt
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('save_dir', help='Directory to save painted images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        'split', help='split configs in BboxToolkit/tools/split_configs')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--method', type=str, default='pyplot', choices=['cv2', 'pyplot'],
                        help='Method used to draw bboxes')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    img_list = glob.glob(os.path.join(args.img_dir, '*.png'))
    
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    # nms_cfg = dict(type='BT_nms', iou_thr=0.5)
    nms_cfg = dict(type='obb_nms', iou_thr=0.1, device_id=args.device.split(':')[-1])
    for img_name in img_list:
        result = inference_detector_huge_image(
            model, img_name, args.split, nms_cfg)
        
        if args.method == 'pyplot':
            # mmdet/models/detectors/obb/obb_base.py
            img = model.show_result(img_name, result, score_thr=args.score_thr, show=False,
                                    thickness=6, font_size=120, colors='red')
        elif args.method == 'cv2':
            img = mmcv.imread(img_name)
            # show the results
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            # draw segmentation masks
            if segm_result is not None and len(labels) > 0:  # non empty
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > args.score_thr)[0]
                np.random.seed(42)
                color_masks = [
                    np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    for _ in range(max(labels) + 1)
                ]
                for i in inds:
                    i = int(i)
                    color_mask = color_masks[labels[i]]
                    mask = segms[i]
                    img[mask] = img[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            bboxes, scores = bboxes[:, :-1], bboxes[:, -1]
            bboxes = bboxes[scores > args.score_thr]
            labels = labels[scores > args.score_thr]
            scores = scores[scores > args.score_thr]
            bboxes = bt.obb2poly(bboxes)
            for bbox, label, score in zip(bboxes, labels, scores):
                img = cv2.drawContours(img, [np.int64(np.array_split(bbox, 4, axis=0))], -1, (255, 0, 255), thickness=6)
                text = f'{model.CLASSES[label]}|{score:.2f}'
                img = cv2.putText(img, text, (int(min(bbox[0::2])), int(min(bbox[1::2]))), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), thickness=6)
        
        mmcv.imwrite(img, os.path.join(args.save_dir, os.path.basename(img_name)))


if __name__ == '__main__':
    main()
