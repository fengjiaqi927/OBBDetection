import time
import os
import numpy as np
import logging
import datetime
import mmcv
from multiprocessing import Pool, Manager
from functools import partial, reduce
import torch
import BboxToolkit as bt

from argparse import ArgumentParser
from xml.dom.minidom import Document

from mmdet.apis import init_detector, show_result_pyplot
from mmdet.apis import inference_detector_huge_image



def result2xml(img_name, height, width,labels,bboxes,output_dir):
    doc = Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    source_list = {'filename': img_name, 'origin': 'GF2/GF3'}
    node_source = doc.createElement('source')
    for source in source_list:
        node_name = doc.createElement(source)
        node_name.appendChild(doc.createTextNode(source_list[source]))
        node_source.appendChild(node_name)
    root.appendChild(node_source)

    research_list = {'version': '1.0', 'provider': 'FAIR1M', 'author': 'www.4399.com',
                     'pluginname': 'FAIR1M', 'pluginclass': 'object detection', 'time': '2021-07-21'}
    node_research = doc.createElement('research')
    for research in research_list:
        node_name = doc.createElement(research)
        node_name.appendChild(doc.createTextNode(research_list[research]))
        node_research.appendChild(node_name)
    root.appendChild(node_research)

    size_list = {'width': width, 'height': height, 'depth': '3'}
    node_size = doc.createElement('size')
    for size in size_list:
        node_name = doc.createElement(size)
        node_name.appendChild(doc.createTextNode(size_list[size]))
        node_size.appendChild(node_name)
    root.appendChild(node_size)

    node_objects = doc.createElement('objects')
    for i in range(len(labels)):
        node_object = doc.createElement('object')
        object_fore_list = {'coordinate': 'pixel', 'type': 'rectangle', 'description': 'None'}
        for object_fore in object_fore_list:
            node_name = doc.createElement(object_fore)
            node_name.appendChild(doc.createTextNode(object_fore_list[object_fore]))
            node_object.appendChild(node_name)

        node_possible_result = doc.createElement('possibleresult')
        node_name = doc.createElement('name')
        node_name.appendChild(doc.createTextNode(labels[i]))
        node_possible_result.appendChild(node_name)
        node_object.appendChild(node_possible_result)

        node_points = doc.createElement('points')
        for j in range(4):
            node_point = doc.createElement('point')
            text = '{:.6f},{:.6f}'.format(bboxes[i][j*2], bboxes[i][j*2+1])
            node_point.appendChild(doc.createTextNode(text))
            node_points.appendChild(node_point)
        node_point = doc.createElement('point')
        text = '{:.6f},{:.6f}'.format(bboxes[i][0], bboxes[i][1])
        node_point.appendChild(doc.createTextNode(text))
        node_points.appendChild(node_point)
        node_object.appendChild(node_points)

        node_objects.appendChild(node_object)
    root.appendChild(node_objects)

    # 开始写xml文档
    filename = output_dir + os.path.splitext(img_name)[0] + '.xml'
    fp = open(filename, 'w', encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
    fp.close()
    return 0


# 尝试写多进程代码实现inference

# 单张大图的测试代码
def single_huge_img_inference(arguments,model,score_thr,output_dir,save_files,split,lock,prog,total):
   
    img = arguments
   
    # test a single image
    nms_cfg = dict(type='BT_nms', iou_thr=0.5)
    result = inference_detector_huge_image(
        model, img, split, nms_cfg)
    
    # result -> labels_name and polys
    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes, scores = bboxes[:, :-1], bboxes[:, -1]
    bboxes = bboxes[scores > score_thr]
    labels = labels[scores > score_thr]
    # scores = scores[scores > score_thr]
    labels_name = [model.CLASSES[label] for label in labels]


    image_name = os.path.split(img)[1]
    height, width = mmcv.imread(img).shape[:2]


    result2xml(image_name, str(height), str(width), labels_name, bt.obb2poly(bboxes),output_dir)
    
    
    # 写日志文件
    lock.acquire()
    prog.value += 1
    msg = f'({prog.value/total:3.1%} {prog.value}:{total})'
    msg += ' - ' + f"Filename: {image_name}"
    msg += ' - ' + f"width: {width:<5d}"
    msg += ' - ' + f"height: {height:<5d}"
    msg += ' - ' + f"Objects: {len(labels):<5d}"
  
    # print(msg)
    with open(save_files,"a") as f:
        f.write(msg) 
        f.write('\n')
    lock.release()
    return 0


def main():

    # parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='Config file')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument(
    #     'split', help='split configs in BboxToolkit/tools/split_configs')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--score-thr', type=float, default=0.3, help='bbox score threshold')
    # args = parser.parse_args()

    # # build the model from a config file and a checkpoint file
    # model = init_detector(args.config, args.checkpoint, device=args.device)
    # # test a single image
    # nms_cfg = dict(type='BT_nms', iou_thr=0.5)
    # result = inference_detector_huge_image(
    #     model, args.img, args.split, nms_cfg)
    # # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)

    # 声明图像序列及日志文件
    torch.multiprocessing.set_start_method('spawn')
    img_dir = "/home/fengjq/workspace/test_for_code/data/split_ss_isprs/train/images/"
    img_dir = "/home/fengjq/workspace/xtb_dataset/test/images/"
    save_files = "/home/fengjq/workspace/test_for_code/data/split_ss_isprs/test_result/temp.txt"
    config_dir = "/home/fengjq/workspace/OBBDetection/fjq_code/faster_rcnn_obb_r50_fpn_1x_dota10_for_test.py"
    checkpoint_dir = "/home/fengjq/workspace/OBBDetection/fjq_code/epoch_12.pth"
    device = "cuda:3"
    split_json = "/home/fengjq/workspace/xtb_dataset/split_isprs_train/annfiles/split_config.json"
    output_dir = "/home/fengjq/workspace/xtb_dataset/test/inference_result/"
    score_thr = 0.3
    img_list = [img_dir+img_name for img_name in os.listdir(img_dir)]
    
    nproc = 4
    
    # 调用多进程
    
    with open(save_files,"w") as f:
        f.write("begin\n") 

    start = time.time()
    manager = Manager()
    model = init_detector(config_dir, checkpoint_dir, device=device)
    worker = partial(single_huge_img_inference,
                     model = model,
                     score_thr=score_thr,
                     output_dir=output_dir,
                     save_files=save_files,
                     split=split_json,
                     lock=manager.Lock(),
                     prog=manager.Value('i',0),
                     total=len(img_list),
                     )
    
    if nproc > 1:
        pool = Pool(nproc)
        patch_infos = pool.map(worker, img_list)
        pool.close()
    else:
        patch_infos = list(map(worker, img_list))
    
    patch_infos = reduce(lambda x, y: x+y, patch_infos)
    stop = time.time()
    with open(save_files,"a") as f:
        f.write(f'Finish splitting images in {int(stop - start)} second!!!') 


if __name__ == '__main__':
    main()
