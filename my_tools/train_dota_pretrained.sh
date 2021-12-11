CUDA_VISIBLE_DEVICES=5 python tools/train.py \
configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_3x_dota10_tzb.py \
--work-dir work_dirs_tzb/faster_rcnn_orpn_r50_fpn_3x_dota10_tzb_001/ \
--options model.pretrained=faster_rcnn_orpn_r50_fpn_1x_dota10_epoch12.pth
