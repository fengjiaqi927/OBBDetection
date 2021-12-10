CUDA_VISIBLE_DEVICES=0 python tools/test.py \
work_dirs_tzb/faster_rcnn_orpn_r50_fpn_1x_dota10_tzb_001/faster_rcnn_orpn_r50_fpn_1x_dota10_tzb.py \
work_dirs_tzb/faster_rcnn_orpn_r50_fpn_1x_dota10_tzb_001/epoch_12.pth \
--out work_dirs_tzb/faster_rcnn_orpn_r50_fpn_1x_dota10_tzb_001/patch_results.pkl \
--eval mAP \
--show-dir work_dirs_tzb/faster_rcnn_orpn_r50_fpn_1x_dota10_tzb_001/patch_results/ \
--options save_dir=work_dirs_tzb/faster_rcnn_orpn_r50_fpn_1x_dota10_tzb_001/img_results/
