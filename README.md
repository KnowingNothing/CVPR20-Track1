# CVPR20-Track1
CVPR20 Chanllege Track1

## requirements
python3
pytorch 1.5

## run training
python train_dilated_cam.py --data /path/to/train/set --gpu

## run evalation
python eval_iou.py --model /path/to/your/model/file --data /path/to/LID/data --gpu 0
