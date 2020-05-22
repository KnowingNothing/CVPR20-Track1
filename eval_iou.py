import os
import argparse
import torch
import time
import numpy as np
from dilated_resnet import CAM
from torch.utils.data import DataLoader
from dataset import DetPixelAnnotationSet
from PIL import Image


im_dir = "peek_image"
if not (os.path.exists(im_dir) and os.path.isdir(im_dir)):
  os.mkdir(im_dir)

for file in os.listdir(im_dir):
  os.remove(os.path.join(im_dir, file))


batch_size = 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", help="model path", type=str, default="")
  parser.add_argument("-c", "--classes", help="how many classes", type=int, default=200)
  parser.add_argument("-g", "--gpu", help="use gpu", action="store_true")
  parser.add_argument("-d", "--dev_id", help="use which gpu", type=int, default=0)
  parser.add_argument("--data", help="data path", type=str, default="/home/E/dataset/LID")
  parser.add_argument("--skip_size", help="skip data with height/width larger than this value", type=int, default=1000)
  args = parser.parse_args()

  print("Preparing model...")
  model = CAM(num_classes=args.classes)
  if args.model != "":
    model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
  if args.gpu:
    model = model.cuda(args.dev_id)

  assert batch_size == 1, "for test on pixel level annotations, must use batch size 1"

  print("Preparing dataset...")
  dataset = DetPixelAnnotationSet(args.data, task="val")

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  count_batch = 0

  model.eval()
  total_iou = []
  label_mark = torch.arange(0, args.classes+1).view(1, -1, 1, 1).int()
  if args.gpu:
    label_mark = label_mark.cuda(args.dev_id)
  print("Begin caluculating...")
  begin = time.time()
  with torch.no_grad():
    for batch_data in data_loader:
      count_batch += 1
      if args.gpu:
        if batch_data["data"].shape[2] > args.skip_size and batch_data["data"].shape[3] > args.skip_size:
          print("skip a data with shape:", batch_data["data"].shape)
          continue
        data = batch_data["data"].cuda(args.dev_id)
        cam_map = model(data)
      else:
        cam_map = model(batch_data["data"])
      end = time.time()
      # print("calculation on model done, using", end - begin, "s")
      begin = end
      mask = torch.tensor(batch_data["mask"]).squeeze()
      if args.gpu:
        mask = mask.cuda(args.dev_id)

      cam_map = torch.nn.functional.interpolate(cam_map, size=mask.shape, mode="bilinear")
      cam_map_min = torch.min(torch.min(cam_map, 3, keepdim=True)[0], 2, keepdim=True)[0]
      cam_map_max = torch.max(torch.max(cam_map, 3, keepdim=True)[0], 2, keepdim=True)[0]
      cam_map = (cam_map - cam_map_min) / (cam_map_max - cam_map_min)
      cam_map = (cam_map > 0.1)

      background = cam_map.sum(dim=1, keepdim=True) == 0
      cam_map = torch.cat([background, cam_map], dim=1).int()
      
      tmp = label_mark * torch.ones_like(cam_map)
      tmp = (torch.abs(mask.unsqueeze(0).unsqueeze(0) - tmp) < 1e-5).int()

      insert = (cam_map.int() * tmp.int()) >= 1
      union = (cam_map.int() + tmp.int()) >= 1

      tmp_iou = []
      sum_insert = insert.int().sum(-1).sum(-1).float().squeeze()
      sum_union = union.int().sum(-1).sum(-1).float().squeeze()

      has_class = 0
      for c in range(1, args.classes+1):
        if c > 0 and tmp[0, c, :, :].sum() >= 100:
          has_class += 1
          tmp_iou.append(sum_insert[c].item() / sum_union[c].item())
          # print(tmp_iou[-1], "=", sum_insert[c].item(), "/", sum_union[c].item())
          if tmp_iou[-1] > 0.5:
            im_map = Image.fromarray(np.uint8(cam_map[0, c, :, :].int().cpu().numpy() * 255))
            im_mask = Image.fromarray(np.uint8(tmp[0, c, :, :].int().cpu().numpy() * 255))
            im_org_mask = Image.fromarray(mask.int().cpu().numpy() * 255)
            im_map.save(os.path.join(im_dir, "-predict-%d-%d.png" % (count_batch, c)), "PNG")
            im_mask.save(os.path.join(im_dir, "-label-%d-%d.png" % (count_batch, c)), "PNG")
            im_org_mask.save(os.path.join(im_dir, "-mask-%d-%d.png" % (count_batch, c)), "PNG")
      if len(tmp_iou) > 0:
        total_iou.append(np.array(tmp_iou).mean())
      else:
        print("No valid class")
        
      if count_batch % 100 == 0:
        if len(total_iou) > 0:
          iou = np.array(total_iou).mean()
          print("parital mean iou over %d batches is: " % count_batch, iou)
        else:
          print("No valid iou")
      end = time.time()
      # print("calculation of iou done, passing", end - begin, "s")
      begin = end

  if count_batch > 0:
    if len(total_iou) > 0:
      iou = np.array(total_iou).mean()
      print("Mean iou is: ", iou)
      with open("iou_record.txt", "a") as fout:
        fout.write("timestamp:%f, mean iou is %f\n" % (time.time(), iou))
    else:
      print("No valid iou")
  else:
    raise RuntimeError("No valid data")