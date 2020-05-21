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


batch_size = 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", help="model path", type=str, default="")
  parser.add_argument("-c", "--classes", help="how many classes", type=int, default=200)
  parser.add_argument("-g", "--gpu", help="use gpu", action="store_true")
  parser.add_argument("--skip_size", help="skip data with height/width larger than this value", type=int, default=1000)
  args = parser.parse_args()

  print("Preparing model...")
  model = CAM(num_classes=args.classes)
  if args.model != "":
    model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
  if args.gpu:
    model = model.cuda()

  assert batch_size == 1, "for test on pixel level annotations, must use batch size 1"

  print("Preparing dataset...")
  dataset = DetPixelAnnotationSet("/home/E/dataset/LID", task="val")

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  count_batch = 0

  model.eval()
  total_iou = [[] for i in range(args.classes)]
  label_mark = torch.arange(0, args.classes+1).view(1, -1, 1, 1).int()
  if args.gpu:
    label_mark = label_mark.cuda()
  print("Begin caluculating...")
  begin = time.time()
  with torch.no_grad():
    for batch_data in data_loader:
      count_batch += 1
      if args.gpu:
        if batch_data["data"].shape[2] > args.skip_size and batch_data["data"].shape[3] > args.skip_size:
          print("skip a data with shape:", batch_data["data"].shape)
          continue
        data = batch_data["data"].cuda()
        cam_map = model(data)
      else:
        cam_map = model(batch_data["data"])
      end = time.time()
      # print("calculation on model done, using", end - begin, "s")
      begin = end
      mask = torch.tensor(batch_data["mask"]).squeeze()
      if args.gpu:
        mask = mask.cuda()
      # print(mask.shape)
      # print(mask.sum())

      cam_map_org = torch.nn.functional.upsample_bilinear(cam_map, mask.shape)
      cam_map = cam_map_org > 0
      background = cam_map.int().sum(dim=1, keepdim=True) == 0
      cam_map = torch.cat([background, cam_map], dim=1)
      
      tmp = label_mark * torch.ones_like(cam_map).int()
      tmp = mask.int() == tmp.int()

      # predict_map = torch.argmax(torch.cat([background.float(), cam_map_org], dim=1), dim=1).squeeze()
      # im = Image.fromarray((predict_map.cpu().numpy().astype("float32") / 200 * 255).astype("int32"))
      # im.save(os.path.join(im_dir, "%d-predict.PNG" % count_batch), "PNG")
      # im = Image.fromarray((mask.cpu().numpy().astype("float32") / 200 * 255).astype("int32"))
      # im.save(os.path.join(im_dir, "%d-truth.PNG" % count_batch), "PNG")

      # for i in range(mask.size(0)):
      #   for j in range(mask.size(1)):
      #     if mask[i, j] > 0 or predict_map[i, j] > 0:
      #       print(predict_map[i, j].item(), mask[i, j].item())

      cam_map = cam_map[:, 1:, :, :]
      tmp = tmp[:, 1:, :, :]
      # print(cam_map.shape, tmp.shape)
      # print(torch.sum(cam_map[:, :1, :, :]), torch.sum(tmp), cam_map.shape)

      insert = cam_map.int() * tmp.int()
      # print("insert", insert.int().sum())
      # print(torch.any(insert > 0))
      union = (cam_map.int() + tmp.int()) > 0
      # print("union", union.int().sum())
      # print("check insertion:", insert.int().sum(-1).sum(-1).float().cpu())
      # print("check union:", union.int().sum(-1).sum(-1).float().cpu())
      sum_insert = insert.int().sum(-1).sum(-1).float().squeeze()
      sum_union = union.int().sum(-1).sum(-1).float().squeeze()
      for c in range(args.classes):
        if sum_union[c] > 1e-5:
          total_iou[c].append(sum_insert[c].item() / sum_union[c].item())
        
      if count_batch % 100 == 0:
        # mean on data
        class_iou = []
        for c in range(args.classes):
          if len(total_iou[c]) > 0:
            class_iou.append(np.array(total_iou[c]).mean())
        # print("IOU on class:", iou.cpu())
        # mean on class
        iou = np.array(class_iou).mean()
        print("parital mean iou over %d batches is: " % count_batch, iou)
      end = time.time()
      # print("calculation of iou done, passing", end - begin, "s")
      begin = end

  if count_batch > 0:
    class_iou = []
    for c in range(args.classes):
      if len(total_iou[c]) > 0:
        class_iou.append(np.array(total_iou[c]).mean())
    iou = np.array(class_iou).mean()
    print("Mean iou is: ", iou)
    with open("iou_record.txt", "a") as fout:
      fout.write("timestamp:%f, mean iou is %f\n" % (time.time(), iou))
  else:
    raise RuntimeError("No valid data")