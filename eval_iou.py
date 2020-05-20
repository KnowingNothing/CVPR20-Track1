import argparse
import torch
import time
from dilated_resnet import CAM
from torch.utils.data import DataLoader
from dataset import DetPixelAnnotationSet


batch_size = 1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", help="model path", type=str, required=True)
  parser.add_argument("-c", "--classes", help="how many classes", type=int, default=201)
  args = parser.parse_args()

  print("Preparing model...")
  model = CAM(num_classes=args.classes)
  model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

  assert batch_size == 1, "for test on pixel level annotations, must use batch size 1"

  print("Preparing dataset...")
  dataset = DetPixelAnnotationSet("/home/E/dataset/LID", task="val")

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  count_batch = 0

  model.eval()
  # from ones to avoid zero division
  total_i = torch.ones([batch_size, args.classes]).float()
  total_u = torch.ones([batch_size, args.classes]).float()
  label_mark = torch.arange(0, args.classes).view(1, -1, 1, 1).int()
  print("Begin caluculating...")
  for batch_data in data_loader:
    _, _, cam_map = model(batch_data["data"])
    mask = torch.tensor(batch_data["mask"])
    cam_map = cam_map > 0
    tmp = label_mark * torch.ones_like(cam_map).int()
    tmp = mask.int() == tmp.int()
    insert = cam_map.int() == tmp.int()
    union = (cam_map.int() + tmp.int()) > 0
    union = union.int() - insert.int()
    total_i += insert.int().sum(-1).sum(-1).float()
    total_u += union.int().sum(-1).sum(-1).float()
    count_batch += 1
    if count_batch % 100 == 0:
      iou = total_i / total_u
      iou = iou.mean().item()
      print("parital mean iou over %d batches is: " % count_batch, iou)

  iou = total_i / total_u
  iou = iou.mean().item()
  print("Mean iou is: ", iou)
  with open("iou_record.txt", "a") as fout:
    fout.write("timestamp:%f, mean iou is %f\n" % (time.time(), iou))