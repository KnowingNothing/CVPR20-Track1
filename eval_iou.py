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
  parser.add_argument("-c", "--classes", help="how many classes", type=int, default=200)
  parser.add_argument("-g", "--gpu", help="use gpu", action="store_true")
  parser.add_argument("--skip_size", help="skip data with height/width larger than this value", type=int, default=1000)
  args = parser.parse_args()

  print("Preparing model...")
  model = CAM(num_classes=args.classes)
  model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
  if args.gpu:
    model = model.cuda()

  assert batch_size == 1, "for test on pixel level annotations, must use batch size 1"

  print("Preparing dataset...")
  dataset = DetPixelAnnotationSet("/home/E/dataset/LID", task="val")

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  count_batch = 0

  model.eval()
  total_iou = torch.zeros([batch_size, args.classes+1]).float()
  label_mark = torch.arange(0, args.classes+1).view(1, -1, 1, 1).int()
  if args.gpu:
    total_iou = total_iou.cuda()
    label_mark = label_mark.cuda()
  print("Begin caluculating...")
  begin = time.time()
  with torch.no_grad():
    for batch_data in data_loader:
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
      cam_map = torch.nn.functional.upsample_bilinear(cam_map, mask.shape)
      cam_map = cam_map > 0
      background = cam_map.int().sum(dim=1, keepdim=True) == 0
      cam_map = torch.cat([background, cam_map], dim=1)
      
      tmp = label_mark * torch.ones_like(cam_map).int()
      tmp = mask.int() == tmp.int()
      # print(cam_map.shape, tmp.shape)
      # print(torch.sum(cam_map[:, :1, :, :]), torch.sum(tmp), cam_map.shape)

      insert = cam_map.int() * tmp.int()
      # print("insert", insert.int().sum())
      # for i in range(201):
      #   for j in range(insert.size(2)):
      #     for k in range(insert.size(3)):
      #       print(insert[0, i, j, k], tmp[0, i, j, k])
      # print(torch.any(insert > 0))
      union = (cam_map.int() + tmp.int()) > 0
      # print("union", union.int().sum())
      # print("check insertion:", insert.int().sum(-1).sum(-1).float().cpu())
      # print("check union:", union.int().sum(-1).sum(-1).float().cpu())
      total_iou = total_iou + (insert.int().sum(-1).sum(-1).float() + 1e-5) / (union.int().sum(-1).sum(-1).float() + 1e-5)
      count_batch += 1
      if count_batch % 100 == 0:
        # mean on data
        iou = total_iou / count_batch
        # print("IOU on class:", iou.cpu())
        # mean on class
        iou = (iou.sum()/(args.classes+1)).item()
        print("parital mean iou over %d batches is: " % count_batch, iou)
      end = time.time()
      # print("calculation of iou done, passing", end - begin, "s")
      begin = end

  if count_batch > 0:
    iou = total_iou / count_batch
    iou = iou.mean().item()
    print("Mean iou is: ", iou)
    with open("iou_record.txt", "a") as fout:
      fout.write("timestamp:%f, mean iou is %f\n" % (time.time(), iou))
  else:
    raise RuntimeError("No valid data")