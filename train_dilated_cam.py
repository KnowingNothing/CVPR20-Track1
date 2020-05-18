import os
import time
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dilated_resnet import CAM, Net
from dataset import DetDataset


dir_name = "train_log"
model_dir = "trained_model"
ts = time.localtime(time.time())
year, mon, day, hour, mini, sec = ts.tm_year, ts.tm_mon, ts.tm_mday, ts.tm_hour, ts.tm_min, ts.tm_sec
if not (os.path.exists(dir_name) and os.path.isdir(dir_name)):
  os.mkdir(dir_name)
if not (os.path.exists(model_dir) and os.path.isdir(model_dir)):
  os.mkdir(model_dir)
date_str = "%d_%d_%d-%d_%d_%d" % (year, mon, day, hour, mini, sec)
file_name = date_str + "-log.log"
model_name = date_str + "-model.pkl"

formatter = logging.Formatter("[%(levelname)s] %(message)s (%(asctime)s)")

fh = logging.FileHandler(os.path.join(dir_name, file_name))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)


epoch = 1
batch_size = 8
num_classes = 201  # include background
dtype = "float32"
map_size = 56
eps = 1e-5
lr1 = 0.0002
wd1 = 0.002
lr2 = 0.001
wd2 = 0.002
loss_report = 100
val_report = 1000
save_report = 100


def validate(model, dataloader, test_num=200):
  model.eval()
  valid_target = np.ones([batch_size, map_size, map_size]).astype(dtype)
  valid_target = torch.tensor(valid_target).cuda()
  count_batch = 0
  total_hard_match = 0
  total_mediate_match = 0
  total_soft_match = 0
  total_num = 0
  for batch_data in dataloader:
    count_batch += 1
    output, cam_logit, cam_map = model(batch_data["data"].cuda())
    output = output.cpu()
    label = batch_data["label"].cpu().int()
    max_k = label.sum(dim=-1).max()
    print(max_k)
    bias = torch.topk(output, max_k)[0][:, -1].unsqueeze(-1)
    mark = output >= bias
    match = label - mark.int() == 0
    count_match = match.sum(dim=-1)
    hard_accurate = (count_match == label.sum(dim=-1)).sum()
    mediate_accurate = (count_match >= label.sum(dim=-1) * 0.8).sum()
    soft_accurate = (count_match > 0).sum()

    total_hard_match += hard_accurate.item()
    total_mediate_match += mediate_accurate.item()
    total_soft_match += soft_accurate.item()
    total_num += batch_data["data"].size(0)

    if count_batch >= test_num:
      break

  logger.info("validation on %d batches.\naccuracy: hard: %f, mediate: %f, soft: %f" % (
        count_batch,
        total_hard_match / float(total_num),
        total_mediate_match / float(total_num),
        total_soft_match / float(total_num)
        ))

  model.train()


def main():
  model = CAM(num_classes=num_classes).cuda()

  param_groups = model.trainable_parameters()
  optimizer = torch.optim.Adam([
        {'params': param_groups[0], 'lr': lr1, 'weight_decay': wd1},
        {'params': param_groups[1], 'lr': lr2, 'weight_decay': wd2},
    ], lr=lr1, weight_decay=wd1)

  print("Get dataset...")

  trainset = DetDataset("/home/E/dataset/ILSVRC", task="train", dtype=dtype)

  train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

  valset = DetDataset("/home/E/dataset/ILSVRC", task="val", dtype=dtype)

  val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

  print("Dataset ready.")
  count_batch = 0
  valid_target = np.ones([batch_size, map_size, map_size]).astype(dtype)
  valid_target = torch.tensor(valid_target).cuda()

  model.train()

  print("Start training, logging into %s..." % file_name)
  for ep in range(epoch):
    logger.info("ep=%d:" % (ep+1))
    for batch_data in train_loader:
      count_batch += 1
      output, cam_logit, cam_map = model(batch_data["data"].cuda())
      label = batch_data["label"].cuda()
      # classification loss
      # no reduction on batch dim
      class_loss = F.multilabel_soft_margin_loss(output, label)
      # one pixel must and only one class
      valid_loss = F.mse_loss(torch.max(cam_logit, 1)[0], valid_target) * batch_size
      # segmentation in the map should be supported by logit
      seg_loss = torch.mean((1.0 - cam_logit) * cam_map) * batch_size
      
      loss = class_loss + valid_loss + seg_loss
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if count_batch % loss_report == 0:
        logger.info("batch count: %d, loss: %f = %f + %f + %f" % (
          count_batch,
          loss.detach().item(),
          class_loss.detach().item(),
          valid_loss.detach().item(),
          seg_loss.detach().item()
          ))

      if count_batch % val_report == 0:
        logger.info("validating...")
        validate(model, val_loader)

      if count_batch % save_report == 0:
        logger.info("saving model to %s..." % model_name)
        torch.save(model.state_dict(), os.path.join(model_dir, model_name))

  print("Done! Totally %d batches" % count_batch)


if __name__ == "__main__":
  main()