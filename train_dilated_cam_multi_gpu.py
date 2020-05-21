# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:20:08 2020

@author: DrLC
"""

import argparse
import os
import tqdm
import time
import torch
import logging
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dilated_resnet import CAM, Net
from dataset import DetDataset
from parallel_wrapper import ParallelWrapper


def validate(model, dataloader, test_num=None):
  model.eval()
  # valid_target = np.ones([batch_size, map_size, map_size]).astype(dtype)
  # valid_target = torch.tensor(valid_target).cuda()
  count_batch = 0
  total_hard_match = 0
  total_mediate_match = 0
  total_soft_match = 0
  total_num = 0
  for batch_data in dataloader:
    count_batch += 1
    output = model(batch_data["data"].cuda())
    output = output.cpu()
    # don't include background
    label = batch_data["label"][1:].cpu().int()
    max_k = label.sum(dim=-1).max()
    # print(max_k)
    bias = torch.topk(output, max_k)[0][:, -1].unsqueeze(-1)
    mark = output >= bias
    match = label * mark.int()
    count_match = match.sum(dim=-1)
    hard_accurate = (count_match == label.sum(dim=-1)).sum()
    mediate_accurate = (count_match >= label.sum(dim=-1) * 0.95).sum()
    soft_accurate = (count_match > 0).sum()

    total_hard_match += hard_accurate.item()
    total_mediate_match += mediate_accurate.item()
    total_soft_match += soft_accurate.item()
    total_num += batch_data["data"].size(0)

    if test_num is not None and count_batch >= test_num:
      break

  logger.info("validation on %d samples.\naccuracy: hard: %f, mediate: %f, soft: %f" % (
        total_num,
        total_hard_match / float(total_num),
        total_mediate_match / float(total_num),
        total_soft_match / float(total_num)
        ))

  model.train()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='CAM TRAIN')
    parser.add_argument('--epoch', type=int, default=1,
                        help="max epoch threshold")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="mini batch size")
    parser.add_argument('--n_class', type=int, default=200,
                        help="class number (excluding background)")
    parser.add_argument('--dtype', type=str, default="float32",
                        help="data type")
    parser.add_argument('--lr1', type=float, default=0.002,
                        help="learning rate 1 initialization")
    parser.add_argument('--wd1', type=float, default=0.002,
                        help="weight decay 1")
    parser.add_argument('--lr2', type=float, default=0.01,
                        help="learning rate 2 initialization")
    parser.add_argument('--wd2', type=float, default=0.002,
                        help="weight decay 2")
    parser.add_argument('--data_dir', type=str, default='./ILSVRC',
                        help='ImageNet data directory')
    
    parser.add_argument('--gpu', type=str, default="1",
                        help="gpu selection (multi-gpu supported)")
    
    parser.add_argument('--log_period', type=int, default=100,
                        help="logging every n iterations")
    parser.add_argument('--val_period', type=int, default=1000,
                        help="validating every n iterations")
    parser.add_argument('--save_period', type=int, default=100,
                        help="saving every n iterations")
    
    parser.add_argument('--log_name', type=str, default='log.log',
                        help='log name')
    parser.add_argument('--model_name', type=str, default='model.pkl',
                        help='model name')
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    data_dir = args.data_dir
    epoch = args.epoch
    batch_size = args.batch_size
    num_classes = args.n_class
    dtype = args.dtype
    lr1 = args.lr1
    wd1 = args.wd1
    lr2 = args.lr2
    wd2 = args.wd2
    
    loss_report = args.log_period
    val_report = args.val_period
    save_report = args.save_period
    
    dir_name = "log"
    model_dir = "model"
    file_name = args.log_name
    model_name = args.model_name
    if not (os.path.exists(dir_name) and os.path.isdir(dir_name)):
      os.mkdir(dir_name)
    if not (os.path.exists(model_dir) and os.path.isdir(model_dir)):
      os.mkdir(model_dir)
    
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
    
    print("Get dataset...")
    
    trainset = DetDataset(data_dir, task="train", dtype=dtype)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = DetDataset(data_dir, task="val", dtype=dtype)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    
    print("Dataset ready.")
    count_batch = 0
    
    model = Net(num_classes=num_classes).cuda()
    pmodel = ParallelWrapper(model, F.multilabel_soft_margin_loss)
    param_groups = model.trainable_parameters()
    optimizer = torch.optim.Adam([
        {'params': param_groups[0], 'lr': lr1, 'weight_decay': wd1},
        {'params': param_groups[1], 'lr': lr2, 'weight_decay': wd2},
    ], lr=lr1, weight_decay=wd1)
    pmodel.train()
    
    print("Start training, logging into %s..." % file_name)
    for ep in range(epoch):
        logger.info("ep=%d:" % (ep+1))
        print ("Epoch "+str(ep)+" ...")
        for batch_data in tqdm.tqdm(train_loader):
            count_batch += 1
            output, loss = pmodel(batch_data["data"].cuda(), batch_data["label"][1:].cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if count_batch % loss_report == 0:
                logger.info("batch count: %d, loss: %f" % (
                        count_batch,
                        loss.detach().item()
                ))
    
            if count_batch % val_report == 0:
                logger.info("validating...")
                validate(pmodel, val_loader, test_num=10000)
    
            if count_batch % save_report == 0:
                logger.info("saving model to %s..." % model_name)
                torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        
        logger.info("testing after one epoch...")
        validate(pmodel, val_loader)
    
    print("Done! Totally %d batches" % count_batch)