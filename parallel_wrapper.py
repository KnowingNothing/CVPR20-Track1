# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:25:22 2020

@author: DrLC
"""

import torch
import torch.nn as nn

class ModelWrapper(nn.Module):
    
    def __init__(self, model, loss_fn):
        
        super(ModelWrapper, self).__init__()
        self.m = model
        self.loss_fn = loss_fn
        
    def forward(self, x):
        
        return self.m.forward(x)
    
    def loss_op(self, x, label):
        
        output = self.forward(x)
        return self.loss_fn(output, label)
    
    def train_op(self, x, label):
        
        output = self.forward(x)
        return output, self.loss_fn(output, label)
    
    def __call__(self, x, label=None):
        
        if label is None:
            return self.forward(x)
        else:
            return self.train_op(x, label)
    
class ParallelWrapper(object):
    
    def __init__(self, model, loss_fn):
        
        self.m = ModelWrapper(model, loss_fn)
        self.ngpu = torch.cuda.device_count()
        if self.ngpu > 1:
            self.m = nn.DataParallel(self.m)
            
    def forward(self, x):
        
        return self.m.forward(x)
    
    def loss_op(self, x, label):
        
        loss = self.m.loss_op
        if self.ngpu > 1:
            loss = torch.mean(loss)
        return loss
    
    def train_op(self, x, label):
        
        output, loss = self.m(x, label)
        if self.ngpu > 1:
            loss = torch.mean(loss)
        return output, loss
    
    def __call__(self, x, label=None):
        
        if label is None:
            return self.forward(x)
        else:
            return self.train_op(x, label)
        
    def eval(self):
        
        self.m.eval()
        
    def train(self):
        
        self.m.train()
        
    def trainable_parameters(self):
        
        return self.m.trainable_parameters()
        
if __name__ == "__main__":
    
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from dilated_resnet import CAM, Net
    from dataset import DetDataset
    import os
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'
    data_dir = "./ILSVRC"

    batch_size = 12
    num_classes = 200  # don't include background
    dtype = "float32"
    map_size = 28
    eps = 1e-5
    lr1 = 0.002
    wd1 = 0.002
    lr2 = 0.01
    wd2 = 0.002
    loss_report = 100
    val_report = 1000
    save_report = 100
    
    #trainset = DetDataset(data_dir, task="train", dtype=dtype)
    #train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)    
    valset = DetDataset(data_dir, task="val", dtype=dtype)    
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    
    model = Net(num_classes=num_classes).cuda()
    pmodel = ParallelWrapper(model, F.multilabel_soft_margin_loss)
    pmodel.eval()
    model.eval()
    
    for batch_data in val_loader:
        x = batch_data["data"].cuda()
        label = batch_data["label"][1:].cuda()
        output, loss = pmodel(x, label)
        print (x.shape)
        print (label.shape)
        print (output.shape)
        print (loss)
        print ()
        output = model(x)
        loss = F.multilabel_soft_margin_loss(output, label)
        print (x.shape)
        print (label.shape)
        print (output.shape)
        print (loss)
        print ()
        break