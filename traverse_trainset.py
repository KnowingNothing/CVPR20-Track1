import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet import resnet50
from dataset import DetDataset


batch_size = 8


def main():
  dataset = DetDataset("/home/E/dataset/ILSVRC", task="train")

  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  count_batch = 0

  for batch_data in train_loader:
    count_batch += 1
  
  print("Done! Totally %d batches" % count_batch)

  


if __name__ == "__main__":
  main()