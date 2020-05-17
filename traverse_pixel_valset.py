import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet import resnet50
from dataset import DetPixelAnnotationSet


batch_size = 1


def main():
  assert batch_size == 1, "for test on pixel level annotations, must use batch size 1"

  dataset = DetPixelAnnotationSet("/home/E/dataset/LID", task="val")

  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  count_batch = 0

  for batch_data in data_loader:
    count_batch += 1
  
  print("Done! Totally %d batches" % count_batch)

  


if __name__ == "__main__":
  main()