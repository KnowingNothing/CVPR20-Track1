import torch
import numpy as np
from resnet import resnet50


def main():
  model = resnet50(pretrained=False, num_classes=200)

  fake_img = np.random.uniform(-1, 1, [2, 3, 224, 224]).astype("float32")
  fake_img = torch.tensor(fake_img)
  print(model(fake_img).shape)


if __name__ == "__main__":
  main()