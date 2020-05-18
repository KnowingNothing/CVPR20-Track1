import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image


def test1():
  # this a known 4 channel picture
  # /home/E/dataset/ILSVRC/Data/DET/train/ILSVRC2013_train/n02105855/n02105855_2933.JPEG
  im = Image.open("./n02105855_2933.JPEG")
  im = np.array(im)
  im = im[:, :, 0:3]  # * np.expand_dims(im[:, :, 3], -1)
  print(im.shape)
  im = Image.fromarray(im)
  im.save("converted_rgb.JPEG", "JPEG")


def test2():
  # see an annotation png file
  im = Image.open("./ILSVRC2012_val_00000026.png")
  im = np.array(im)
  print(im.shape)
  print(im.dtype)
  print(im.max())
  print(im.min())


def test3():
  img = torch.rand([4, 5])
  y = torch.randint(0, 2, [4, 5])
  loss = F.multilabel_soft_margin_loss(img, y)


if __name__ == "__main__":
  test1()
  test2()
  test3()