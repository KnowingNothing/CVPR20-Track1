import numpy as np
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



if __name__ == "__main__":
  test1()
  test2()