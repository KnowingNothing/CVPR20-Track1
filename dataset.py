import os
import time
import logging
import torch
import numpy as np
import xml.etree.ElementTree as ET
import scipy.io as scio
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ts = time.localtime(time.time())
year, mon, day, hour, mini, sec = ts.tm_year, ts.tm_mon, ts.tm_mday, ts.tm_hour, ts.tm_min, ts.tm_sec
if not (os.path.exists("dataset_log") and os.path.isdir("dataset_log")):
  os.mkdir("dataset_log")
logging.basicConfig(filename="dataset_log/%d_%d_%d-%d_%d_%d-log.log" % (year, mon, day, hour, mini, sec), level=logging.INFO)
# logging.basicConfig(filename="dataset.log", level=logging.INFO)


class DetDataset(Dataset):
  """
  ILSVRC DET Dataset
  """
  def __init__(self, root_dir="./", task="train", transform=None, dtype="float32"):
    assert task in ["train", "val", "test"]
    self.dtype = dtype
    self.transform = transform
    if task == "train":
      self.img_dir = os.path.join(root_dir, "Data", "DET", "train")
      self.label_dir = os.path.join(root_dir, "Annotations", "DET", "train")
      self.xml_files = sorted(glob(os.path.join(self.label_dir, "ILSVRC2013_train", "*", "*.xml")))
      xml_files_2014 = sorted(glob(os.path.join(self.label_dir, "*", "*.xml")))
      self.xml_files.extend(xml_files_2014)
      logging.info("totally %d xml files for training" % len(self.xml_files))
    elif task == "val":
      self.img_dir = os.path.join(root_dir, "Data", "DET", "val")
      self.label_dir = os.path.join(root_dir, "Annotations", "DET", "val")
      self.xml_files = sorted(glob(os.path.join(self.label_dir, '*.xml')))
      logging.info("totally %d xml files for validation" % len(self.xml_files))
    else:
      raise ValueError("No test annotations")

    self.task = task

    # load meta data
    data = scio.loadmat('meta_det.mat')
    self.name_to_detlabelid = {}
    for item in data['synsets'][0]:
      det_label_id = item[0][0][0]
      name = item[1][0]
      cat_name = item[2][0]
      # print(det_label_id)
      # print(cat_name)
      # print(name)
      self.name_to_detlabelid[name] = det_label_id

  def __len__(self):
    return len(self.xml_files)

  def __getitem__(self, idx):
    xml_file = self.xml_files[idx]
    tree = ET.parse(xml_file)
    root = tree.getroot()
    source = root.findall('source')
    # label and bounding box
    label = np.zeros(201).astype(self.dtype)
    objects = root.findall('object')

    for object in objects:
      object_name = object[0].text
      label_id = self.name_to_detlabelid[object_name]
      label[label_id] = 1
    # background
    label[0] = 1

    # img
    if self.task == "train":
      if source[0][0].text=='ILSVRC_2013':
        filename = root.findall('filename')[0].text
        folder = root.findall('folder')[0].text
        pic_path = os.path.join(self.img_dir, 'ILSVRC2013_train', folder, filename+'.JPEG')
        imt = Image.open(pic_path)
      else:
        filename = root.findall('filename')[0].text
        folder = root.findall('folder')[0].text
        pic_path = os.path.join(self.img_dir, folder, filename+'.JPEG')
        imt = Image.open(pic_path)
    elif self.task == "val":
      filename = root.findall('filename')[0].text
      pic_path = os.path.join(self.img_dir, filename+'.JPEG')
      imt = Image.open(pic_path)
    else:
      raise ValueError("No test set")
    
    if self.transform is not None:
      imt = self.transform(imt)
    
    imt = imt.resize([224, 224], resample=Image.BICUBIC)
    imt = np.array(imt).astype(self.dtype)
    imt = torch.tensor(imt)
    if len(imt.shape) != 3:
      logging.warn("image shape mismatch: %s, shape: %s" % (pic_path, str(imt.shape)))
    if len(imt.shape) == 2:
      imt = imt.unsqueeze(-1).expand(*imt.shape, 3)
    # rgba
    if imt.shape[2] != 3:
      imt = imt[:,:,:3]
    # CHW
    imt = imt.permute(2, 0, 1)
    imt = imt / 255
    return {"data": imt, "label": label}
    

class DetPixelAnnotationSet(Dataset):
  """
  DET Pixel Level Annotation Dataset
  """
  def __init__(self, root_dir="./", task="val", transform=None, dtype="float32"):
    assert task in ["val", "test"]
    self.task = task
    self.dtype = dtype
    self.transform = transform
    self.img_dir = os.path.join(root_dir, "LID_track1_imageset", "LID_track1", task)
    self.label_dir = os.path.join(root_dir, "LID_track1_annotations", "track1_"+task+"_annotations_raw")
    self.png_files = sorted(glob(os.path.join(self.label_dir, '*.png')))
    initial_num = len(self.png_files)
    logging.info("Pixel level dataset totally %d xml files" % initial_num)
    self.clear_dataset()
    logging.info("However, %d files not found" % (initial_num - len(self.png_files)))

  def clear_dataset(self):
    new_files = []
    for png_file in self.png_files:
      file_name = os.path.split(png_file)[1].split(".")[0] + ".JPEG"
      img_path = os.path.join(self.img_dir, file_name)

      if (os.path.exists(img_path) and os.path.isfile(img_path)):
        try:
          Image.open(img_path)
          new_files.append(png_file)
        except Exception:
          pass
    self.png_files = new_files

  def __len__(self):
    return len(self.png_files)

  def __getitem__(self, idx):
    png_file = self.png_files[idx]
    file_name = os.path.split(png_file)[1].split(".")[0] + ".JPEG"
    img_path = os.path.join(self.img_dir, file_name)

    imt = Image.open(img_path)
    imt = imt.resize([224, 224], resample=Image.BICUBIC)
    imt = np.array(imt).astype(self.dtype)
    imt = torch.tensor(imt)
    if len(imt.shape) != 3:
      logging.warn("image shape mismatch: %s, shape: %s" % (img_path, str(imt.shape)))
    if len(imt.shape) == 2:
      imt = imt.unsqueeze(-1).expand(*imt.shape, 3)
    # rgba
    if imt.shape[2] != 3:
      imt = imt[:,:,:3]
    # CHW
    imt = imt.permute(2, 0, 1)
    imt = imt / 255

    mask = Image.open(png_file)
    mask = np.array(mask)
    return {"data": imt, "mask": mask}
    