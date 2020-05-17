import os
from glob import glob

for file in glob("./dataset_log/*.log"):
  os.remove(file)