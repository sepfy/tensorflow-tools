import cv2
import numpy as np
import os

def load_images(image_dir, num):

  batch = np.empty((0, 224, 224, 3), dtype=np.float32)
  for image_name in os.listdir(image_dir)[:num]:

    img_path = image_dir + "/" + image_name
    #print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    npimg = np.asarray(img, dtype=np.float32)
    npimg = (npimg - 127.5)/127.5
    npimg.resize(1, 224, 224, 3)

    #print(batch.shape) 
    batch = np.concatenate((batch, npimg), 0)
  return batch

def create_label(shape, index):
  labels = np.zeros(shape, dtype=np.float32)
  labels[:, index] = 1
  return labels 

