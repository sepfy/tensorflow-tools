import cv2
from keras.preprocessing.image import ImageDataGenerator
import sys
import os


pics = 5
datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

def gen(folder):
  all_files = os.listdir(folder)
  for f in all_files:
    filename = folder + "/" + f
    im = cv2.imread(filename)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.reshape((1,) + im.shape)
    #print(im.shape)
    count = 0
    for b in datagen.flow(im, batch_size=1, save_to_dir=folder, save_prefix=f.split(".")[0]):
      count += 1
      if count > pics:
        break


if __name__ == "__main__":
  gen(sys.argv[1])
