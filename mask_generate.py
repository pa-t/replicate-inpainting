import logging
import os

import cv2
from rembg import remove
from PIL import Image


# set up logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


INPUT_IMAGE_DIR = "background-images"
NO_BG_IMAGE_DIR = "no-bg-images"
MASK_IMAGE_DIR = "mask-images"


# generate a list of filenames to create masks for
filename_list = []
for filename in os.listdir(INPUT_IMAGE_DIR):
  image = os.path.join(INPUT_IMAGE_DIR, filename)
  mask = os.path.join(MASK_IMAGE_DIR, filename)
  # check if both paths are valid files
  if os.path.isfile(image) and not os.path.isfile(mask) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
    logger.info(f"no mask found for: {filename}, adding to list")
    filename_list.append(filename)


for filename in filename_list:
  input_path = f"{INPUT_IMAGE_DIR}/{filename}"
  no_bg_path = f"{NO_BG_IMAGE_DIR}/{filename}"
  mask_path = f"{MASK_IMAGE_DIR}/{filename}"
  
  logger.info(f"reading in image: {filename}")
  # read in image
  input = Image.open(input_path)
  
  logger.info(f"removing background from image: {filename}")
  # remove background
  output = remove(input)

  # write file out
  try:
    logger.info(f"writing no background image: {filename} to path: {no_bg_path}")
    output.save(no_bg_path)
  except Exception as e:
    # depending on filetype this might break, convert and write
    logger.info(f"exception writing no background image: {filename}. exception: {e}")
    logger.info(f"converting image and trying write again...")
    output = output.convert("RGB")
    output.save(no_bg_path)
    logger.info(f"{no_bg_path} written")

  # convert to mask
  logger.info(f"reading in no bg photo to opencv: {no_bg_path}")
  cv2_image = cv2.imread(no_bg_path)
  logger.info(f"converting {filename} to binary mask")
  img_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
  ret, im = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)
  logger.info(f"writing mask {filename} to file")
  cv2.imwrite(mask_path, im)
  logger.info(f"write successful: {mask_path}")