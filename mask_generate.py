import logging
import os

import cv2
from rembg import remove
from PIL import Image


"""
Using OpenCV, rembg, and PIL find and remove the background from a source directory, then make a binary mask of the subject
"""

# set up logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# constants
INPUT_IMAGE_DIR = "background-images"
NO_BG_IMAGE_DIR = "no-bg-images"
MASK_IMAGE_DIR = "mask-images"


# generate a list of filenames to create masks for
def get_filenames():
  filename_list = []
  for filename in os.listdir(INPUT_IMAGE_DIR):
    image = os.path.join(INPUT_IMAGE_DIR, filename)
    mask = os.path.join(MASK_IMAGE_DIR, filename)
    # check if image file exists, is an image type, and no mask already exists
    if os.path.isfile(image) and not os.path.isfile(mask) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
      logger.info(f"no mask found for: {filename}, adding to list")
      filename_list.append(filename)
  
  return filename_list


# remove the background of an image using rembg
def remove_background(file_path):
  logger.info(f"reading in image: {file_path}")
  # read in image
  input = Image.open(file_path)
  
  # remove background
  logger.info(f"removing background from image: {file_path}")
  return remove(input)


# write file out
def save_no_bg_image(filename, image, path):
  try:
    logger.info(f"writing no background image: {filename} to path: {path}")
    image.save(path)
  except Exception as e:
    # depending on filetype this might break, convert and write
    logger.info(f"exception writing no background image: {filename}. exception: {e}")
    logger.info(f"converting image and trying write again...")
    image = image.convert("RGB")
    image.save(path)
    logger.info(f"{path} written")


def create_binary_mask(filename, input_path, output_path):
  # convert to mask
  logger.info(f"reading in no bg photo to opencv: {input_path}")
  cv2_image = cv2.imread(input_path)

  logger.info(f"converting {filename} to binary mask")
  img_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
  ret, im = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)

  logger.info(f"writing mask {filename} to file")
  cv2.imwrite(output_path, im)

  logger.info(f"write successful: {output_path}")


def main():
  filename_list = get_filenames()
  for filename in filename_list:
    input_path = f"{INPUT_IMAGE_DIR}/{filename}"
    no_bg_path = f"{NO_BG_IMAGE_DIR}/{filename}"
    mask_path = f"{MASK_IMAGE_DIR}/{filename}"
    
    output = remove_background(file_path=input_path)

    save_no_bg_image(filename, output, no_bg_path)

    create_binary_mask(filename, no_bg_path, mask_path)


if __name__ == '__main__':
  main()