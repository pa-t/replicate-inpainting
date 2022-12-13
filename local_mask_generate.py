import logging
import os

import cv2
from rembg import remove
from PIL import Image


"""
Using OpenCV, rembg, and PIL find and remove the background from a source directory, then make a binary mask of the subject
"""

class LocalMaskGen:
  def __init__(self):
    # set up logger
    logging.basicConfig()
    self.logger = logging.getLogger(__name__)
    self.logger.setLevel(logging.INFO)

  def set_constants(self, batch: bool, input_path: str, no_bg_path: str, mask_path: str):
    """set constants used during mask generation"""
    self.BATCH = batch
    self.INPUT_PATH = input_path
    self.NO_BG_PATH = no_bg_path
    self.MASK_PATH = mask_path

  # generate a list of filenames to create masks for
  def get_filenames(self):
    filename_list = []
    for filename in os.listdir(self.INPUT_PATH):
      image = os.path.join(self.INPUT_PATH, filename)
      mask = os.path.join(self.MASK_PATH, filename)
      # check if image file exists, is an image type, and no mask already exists
      if os.path.isfile(image) and not os.path.isfile(mask) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        self.logger.info(f"no mask found for: {filename}, adding to list")
        filename_list.append(filename)
    
    return filename_list


  # remove the background of an image using rembg
  def remove_background(self, file_path):
    self.logger.info(f"reading in image: {file_path}")
    # read in image
    input = Image.open(file_path)
    
    # remove background
    self.logger.info(f"removing background from image: {file_path}")
    return remove(input)


  # write file out
  def save_no_bg_image(self, filename, image, path):
    try:
      self.logger.info(f"writing no background image: {filename} to path: {path}")
      image.save(path)
    except Exception as e:
      # depending on filetype this might break, convert and write
      self.logger.info(f"exception writing no background image: {filename}. exception: {e}")
      self.logger.info(f"converting image and trying write again...")
      image = image.convert("RGB")
      image.save(path)
      self.logger.info(f"{path} written")


  def create_binary_mask(self, filename, input_path, output_path):
    # convert to mask
    self.logger.info(f"reading in no bg photo to opencv: {input_path}")
    cv2_image = cv2.imread(input_path)

    self.logger.info(f"converting {filename} to binary mask")
    img_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)

    self.logger.info(f"writing mask {filename} to file")
    cv2.imwrite(output_path, im)

    self.logger.info(f"write successful: {output_path}")


  def run_batch(self, mask_images, target_images):
    while len(mask_images) < len(target_images):
      # get all valid image filenames in list that need masks
      filename_list = self.get_filenames()

      self.logger.info(f"found {len(filename_list)} valid image names needing masks...")

      for filename in filename_list:
        input_path = f"{self.INPUT_PATH}/{filename}"
        no_bg_path = f"{self.NO_BG_PATH}/{filename}"
        mask_path = f"{self.MASK_PATH}/{filename}"
        
        output = self.remove_background(file_path=input_path)

        self.save_no_bg_image(filename, output, no_bg_path)

        self.create_binary_mask(filename, no_bg_path, mask_path)
      
      # get all mask images now for comparison
      mask_images = [filename for filename in os.listdir(self.MASK_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

  
  def run_single(self, mask_images):
    # just looking for one image in the mask directory
    filename = self.INPUT_PATH.split("/")[-1]
    while filename not in mask_images:
      output = self.remove_background(file_path=self.INPUT_PATH)

      no_bg_path = f"{self.NO_BG_PATH}/{filename}"
      no_bg_path = f"{self.MASK_PATH}/{filename}"

      self.save_no_bg_image(self.INPUT_PATH, output, no_bg_path)

      self.create_binary_mask(filename, no_bg_path, no_bg_path)

      mask_images = [filename for filename in os.listdir(self.MASK_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]


  def run(self):
    mask_images = [filename for filename in os.listdir(self.MASK_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if self.BATCH:
      target_images = [filename for filename in os.listdir(self.INPUT_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
      self.run_batch(mask_images=mask_images, target_images=target_images)
    else:
      self.run_single(mask_images)


if __name__ == '__main__':
  mask_gen = LocalMaskGen()
  mask_gen.run()