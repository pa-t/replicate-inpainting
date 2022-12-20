from collections import defaultdict
import cv2
from io import BytesIO
import numpy as np
import os
from PIL import Image, ImageOps
import replicate
from time import perf_counter

from components.base_mask_gen import BaseMaskGen
from components.replicate_base import ReplicateBase, DICT_DEFAULT_VAL, def_value

class ReplicateMaskGen(ReplicateBase, BaseMaskGen):
  def __init__(self):
    super().__init__()
    self.model_name = "arielreplicate/dichotomous_image_segmentation"
    self.model_version_id = "69bd4043d3ff604dcf5abeb27e10d959d520f323cf990a188f072c578348c7fd"
    self.NUM_INFERENCE_STEPS = 25
    # initialize replicate model obj
    self.model = replicate.models.get(self.model_name)
    # get latest version (can be found with model.versions.list())
    self.version = self.model.versions.get(self.model_version_id)
  

  def get_filename_list(self):
    """
    get all image names that need a mask
    IMAGES AND MASKS MUST SHARE SAME NAME
    """
    filename_list = []

    for filename in os.listdir(self.INPUT_PATH):
      if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask = os.path.join(self.MASK_PATH, filename)
        image = os.path.join(self.INPUT_PATH, filename)
        # check if both paths are valid files
        if not os.path.isfile(mask) and os.path.isfile(image):
          filename_list.append(filename)
    
    return filename_list
  

  def run_pipeline(self, filename_list):
    """
    models will run in background so we don't have to wait for each prediction result
    can get later run model for each file found earlier
    """
    # default dict to hold 
    predictions = defaultdict(def_value)

    for filename in filename_list:
      if self.BATCH:
        curr_image_path = f"{self.INPUT_PATH}/{filename}"
      else:
        curr_image_path = filename
      with open(curr_image_path, "rb") as image:
        try:
          prediction = replicate.predictions.create(
            version=self.version,
            input={
              "input_image": image,
              "num_inference_steps": self.NUM_INFERENCE_STEPS
            }
          )
          predictions[filename] = prediction
          self.logger.info(f"prediction triggered for {filename}")
        except Exception as e:
          self.logger.exception(e)
          self.logger.info(f"exception calling prediction for {filename}: {e}")
    
    return predictions
  

  def remove_background(self, image_path, mask_path, no_bg_path):
    # read in original image and mask
    im = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    # subtract the images to keep what was masked out
    diff = cv2.subtract(im, mask)
    # make image background transparent instead of black
    tmp = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(diff)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    # write no background image to path
    cv2.imwrite(no_bg_path, dst)


  def write_output(self, filename_list, predictions):
    for filename in filename_list:
      curr_prediction = predictions[filename]
      # check prediciton in dict
      if curr_prediction != DICT_DEFAULT_VAL:
        if curr_prediction.status == "failed":
          self.logger.error(f"Error with replicate: {curr_prediction.error}")
        else:
          try:
            replicate_img = self.request_image(curr_prediction.output)
            inverted_image = ImageOps.invert(replicate_img)
            if self.BATCH:
              # if we are running as batch, use filenames and treat paths as directories
              image_path = f"{self.INPUT_PATH}/{filename}"
              new_mask_path = f"{self.MASK_PATH}/{filename}"
              no_bg_path = f"{self.NO_BG_PATH}/{filename}"
            else:
              # if we are running for single files, need to split the path and get filename
              short_filename = filename.split("/")[-1]
              # use filename as full path for initial image
              image_path = filename
              new_mask_path = f"{self.MASK_PATH}/{short_filename}"
              no_bg_path = f"{self.NO_BG_PATH}/{short_filename}"
            # write 
            self.logger.info(f"writing image: {new_mask_path}")
            inverted_image.save(new_mask_path)
            self.remove_background(image_path=image_path, mask_path=new_mask_path, output_path=no_bg_path)
          except Exception as e:
            self.logger.info(f"exception getting output from prediction: {curr_prediction.id}. Prediction status: {curr_prediction.status}, Output: {curr_prediction.output}")
            self.logger.exception(e)
  

  def create_binary_mask_endpoint(self, input: bytes):
    self.logger.info("opening image...")
    input_image = Image.open(input)
    prediction = replicate.predictions.create(
      version=self.version,
      input={
        "input_image": input,
        "num_inference_steps": self.NUM_INFERENCE_STEPS
      }
    )
    self.logger.info("waiting for predictions to complete...")
    start_time = perf_counter()
    prediction.wait()
    stop_time = perf_counter()
    self.logger.info(f"waited for predictions for {stop_time - start_time} seconds...")
    if prediction.status != 'succeeded':
      self.logger.error(f"Error from prediction pipeline: {prediction.error}")
    
    binary_mask_image = self.request_image(prediction.output)
    blank = input_image.point(lambda _: 0)
    segmented = Image.composite(input_image, blank, binary_mask_image.convert("1"))

    return binary_mask_image, segmented


  def wait_for_pipeline(self, predictions, filename_list):
    # need predictions in list form for some operations
    prediction_list = [value for value in predictions.values()]

    # wait for all predictions to complete
    self.wait_for_predictions(prediction_list=prediction_list)

    self.logger.info("predictions complete, fetching results...")

    self.write_output(filename_list=filename_list, predictions=predictions)
    
    # get all mask images now for comparison
    return [filename for filename in os.listdir(self.MASK_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

  
  def run_batch(self, mask_images, target_images):
    while len(mask_images) < len(target_images):
      # get all valid image filenames in list that need masks
      filename_list = self.get_filename_list()
        
      self.logger.info(f"found {len(filename_list)} valid image names needing masks...")

      # run replicate pipeline
      predictions = self.run_pipeline(
        filename_list=filename_list,
      )

      mask_images = self.wait_for_pipeline(predictions=predictions, filename_list=filename_list)

      
  
  def run_single(self, mask_images):
    # just looking for one image in the mask directory
    filename = self.INPUT_PATH.split("/")[-1]
    while filename not in mask_images:
      # run the replicate pipeline on the single image
      predictions = self.run_pipeline(
        filename_list=[self.INPUT_PATH],
      )

      # update mask_image list 
      mask_images = self.wait_for_pipeline(predictions=predictions, filename_list=[self.INPUT_PATH])


if __name__ == '__main__':
  mask_gen = ReplicateMaskGen()
  mask_gen.run()


