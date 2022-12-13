from collections import defaultdict
import cv2
from io import BytesIO
import os
from PIL import Image, ImageOps
import replicate
import requests

from replicate_base import ReplicateBase, DICT_DEFAULT_VAL, def_value

class ReplicateMaskGen(ReplicateBase):
  def __init__(self):
    super().__init__()
    self.model_name = "arielreplicate/dichotomous_image_segmentation"
    self.model_version_id = "69bd4043d3ff604dcf5abeb27e10d959d520f323cf990a188f072c578348c7fd"
    self.NUM_INFERENCE_STEPS = 25
    # initialize replicate model obj
    self.model = replicate.models.get(self.model_name)
    # get latest version (can be found with model.versions.list())
    self.version = self.model.versions.get(self.model_version_id)
  

  def set_constants(self, batch: bool, input_path: str, no_bg_path: str, mask_path: str):
    """set constants used during mask generation"""
    self.BATCH = batch
    self.INPUT_PATH = input_path
    self.NO_BG_PATH = no_bg_path
    self.MASK_PATH = mask_path
  

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
  

  def remove_background(self, image_path, mask_path, output_path):
    im = cv2.imread(image_path)
    mask = cv2.imread(mask_path)
    diff = cv2.subtract(im, mask)
    tmp = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(diff)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite(output_path, dst)


  def write_output(self, filename_list, predictions):
    for filename in filename_list:
      curr_prediction = predictions[filename]
      # check prediciton in dict
      if curr_prediction != DICT_DEFAULT_VAL:
        if curr_prediction.status == "failed":
          self.logger.error(f"Error with replicate: {curr_prediction.error}")
        else:
          try:
            response = requests.get(curr_prediction.output)
            img = Image.open(BytesIO(response.content))
            inverted_image = ImageOps.invert(img)
            if self.BATCH:
              image_path = f"{self.INPUT_PATH}/{filename}"
              new_mask_img_filepath = f"{self.MASK_PATH}/{filename}"
              output_path = f"{self.NO_BG_PATH}/{filename}"
            else:
              short_filename = filename.split("/")[-1]
              image_path = filename
              new_mask_img_filepath = f"{self.MASK_PATH}/{short_filename}"
              output_path = f"{self.NO_BG_PATH}/{short_filename}"
            self.logger.info(f"writing image: {new_mask_img_filepath}")
            inverted_image.save(new_mask_img_filepath)
            self.remove_background(image_path=image_path, mask_path=new_mask_img_filepath, output_path=output_path)
          except Exception as e:
            self.logger.info(f"exception getting output from prediction: {curr_prediction.id}. Prediction status: {curr_prediction.status}, Output: {curr_prediction.output}")
            self.logger.exception(e)
  

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



  def run(self):
    mask_images = [filename for filename in os.listdir(self.MASK_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if self.BATCH:
      target_images = [filename for filename in os.listdir(self.INPUT_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
      self.run_batch(mask_images=mask_images, target_images=target_images)
    else:
      self.run_single(mask_images)
    
    self.logger.info(f"{len(mask_images)} masks found for {len(target_images)}")


if __name__ == '__main__':
  mask_gen = ReplicateMaskGen()
  mask_gen.run()


