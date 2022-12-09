from collections import defaultdict
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
  

  def get_filename_list(self):
    """
    get all image names that need a mask
    IMAGES AND MASKS MUST SHARE SAME NAME
    """
    filename_list = []

    for filename in os.listdir(self.IMAGE_DIR):
      if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask = os.path.join(self.MASK_IMAGE_DIR, filename)
        image = os.path.join(self.IMAGE_DIR, filename)
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
      with open(f"{self.IMAGE_DIR}/{filename}", "rb") as image:
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


  def write_output(self, filename_list, predictions):
    for filename in filename_list:
      curr_prediction = predictions[filename]
      # check prediciton in dict
      if curr_prediction != DICT_DEFAULT_VAL:
        try:
          new_mask_img_filepath = f"{self.MASK_IMAGE_DIR}/{filename}"
          self.logger.info(f"writing image: {new_mask_img_filepath}")
          response = requests.get(curr_prediction.output)
          img = Image.open(BytesIO(response.content))
          inverted_image = ImageOps.invert(img)
          inverted_image.save(new_mask_img_filepath)
        except Exception as e:
          self.logger.info(f"exception getting output from prediction: {curr_prediction.id}. Prediction status: {curr_prediction.status}, Output: {curr_prediction.output}")
          self.logger.exception(e)
  

  def run(self):
    mask_images = [filename for filename in os.listdir(self.MASK_IMAGE_DIR) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    target_images = [filename for filename in os.listdir(self.IMAGE_DIR) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    while len(mask_images) < len(target_images):
      # get all valid image filenames in list that need masks
      filename_list = self.get_filename_list()

      self.logger.info(f"found {len(filename_list)} valid image names needing masks...")

      # run replicate pipeline
      predictions = self.run_pipeline(
        filename_list=filename_list,
      )

      # need predictions in list form for some operations
      prediction_list = [value for value in predictions.values()]

      # wait for all predictions to complete
      self.wait_for_predictions(prediction_list=prediction_list)

      self.logger.info("predictions complete, fetching results...")

      self.write_output(filename_list=filename_list, predictions=predictions)
      
      # get all mask images now for comparison
      mask_images = [filename for filename in os.listdir(self.MASK_IMAGE_DIR) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    self.logger.info(f"{len(mask_images)} masks found for {len(target_images)}")


if __name__ == '__main__':
  mask_gen = ReplicateMaskGen()
  mask_gen.run()


