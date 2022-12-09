import logging
from time import perf_counter
import os

DICT_DEFAULT_VAL = "Not Present"


# set up defaultdict default value
def def_value():
  return DICT_DEFAULT_VAL


def default_prompt():
  return "The snow-capped peaks of the Rocky Mountains rise majestically above a valley blanketed in powdery snow with a frozen river glinting in the sunshine, photorealistic, 8k, high resolution"


class ReplicateBase:
  IMAGE_DIR = "background-images"
  MASK_IMAGE_DIR = "mask-images"
  OUTPUT_IMAGE_DIR = "output-images"
  logging.basicConfig()
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.INFO)

  def __init__(self):
    pass

  def get_filename_list(self):
    """
    create a list of filenames to be used later. add filenames to list only
    if both mask and image exist
    """
    filename_list = []

    for filename in os.listdir(self.MASK_IMAGE_DIR):
      if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        mask = os.path.join(self.MASK_IMAGE_DIR, filename)
        image = os.path.join(self.IMAGE_DIR, filename)
        # check if both paths are valid files
        if os.path.isfile(mask) and os.path.isfile(image):
          filename_list.append(filename)
    
    return filename_list

  def run_pipeline(self, filename_list):
    raise NotImplemented

  def wait_for_predictions(self, prediction_list):
    """
    check that all predictions are complete in list
    """
    start_time = perf_counter()

    # wait for first prediction to be finished
    prediction_list[0].wait()

    # other predictions should be close since first has finished
    waiting = True
    while waiting:
      waiting = False
      for prediction in prediction_list:
        # reload model to get most recent
        prediction.reload()
        # only check for processing state, if in succeeded or other state
        # this will be considered 'done'
        if prediction.status == 'processing' or prediction.status == 'starting':
          waiting = True

    stop_time = perf_counter()
    self.logger.info(f"waited for predictions for {stop_time - start_time} seconds...")

  def write_output(self, filename_list, predictions):
    raise NotImplemented
  
  def run(self):
    raise NotImplemented