from collections import defaultdict
from datetime import datetime
from io import BytesIO
import logging
from time import perf_counter
import os
import requests

import replicate
from PIL import Image

"""
set the following env var with your correct token:

export REPLICATE_API_TOKEN=<token>
"""

# set up logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# set constants
DICT_DEFAULT_VAL = "Not Present"
IMAGE_DIR = "background-images"
MASK_IMAGE_DIR = "mask-images"
OUTPUT_IMAGE_DIR = "output-images"


# set up defaultdict default value
def def_value():
  return DICT_DEFAULT_VAL


# get all image names that have a mask
# IMAGES AND MASKS MUST SHARE SAME NAME
def get_filename_list():
  filename_list = []

  for filename in os.listdir(MASK_IMAGE_DIR):
    mask = os.path.join(MASK_IMAGE_DIR, filename)
    image = os.path.join(IMAGE_DIR, filename)
    # check if both paths are valid files
    if os.path.isfile(mask) and os.path.isfile(image):
      filename_list.append(filename)
  
  return filename_list


# models will run in background so we don't have to wait for each prediction result, can get later
# run model for each file found earlier
def run_pipeline(model_version, filename_list, prompt_dict):
  # default dict to hold 
  predictions = defaultdict(def_value)

  for filename in filename_list:
    with open(f"{IMAGE_DIR}/{filename}", "rb") as image, open(f"{MASK_IMAGE_DIR}/{filename}", "rb") as mask:
      try:
        prediction = replicate.predictions.create(
          version=model_version,
          input={
            "prompt": prompt_dict[filename],
            "image": image,
            "mask": mask,
            "prompt_strength": 0.8,
            "num_outputs": 3,
            "num_ingerence_steps": 25,
            "guidance_scale": 7.5,  
          }
        )
        predictions[filename] = prediction
        logger.info(f"prediction triggered for {filename}")
      except Exception as e:
        logger.exception(e)
        logger.info(f"exception calling prediction for {filename}: {e}")
  
  return predictions


# check that all predictions are complete 
def wait_for_predictions(prediction_list):
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
      if prediction.status == 'processing':
        waiting = True

  stop_time = perf_counter()
  logger.info(f"waited for predictions for {stop_time - start_time} seconds...")


# write outputs of predictions to output dir
def write_outputs(filename_list, predictions):
  for filename in filename_list:
    curr_prediction = predictions[filename]
    # check prediciton in dict
    if curr_prediction != DICT_DEFAULT_VAL:
      prediction_output = curr_prediction.output
      for output in prediction_output:
        # wrap this in retry
        response = requests.get(output)
        img = Image.open(BytesIO(response.content))
        img.save(f"{OUTPUT_IMAGE_DIR}/{datetime.now().isoformat()}-{filename}")


def main():
  # initialize replicate model obj
  model = replicate.models.get("stability-ai/stable-diffusion-inpainting")
  # get latest version (can be found with model.versions.list())
  version = model.versions.get("e5a34f913de0adc560d20e002c45ad43a80031b62caacc3d84010c6b6a64870c")

  # store prompt(s) for each file name, can be read in later
  prompt_dict = {
    "image (60).png": "A peaceful lake nestled in a valley surrounded by the towering snowing mountains of the Alps, a mist is rising from the water with a golden sunrise illuminating the sky, photorealistic, 8k",
    "image (64).png": "A sandy beach with crystal-clear water and palm trees swaying in the breeze with a sunset casting a warm glow over the scene, photorealistic, 8k"
  }

  # get all valid image filenames in list that have masks
  filename_list = get_filename_list()

  logger.info(f"found {len(filename_list)} valid image names with masks...")

  # run replicate pipeline
  predictions = run_pipeline(
    model_version=version,
    filename_list=filename_list,
    prompt_dict=prompt_dict
  )

  # need predictions in list form for some operations
  prediction_list = [value for value in predictions.values()]

  # wait for all predictions to complete
  wait_for_predictions(prediction_list=prediction_list)

  logger.info("predictions complete, fetching results...")

  write_outputs(filename_list=filename_list, predictions=predictions)


if __name__ == '__main__':
  main()











# can also just call model.predict() if we want to wait on each run
# model_output = model.predict(
#   prompt="",
#   image=image,
#   mask=mask,
#   prompt_strength=0.8,
#   num_outputs=3,
#   num_ingerence_steps=25,
#   guidance_scale=7.5,
# )
# for output in model_output:
#   # wrap this in retry
#   response = requests.get(output)
#   img = Image.open(BytesIO(response.content))
#   img.save(f"output-images/image (60) - {datetime.now().isoformat()}.png")
  # img.show()