from collections import defaultdict
from datetime import datetime
from io import BytesIO
from PIL import Image
from time import perf_counter
import replicate
import requests

from components.replicate_base import ReplicateBase, DICT_DEFAULT_VAL, default_prompt, def_value

class ReplicateInPainting(ReplicateBase):
  model_name = "stability-ai/stable-diffusion-inpainting"
  model_version_id = "e5a34f913de0adc560d20e002c45ad43a80031b62caacc3d84010c6b6a64870c"
  NUM_IMG_OUTPUTS = 1
  PROMPT_STRENGTH = 0.8
  NUM_INFERENCE_STEPS = 25
  GUIDANCE_SCALE = 7.5

  def __init__(self):
    super().__init__()
    # initialize replicate model obj
    self.model = replicate.models.get(self.model_name)
    # get latest version (can be found with model.versions.list())
    self.version = self.model.versions.get(self.model_version_id)
    # store prompt for each file name, can be read in later
    self.prompt_dict = defaultdict(default_prompt)
    self.prompt_dict["image (60).png"] = "A peaceful lake nestled in a valley surrounded by the towering snowing mountains of the Alps, a mist is rising from the water with a golden sunrise illuminating the sky, photorealistic, 8k"
    self.prompt_dict["image (59).png"] = "A narrow cobblestone alley way in New York lined with brick buildings that are several stories tall with ivy climbing the walls and a small cafe with tables illuminated by an old-fashioned street lamp, photorealistic, 8k"
    self.prompt_dict["image (63).png"] = "A winding trail leads through a dense, verdant forest, with a stunning waterfall tumbling down a rocky cliff in the distance, photorealistic, 8k"
    self.prompt_dict["image (64).png"] = "A sandy beach with crystal-clear water and palm trees swaying in the breeze with a sunset casting a warm glow over the scene, photorealistic, 8k"
  

  def run_pipeline(self, filename_list, prompt_dict):
    """
    models will run in background so we don't have to wait for each prediction result
    can get later run model for each file found earlier
    """
    # default dict to hold 
    predictions = defaultdict(def_value)

    for filename in filename_list:
      with open(f"{self.IMAGE_DIR}/{filename}", "rb") as image, open(f"{self.MASK_IMAGE_DIR}/{filename}", "rb") as mask:
        try:
          prediction = replicate.predictions.create(
            version=self.version,
            input={
              "prompt": prompt_dict[filename],
              "image": image,
              "mask": mask,
              "prompt_strength": self.PROMPT_STRENGTH,
              "num_outputs": self.NUM_IMG_OUTPUTS,
              "num_inference_steps": self.NUM_INFERENCE_STEPS,
              "guidance_scale": self.GUIDANCE_SCALE,
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
        prediction_output = curr_prediction.output
        for output in prediction_output:
          # create filepath for new mask image
          new_mask_img_filepath = f"{self.OUTPUT_IMAGE_DIR}/{datetime.now().isoformat()}-{filename}"
          self.logger.info(f"writing image: {new_mask_img_filepath}")
          # retrieve output from prediction
          try:
            response = requests.get(output)
            img = Image.open(BytesIO(response.content))
            img.save(new_mask_img_filepath)
          except Exception as e:
            self.logger.info(f"exception writing {new_mask_img_filepath}")
            self.logger.exception(e)
  

  def run(self):
    # get all valid image filenames in list that have masks
    filename_list = self.get_filename_list()

    self.logger.info(f"found {len(filename_list)} valid image names with masks...")

    # run replicate pipeline
    predictions = self.run_pipeline(
      filename_list=filename_list,
      prompt_dict=self.prompt_dict
    )

    # need predictions in list form for some operations
    prediction_list = [value for value in predictions.values()]

    # wait for all predictions to complete
    self.wait_for_predictions(prediction_list=prediction_list)

    self.logger.info("predictions complete, fetching results...")

    self.write_output(filename_list=filename_list, predictions=predictions)
  

  def run_endpoint(
    self, 
    image: bytes,
    mask_image: bytes,
    prompt: str = "",
    num_outputs: int = 2,
  ):
    self.logger.info("reading in images...")
    try:
      img_tmp = BytesIO(image)
      mask_tmp = BytesIO(mask_image)
      prediction = replicate.predictions.create(
        version=self.version,
        input={
          "prompt": prompt,
          "image": img_tmp,
          "mask": mask_tmp,
          "prompt_strength": self.PROMPT_STRENGTH,
          "num_outputs": num_outputs,
          "num_inference_steps": self.NUM_INFERENCE_STEPS,
          "guidance_scale": self.GUIDANCE_SCALE,
        }
      )
      self.logger.info("waiting for pipeline to complete...")
      start_time = perf_counter()
      prediction.wait()
      stop_time = perf_counter()
      self.logger.info(f"waited for predictions for {stop_time - start_time} seconds...")
      if prediction.status != 'succeeded':
        self.logger.error(f"Error from prediction pipeline: {prediction.error}")
      return prediction.output
    except Exception as e:
      self.logger.error("Exception running inpaint prediction:")
      self.logger.exception(e)
    return None



if __name__ == '__main__':
  inpainting = ReplicateInPainting()
  inpainting.run()











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