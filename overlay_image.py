from datetime import datetime
import requests
from io import BytesIO
from PIL import Image
import replicate
from time import perf_counter

from replicate_base import ReplicateBase


class OverlayImage(ReplicateBase):
  def __init__(self):
    super().__init__()
    self.model_name = "stability-ai/stable-diffusion"
    self.model = replicate.models.get(self.model_name)
    self.model_version_id = self.model.versions.list()[0].id
    self.version = self.model.versions.get(self.model_version_id)


  def generate_scenes(
    self,
    prompt: str = "A peaceful lake nestled in a valley surrounded by the towering snowing mountains of the Alps, a mist is rising from the water with a golden sunrise illuminating the sky, photorealistic, 8k",
    num_outputs : int = 3
  ):
    self.logger.info(f"generating {num_outputs} for the prompt: {prompt}")
    prediction = replicate.predictions.create(
      version=self.version,
      input={
        "prompt": prompt,
        "width": 768,
        "height": 768,
        "prompt_strength": 0.8,
        "num_outputs": num_outputs,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "scheduler": "K_EULER"
      }
    )
    self.logger.info("waiting for pipeline to complete...")
    start_time = perf_counter()
    prediction.wait()
    stop_time = perf_counter()
    self.logger.info(f"waited for predictions for {stop_time - start_time} seconds...")
    if num_outputs > 1:
      for output in prediction.output:
        scene_img = self.request_image(output)
        path = f"scenes/{datetime.now().isoformat()}.png"
        self.logger.info(f"writing generated scene to: {path}")
        scene_img.save(path)
    else:
      scene_img = self.request_image(prediction.output)
      path = f"scenes/{datetime.now().isoformat()}.png"
      self.logger.info(f"writing generated scene to: {path}")
      scene_img.save(path)
  

  def overlay_image(
    self,
    background_path: str,
    foreground_path: str,
    output_path: str,
    x_pos: int = 100,
    y_pos: int = 50,
  ):
    self.logger.info("reading in images...")
    background_image = Image.open(background_path)
    foreground_image = Image.open(foreground_path).resize(background_image.size)

    back_im = background_image.copy()
    self.logger.info(f"overlaying image: {foreground_path} over: {background_path} at position ({x_pos}, {y_pos})")
    back_im.paste(foreground_image, (x_pos, y_pos), mask=foreground_image)
    self.logger.info(f"writing overlain image to {output_path}")
    back_im.save(output_path)
  

if __name__ == "__main__":
  overlay = OverlayImage()
  overlay.overlay_image(
    background_path="scenes/lake_scene_1.png",
    foreground_path="no-bg-images/image (60).png",
    output_path="output-images/lake_scene_1.png"
  )
