from datetime import datetime
import requests
from io import BytesIO
from PIL import Image
import replicate

from replicate_base import ReplicateBase


class OverlayImage(ReplicateBase):
  def __init__(self):
    super().__init__()
    self.model_name = "stability-ai/stable-diffusion"
    self.model = replicate.models.get(self.model_name)
    self.model_version_id = self.model.versions.list()[0].id
    self.version = self.model.versions.get(self.model_version_id)
    self.NUM_OUTPUTS = 3


  def generate_scenes(self):
    prediction = replicate.predictions.create(
      version=self.version,
      input={
        "prompt": "A peaceful lake nestled in a valley surrounded by the towering snowing mountains of the Alps, a mist is rising from the water with a golden sunrise illuminating the sky, photorealistic, 8k",
        "width": 768,
        "height": 768,
        "prompt_strength": 0.8,
        "num_outputs": self.NUM_OUTPUTS,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "scheduler": "K_EULER"
      }
    )
    prediction.wait()
    if self.NUM_OUTPUTS > 1:
      for output in prediction.output:
        scene_img = self.request_image(output)
        scene_img.save(f"scenes/{datetime.now().isoformat()}.png")
    else:
      scene_img = self.request_image(prediction.output)
      scene_img.save(f"scenes/{datetime.now().isoformat()}.png")


background_image = Image.open("scenes/lake_scene_1.png")
foreground_image = Image.open("no-bg-images/image (60).png").resize(background_image.size)

back_im = background_image.copy()
back_im.paste(foreground_image, (100, 50), mask=foreground_image)
back_im.save("output-images/lake_scene_1.png")