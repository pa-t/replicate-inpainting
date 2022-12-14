import os

class BaseMaskGen:
  def __init__(self):
    self.BATCH = None
    self.MASK_PATH = None
    self.INPUT_PATH = None


  def set_constants(self, batch: bool, input_path: str, no_bg_path: str, mask_path: str):
    """set constants used during mask generation"""
    self.BATCH = batch
    self.INPUT_PATH = input_path
    self.NO_BG_PATH = no_bg_path
    self.MASK_PATH = mask_path


  def run(self):
    mask_images = [filename for filename in os.listdir(self.MASK_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if self.BATCH:
      target_images = [filename for filename in os.listdir(self.INPUT_PATH) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]
      self.run_batch(mask_images=mask_images, target_images=target_images)
    else:
      self.run_single(mask_images)
    
    self.logger.info(f"{len(mask_images)} masks found for {len(target_images)}")