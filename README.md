# replicate-inpainting

## Getting started
Install requirements
```
pip install -r requirements.txt
```

Get token from [replicate.com/account](https://replicate.com/account) and set environment variable:
```
export REPLICATE_API_TOKEN=[token]
```


For now the prompts are stored in a dict in the `__init__()` method in `replicate_inpaint.py` where the key is the image file name and the value is the prompt, eventually read this in from elsewhere



## File structure
Need to have the following directories
```
background-images/
mask-images/
no-bg-images/
output-images/
```
Need to add images to the `background-images` directory and the corresponding mask images to `mask-images` with the same name in each directory


## Run pipeline
```
python3 pipeline.py

-- or --

python3 pipeline.py --mask replicate

-- or --

python3 pipeline.py --mask local ---no-inpainting

ARGUMENTS:
  -  `--mask` : default='local', options=['local', 'replicate'], specify which mask generator to use
  -  `--batch` : Runs mask generation as batch (against directories)
  -  `--single-file` : Runs mask generation against a single file
  -  `--input-path` : default="background-images", Path for mask generation input
  -  `--no-bg-path` : default="no-bg-images", Path for mask generation to write no background images to
  -  `--mask-path` : default="mask-images", Path for mask generation to write masks to
  -  `--inpainting` : default=True, Enable inpainting to run
  <!-- -  `--no-inpainting` : default=True, Disable inpainting to run -->
  -  `--overlay` : Run overlay, disabled by default
  -  `--generate` : Run image generation in overlay module (stable diffusion model)
  -  `--no-generate` : Disbale image generation in overlay module
  -  `--prompt` : Prompt for image generation in overlay module
  -  `--num-outputs` : Number of outputs from generation in overlay module
  -  `--x-pos` : default=0, X position to place overlain image
  -  `--y-pos` : default=0, Y position to place overlain image
  -  `--background-path` : Path to background image for overlaying
  -  `--foreground-path` : Path to foreground image for overlaying
  -  `--output-path` :  Path to output image from overlaying
```


## Individually run generate mask and no background images
```
python3 local_mask_generate.py
```
Not perfect, but does a good job at removing most backgrounds and masking the subject. The subject needs to be clear or else it wont give good results and does it all locally

Can also use the `arielreplicate/dichotomous_image_segmentation` model hosted on replicate
```
python3 replicate_mask_generate.py
```


## Individually running inpainting script
```
python3 replicate_inpaint.py
```



## Run Overlay image
Two parts to the `OverlayImage` in `overlay_image.py`
- `generate_scenes()` : generate background scenes using stable diffusion
- `overlay_image()` : overlay two images and then write to a specific output


Generating scene
```
python3 pipeline.py --overlay --generate --prompt "Summer time on the water in the south of france, sun is glistening on the water and a road winds around the coastline, in the style of an oil painting" --num-outputs 3
```


Overlaying images
```
python3 pipeline.py --overlay --no-generate --background-path 'scenes/lake_scene_1.png' --foreground-path 'no-bg-images/image (60).png' --output-path 'output-images/lake_scene_1.png' --x-pos 100 --y-pos 100
```


## More resources
[replicate python docs](https://github.com/replicate/replicate-python#readme)

[replicate web ui](https://replicate.com/stability-ai/stable-diffusion-inpainting)