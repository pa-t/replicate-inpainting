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

python3 pipeline.py --mask local --inpainting False

ARGUMENTS:
  -  `--mask` : default='local', options=['local', 'replicate'], specify which mask generator to use
  -  `--inpainting` : default=True, options=[True, False], run inpainting or not
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

## More resources
[replicate python docs](https://github.com/replicate/replicate-python#readme)

[replicate web ui](https://replicate.com/stability-ai/stable-diffusion-inpainting)