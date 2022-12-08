# replicate-inpainting

## Getting started
Install `replicate` python library
```
pip install replicate
```

Get token from [replicate.com/account](https://replicate.com/account) and set environment variable:
```
export REPLICATE_API_TOKEN=[token]
```

Need to add images to the `background-images` directory and the corresponding mask images to `mask-images` with the same name in each directory


For now the prompts are stored in a dict in the `main()` method in `main.py` where the key is the image file name and the value is the prompt, eventually read this in from elsewhere


## Running
```
python3 main.py
```

## More resources
[replicate python docs](https://github.com/replicate/replicate-python#readme)
[replicate web ui](https://replicate.com/stability-ai/stable-diffusion-inpainting)