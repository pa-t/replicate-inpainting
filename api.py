from io import BytesIO
import os
from PIL import Image
import requests
from uuid import uuid4 as uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from google.cloud import storage
import uvicorn

from components import (
  ReplicateMaskGen,
  ReplicateInPainting,
  LocalMaskGen,
  OverlayImage
)
from domain.schemas import (
  OverlayRequestGenerate,
  ImageListResponse,
)
from domain.enums import MaskGen


# initialize fastapi app
app = FastAPI()


# Google Cloud Provider constants
BUCKET_NAME = "width-image-bucket"
PROJECT_NAME = "triple-whale-staging"
SA_JSON_PATH = "triple-whale-staging-83e688a363fe.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = SA_JSON_PATH


# convert image to byte array and return
# used for upload byte data as string
def convert_image_to_bytes(image: Image):
  imgByteArr = BytesIO()
  image_format = 'PNG' if not image.format else image.format
  image.save(imgByteArr, format=image_format)
  return imgByteArr.getvalue()


# upload byte data as string to gcs bucket
# returns path to object
def upload_data_to_gcs(data, target_key):
  try:
    client = storage.Client(project=PROJECT_NAME)
    bucket = client.bucket(BUCKET_NAME)
    bucket.blob(target_key).upload_from_string(data)
    return bucket.blob(target_key).public_url
  except Exception as e:
    print(e)
  return None


# simple method used to download img from url
# and upload to gcs
def request_and_upload_image(url):
  try:
    response = requests.get(url)
    img_path = upload_data_to_gcs(
      data=response.content,
      target_key=f'{uuid()}.png'
    )
    if img_path:
      return img_path
  except Exception as e:
    print(e)
  return None


# calls convert and gcs upload functions for each image passed in a list
def convert_and_upload_images(images):
  image_paths = []
  for image in images:
    # convert image to bytes
    image_bytes = convert_image_to_bytes(image)
    # upload byte data to google cloud storage bucket
    image_path = upload_data_to_gcs(
      data=image_bytes,
      target_key=f'{uuid()}.png'
    )
    if image_path:
      image_paths.append(image_path)
  return image_paths

@app.get("/health")
def health():
  return "ok"


@app.post("/create-binary-mask")
async def create_binary_mask(
  input_image: UploadFile = File(...),
  mask_gen: MaskGen = MaskGen.LOCAL
):
  # read in image data
  input_image_data = await input_image.read()
  # check which mask gen to use
  if mask_gen.value == MaskGen.LOCAL.value:
    local_mask_gen = LocalMaskGen()
    # generate mask image
    mask_image = local_mask_gen.create_binary_mask_endpoint(
      input_image=BytesIO(input_image_data)
    )
    # generate no background image
    no_background_image = local_mask_gen.remove_background(
      BytesIO(input_image_data)
    )
    # convert images to bytes and upload to gcs, get image paths
    image_paths = convert_and_upload_images(images=[mask_image, no_background_image])
    return ImageListResponse(output = image_paths)
  elif mask_gen.value == MaskGen.REPLICATE.value:
    replicate_mask_gen = ReplicateMaskGen()
    # TODO: fix no_background_image (inverted)
    mask_image, no_background_image = replicate_mask_gen.create_binary_mask_endpoint(
      input=BytesIO(input_image_data)
    )
    # convert images to bytes and upload to gcs, get image paths
    image_paths = convert_and_upload_images(images=[mask_image, no_background_image])
    return ImageListResponse(output = image_paths)


@app.post("/infill-background")
async def infill_background(
  input_image: UploadFile = File(...),
  mask_image: UploadFile = File(...),
  prompt: str = "",
  num_outputs: int = 2,
):
  # read in images
  input_image_data = await input_image.read()
  mask_image_data = await mask_image.read()
  inpainter = ReplicateInPainting()
  output = inpainter.run_endpoint(
    image=input_image_data,
    mask_image=mask_image_data,
    prompt=prompt,
    num_outputs=num_outputs
  )

  image_paths = [request_and_upload_image(url) for url in output]
  if image_paths:
    return ImageListResponse(output=image_paths)
  else:
    raise HTTPException(500, detail="No output from model")


@app.post("/generate-background")
def generate_background(request: OverlayRequestGenerate):
  overlay = OverlayImage()
  output = overlay.generate_scenes_endpoint(prompt=request.prompt, num_outputs=request.num_outputs)
  image_paths = [request_and_upload_image(url) for url in output]
  return ImageListResponse(output = image_paths)


@app.post("/overlay-image")
async def overlay_image(
  background_file: UploadFile = File(...),
  foreground_file: UploadFile = File(...),
  x_pos: int = 0,
  y_pos: int = 0,
):
  # read in images
  background_file_data = await background_file.read()
  foreground_file_data = await foreground_file.read()
  # start overlay process
  overlay = OverlayImage()
  output = overlay.overlay_image_endpoint(
    background_img=background_file_data,
    foreground_img=foreground_file_data,
    x_pos=x_pos,
    y_pos=y_pos
  )
  # convert image to bytes and upload to gcs, get image path
  image_paths = convert_and_upload_images(images=[output])
  return ImageListResponse(output = image_paths)


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8000)