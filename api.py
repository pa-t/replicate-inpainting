from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
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
  OverlayResponse,
)
from domain.enums import MaskGen


# initialize fastapi app
app = FastAPI()


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
    mask_image = local_mask_gen.create_binary_mask_endpoint(
      input_image=BytesIO(input_image_data)
    )
    no_background_image = local_mask_gen.remove_background(BytesIO(input_image_data))
    # TODO: write output to s3
    return ImageListResponse(output = ['path/to/s3.png'])
  elif mask_gen.value == MaskGen.REPLICATE.value:
    replicate_mask_gen = ReplicateMaskGen()
    # TODO: fix no_background_image
    mask_image, no_background_image = replicate_mask_gen.create_binary_mask_endpoint(
      input=BytesIO(input_image_data)
    )
    # TODO: write output to s3
    return ImageListResponse(output = ['path/to/s3.png'])


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
  if output:
    return ImageListResponse(output = output)
  else:
    raise HTTPException(500, detail="No output from model")


@app.post("/generate-background")
def generate_background(request: OverlayRequestGenerate):
  overlay = OverlayImage()
  output = overlay.generate_scenes_endpoint(prompt=request.prompt, num_outputs=request.num_outputs)
  return OverlayResponse(output = output)


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
  return ImageListResponse(output = output)


if __name__ == '__main__':
  uvicorn.run(app, host="0.0.0.0", port=8000)