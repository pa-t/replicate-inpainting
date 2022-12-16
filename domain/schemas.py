from pydantic import BaseModel
from typing import List


class OverlayRequestGenerate(BaseModel):
    """schema to be used for generating background images
    using the overlay module implementation of stable diffusion
    """
    prompt: str
    num_outputs: int = 1


class ImageListResponse(BaseModel):
    """schema for returning a list of urls to generated images
    from stable diffusion
    """
    output: List[str]


class OverlayResponse(BaseModel):
    output: str

