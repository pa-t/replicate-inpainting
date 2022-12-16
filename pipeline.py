import argparse
import logging

from components.local_mask_generate import LocalMaskGen
from components.replicate_inpaint import ReplicateInPainting
from components.replicate_mask_generate import ReplicateMaskGen
from components.overlay_image import OverlayImage


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
  parser = argparse.ArgumentParser(description='Stable diffusion pipeline')
  # mask args
  parser.add_argument('--mask', type=str, default='', help='[MASK] `local` or `replicate` mask generator?')
  parser.add_argument('--batch', action='store_true', help="[MASK] Enable batch run to run against multiple files in a directory")
  parser.add_argument('--single-file', dest='batch', action='store_false', help="[MASK] Disable batch run and run against a specific path")
  parser.set_defaults(batch=True)
  parser.add_argument('--input-path', type=str, default='background-images', help='[MASK] Path to input image(s)')
  parser.add_argument('--no-bg-path', type=str, default='no-bg-images', help='[MASK] Path to no background image(s)')
  parser.add_argument('--mask-path', type=str, default='mask-images', help='[MASK] Path to mask image(s)')
  # inpainting args
  parser.add_argument('--inpainting', action='store_true', help="[In-Painting] Enable inpainting to run")
  # parser.add_argument('--no-inpainting', dest='inpainting', action='store_false', help="[In-Painting] Disable inpainting from running")
  # parser.set_defaults(inpainting=False)
  # overlay args
  parser.add_argument('--overlay', action='store_true', help='[Overlay] run the overlay module')
  parser.set_defaults(overlay=False)
  parser.add_argument('--generate', action='store_true', help='[Overlay] disable image generation for overlay module')
  parser.set_defaults(generate=True)
  parser.add_argument('--no-generate', dest='generate', action='store_false', help='[Overlay] disable image generation for overlay module')
  parser.add_argument('--prompt', type=str, default='River in a valley surrounded by mountains', help='[Overlay] prompt for image generation')
  parser.add_argument('--num-outputs', type=int, default=3, help='[Overlay] number of outputs for image generation')
  parser.add_argument('--x-pos', type=int, default=0, help='[Overlay] X coordinate for placing overlain image')
  parser.add_argument('--y-pos', type=int, default=0, help='[Overlay] Y coordinate for placing overlain image')
  parser.add_argument('--background-path', type=str, default=None, help='[Overlay] Path to background image for overlaying')
  parser.add_argument('--foreground-path', type=str, default=None, help='[Overlay] Path to foreground image for overlaying')
  parser.add_argument('--output-path', type=str, default=None, help='[Overlay] Path to output image from overlaying')
  args = parser.parse_args()


  # set to None for later check
  mask_gen = None

  if args.mask.lower() == 'local':
    logger.info("using local mask generator...")
    mask_gen = LocalMaskGen()
  elif args.mask.lower() == 'replicate':
    logger.info("using replicate hosted mask generator...")
    mask_gen = ReplicateMaskGen()
  else:
    logger.warning("`--mask` argument not set to either `local` or `replicate`, not generating masks")

  if mask_gen:
    try:
      mask_gen.set_constants(
        batch=args.batch,
        input_path=args.input_path,
        no_bg_path=args.no_bg_path,
        mask_path=args.mask_path
      )
      mask_gen.run()
    except Exception as e:
      logger.exception(e)

  
  if args.inpainting:
    logger.info("inpainting enabled...")
    inpainter = ReplicateInPainting()
    logger.info("starting inpainting...")
    inpainter.run()
  else:
    logger.warning("`--inpainting` argument not set, not running inpaiting module")
  

  if args.overlay:
    overlay = OverlayImage()
    if args.generate:
      _ = overlay.generate_scenes(prompt=args.prompt, num_outputs=args.num_outputs)
    if args.background_path and args.foreground_path and args.output_path:
      _ = overlay.overlay_image(
        background_path=args.background_path,
        foreground_path=args.foreground_path,
        output_path=args.output_path,
        x_pos=args.x_pos,
        y_pos=args.y_pos
      )
    else:
      logger.warning("need to set --background-path, --foreground-path, and --output-path to overlay images")
  else:
    logger.warning("`--overlay` argument not set, not running overlay module")



if __name__ == '__main__':
  main()