import argparse
import logging

from local_mask_generate import LocalMaskGen
from replicate_inpaint import ReplicateInPainting
from replicate_mask_generate import ReplicateMaskGen


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('--mask', type=str, default='local', help='`local` or `replicate` mask generator?')
  parser.add_argument('--batch', action='store_true', help="Enable batch run to run against multiple files in a directory")
  parser.add_argument('--single-file', dest='batch', action='store_false', help="Disable batch run and run against a specific path")
  parser.set_defaults(batch=True)
  parser.add_argument('--input-path', type=str, default='background-images', help='Path to input image(s)')
  parser.add_argument('--no-bg-path', type=str, default='no-bg-images', help='Path to no background image(s)')
  parser.add_argument('--mask-path', type=str, default='mask-images', help='Path to mask image(s)')
  parser.add_argument('--inpainting', action='store_true', help="Enable inpainting to run")
  parser.add_argument('--no-inpainting', dest='inpainting', action='store_false', help="Disable inpainting from running")
  parser.set_defaults(inpainting=True)
  args = parser.parse_args()

  if args.mask.lower() == 'local':
    logger.info("using local mask generator...")
    mask_gen = LocalMaskGen()
  elif args.mask.lower() == 'replicate':
    logger.info("using replicate hosted mask generator...")
    mask_gen = ReplicateMaskGen()
  else:
    logger.warning("`--mask` argument were not set to either `local` or `replicate`, not generating masks")

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


if __name__ == '__main__':
  main()