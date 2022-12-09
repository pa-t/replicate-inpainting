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
  parser.add_argument('--inpainting', type=bool, default=True, help="True/False, enable or disable inpainting to run")
  args = parser.parse_args()

  if args.mask.lower() == 'local':
    logger.info("using local mask generator...")
    local_mask_gen = LocalMaskGen()
    local_mask_gen.run()
  elif args.mask.lower() == 'replicate':
    logger.info("using replicate hosted mask generator...")
    replicate_mask_gen = ReplicateMaskGen()
    replicate_mask_gen.run()
  else:
    logger.warning("`--mask` argument must either be `local` or `replicate`")
  
  if args.inpainting:
    logger.info("inpainting enabled...")
    inpainter = ReplicateInPainting()
    logger.info("starting inpainting...")
    inpainter.run()


if __name__ == '__main__':
  main()