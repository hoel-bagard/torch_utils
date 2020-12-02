from .data_config import DataConfig
from src.networks.build_network import ModelHelper


class ModelConfig:
    # Training parameters
    BATCH_SIZE         = 4        # Batch size
    MAX_EPOCHS         = 2000     # Number of Epochs
    BUFFER_SIZE        = 256      # Buffer Size, used for the shuffling
    LR                 = 4e-4     # Learning Rate
    LR_DECAY           = 0.997
    DECAY_START        = 10
    REG_FACTOR         = 0.005    # Regularization factor (Used to be 0.005 for the fit mode)

    IMAGE_SIZES    = (256, 256)   # All images will be resized to this size
    OUTPUT_CLASSES = len(DataConfig.LABEL_MAP)
    N_TO_N         = True         # If True then there must be a label for each frame

    # Network part
    MODEL           = ModelHelper.Transformer
    NETWORK         = MODEL.__name__  # TODO: remove this if possible
    USE_GRAY_SCALE  = False  # Cannot be used if using the DALI dataloader
    # First channel should be the number of channels in the input image,
    # there should be one more value in CHANNELS compared to the other list
    CHANNELS        = [1 if USE_GRAY_SCALE else 3, 24, 32, 64, 64, 48, 32, 24, 16, 12]
    SIZES           = [3, 3, 3, 3, 3, 3, 3, 3, 3]   # Kernel sizes
    STRIDES         = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    PADDINGS        = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    NB_BLOCKS       = [2, 2, 2, 1, 1, 1, 1, 1]
    SEQUENCE_LENGTH = 15   # Video size / Number of frames in each sample
    WORKERS         = 8    # Number of workers for PyTorch dataloader
