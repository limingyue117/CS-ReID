import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = '/public/home/G19830015/VideoGroup/lmy/TransReID-1/data'
# Dataset for evaluation
_C.DATA.DATASET = 'market1501'
# Split index
_C.DATA.SPLIT_ID = 0
# Whether to use labeled images, if false, detected images are used
_C.DATA.CUHK03_LABELED = False
# Whether to use classic split by Li et al. CVPR'14 (default: False)
_C.DATA.CUHK03_CLASSIC_SPLIT = False
# Workers for dataloader
_C.DATA.NUM_WORKERS = 4
# Height of input image
_C.DATA.HEIGHT = 256
# Width of input image
_C.DATA.WIDTH = 128
# Batch size for training
_C.DATA.TRAIN_BATCH = 64
# Batch size for testing
_C.DATA.TEST_BATCH = 1024
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 8
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'resnet50'
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim
_C.MODEL.FEATURE_DIM = 4096
# Choose in ['avg', 'max', 'gem', 'maxavg']
_C.MODEL.POOLINGNAME = 'maxavg'
# Model path for resuming
_C.MODEL.RESUME = ''
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropy'
_C.LOSS.CENTER_LOSS = False
_C.LOSS.CENTER_LOSS_WEIGHT = 0.0005
# Scale
_C.LOSS.CLA_S = 16.
# Margin
_C.LOSS.CLA_M = 0.
# Pairwise loss
_C.LOSS.PAIR_LOSS = 'triplet'
_C.LOSS.MA = 0.0
# Scale
_C.LOSS.PAIR_S = 16.
# Margin
_C.LOSS.PAIR_M = 0.3
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.method = 'cal'
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 60
_C.TRAIN.lr_type = 'step'
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [20, 40]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Similarity for testing
_C.TEST.DISTANCE = 'cosine'
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 5
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 2
# Perform evaluation only
_C.EVAL_MODE = False
_C.SAVE_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = './logs/'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'res50-ce-tri'


def update_config(config, args):
    config.defrost()

    print('=> merge config from {}'.format(args.cfg))
    config.merge_from_file(args.cfg)

    # merge from specific arguments
    if args.root:
        config.DATA.ROOT = args.root
    if args.output:
        config.OUTPUT = args.output

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.gpu:
        config.GPU = args.gpu

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
