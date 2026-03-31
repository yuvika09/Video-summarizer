import argparse
import datetime
import logging
import random
from pathlib import Path
import numpy as np
import torch

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_logger(log_dir: str) -> None:
    """Initialize logger with file and console handlers"""
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    now_time = now.strftime('log_%Y-%m-%d-%H-%M-%S.log')
    fh = logging.FileHandler(str(log_dir / now_time))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)

def get_parser() -> argparse.ArgumentParser:
    """Create argument parser with all parameters"""
    parser = argparse.ArgumentParser(description='STeMI: Few-shot Video Summarization')
    
    # Basic settings
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=['splits/tvsum_few_shot.yml'])
    parser.add_argument('--max-epoch', type=int, default=300)
    parser.add_argument('--model-dir', type=str, default='models/tvsum')
    
    # Optimizer settings
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    
    # Model architecture
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--temporal_scales', type=int, default=4)
    parser.add_argument('--spatial_scales', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Original loss weights
    parser.add_argument('--lambda-reg', type=float, default=1.0)
    parser.add_argument('--lambda-ctr', type=float, default=1.0)
    parser.add_argument('--lambda_rec_x', type=float, default=1.0)
    parser.add_argument('--lambda_rec_s', type=float, default=1.0)
    
    # New loss weights
    parser.add_argument('--lambda-diversity', type=float, default=0.01)
    parser.add_argument('--lambda-rep', type=float, default=0.01)
    parser.add_argument('--lambda-sparsity', type=float, default=0.05)
    parser.add_argument('--lambda-smooth', type=float, default=0.01)
    parser.add_argument('--lambda-consistency', type=float, default=0.05)
    
    # Loss options
    parser.add_argument('--use-focal-loss', action='store_true', default=False)
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--sparsity-target', type=float, default=0.15)
    
    # Training options
    parser.add_argument('--early-stopping', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    
    # Evaluation settings
    parser.add_argument('--nms-thresh', type=float, default=0.4)
    parser.add_argument('--sample-rate', type=int, default=15)
    
    # Checkpoint settings
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--save-path', type=str, default=None)
    
    # Scheduler settings
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'cosine_warm_restarts', 'step', 'plateau', 'none'])
    parser.add_argument('--scheduler-t0', type=int, default=50)
    parser.add_argument('--scheduler-tmult', type=int, default=2)
    parser.add_argument('--scheduler-eta-min', type=float, default=1e-6)
    
    return parser

def get_arguments() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = get_parser()
    args = parser.parse_args()
    return args
