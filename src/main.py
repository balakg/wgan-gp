import os
import argparse
from solver import Solver
from torch.backends import cudnn
import torch


def call(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    solver = Solver(config)
    solver.train() if config.train  else solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--z_dim', type=int, default=1024, help='latent z dimension')   
    parser.add_argument('--im_size', type=int, default=32, help='image size')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations for training D')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')

    # Miscellaneous.
    parser.add_argument('--train', type=int, default=1, choices=[0,1])

    # Directories.
    parser.add_argument('--log_dir', type=str, default='../outputs/logs')
    parser.add_argument('--model_dir', type=str, default='../outputs/models')
    parser.add_argument('--sample_dir', type=str, default='../outputs/samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=5000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=0)

    config = parser.parse_args()
    call(config)
