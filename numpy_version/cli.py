"""Shared CLI parser for FLplatform_npy runners.

This module centralizes common arguments (model, optimizer, lr, optimizer hyperparams,
num-workers, epochs, dataset size) so the local and MPI runners use the same flags.
"""
import argparse


def create_common_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--model', choices=['least', 'logistic', 'dnn'], default='least')
    parser.add_argument('--optimizer', choices=['sgd', 'svrg'], default='sgd')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', '--momentum', dest='momentum', type=float, default=0.0, help='momentum for server-side SGD optimizer')
    parser.add_argument('--weight-decay', '--weight_decay', dest='weight_decay', type=float, default=0.0, help='weight decay for server-side SGD optimizer')
    parser.add_argument('--dampening', '--dampening', dest='dampening', type=float, default=0.0, help='dampening for server-side SGD optimizer')
    parser.add_argument('--nesterov', '--nesterov', dest='nesterov', action='store_true', help='enable Nesterov momentum on server-side SGD optimizer')
    parser.add_argument('--one-worker', action='store_true', help='Force single-worker mode (for testing)')
    parser.add_argument('--n', type=int, default=200, help='number of samples to generate')
    return parser


def parse_common_args():
    parser = create_common_parser()
    return parser.parse_args()
