"""Dispatch runner that centralizes CLI and lets you choose mode: local or mpi.

Usage examples:
  python3 run_federated.py --mode local --model least --optimizer sgd --num-workers 2 --epochs 5 --lr 0.01
  mpirun -n 3 python3 run_federated.py --mode mpi --num-workers 2 --model least --epochs 5 --lr 0.01
"""
from cli import parse_common_args


def main():
    # parse common args first to get flags, then dispatch based on mode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['local', 'mpi'], default='local', help='run mode: local (multiprocessing) or mpi')
    # parse only the mode here, then parse the rest via common parser
    known, rest = parser.parse_known_args()

    args = parse_common_args()
    # override mode from the initial parse
    args.mode = known.mode

    if args.mode == 'local':
        from local_federated import run_with_args
        run_with_args(args)
    else:
        from mpi_federated import run_with_args
        run_with_args(args)


if __name__ == '__main__':
    main()
