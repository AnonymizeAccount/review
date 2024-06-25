import argparse


def arg_parse(parser):
    parser.add_argument('--base_dir', type=str, default='/home/yihong/RNTraj/', help='base dir')
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--a", type=float, default=1, help='connection; weight for the loss function')
    parser.add_argument('--save', action='store_true', default=False, help='save ckpt')

    return parser.parse_args()


args = arg_parse(argparse.ArgumentParser())

print(args.save)
print(args.a)