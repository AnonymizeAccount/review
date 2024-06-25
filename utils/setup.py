import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim


def arg_parse(parser):
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--data_dir', type=str, default='/home/yihong/RNTraj/data/', help='dataset')
    parser.add_argument('--base_dir', type=str, default='/home/yihong/RNTraj/', help='base dir')
    parser.add_argument('--result_dir', type=str, default='/home/yihong/RNTraj/log/', help='result dir')
    parser.add_argument('--dataset', type=str, default='chengdu', help='dataset')
    parser.add_argument('--model', type=str, default='kg', help='model')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='')
    parser.add_argument('--labelrate', type=float, default=23, help='percent')
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument('--patience', type=int, default=100, help='patience')
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--node_dim", type=int, default=512)
    parser.add_argument("--edge_dim", type=int, default=64)
    parser.add_argument("--direction_dim", type=int, default=64)
    parser.add_argument("--connection_dim", type=int, default=64)
    parser.add_argument("--distance_dim", type=int, default=64)
    parser.add_argument("--consistent_dim", type=int, default=64)
    parser.add_argument("--margin", type=float, default=6)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--min_length", type=int, default=15)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--obs_len", type=int, default=10)
    parser.add_argument("--pre_len", type=int, default=5)
    parser.add_argument("--direction", type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0, help='')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--multi', type=int, default=2, help='')
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--mlpk", type=int, default=1)
    parser.add_argument('--weight', type=float, default=0.9, help='')
    parser.add_argument('--temp', type=float, default=0.1, help='')
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3)
    parser.add_argument('--weight_decay', "--wd", type=float, default=1e-2)
    parser.add_argument("--a", type=float, default=1, help='connection; weight for the loss function')
    parser.add_argument("--b", type=float, default=1, help='direction; weight for the loss function')
    parser.add_argument("--c", type=float, default=1, help='consistent; weight for the loss function')
    parser.add_argument("--d", type=float, default=1, help='distance; weight for the loss function')
    parser.add_argument("--e", type=float, default=1, help='rank; weight for the loss function')
    parser.add_argument("--f", type=float, default=1, help='topk1; weight for the loss function')
    parser.add_argument("--g", type=float, default=1, help='inference; weight for the loss function')
    parser.add_argument("--h", type=float, default=1, help='direction estimation; weight for the loss function')
    parser.add_argument("--goal_type", type=int, default=0, help='0: no goal, 1: goal direction, 2: goal')
    parser.add_argument('--goal', action='store_true', default=False, help='add goal')
    parser.add_argument('--ed', action='store_true', default=False, help='add goal')
    parser.add_argument('--shuffle', action='store_true', default=True, help='')
    parser.add_argument('--efficient', action='store_true', default=False, help='space efficient')
    parser.add_argument('--mask_node', action='store_true', default=False, help='')
    parser.add_argument('--rand', action='store_true', default=False, help='')
    parser.add_argument('--mask_edge', action='store_true', default=True, help='')
    parser.add_argument('--worerank', action='store_true', default=False, help='')
    parser.add_argument('--wo_rn', action='store_true', default=False, help='')
    parser.add_argument('--wo_d', action='store_true', default=False, help='')
    parser.add_argument('--mgpu', action='store_true', default=False, help='multiple gpu training')
    parser.add_argument('--val', action='store_true', default=False, help='eval')
    parser.add_argument('--test', action='store_true', default=False, help='test')
    parser.add_argument('--train', action='store_true', default=False, help='train')
    parser.add_argument('--time', action='store_true', default=False, help='check time')
    parser.add_argument('--save', action='store_true', default=False, help='save ckpt')

    return parser.parse_args()


def setup_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
    args.device = torch.device(f"cuda:0")


def weight_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1)
        nn.init.constant_(model.bias, 0)


def new_get_model(args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix, loc_dlabels_matrix, TM, edge_A,
                  length_shortest_paths):
    from baselines.kg.transH.KG_0509.mlp_model import MLP_model
    MLP = MLP_model(input_dim=(args.edge_dim * 2 + args.direction_dim * 2), args=args).to(args.device)
    # MLP = MLP_model(input_dim=(args.edge_dim * 2 + args.direction_dim) * args.obs_len, args=args).to(args.device)

    from baselines.kg.transH.KG_0509.SpKG_Goal import RNN
    model = RNN(args=args, graph=graph, node_adj_edges=node_adj_edges, loc_dlabels_matrix=loc_dlabels_matrix, direction_labels=direction_labels,
                loc_direct_matrix=loc_direct_matrix, loc_dist_matrix=loc_dist_matrix, TM=TM, edge_A=edge_A,
                length_shortest_paths=length_shortest_paths).to(args.device)
    opt_mlp = optim.Adam(params=MLP.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    opt_model = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return MLP, opt_mlp, model, opt_model


def get_model(args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix, loc_dlabels_matrix, TM, edge_A,
              length_shortest_paths):
    assert args.model in ['rnn', 'gru', 'lstm', 'nettraj', 'kg', 'mlp']

    if args.model in ['rnn', 'gru', 'lstm', 'mlp', 'nettraj']:
        if args.goal_type == 0:
            from baselines.rnns.rnns_manual.rnns import RNN
        elif args.goal_type == 1:
            from baselines.rnns.rnns_manual.rnns_GoalDirection import RNN
        elif args.goal_type == 2:
            from baselines.rnns.rnns_manual.rnns_Goal import RNN
        else:
            raise ValueError

        model = RNN(args=args, graph=graph, node_adj_edges=node_adj_edges, loc_dlabels_matrix=loc_dlabels_matrix).to(args.device)
    # elif args.model in ['nettraj',]:
    #     from baselines.rnns.nettraj import RNN
    #     model = RNN(args=args, graph=graph, node_adj_edges=node_adj_edges, loc_dlabels_matrix=loc_dlabels_matrix).to(args.device)
    else:
        # from baselines.kg.transH.KG_0313.SpKG import RNN
        # from baselines.kg.transH.KG_0314.SpKG import RNN
        # from baselines.kg.transH.KG_0314.SpKG_Goal import RNN
        # from baselines.kg.transH.KG_0317.SpKG import RNN

        # from baselines.kg.transH.KG_0509.SpKG_Goal import RNN
        # from baselines.kg.transH.KG_0511.SpKG_Goal import RNN
        # from baselines.kg.transH.KG_0512.SpKG_Goal import RNN
        if args.ed:
            from baselines.kg.transH.KG_0514.SpKG_EGoalD import RNN
        else:
            from baselines.kg.transH.KG_0512.SpKG_Goal import RNN

        model = RNN(args=args, graph=graph, node_adj_edges=node_adj_edges, loc_dlabels_matrix=loc_dlabels_matrix, direction_labels=direction_labels,
                    loc_direct_matrix=loc_direct_matrix, loc_dist_matrix=loc_dist_matrix, TM=TM, edge_A=edge_A,
                    length_shortest_paths=length_shortest_paths).to(args.device)
        # pass

    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return model, optimizer