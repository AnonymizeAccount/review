import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim


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
        # from flow.models.SpKG import RNN


        # if args.ed:
        #     from baselines.kg.transH.KG_0514.SpKG_EGoalD import RNN
        # else:
        #     from baselines.kg.transH.KG_0512.SpKG_Goal import RNN


        if args.ed:
            from flow.models.SpKG_EGoalD import RNN
        else:
            from flow.models.SpKG_Goal import RNN

        model = RNN(args=args, graph=graph, node_adj_edges=node_adj_edges, loc_dlabels_matrix=loc_dlabels_matrix, direction_labels=direction_labels,
                    loc_direct_matrix=loc_direct_matrix, loc_dist_matrix=loc_dist_matrix, TM=TM, edge_A=edge_A,
                    length_shortest_paths=length_shortest_paths).to(args.device)
        # pass

    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return model, optimizer