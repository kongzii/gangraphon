import argparse
import numpy as np
import os
import pickle
import time
import json
import networkx as nx
import methods.learner as learner
import methods.simulator as simulator

from pathlib import Path
from collections import defaultdict, Counter
from common.evaluators.eval import eval_tdas
from common import graphons, data, tools


parser = argparse.ArgumentParser(description='Comparison for various methods on synthetic data')
parser.add_argument('--g', type=str)
parser.add_argument('--source', choices=['graphon', 'tudataset', 'single'])
parser.add_argument('--f-result', type=str,
                    default='/app/repos/SGWB-Graphon/results',
                    help='the root path saving learning results')
parser.add_argument('--r', type=int,
                    default=1000,
                    help='the resolution of graphon')
parser.add_argument('--num-graphs', type=int,
                    default=10,
                    help='the number of synthetic graphs')
parser.add_argument('--num-nodes', type=int, default=200,
                    help='the number of nodes per graph')
parser.add_argument('--graph-size', type=str, default='random',
                    help='the size of each graph, random or fixed')
parser.add_argument('--threshold-sba', type=float, default=0.1,
                    help='the threshold of sba method')
parser.add_argument('--threshold-usvt', type=float, default=0.1,
                    help='the threshold of usvt method')
parser.add_argument('--alpha', type=float, default=0.0003,
                    help='the weight of smoothness regularizer')
parser.add_argument('--beta', type=float, default=5e-3,
                    help='the weight of proximal term')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='the weight of gw term')
parser.add_argument('--inner-iters', type=int, default=50,
                    help='the number of inner iterations')
parser.add_argument('--outer-iters', type=int, default=20,
                    help='the number of outer iterations')
parser.add_argument('--n-trials', type=int, default=1,
                    help='the number of trials')
args = parser.parse_args()

Path(args.f_result).mkdir(exist_ok=True, parents=True)

methods = [
    'SBA',
    'SAS',
    'LG',
    'MC',
    'USVT',
    'GWB',
    'SGWB',
    'FGWB',
    'SFGWB',
]
best_tda = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float('inf'))))

errors = np.zeros((13, len(methods), args.n_trials))
i = args.g

for n in range(args.n_trials):
    if args.source == 'graphon':
        graphon = simulator.synthesize_graphon(r=args.r, type_idx=i)
        simulator.visualize_graphon(graphon, save_path=os.path.join(args.f_result, 'graphon_{}.pdf'.format(i)))
        graphs = simulator.simulate_graphs(graphon,
            num_graphs=args.num_graphs,
            num_nodes=args.num_nodes,
            graph_size=args.graph_size,
        )
        test_graphs = [
            nx.from_numpy_matrix(adj)
            for adj in simulator.simulate_graphs(graphon,
                num_graphs=args.num_graphs,
                num_nodes=args.num_nodes,
                graph_size=args.graph_size,
            )
        ]

    elif args.source == 'tudataset':
        train_graphs, test_graphs = data.DatasetTUDataset2.split(
            name=i,
            class_idx=0,
            train_size=0.5,
            stats=False,
            limit_graphs=None,
            max_graph_size=500,
            allow_directed=False,
        )
        graphs = [np.squeeze(np.asarray(nx.to_numpy_matrix(g).astype('float'))) for g in train_graphs.graphs]
        test_graphs = test_graphs.graphs

    elif args.source == 'single':
        train_graphs, test_graphs = data.InducedGraphsDataset.split(
            graph_file=i,
            n_induced=100,
            induced_size_range=(100, 200),
            stats=False,
        )
        for g in train_graphs.graphs:
            g.remove_edges_from(nx.selfloop_edges(g))
        graphs = [nx.to_numpy_matrix(g) for g in train_graphs.graphs]
        graphs = [np.squeeze(np.asarray(nx.to_numpy_matrix(g).astype('float'))) for g in train_graphs.graphs]
        test_graphs = test_graphs.graphs

    # simulator.visualize_unweighted_graph(graphs[0],
    #                                      save_path=os.path.join(args.f_result, 'adj_{}_{}.pdf'.format(i, n)))

    print('lengraphs', len(graphs))
    print('len unique sizes', len(set([tuple(g.shape) for g in graphs])))
    print('total edges', sum(g.sum() for g in graphs))

    for m in range(len(methods)):
        since = time.time()
        print('estimating')
        _, estimation = learner.estimate_graphon(graphs, method=methods[m], args=args)
        print('visualizing')
        simulator.visualize_graphon(estimation,
                                    title=methods[m],
                                    save_path=os.path.join(args.f_result,
                                                            'estimation_{}_{}_{}.pdf'.format(i, n, methods[m])))
        # if m <= 8:
        #     errors[i, m, n] = simulator.mean_square_error(graphon, estimation)
        # else:
        #     errors[i, m, n] = simulator.gw_distance(graphon, estimation)

        print('generating')
        estimated_graphs = [
            nx.from_numpy_matrix(adj)
            for adj in simulator.simulate_graphs(
                estimation,
                num_graphs=args.num_graphs,
                num_nodes=args.num_nodes,
                graph_size=args.graph_size,
            )
        ]

        tdas = eval_tdas(test_graphs, estimated_graphs)

        for key, value in tdas.items():
            if best_tda[i][methods[m]][key] > value:
                best_tda[i][methods[m]][key] = value

        with open(f'{args.f_result}/tdas.{i}.json', 'w') as f:
            json.dump(best_tda, f, indent=2)
