import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import graphrnn.data as data
from graphrnn.baselines.graphvae.model import GraphVAE
from graphrnn.baselines.graphvae.data import GraphAdjSampler

GPU_AVAILABLE = torch.cuda.is_available() and torch.cuda.device_count() > 0
LR_milestones = [500, 1000]


def build_model(args, max_num_nodes):
    out_dim = max_num_nodes * (max_num_nodes + 1) // 2
    if args.feature_type == "id":
        input_dim = max_num_nodes
    elif args.feature_type == "deg":
        input_dim = 1
    elif args.feature_type == "struct":
        input_dim = 2
    model = GraphVAE(input_dim, 64, 256, max_num_nodes)
    return model


def train(args, dataloader, model):
    if GPU_AVAILABLE:
        model = model.to("cuda")
    epoch = 1
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=LR_milestones, gamma=args.lr)

    model.train()
    for epoch in range(5000):
        for batch_idx, data in enumerate(dataloader):
            model.zero_grad()
            features = data["features"].float()
            adj_input = data["adj"].float()

            if GPU_AVAILABLE:
                features = features.cuda()
                adj_input = adj_input.cuda()

            loss = model(features, adj_input)
            print("Epoch: ", epoch, ", Iter: ", batch_idx, ", Loss: ", loss)
            loss.backward()

            optimizer.step()
            scheduler.step()


def arg_parse():
    parser = argparse.ArgumentParser(description="GraphVAE arguments.")
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
    )
    parser.add_argument(
        "--max_num_nodes",
        dest="max_num_nodes",
        type=int,
        help="Predefined maximum number of nodes in train/test graphs. -1 if determined by \
                  training data.",
    )
    parser.add_argument(
        "--feature",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    parser.add_argument("--output-predictions", type=str, required=True)

    parser.set_defaults(
        dataset="grid",
        feature_type="id",
        lr=0.001,
        batch_size=1,
        num_workers=1,
        max_num_nodes=-1,
    )
    return parser.parse_args()


def main():
    prog_args = arg_parse()
    ### running log

    with open(prog_args.dataset_file, "r") as f:
        graph = nx.Graph()

        if prog_args.dataset_file.endswith(".mtx"):
            next(f)

        for line in f:
            a, b, *_ = line.split()
            graph.add_edge(a, b)

        graphs_train = [graph, graph]
        graphs_test = [graph, graph]
        graphs = graphs_train + graphs_test

        max_num_nodes = max([graphs[i].number_of_nodes() for i in range(len(graphs))])

    print(
        "total graph num: {}, training set: {}".format(len(graphs), len(graphs_train))
    )
    print("max number node: {}".format(max_num_nodes))

    dataset = GraphAdjSampler(
        graphs_train, max_num_nodes, features=prog_args.feature_type
    )
    # sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
    #        [1.0 / len(dataset) for i in range(len(dataset))],
    #        num_samples=prog_args.batch_size,
    #        replacement=False)
    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=prog_args.batch_size, num_workers=prog_args.num_workers
    )
    model = build_model(prog_args, max_num_nodes)
    train(prog_args, dataset_loader, model)


if __name__ == "__main__":
    main()
