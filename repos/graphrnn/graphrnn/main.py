import argparse
import os
import networkx as nx
import pickle
import uuid
from .train import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--small", action="store_true", default=False)
    parser.add_argument(
        "--model",
        choices=["GraphRNN_RNN", "GraphRNN_MLP", "GraphRNN_VAE_conditional"],
        type=str,
        default="GraphRNN_RNN",
    )
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument(
        "--test-total-size", type=int, default=100, help="Number of graphs to generate."
    )
    parser.add_argument("--output-predictions", type=str, required=False, default=None)

    args = Args()
    args = parser.parse_args(args=None, namespace=args)
    # All necessary arguments are defined in args.py
    # print('CUDA', args.cuda)
    args.cuda = False if args.no_cuda else True
    args.note = args.model
    args.graph_type = os.path.splitext(os.path.basename(args.dataset_file))[0]
    args.update_parameters()
    print("File name prefix", args.fname)
    # check if necessary directories exist
    # if not os.path.exists(os.path.dirname(args.output_predictions)):
    #     os.makedirs(args.output_predictions)
    # if not os.path.isdir(args.model_save_path):
    #     os.makedirs(args.model_save_path)
    # if not os.path.isdir(args.graph_save_path):
    #     os.makedirs(args.graph_save_path)
    # if not os.path.isdir(args.figure_save_path):
    #     os.makedirs(args.figure_save_path)
    # if not os.path.isdir(args.timing_save_path):
    #     os.makedirs(args.timing_save_path)
    # if not os.path.isdir(args.figure_prediction_save_path):
    #     os.makedirs(args.figure_prediction_save_path)
    # if not os.path.isdir(args.nll_save_path):
    #     os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    time += "-{}".format(str(uuid.uuid4())[:8])
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run" + time, flush_secs=5)

    if args.dataset_file.endswith(".pickle"):
        with open(args.dataset_file, "rb") as f:
            dataset = pickle.load(f)

            graphs_train = dataset["train"]
            graphs_test = dataset["test"]

        args.output_predictions = (
            "/".join(args.dataset_file.split("/")[:-1]) + "/graphrnn"
        )

    else:
        with open(args.dataset_file, "r") as f:
            graph = nx.Graph()

            if args.dataset_file.endswith(".mtx"):
                next(f)

            for line in f:
                a, b, *_ = line.split()
                graph.add_edge(a, b)

            graphs_train = [graph, graph]
            graphs_test = [graph, graph]

    assert args.output_predictions

    graphs = graphs_train + graphs_test

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print("graph_test_len", graph_test_len)

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # args.max_num_node = 2000
    # show graphs statistics
    print(
        "total graph num: {}, training set: {}".format(len(graphs), len(graphs_train))
    )
    print("max number node: {}".format(args.max_num_node))
    print("max/min number edge: {}; {}".format(max_num_edge, min_num_edge))
    print("max previous node: {}".format(args.max_prev_node))

    ### dataset initialization
    # if 'nobfs' in args.note:
    #     print('nobfs')
    #     dataset = Graph_sequence_sampler_pytorch_nobfs(graphs_train, max_num_node=args.max_num_node)
    #     args.max_prev_node = args.max_num_node-1
    # if 'barabasi_noise' in args.graph_type:
    #     print('barabasi_noise')
    #     dataset = Graph_sequence_sampler_pytorch_canonical(graphs_train,max_prev_node=args.max_prev_node)
    #     args.max_prev_node = args.max_num_node - 1
    # else:
    dataset = Graph_sequence_sampler_pytorch(
        graphs_train, max_prev_node=args.max_prev_node, max_num_node=args.max_num_node
    )
    # Get the automatically computed value
    args.max_prev_node = dataset.max_prev_node
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler(
        [1.0 / len(dataset) for i in range(len(dataset))],
        num_samples=args.batch_size * args.batch_ratio,
        replacement=True,
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sample_strategy,
    )

    ### model initialization
    ## Graph RNN VAE model
    # lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
    #                   hidden_size=args.hidden_size, num_layers=args.num_layers).cuda()

    if "GraphRNN_VAE_conditional" in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            has_input=True,
            has_output=False,
        )
        output = MLP_VAE_conditional_plain(
            h_size=args.hidden_size_rnn,
            embedding_size=args.embedding_size_output,
            y_size=args.max_prev_node,
        )
    elif "GraphRNN_MLP" in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            has_input=True,
            has_output=False,
        )
        output = MLP_plain(
            h_size=args.hidden_size_rnn,
            embedding_size=args.embedding_size_output,
            y_size=args.max_prev_node,
        )
    elif "GraphRNN_RNN" in args.note:
        rnn = GRU_plain(
            input_size=args.max_prev_node,
            embedding_size=args.embedding_size_rnn,
            hidden_size=args.hidden_size_rnn,
            num_layers=args.num_layers,
            has_input=True,
            has_output=True,
            output_size=args.hidden_size_rnn_output,
        )
        output = GRU_plain(
            input_size=1,
            embedding_size=args.embedding_size_rnn_output,
            hidden_size=args.hidden_size_rnn_output,
            num_layers=args.num_layers,
            has_input=True,
            has_output=True,
            output_size=1,
        )
    if args.cuda:
        rnn = rnn.cuda()
        output = output.cuda()

    ### start training
    train(args, dataset_loader, rnn, output)

    ### graph completion
    # train_graph_completion(args,dataset_loader,rnn,output)

    ### nll evaluation
    # train_nll(args, dataset_loader, dataset_loader, rnn, output, max_iter = 200, graph_validate_len=graph_validate_len,graph_test_len=graph_test_len)


if __name__ == "__main__":
    main()
