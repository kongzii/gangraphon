import os
import json
import glob
import typer
import pickle
import numpy as np

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from common import data, tools


def compute_hds(graphlets, graphs):
    true_hds = {}

    for graphlet in tqdm(graphlets):
        values = []

        for graph in tqdm(graphs):
            try:
                values.append(data.DatasetMC.estimate_naive_monte_carlo_homomorphism_density(
                    graphlet.graph.to_nx(),
                    graph,
                ))

            except ValueError:
                pass

        if not values:
            continue

        true_hds[graphlet.name] = np.mean(values)

    return true_hds


def main(
    n_graphlets: int = 4
):
    structure = defaultdict(lambda: defaultdict(list))
    graphs_paths = list(Path('data').rglob('*.pickle'))

    for path in graphs_paths:
        str_path = str(path)
        graph_name = str_path.split('/')[2]

        if '/GAN/' in str_path:
            output_path = ''.join(str_path.partition('/GAN/')[:2])
        elif '/gran/' in str_path:
            output_path = ''.join(str_path.partition('/gran/')[:2])
        elif '/graphrnn/' in str_path:
            output_path = ''.join(str_path.partition('/graphrnn/')[:2])
        elif str_path.endswith('train_val_graphs.pickle') or str_path.endswith('test_count=100.pickle'):
            continue
        else:
            raise RuntimeError(str_path)

        structure[graph_name][output_path].append(path)

    graphlets = [
        data.Graphlet(
            data.INDEX_TO_PGD_GRAPHLET_NAME.get(i, f"graphlet_{i}"),
            g,
            tools.complement_graph(*g),
        )
        for i, g in enumerate(
            tqdm(tools.create_all_graphs(n_graphlets), desc="Graphlets")
        )
    ]

    for graph_name, output_path in structure.items():
        true_hds = None

        for path, graphs_paths in output_path.items():
            path = Path(path)

            if true_hds is None:
                if (path.parent / 'train_val_graphs.pickle').is_file():
                    train_val_graphs_path = path.parent / 'train_val_graphs.pickle'
                elif (path.parent.parent / 'train_val_graphs.pickle').is_file():
                    train_val_graphs_path = path.parent.parent / 'train_val_graphs.pickle'
                elif (path.parent.parent.parent / 'train_val_graphs.pickle').is_file():
                    train_val_graphs_path = path.parent.parent.parent / 'train_val_graphs.pickle'
                else:
                    raise RuntimeError(str_path)

                with open(train_val_graphs_path, 'rb') as f:
                    true_graphs = pickle.load(f)['test']

                true_hds = compute_hds(graphlets, true_graphs)

                with open(f"{path}/true_hds.json", "w") as f:
                    json.dump(true_hds, f, indent=2)

            gen_hds_diff = {}

            for file in graphs_paths:
                with open(file, "rb") as f:
                    file_graphs = pickle.load(f)

                file_hds = compute_hds(graphlets, file_graphs)

                for key, value in file_hds.items():
                    old_diff = gen_hds_diff.get(key, float('inf'))
                    new_diff = abs(true_hds[key] - value)

                    if old_diff > new_diff:
                        gen_hds_diff[key] = new_diff

            with open(f"{path}/gen_hds_diff.json", "w") as f:
                json.dump(gen_hds_diff, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
