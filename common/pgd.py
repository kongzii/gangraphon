import os
import typer
import networkx as nx

from common import tools


def main(graph_file: str):
    graph = tools.nx_from_file(graph_file)

    # PGD does not accept node starting at 0
    graph = nx.convert_node_labels_to_integers(graph, first_label=1)

    tmp_mtx = "/tmp/graph.mtx"
    with open(tmp_mtx, "w") as f:
        f.write(f"{len(graph)} {len(graph)} {len(graph.edges())}")

        for a, b in graph.edges():
            f.write(f"{a} {b}\n")

    output_directory = "/".join(graph_file.split("/")[:-1])
    os.system(
        f"""
        ./pgd \
        -f {tmp_mtx}\
        --macro {output_directory}/pgd.macro \
        > {output_directory}/pgd.macro.stdout
    """
    )


if __name__ == "__main__":
    typer.run(main)
