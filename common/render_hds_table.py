import glob
import json
import math
import typer
import pickle
import numpy as np
import typing as t

from collections import defaultdict, Counter
from pathlib import Path

TABLE = '''
\\begin{table*}[]
	\\centering
	\\caption{Graphlet NAME}
	\\begin{adjustbox}{max width=\\textwidth}
		\\begin{tabular}{l|ccc}
            \\toprule
            GRAPH        & GanGraphon & GRAN & GraphRNN \\\\
            \\midrule
ROWS
            \\bottomrule
        \\end{tabular}
	\\end{adjustbox}
\\end{table*}
'''
ROW = '{NAME}     & {GAN} & {gran}  & {graphrnn}\\\\ \n'


def cround(value):
    if value is None:
        return None

    if isinstance(value, str):
        return value

    try:
        zeros = math.ceil(-math.log10(abs(value) - abs(math.floor(value)))) - 1
        return round(value, zeros + 3)

    except ValueError:
        return value


def main():
    paths = list(Path('data').rglob('gen_hds_diff.json'))

    table = defaultdict(lambda: defaultdict(dict))

    for path in paths:
        model_name = str(path).split('/')[-2]
        graph_name = str(path).split('/')[2]

        with open(path) as f:
            diffs = json.load(f)

        for key, value in diffs.items():
            table[key][graph_name][model_name] = value

    for graphlet, graphs in table.items():
        rows = []

        for graph, value in graphs.items():
            raw_values = [value.get('GAN', float('inf')), value.get('graphrnn', float('inf')), value.get('gran', float('inf'))]
            min_idx = np.argmin(raw_values)

            raw_values[min_idx] = float('inf')
            second_min_idx = np.argmin(raw_values)

            values = [
                cround(value.get('GAN')),
                cround(value.get('graphrnn')),
                cround(value.get('gran')),
            ]

            values[min_idx] = '\\textbf{' + str(values[min_idx]) + '}'
            values[second_min_idx] = '\\underline{' + str(values[second_min_idx]) + '}'

            row = ROW.format(
                NAME=graph,
                GAN=values[0],
                graphrnn=values[1],
                gran=values[2],
            )
            rows.append(row)

        latex_table = TABLE.replace('NAME', graphlet).replace('ROWS', ''.join(rows))

        print(latex_table)
        print()


if __name__ == '__main__':
    typer.run(main)
