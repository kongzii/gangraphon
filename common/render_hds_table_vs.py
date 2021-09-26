import json
import math
import typer
import itertools as it

from collections import defaultdict
from pathlib import Path

TABLE = '''
\\begin{table}
	\\centering
	\\caption{ABOUT. Values are win/tie/loss. Row indicates winning method, column lossing one.}
	\\begin{adjustbox}{max width=\\textwidth}
		\\begin{tabular}{l|cc}
\\toprule
            & GraphRNN & GRAN \\\\
\\midrule
ROWS
            \\bottomrule
        \\end{tabular}
	\\end{adjustbox}
\\end{table}
'''
# TABLE = '''
# \\begin{tabular}{l|cc}
# ABOUT \\\\
#  & GraphRNN & GRAN \\\\
# ROWS
# \\end{tabular}
# '''
ROW = '{METHOD} & {GraphRNN} & {GraphRNN} \\\\ \n'


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

    gan_wins, gan_loss = 0, 0
    tables = []

    for artificial in [True, False]:
        table = defaultdict(lambda: defaultdict(dict))

        graphlets = set()

        for path in paths:
            if not artificial and 'artificial' in str(path):
                continue

            if artificial and 'artificial' not in str(path):
                continue

            model_name = str(path).split('/')[-2]
            graph_name = str(path).split('/')[2]

            with open(path) as f:
                diffs = json.load(f)

            for graphlet, value in diffs.items():
                graphlets.add(graphlet)

                if 'randomized' in graph_name:
                    continue

                table[graphlet][graph_name][model_name] = value

        methods = ['GraphonGAN', 'GraphRNN', 'GRAN']

        for graphlet in sorted(graphlets) + ['all_graphlets']:
            table_win_loss_counts = defaultdict(lambda: {'win': 0, 'tie': 0, 'loss': 0})

            # vs0 is better then vs1 then +1
            for vs0, vs1 in it.product(methods, methods):
                if vs0 == vs1:
                    continue

                if graphlet == 'all_graphlets':
                    values = [x for g in graphlets for x in table[g].values()]

                else:
                    values = table[graphlet].values()

                for value in values:
                    raw_values = [value.get('GAN', float('inf')), value.get('graphrnn', float('inf')), value.get('gran', float('inf'))]

                    value_vs0 = raw_values[methods.index(vs0)]
                    value_vs1 = raw_values[methods.index(vs1)]

                    # Less is better (distance metrics)
                    table_win_loss_counts[(vs0, vs1)]['win'] += 1 if value_vs1 < value_vs0 else 0
                    table_win_loss_counts[(vs0, vs1)]['tie'] += 1 if value_vs1 == value_vs0 else 0
                    table_win_loss_counts[(vs0, vs1)]['loss'] += 1 if value_vs1 > value_vs0 else 0

            rows = []
            table_win_loss_counts = dict(table_win_loss_counts)

            v0 = None
            for v in table_win_loss_counts.values():
                if v0 is None:
                    v0 = sum(v.values())
                else:
                    assert sum(v.values()) == v0 == (9 if not artificial else 12) if graphlet != 'all_graphlets' else (14 * 9 if not artificial else 14 * 12)

            for vs1 in methods:
                row = f'{vs1}    '

                if vs1 == 'GRAN':
                    continue

                for vs0 in methods:
                    if vs0 == 'GraphonGAN':
                        continue

                    result = table_win_loss_counts.get((vs0, vs1), None)

                    if result is None:
                        row += ' & -'

                    else:
                        row += f' & {result["win"]}/{result["tie"]}/{result["loss"]}'

                        if vs1 == 'GraphonGAN' and result["win"] > result["loss"]:
                            gan_wins += 1

                        if vs1 == 'GraphonGAN' and result["win"] < result["loss"]:
                            gan_loss += 1

                rows.append(row + '  \\\\')

            about = f'{graphlet} ' + ('artificial' if artificial else 'real')
            tables.append(TABLE.replace('ABOUT', about).replace('ROWS', '\n'.join(rows)))
            print(tables[-1])
            print()


if __name__ == '__main__':
    typer.run(main)
