import glob
import json
import math
import typer
import pickle
import numpy as np
import typing as t

TABLE = '''
\\toprule
                  &     &     &      &      & {NAME} &    &      \\\\
\\midrule
GRAPH NAME        & |V| & |E| & TDA1 & TDA2 & TDA3 & TDA4 & TDA5 \\\\
\\midrule
{ROWS}
\\bottomrule
'''
ROW = '{NAME} & {V} & {E}           & {TDA1} & {TDA2}  & {TDA3} & {TDA4} & {TDA5} \\\\ \n'
ROW_NOT_FOUND = '{NAME} & {V} & {E} & \\multicolumn{{5}}{{c}}{{TRAINING ERROR *}} \\\\ \n'


def cround(value):
    if isinstance(value, str):
        return value

    try:
        zeros = math.ceil(-math.log10(abs(value) - abs(math.floor(value)))) - 1
        return round(value, zeros + 3)

    except ValueError:
        return value


def find_best_tda(path: str, model: str):
    try:
        if 'gan' in model.lower():
            tdas_paths = glob.glob(f'{path}/{model}/epoch=**/*tda.best.*.json', recursive=True)
            tdas_paths = sorted(tdas_paths, key=lambda p: int(p.split('epoch=')[1].split('/')[0]), reverse=True)

        else:
            tdas_paths = glob.glob(f'{path}/{model}/tda.best.*.json', recursive=True)
            assert len(tdas_paths) <= 1

        with open(tdas_paths[0]) as f:
            data = json.load(f)

        pertubations_tda_path = tdas_paths[0].split('/training')[0]

        with open(pertubations_tda_path + '/pertubations.tda.json') as f:
            pertubations_tda = json.load(f)

        best_correlation = max(
            x["spearmanr"]["correlation"] for x in pertubations_tda["RewireEdges"].values()
        )
        best_metrics = [
            name
            for name, v in pertubations_tda["RewireEdges"].items()
            if v["spearmanr"]["correlation"] == best_correlation
        ]

        return {
            m: data[m]
            for m in best_metrics
        }

    except Exception as e:
        print(e)
        print(path, model)

        return None


def find_stats(path: str):
    with open(f'{path.rstrip("/")}/train_val_graphs.pickle', 'rb') as f:
        graphs = pickle.load(f)

    vertices = [len(g.nodes()) for g in graphs['test']]
    edges = [len(g.edges()) for g in graphs['test']]

    return float(np.mean(vertices)), float(np.mean(edges))


def main(
    path: t.List[str],
    model: str = 'GAN'
):
    rows = ''

    for p in path:
        name = p.split('data/')[1].split('/')[1]
        tda = find_best_tda(p, model)
        stats = find_stats(p)

        if tda:
            rows += ROW.format(
                NAME=name,
                V=cround(stats[0]),
                E=cround(stats[1]),
                TDA1=cround(tda.get("degree_filtration_betti_curve", {}).get('value', '-')),
                TDA2=cround(tda.get("clustering_coefficient_filtration_betti_curve", {}).get('value', '-')),
                TDA3=cround(tda.get("hks_filtration_betti_curve_{'t': 0.1}", {}).get('value', '-')),
                TDA4=cround(tda.get("hks_filtration_betti_curve_{'t': 1.0}", {}).get('value', '-')),
                TDA5=cround(tda.get("hks_filtration_betti_curve_{'t': 10.0}", {}).get('value', '-')),
            )

        else:
            rows += ROW_NOT_FOUND.format(NAME=name, V=cround(stats[0]), E=cround(stats[1]))

    print(TABLE.format(NAME=model, ROWS=rows))


if __name__ == '__main__':
    typer.run(main)
