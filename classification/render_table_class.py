import typer
import glob
import json
import math

TABLE = '''
\\begin{table*}[]
	\\centering
	\\caption{blabla}
	\\begin{adjustbox}{max width=\\textwidth}
		\\begin{tabular}{l|cc}
            \\toprule
            GRAPHS        & Test Accuracy & Gen. Test Accuracy \\\\
            \\midrule
ROWS
            \\bottomrule
        \\end{tabular}
	\\end{adjustbox}
\\end{table*}
'''
ROW = '{NAME}     & {TA} & {GTA} \\\\ \n'


def cround(value):
    if value is None:
        return None

    if isinstance(value, str):
        return value

    try:
        zeros = math.ceil(-math.log10(abs(value) - abs(math.floor(value)))) - 1
        return round(value, zeros + 2)

    except ValueError:
        return value


def main(n_sample: int = 0):
    files = glob.glob(f"data/classification/datasets.*n_sample={n_sample}*.json")
    rows = []

    for file in files:
        with open(file) as f:
            res = json.load(f)

        name = file.split(".")[1].replace(",_", ", ").replace("(", "").replace(")", "").replace("'", "")

        rows.append(ROW.format(
            NAME=name,
            TA=cround(res["test_accuracy"]),
            GTA=cround(res["test_aug_accuracy"]),
        ))

    print(TABLE.replace("ROWS", "".join(rows)))


if __name__ == "__main__":
    typer.run(main)
