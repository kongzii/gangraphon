import typer
import glob
import json
import math

TABLE = '''
\\begin{table*}[]
	\\centering
	\\caption{Experiments are sorted by decreasing accuracy of the opposite test set. I.e. if the experiment was trained on real data, we aim for augmented data accuracy and vice-versa. The opposite accuracy is underlined. Bold indicates if real and aug data accuracy is close.}
	\\begin{adjustbox}{max width=\\textwidth}
		\\begin{tabular}{l|ccccc}
            \\toprule
GRAPHS & \\begin{tabular}[c]{@{}l@{}}Trained\\\\ On\\end{tabular} & \\begin{tabular}[c]{@{}l@{}}Test\\\\ Acc.\\end{tabular} & \\begin{tabular}[c]{@{}l@{}}Test\\\\ Acc. (major class)\\end{tabular} & \\begin{tabular}[c]{@{}l@{}}Gen. Test\\\\ Acc.\\end{tabular} & \\begin{tabular}[c]{@{}l@{}}Gen. Test\\\\ Acc. (major class)\\end{tabular} \\\\
            \\midrule
ROWS
            \\bottomrule
        \\end{tabular}
	\\end{adjustbox}
\\end{table*}
'''
ROW = '{NAME}  & {TO}   & {TA} & {TAC} & {GTA} & {GTAC} \\\\ \n'


def cround(value, underline: bool = False, bold: bool = False):
    to_return = None

    if value is None:
        return to_return

    if isinstance(value, str):
        to_return = value

    try:
        zeros = math.ceil(-math.log10(abs(value) - abs(math.floor(value)))) - 1
        to_return =  str(round(value, zeros + 2))

    except ValueError:
        to_return = str(value)

    if underline:
        to_return = '\\underline{' + to_return + '}'

    if bold:
        to_return = '\\textbf{' + to_return + '}'

    return to_return


def main():
    files = glob.glob("data/classification2/*")
    rows = []

    loaded = []

    for file in files:
        name = file.split(".")[1].replace(",_", ", ").replace("(", "").replace(")", "").replace("'", "")
        aug = 'reversed=True' in file

        with open(file) as f:
            loaded.append((name, aug, json.load(f)))

    loaded = sorted(loaded, key=lambda x: -x[2]['test_accuracy' if x[1] else 'test_aug_accuracy'])

    for name, aug, res in loaded:
        major_class = max(res['ratios']['train_data'].keys(), key=lambda x: res['ratios']['train_data'][x])
        major_class_acc_real = res['orig_accuracies'][f'test/accuracy/class_{major_class}']
        major_class_acc_aug = res['aug_accuracies'][f'test/accuracy/class_{major_class}']

        if abs(res["test_accuracy"] - res["test_aug_accuracy"]) < 0.2 or abs(major_class_acc_real - major_class_acc_aug) < 0.2:
            bold = True

        else:
            bold = False

        rows.append(ROW.format(
            NAME=name,
            TO='Aug' if aug else 'Real',
            TA=cround(res["test_accuracy"], underline=aug, bold=aug and bold),
            TAC=cround(major_class_acc_real, underline=aug, bold=aug and bold),
            GTA=cround(res["test_aug_accuracy"], underline=not aug, bold=not aug and bold),
            GTAC=cround(major_class_acc_aug, underline=not aug, bold=not aug and bold),
        ))

    print(TABLE.replace("ROWS", "".join(rows)))


if __name__ == "__main__":
    typer.run(main)
