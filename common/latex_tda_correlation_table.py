import glob
import json

THRESHOLD = 0.98


def main():
    files = glob.glob("data/*/*/pertubations.tda.json")

    indexes = [(0, 3), (3, 6)]

    for a, b in indexes:
        tests = None
        pertubations = None

        for file in files:
            split = file.split("/")
            name = split[2].replace("_", "-")

            with open(file) as f:
                values = json.load(f)

            if pertubations is None:
                pertubations = list(values.keys())

            if tests is None:
                tests = list(values[pertubations[0]].keys())[a:b]

            line = f"{name}"

            for test in tests:
                coef_tuple = []

                for i, pertubation in enumerate(pertubations):
                    value = values[pertubation][test]["spearmanr"]["correlation"]

                    if value is None:
                        coef_tuple.append("-")
                    elif value >= THRESHOLD and i == 2:
                        coef_tuple.append("\\textbf{" + f"{round(value, 4):.4f}" + "}")
                    else:
                        coef_tuple.append(f"{round(value, 4):.4f}")

                line += f" & {coef_tuple[0]}, {coef_tuple[1]}, {coef_tuple[2]}"

            line += " \\\\"

            print(line)

        print("---")


if __name__ == "__main__":
    main()
