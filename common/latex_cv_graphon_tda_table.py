import glob
import json


def main():
    files = glob.glob("data/*/*/cv.tda.json")

    print("---")

    for file in files:
        split = file.split("/")
        graphon_name = split[2].replace("_", "-")

        with open(file) as f:
            values = json.load(f)

        line = f"{graphon_name}"

        for name, test in values.items():
            line += f"  &  {round(test['mean'], 2)}  &  {round(test['std'], 2)}"

        line += "  \\\\"

        print(line)

    print("---")


if __name__ == "__main__":
    main()
