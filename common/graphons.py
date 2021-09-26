import numpy as np

from enum import Enum
from PIL import Image


class Graphon(Enum):
    complete_bipartite = "complete_bipartite"
    chess_board_3 = "chess_board_3"
    chess_board_4 = "chess_board_4"
    chess_board_5 = "chess_board_5"
    chess_board_6 = "chess_board_6"
    chess_board_3_randomized = "chess_board_3_randomized"
    chess_board_4_randomized = "chess_board_4_randomized"
    chess_board_5_randomized = "chess_board_5_randomized"
    chess_board_6_randomized = "chess_board_6_randomized"
    random_board_3 = "random_board_3"
    random_board_4 = "random_board_4"
    random_board_5 = "random_board_5"
    random_board_6 = "random_board_6"
    hearth = "hearth"
    spades = "spades"
    clubs = "clubs"
    diamond = "diamond"
    eta_0 = "eta_0"
    eta_1 = "eta_1"
    eta_2 = "eta_2"
    eta_3 = "eta_3"
    eta_4 = "eta_4"
    eta_5 = "eta_5"
    eta_6 = "eta_6"
    eta_7 = "eta_7"
    eta_8 = "eta_8"
    hta_0 = "hta_0"
    hta_1 = "hta_1"
    hta_2 = "hta_2"
    hta_3 = "hta_3"


def create_board(
    n: int, randomize_blocks: bool = False, randomize_arrangement: bool = False,
):
    np.random.seed(0)

    if randomize_blocks:
        values = np.random.rand(n, n).flatten()
    else:
        values = np.array([[(i + j) % 2 for i in range(n)] for j in range(n)]).flatten()

    if randomize_arrangement:
        np.random.shuffle(values)

    values = np.triu(values.reshape(n, n))

    # Make it symmetric, graphon requirement
    for i in range(n):
        for j in range(i, n):
            values[j][i] = values[i][j]

    def board(x, y):
        x[x == 1.0] = 0.99
        y[y == 1.0] = 0.99

        bins = np.linspace(0, 1, n)
        x_index, y_index = np.digitize([x, y], bins) - 1

        return values[x_index, y_index]

    board.__name__ = (
        ("chess" if not randomize_arrangement else "random")
        + f"_board_{n}"
        + ("_randomized" if randomize_blocks else "")
    )

    return board


def create_from_image(
    path: str, invert: bool, round: bool, make_symmetric: bool = True
):
    image = Image.open(path).convert("L")

    image_a = np.array(image).astype(float)
    assert image_a.shape[0] == image_a.shape[1], "Must be square."

    image_a = image_a / np.max(image_a)
    image_a[image_a < 0] = 0

    if round:
        image_a = np.around(image_a, 0)

    if invert:
        image_a = 1 - image_a

    image_a = np.rot90(image_a, 3)

    if make_symmetric:
        image_a = np.triu(image_a)

        # Make it symmetric, graphon requirement
        for i in range(image_a.shape[0]):
            for j in range(i, image_a.shape[0]):
                image_a[j][i] = image_a[i][j]

    def graphon(x, y):
        bins = np.linspace(0, 1, image_a.shape[0])
        x_index, y_index = np.digitize([x, y], bins) - 1

        return image_a[x_index, y_index]

    graphon.__name__ = path.split("/")[-1].split(".")[0]

    return graphon


chess_board_3 = create_board(3, randomize_blocks=False)

chess_board_4 = create_board(4, randomize_blocks=False)

chess_board_5 = create_board(5, randomize_blocks=False)

chess_board_6 = create_board(6, randomize_blocks=False)

chess_board_3_randomized = create_board(3, randomize_blocks=True)

chess_board_4_randomized = create_board(4, randomize_blocks=True)

chess_board_5_randomized = create_board(5, randomize_blocks=True)

chess_board_6_randomized = create_board(6, randomize_blocks=True)

random_board_3 = create_board(3, randomize_blocks=False, randomize_arrangement=True)

random_board_4 = create_board(4, randomize_blocks=False, randomize_arrangement=True)

random_board_5 = create_board(5, randomize_blocks=False, randomize_arrangement=True)

random_board_6 = create_board(6, randomize_blocks=False, randomize_arrangement=True)

hearth = create_from_image("/app/materials/hearth.png", invert=True, round=True)

spades = create_from_image("/app/materials/spades.png", invert=True, round=True)

clubs = create_from_image("/app/materials/clubs.png", invert=True, round=True)

diamond = create_from_image("/app/materials/diamond.png", invert=True, round=True)


def complete_bipartite(x, y):
    """Complete bipartite graphon"""

    return (((x <= 0.5) & (y >= 0.5)) | ((x > 0.5) & (y < 0.5))).astype(float)


# ETA and HTA graphons stand for Easy To Align and Hard to Align from
# https://www.aaai.org/AAAI21Papers/AAAI-8648.XuH.pdf
def eta_0(x, y):
    return x * y


def eta_1(x, y):
    return np.exp(-(np.power(x, 0.7) + np.power(y, 0.7)))


def eta_2(x, y):
    return (x ** 2 + y ** 2 + np.sqrt(x) + np.sqrt(y)) / 4


def eta_3(x, y):
    return 1 / 2 * (x + y)


def eta_4(x, y):
    return 1 / (1 + np.exp(-10 * (x ** 2 + y ** 2)))


def eta_5(x, y):
    return 1 / (1 + np.exp(-10 * np.maximum(x, y) ** 2))


def eta_6(x, y):
    return np.exp(-np.power(np.maximum(x, y), 3/4))


def eta_7(x, y):
    return np.exp(- (np.minimum(x, y) + np.sqrt(x) + np.sqrt(y)) / 2)


def eta_8(x, y):
    return np.log(1 + np.maximum(x, y))


def hta_0(x, y):
    return abs(x - y)


def hta_1(x, y):
    return 1 - abs(x - y)


def hta_2(x, y):
    return (((x <= 0.5) & (y >= 0.5)) | ((x > 0.5) & (y < 0.5))).astype(float) * 0.8


def hta_3(x, y):
    return (((x >= 0.5) & (y <= 0.5)) | ((x < 0.5) & (y > 0.5))).astype(float) * 0.8


ALL = [
    chess_board_3,
    chess_board_4,
    chess_board_5,
    chess_board_6,
    chess_board_3_randomized,
    chess_board_4_randomized,
    chess_board_5_randomized,
    chess_board_6_randomized,
    random_board_3,
    random_board_4,
    random_board_5,
    random_board_6,
    hearth,
    spades,
    clubs,
    diamond,
    complete_bipartite,
]