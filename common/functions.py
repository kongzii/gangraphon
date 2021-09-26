import torch
import typing as t

from common import tools

CACHE = tools.Cache()


class Block(torch.nn.Module):
    def __init__(
        self, input_dim: int = 8, hidden_size: int = 8, activation: str = "ReLU"
    ):
        super().__init__()

        self.input = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            getattr(torch.nn, activation)(),
        )

        self.sequence = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 2 * hidden_size),
            getattr(torch.nn, activation)(),
            # ---
            torch.nn.Linear(2 * hidden_size, 4 * hidden_size),
            getattr(torch.nn, activation)(),
            # ---
            torch.nn.Linear(4 * hidden_size, 2 * hidden_size),
            getattr(torch.nn, activation)(),
            # ---
            torch.nn.Linear(2 * hidden_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            getattr(torch.nn, activation)(),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.sequence(x) + x

        return x


def scaled_degree(
    graphon: torch.nn.Module,
    vertices: t.Union[float, torch.Tensor],
    resolution: int = 100,
):
    # `Large networks and graph limits` equation (7.1)
    # If the graphon is associated with a simple graph G, this corresponds to the scaled degree dG(x) / v(G)
    device = next(graphon.parameters()).device

    if isinstance(vertices, float):
        vertices = torch.Tensor([[vertices]])

    indexes = CACHE(lambda res: torch.LongTensor([0] * res), resolution)

    x = torch.index_select(vertices, 1, indexes).to(device)
    y = CACHE(
        lambda len_vert, res: torch.cat(
            [torch.linspace(0, 1, res).unsqueeze(0) for _ in range(len_vert)]
        ),
        len(vertices),
        resolution,
    ).to(device)

    if "Generator" in graphon.__class__.__name__:
        edges = torch.hstack([x.reshape(-1, 1), y.reshape(-1, 1)]).unsqueeze(0)
        res = graphon(edges)[:, :, 1]

    else:
        res = graphon(x.reshape(-1), y.reshape(-1))

    res = res.reshape(len(vertices), resolution)

    return torch.sum(res * (1 / resolution), dim=1)


def degree_distribution(
    graphon: torch.nn.Module, n_vertices: int, resolution: int = 100
):
    device = next(graphon.parameters()).device

    vertices = torch.rand(resolution).view(-1, 1)

    k_degrees_list = CACHE(lambda max_k_degree: list(range(max_k_degree)), n_vertices)
    k_degrees = CACHE(lambda x: torch.LongTensor(x), k_degrees_list)
    stacked_k_degrees = CACHE(
        lambda k_degrees, len_vertices: torch.index_select(
            k_degrees.cpu().unsqueeze(1), 1, torch.LongTensor([0] * len_vertices)
        ),
        k_degrees,
        len(vertices),
    ).to(device)
    n_over_k = CACHE(
        lambda n_vertices, k_degrees_list: (
            torch.Tensor(
                [
                    torch.log(
                        torch.Tensor(
                            list(range(n_vertices - 1, n_vertices - 1 - k, -1))
                        )
                    ).sum()
                    for k in k_degrees_list
                ]
            )
            - torch.Tensor(
                [
                    torch.log(torch.Tensor(list(range(k, 0, -1)))).sum()
                    for k in k_degrees_list
                ]
            )
        ),
        n_vertices,
        k_degrees_list,
    )
    stacked_n_over_k = CACHE(
        lambda n_over_k, len_vertices: torch.index_select(
            n_over_k.unsqueeze(1), 1, torch.LongTensor([0] * len_vertices)
        ),
        n_over_k,
        len(vertices),
    ).to(device)
    degrees = scaled_degree(graphon, vertices, resolution=resolution)
    stacked_degrees = torch.index_select(
        degrees.unsqueeze(0),
        0,
        torch.LongTensor([0] * len(k_degrees)).to(device),
    )

    steps = (
        stacked_n_over_k
        + stacked_k_degrees * torch.log(stacked_degrees)
        + (n_vertices - 1 - stacked_k_degrees) * torch.log(1 - stacked_degrees)
    )

    degree_distribution = torch.sum(torch.exp(steps) * (1 / len(degrees)), dim=1)

    return degree_distribution
