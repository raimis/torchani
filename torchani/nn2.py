from collections import OrderedDict
import numpy as np
import torch
from .nn import ANIModel, Ensemble, SpeciesEnergies

class ANIModel2(ANIModel):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, species_aev, cell=None, pbc=None):

        species, aev = species_aev

        species_ = species.flatten()
        aev = aev.flatten(0, 1)
        num_atoms = int(species.shape[1])

        energies = []

        for i, (_, nn) in enumerate(self.items()):
            midx = torch.tensor(np.flatnonzero(species_.cpu() == i), device=species_.device)
            if midx.shape[0] > 0:
                input_ = aev.index_select(0, midx)
                energies.append(nn(input_).flatten())

        energies = torch.cat(energies).reshape((1, num_atoms))
        energy = torch.sum(energies, dim=1)

        return SpeciesEnergies(species, energy)

def pad(tensor, size):

    padded_tensor = torch.zeros(size, dtype=tensor.dtype, device=tensor.device)

    if type(size) is int:
        assert len(tensor.shape) == 1
        assert tensor.shape[0] <= size
        padded_tensor[:tensor.shape[0]] = tensor
    elif type(size) is tuple and len(size) == 2:
        assert tensor.shape[0] <= size[0]
        assert tensor.shape[1] <= size[1]
        padded_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
    else:
        assert False

    return padded_tensor

def celu(x):
    return 0.1 * torch.nn.functional.elu(10 * x, alpha=1)

class Ensemble2(Ensemble):

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, species_aev, cell=None, pbc=None):

        assert cell is None
        assert pbc is None

        species_, aev_ = species_aev

        assert len(species_.shape) == 2
        assert species_.shape[0] == 1

        assert len(aev_.shape) == 3
        assert aev_.shape[0] == 1
        assert aev_.shape[1] == species_.shape[1]

        num_atoms = int(species_.shape[1])
        num_ave = int(aev_.shape[2])
        species = species_.reshape(num_atoms)

        # for ilayer, in_size, out_size in [(0, 384, 160), (2, 160, 128), (4, 128, 96), (6, 96, 1)]:
        #     weights, biases = [], []
        #     for iatom in species:
        #         subweights, subbiases = [], []
        #         for ani in self:
        #             weight = list(ani.values())[iatom][ilayer].weight
        #             bias   = list(ani.values())[iatom][ilayer].bias
        #             assert len(weight.shape) == 2
        #             assert weight.shape[0] <= out_size
        #             assert weight.shape[1] <= in_size
        #             assert len(bias.shape) == 1
        #             assert bias.shape[0] == weight.shape[0]
        #             weight = pad(weight, (out_size, in_size)).reshape((1, 1, 1, out_size, in_size))
        #             bias   = pad(bias,    out_size          ).reshape((1, 1, 1, out_size, 1      ))
        #             subweights.append(weight)
        #             subbiases.append(bias)
        #         weights.append(torch.cat(subweights, dim=2))
        #         biases.append(torch.cat(subbiases  , dim=2))
        #     weights = torch.cat(weights, dim=1)
        #     biases = torch.cat(biases  , dim=1)
        #     self.register_buffer(f'weights_{ilayer}', weights)
        #     self.register_buffer(f'biases_{ilayer}', biases)

        # vectors = aev_.reshape((1, num_atoms, 1, num_ave, 1))
        # vectors = torch.matmul(self.weights_0, vectors) + self.biases_0
        # vectors = celu(vectors)
        # vectors = torch.matmul(self.weights_2, vectors) + self.biases_2
        # vectors = celu(vectors)
        # vectors = torch.matmul(self.weights_4, vectors) + self.biases_4
        # vectors = celu(vectors)
        # vectors = torch.matmul(self.weights_6, vectors) + self.biases_6
        # energy = torch.sum(vectors).reshape(1) / len(self)

        aev = aev_.reshape((num_atoms, num_ave))
        energies = []
        for i, submodel in enumerate(self):
            for j, nn in enumerate(self[i].values()):
                midx = torch.tensor(np.flatnonzero(species.cpu() == j), device=species.device)
                if midx.shape[0] > 0:
                    vector = aev.index_select(0, midx)
                    assert len(nn) == 7
                    vector = torch.matmul(vector, nn[0].weight.t())
                    vector += nn[0].bias
                    vector = celu(vector)
                    vector = torch.matmul(vector, nn[2].weight.t())
                    vector += nn[2].bias
                    vector = celu(vector)
                    vector = torch.matmul(vector, nn[4].weight.t())
                    vector += nn[4].bias
                    vector = celu(vector)
                    vector = torch.matmul(vector, nn[6].weight.t())
                    vector += nn[6].bias
                    energies.append(vector.flatten())
        energies = torch.cat(energies)
        energy = torch.sum(energies).reshape(1) / len(self)

        return SpeciesEnergies(species_, energy)