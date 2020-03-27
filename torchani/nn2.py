from collections import OrderedDict
import numpy as np
import torch
from .nn import ANIModel, Ensemble, SpeciesEnergies


class CELU:

    def __init__(self, alpha):

        self._alpha = float(alpha)
        self._input = None
        self._output = None

    def forward(self, input):

        self._input = input
        self._output = 0.1 * torch.nn.functional.elu(10 * self._input, alpha=1)

        return self._output

    def backward(self, grad_output):

        grad_input = torch.autograd.grad(self._output, self._input, grad_output, retain_graph=True)[0]

        return grad_input

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

        self._aevs = aev_

        num_atoms = int(species_.shape[1])
        num_ave = int(aev_.shape[2])

        species = species_.reshape(num_atoms)
        aev = aev_.reshape((num_atoms, num_ave))


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

        assert np.all(np.sort(species.cpu().numpy())[::-1] == species.cpu().numpy())

        self._nns = []
        self._celu1 = []
        self._celu3 = []
        self._celu5 = []

        energies = []
        for i, submodel in enumerate(self):
            nns = reversed(self[i].values())
            js = reversed(range(len(self[i])))
            for j, nn in zip(js, nns):
                assert len(nn) == 7
                midx = torch.tensor(np.flatnonzero(species.cpu() == j), device=species.device)
                if midx.shape[0] > 0:
                    self._nns.append(nn)
                    vector0 = aev.index_select(0, midx)
                    vector1 = torch.matmul(vector0, nn[0].weight.t()) + nn[0].bias
                    self._celu1.append(CELU(10))
                    vector2 = self._celu1[-1].forward(vector1)
                    vector3 = torch.matmul(vector2, nn[2].weight.t()) + nn[2].bias
                    self._celu3.append(CELU(10))
                    vector4 = self._celu3[-1].forward(vector3)
                    vector5 = torch.matmul(vector4, nn[4].weight.t()) + nn[4].bias
                    self._celu5.append(CELU(10))
                    vector6 = self._celu5[-1].forward(vector5)
                    vector7 = torch.matmul(vector6, nn[6].weight.t()) + nn[6].bias
                    energies.append(vector7.flatten())
        self._energies = energies
        energies = torch.cat(energies)
        energy = torch.sum(energies).reshape(1) / len(self)

        return SpeciesEnergies(species_, energy)

    def backward(self):

        num_atoms = int(self._aevs.shape[1])
        num_aevs = int(self._aevs.shape[2])

        grad_aevs = []
        for nn, celu1, celu3, celu5, energy in zip(
                self._nns,
                self._celu1, self._celu3, self._celu5,
                self._energies):
            num_energies = int(energy.shape[0])
            grad_vec7 = torch.ones((num_energies, 1), dtype=energy.dtype, device=energy.device)
            grad_vec6 = torch.matmul(grad_vec7, nn[6].weight)
            grad_vec5 = celu5.backward(grad_vec6)
            grad_vec4 = torch.matmul(grad_vec5, nn[4].weight)
            grad_vec3 = celu3.backward(grad_vec4)
            grad_vec2 = torch.matmul(grad_vec3, nn[2].weight)
            grad_vec1 = celu1.backward(grad_vec2)
            grad_vec0 = torch.matmul(grad_vec1, nn[0].weight)
            grad_aevs.append(grad_vec0)
        grad_aevs = torch.cat(grad_aevs)
        grad_aevs = grad_aevs.reshape((1, len(self), num_atoms, num_aevs))
        grad_aevs = grad_aevs.mean(dim=1)

        return grad_aevs