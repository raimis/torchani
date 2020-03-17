import numpy as np
import torch

from .aev import AEVComputer, SpeciesAEV


class AEVComputer2(AEVComputer):

    def __init__(self, **args):
        super().__init__(**args)

    def compute_cutoff(self, distances, cutoff):

        assert type(cutoff) is float

        value = 0.5 * torch.cos(distances * (np.pi / cutoff)) + 0.5
        zero = torch.tensor([0], dtype=value.dtype, device=value.device)
        value = torch.where(distances < cutoff, value, zero)

        return value

    def construct_radial_mapping(self, species):

        assert len(species.shape) == 1

        num_atoms = species.shape[0]

        mapping = np.zeros((num_atoms, self.radial_sublength, self.num_species, self.radial_sublength), dtype=np.float32)
        for i1, s1 in enumerate(species.cpu().numpy()):
            for i in range(self.radial_sublength):
                mapping[i1, i, s1, i] = 1
        mapping = mapping.reshape(((num_atoms * self.radial_sublength, self.radial_length)))

        self.register_buffer('radial_mapping', torch.tensor(mapping, device=species.device))

    def construct_angular_mapping(self, species):

        assert len(species.shape) == 1

        num_atoms = species.shape[0]
        num_species_pairs = self.angular_length // self.angular_sublength

        assert len(self.triu_index.shape) == 2
        assert self.triu_index.shape[0] == self.num_species
        assert self.triu_index.shape[1] == self.num_species

        mapping = np.zeros((num_atoms, num_atoms, self.angular_sublength,
                            num_species_pairs, self.angular_sublength), dtype=np.float32)
        for i1, s1 in enumerate(species.cpu().numpy()):
            for i2, s2 in enumerate(species.cpu().numpy()):
                for i in range(self.angular_sublength):
                    mapping[i1, i2, i, self.triu_index[s1, s2], i] = 1
        mapping = mapping.reshape(((num_atoms ** 2 * self.angular_sublength,
                                    num_species_pairs * self.angular_sublength)))

        self.register_buffer('angular_mapping', torch.tensor(mapping, device=species.device))

    def compute_radial_aev(self, distances):

        assert len(distances.shape) == 2
        assert distances.shape[0] == distances.shape[1]

        num_atoms = int(distances.shape[0])
        distances = distances.reshape((num_atoms, num_atoms, 1))

        # Compute cutoff matrix
        cutoff = self.compute_cutoff(distances, self.Rcr)

        # Compute radial terms
        assert len(self.EtaR.shape) == 2
        assert self.EtaR.shape[0] == 1
        assert self.EtaR.shape[1] == 1
        EtaR = float(self.EtaR)
        assert len(self.ShfR.shape) == 2
        assert self.ShfR.shape[0] == 1
        assert self.ShfR.shape[1] == 16
        ShfR = self.ShfR.reshape(16)
        terms = 0.25 * torch.exp(-EtaR * (distances - ShfR) ** 2) * cutoff

        # Filter self-interaction terms
        zero = torch.tensor([0], dtype=terms.dtype, device=terms.device)
        terms = torch.where(distances == 0.0, zero, terms)

        # Compute radial AEV
        terms = terms.reshape(num_atoms, num_atoms * self.radial_sublength)
        radial_aev = torch.matmul(terms, self.radial_mapping)

        return radial_aev

    def compute_grad_coords_radial(self, grad_aev):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_aev.shape) == 2
        assert grad_aev.shape[0] == num_atoms
        assert grad_aev.shape[1] == self.radial_length

        grad_coords = torch.autograd.grad(self._aev_radial, self._coordinates, grad_aev, retain_graph=True)[0]

        return grad_coords

    def compute_angular_aev(self, distances, vectors):

        assert len(distances.shape) == 2
        assert distances.shape[0] == distances.shape[1]

        assert len(vectors.shape) == 3
        assert vectors.shape[0] == vectors.shape[1]
        assert vectors.shape[0] == distances.shape[0]
        assert vectors.shape[2] == 3

        num_atoms = int(distances.shape[0])

        # Compute mean distance tensor
        dist1 = distances.reshape((num_atoms, 1, num_atoms, 1, 1))
        dist2 = distances.reshape((num_atoms, num_atoms, 1, 1, 1))
        mean_dists = 0.5 * (dist1 + dist2)

        # Compute angle tensor
        vec1 = vectors.reshape((num_atoms, 1, num_atoms, 1, 3))
        vec2 = vectors.reshape((num_atoms, num_atoms, 1, 1, 3))
        epsilon = torch.tensor(1e-45, dtype=dist1.dtype, device=dist1.device)
        similarity = torch.sum(vec1 * vec2, dim=4, keepdim=True)/torch.max(dist1 * dist2, epsilon)
        angles = torch.acos(0.95 * similarity)

        # Compute the factors
        assert len(self.ShfZ.shape) == 4
        assert self.ShfZ.shape[0] == 1
        assert self.ShfZ.shape[1] == 1
        assert self.ShfZ.shape[2] == 1
        assert self.ShfZ.shape[3] == 8
        ShfZ = self.ShfZ.reshape((1, 8))
        assert len(self.Zeta.shape) == 4
        assert self.Zeta.shape[0] == 1
        assert self.Zeta.shape[1] == 1
        assert self.Zeta.shape[2] == 1
        assert self.Zeta.shape[3] == 1
        Zeta = float(self.Zeta)
        assert len(self.EtaA.shape) == 4
        assert self.EtaA.shape[0] == 1
        assert self.EtaA.shape[1] == 1
        assert self.EtaA.shape[2] == 1
        assert self.EtaA.shape[3] == 1
        EtaA = float(self.EtaA)
        assert len(self.ShfA.shape) == 4
        assert self.ShfA.shape[0] == 1
        assert self.ShfA.shape[1] == 1
        assert self.ShfA.shape[2] == 4
        assert self.ShfA.shape[3] == 1
        ShfA = self.ShfA.reshape((4, 1))
        factor1 = (0.5 * (1 + torch.cos(angles - ShfZ))) ** Zeta
        factor2 = torch.exp(-EtaA * (mean_dists - ShfA) ** 2)

        # Compute cutoff tensor
        cutoff = self.compute_cutoff(distances, self.Rca)
        cutoff = cutoff.reshape((num_atoms, 1, num_atoms, 1, 1)) *\
                 cutoff.reshape((num_atoms, num_atoms, 1, 1, 1))

        # Compute terms
        terms = factor1 * factor2 * cutoff
        terms = terms.reshape((num_atoms, num_atoms, num_atoms, self.angular_sublength))

        # Filter self-interaction terms
        valid = (distances.reshape((1, num_atoms, num_atoms, 1)) == 0.0) |\
                (distances.reshape((num_atoms, 1, num_atoms, 1)) == 0.0) |\
                (distances.reshape((num_atoms, num_atoms, 1, 1)) == 0.0)
        zero = torch.tensor([0], dtype=terms.dtype, device=terms.device)
        terms = torch.where(valid, zero, terms)

        # Compute angular AEV
        terms = terms.reshape(num_atoms, num_atoms ** 2 * self.angular_sublength)
        angular_aev = torch.matmul(terms, self.angular_mapping)

        return angular_aev

    def compute_grad_coords_angular(self, grad_aev):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_aev.shape) == 2
        assert grad_aev.shape[0] == num_atoms
        assert grad_aev.shape[1] == self.angular_length

        grad_coords = torch.autograd.grad(self._aev_angular, self._coordinates, grad_aev, retain_graph=True)[0]

        return grad_coords

    def forward(self, species_coordinates, cell=None, pbc=None):

        species_, coordinates_ = species_coordinates

        assert len(species_.shape) == 2
        assert species_.shape[0] == 1

        assert len(coordinates_.shape) == 3
        assert coordinates_.shape[0] == 1
        assert coordinates_.shape[1] == species_.shape[1]
        assert coordinates_.shape[2] == 3
        self._coordinates = coordinates_

        assert cell is None
        assert pbc is None

        num_atoms = int(species_.shape[1])
        species = species_.reshape(num_atoms)
        coordinates = coordinates_.reshape((num_atoms, 3))

        # Construct mapping matrices
        self.construct_radial_mapping(species)
        self.construct_angular_mapping(species)

        # Compute vector and distance matrices
        vectors = coordinates.reshape((num_atoms, 1, 3)) - coordinates.reshape((1, num_atoms, 3))
        distances = vectors.norm(2, dim=2)

        # Compute AEV components
        radial_aev = self.compute_radial_aev(distances)
        self._aev_radial = radial_aev
        angular_aev = self.compute_angular_aev(distances, vectors)
        self._aev_angular = angular_aev

        # Merge AEV components
        aev = torch.cat([radial_aev, angular_aev], dim=1)
        aev = aev.reshape((1, num_atoms, self.aev_length))

        return SpeciesAEV(species_, aev)

    def backward(self, grad_aevs):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_aevs.shape) == 3
        assert grad_aevs.shape[0] == 1
        assert grad_aevs.shape[1] == num_atoms
        assert grad_aevs.shape[2] == self.aev_length

        grad_aev_radial, grad_aev_angular = torch.split(grad_aevs[0], [self.radial_length, self.angular_length], dim=1)

        # Compute the gradient of AEV components
        grad_coords_radial = self.compute_grad_coords_radial(grad_aev_radial)
        grad_coords_angular = self.compute_grad_coords_angular(grad_aev_angular)

        # Sum the gradients of AEV components
        grad_coords = grad_coords_radial + grad_coords_angular
        grad_coords.reshape((1, num_atoms, 3))

        return grad_coords
