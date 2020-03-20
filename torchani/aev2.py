import numpy as np
import torch

from .aev import AEVComputer, SpeciesAEV


class AEVComputer2(AEVComputer):

    def __init__(self, **args):
        super().__init__(**args)

    def compute_scale(self, distances, cutoff):

        assert type(cutoff) is float

        scale = 0.5 * torch.cos(distances * (np.pi / cutoff)) + 0.5
        zero = torch.tensor([0], dtype=scale.dtype, device=scale.device)
        scale = torch.where(distances < cutoff, scale, zero)

        return scale

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

    def compute_radial_terms(self, distances):

        num_atoms = int(self._coordinates.shape[1])

        assert len(distances.shape) == 3
        assert distances.shape[0] == num_atoms
        assert distances.shape[1] == num_atoms
        assert distances.shape[2] == 1

        assert len(self.EtaR.shape) == 2
        assert self.EtaR.shape[0] == 1
        assert self.EtaR.shape[1] == 1
        EtaR = float(self.EtaR)
        assert len(self.ShfR.shape) == 2
        assert self.ShfR.shape[0] == 1
        assert self.ShfR.shape[1] == 16
        ShfR = self.ShfR.reshape(16)

        # Compute radial terms
        self._radial_centers = distances - ShfR
        self._radial_exponents = -EtaR * self._radial_centers ** 2
        terms = 0.25 * torch.exp(self._radial_exponents)

        return terms

    def compute_grad_radial_terms(self, grad_terms):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_terms.shape) == 3
        assert grad_terms.shape[0] == num_atoms
        assert grad_terms.shape[1] == num_atoms
        assert grad_terms.shape[2] == self.radial_sublength

        grad_exps = self._aev_radial_terms * grad_terms
        grad_cents = -2 * float(self.EtaR) * self._radial_centers * grad_exps
        grad_dists = grad_cents.sum(2)

        return grad_dists

    def compute_grad_radial_scale(self, grad_terms):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_terms.shape) == 3
        assert grad_terms.shape[0] == num_atoms
        assert grad_terms.shape[1] == num_atoms
        assert grad_terms.shape[2] == self.radial_sublength

        scale = self._ave_radial_scale.repeat_interleave(self.radial_sublength, dim=2)
        grad_dists = torch.autograd.grad(scale, self._distances, grad_terms, retain_graph=True)[0]

        return grad_dists

    def compute_radial_aev(self, distances):

        num_atoms = int(self._coordinates.shape[1])

        assert len(distances.shape) == 2
        assert distances.shape[0] == num_atoms
        assert distances.shape[1] == num_atoms

        # Compute radial terms
        distances = distances.reshape((num_atoms, num_atoms, 1))
        self._aev_radial_terms = self.compute_radial_terms(distances)

        # Scale terms
        self._ave_radial_scale = self.compute_scale(distances, self.Rcr)
        terms = self._aev_radial_terms * self._ave_radial_scale

        # Filter self-interaction terms
        self._aev_radial_valid = distances != 0.0
        zero = torch.tensor([0], dtype=terms.dtype, device=terms.device)
        terms = torch.where(self._aev_radial_valid, terms, zero)

        # Compute radial AEV
        terms = terms.reshape(num_atoms, num_atoms * self.radial_sublength)
        radial_aev = torch.matmul(terms, self.radial_mapping)

        return radial_aev

    def compute_grad_radial(self, grad_aev):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_aev.shape) == 2
        assert grad_aev.shape[0] == num_atoms
        assert grad_aev.shape[1] == self.radial_length

        # Compute the gradient of radial AEV
        grad_terms = torch.matmul(grad_aev, self.radial_mapping.t())
        grad_terms = grad_terms.reshape((num_atoms, num_atoms, self.radial_sublength))

        # Filter the gradient of self-interaction terms
        zero = torch.tensor([0], dtype=grad_terms.dtype, device=grad_terms.device)
        grad_terms = torch.where(self._aev_radial_valid, grad_terms, zero)

        # Compte the gradient of scaling
        grad_dists_terms = self.compute_grad_radial_terms(self._ave_radial_scale * grad_terms)
        grad_dists_scale = self.compute_grad_radial_scale(self._aev_radial_terms * grad_terms)
        grad_dists = grad_dists_terms + grad_dists_scale

        grad_vecs = torch.autograd.grad(self._distances, self._vectors, grad_dists, retain_graph=True)[0]

        return grad_vecs

    def compute_angles(self, distances, vectors):

        num_atoms = int(self._coordinates.shape[1])

        assert len(distances.shape) == 2
        assert distances.shape[0] == num_atoms
        assert distances.shape[1] == num_atoms

        assert len(vectors.shape) == 3
        assert vectors.shape[0] == num_atoms
        assert vectors.shape[0] == num_atoms
        assert vectors.shape[2] == 3

        # Compute scaling
        dist12 = distances.reshape((num_atoms, 1, num_atoms)) *\
                 distances.reshape((num_atoms, num_atoms, 1))

        self._angles_valid = torch.abs(dist12) > 0
        one = torch.tensor([1], dtype=dist12.dtype, device=dist12.device) # Has to be non-zero
        self._angles_dist12 = torch.where(self._angles_valid, dist12, one)
        self._angles_scale = 0.95/self._angles_dist12

        # Compute dot product
        vec_prod = vectors.reshape((num_atoms, 1, num_atoms, 3)) *\
                   vectors.reshape((num_atoms, num_atoms, 1, 3))
        self._angles_dot_prod = vec_prod.sum(3)

        # Compute angles
        self._angular_cosines =  self._angles_scale * self._angles_dot_prod
        angles = torch.acos(self._angular_cosines)

        return angles

    def compute_grad_angles(self, grad_angles):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_angles.shape) == 3
        assert grad_angles.shape[0] == num_atoms
        assert grad_angles.shape[1] == num_atoms
        assert grad_angles.shape[2] == num_atoms

        # Compute the gradients of cosines
        grad_cosines = -1 / torch.sqrt(1 - self._angular_cosines ** 2) * grad_angles
        grad_cosine_scale = self._angles_dot_prod * grad_cosines
        grad_cosine_dot_prod = self._angles_scale * grad_cosines

        # Compute the gradients of scales
        grad_dists12 = -0.95 / self._angles_dist12 ** 2 * grad_cosine_scale
        zero = torch.tensor([0], dtype=grad_dists12.dtype, device=grad_dists12.device)
        grad_dists12 = torch.where(self._angles_valid, grad_dists12, zero)

        dist1 = self._distances.reshape((num_atoms, 1, num_atoms))
        dist2 = self._distances.reshape((num_atoms, num_atoms, 1))
        grad_dist_dist1 = torch.sum(dist2 * grad_dists12, 1)
        grad_dist_dist2 = torch.sum(dist1 * grad_dists12, 2)

        grad_vecs_dist1 = torch.autograd.grad(self._distances, self._vectors, grad_dist_dist1, retain_graph=True)[0]
        grad_vecs_dist2 = torch.autograd.grad(self._distances, self._vectors, grad_dist_dist2, retain_graph=True)[0]

        # Compute the gradient of dot products
        vec1 = self._vectors.reshape((num_atoms, 1, num_atoms, 3))
        vec2 = self._vectors.reshape((num_atoms, num_atoms, 1, 3))
        grad_cosine_dot_prod = grad_cosine_dot_prod.reshape((num_atoms, num_atoms, num_atoms, 1))
        grad_vecs_vec1 = torch.sum(vec2 * grad_cosine_dot_prod, 1)
        grad_vecs_vec2 = torch.sum(vec1 * grad_cosine_dot_prod, 2)

        grad_vecs = (grad_vecs_dist1 + grad_vecs_dist2) + (grad_vecs_vec1 + grad_vecs_vec2)

        return grad_vecs

    def compute_angular_terms(self, distances, vectors):

        num_atoms = int(self._coordinates.shape[1])

        assert len(distances.shape) == 2
        assert distances.shape[0] == num_atoms
        assert distances.shape[1] == num_atoms

        assert len(vectors.shape) == 3
        assert vectors.shape[0] == num_atoms
        assert vectors.shape[0] == num_atoms
        assert vectors.shape[2] == 3

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
        Zeta = int(self.Zeta)
        assert len(self.EtaA.shape) == 4
        assert self.EtaA.shape[0] == 1
        assert self.EtaA.shape[1] == 1
        assert self.EtaA.shape[2] == 1
        assert self.EtaA.shape[3] == 1
        EtaA = int(self.EtaA)
        assert len(self.ShfA.shape) == 4
        assert self.ShfA.shape[0] == 1
        assert self.ShfA.shape[1] == 1
        assert self.ShfA.shape[2] == 4
        assert self.ShfA.shape[3] == 1
        ShfA = self.ShfA.reshape((4, 1))

        # Computer factor1
        angles = self.compute_angles(distances, vectors).reshape((num_atoms, num_atoms, num_atoms, 1, 1))
        self._angular_factor1_center = angles - ShfZ
        self._angular_factor1_base = 0.5 * (1 + torch.cos(self._angular_factor1_center))
        self._angular_factor1 = self._angular_factor1_base ** Zeta

        # Compute mean distances
        mean_distances = 0.5 * (distances.reshape((num_atoms, 1, num_atoms, 1, 1)) +\
                                distances.reshape((num_atoms, num_atoms, 1, 1, 1)))

        # Computer factor2
        self._angular_factor2_center = mean_distances - ShfA
        self._angular_factor2_base = -EtaA * self._angular_factor2_center ** 2
        self._angular_factor2 = torch.exp(self._angular_factor2_base)

        # Compute terms
        terms = self._angular_factor1 * self._angular_factor2
        terms = terms.reshape((num_atoms, num_atoms, num_atoms, self.angular_sublength))

        return terms

    def compute_angular_aev(self, distances, vectors):

        assert len(distances.shape) == 2
        assert distances.shape[0] == distances.shape[1]

        assert len(vectors.shape) == 3
        assert vectors.shape[0] == vectors.shape[1]
        assert vectors.shape[0] == distances.shape[0]
        assert vectors.shape[2] == 3

        num_atoms = int(distances.shape[0])

        # Compute terms
        self._aev_angular_terms = self.compute_angular_terms(distances, vectors)

        # Scale terms
        self._aev_angular_scale = self.compute_scale(distances, self.Rca)
        self._aev_angular_scale = self._aev_angular_scale.reshape((num_atoms, 1, num_atoms, 1)) *\
                                  self._aev_angular_scale.reshape((num_atoms, num_atoms, 1, 1))
        terms = self._aev_angular_terms * self._aev_angular_scale

        # Filter self-interaction terms
        self._aev_angular_valid = (distances.reshape((1, num_atoms, num_atoms, 1)) != 0.0) &\
                                  (distances.reshape((num_atoms, 1, num_atoms, 1)) != 0.0) &\
                                  (distances.reshape((num_atoms, num_atoms, 1, 1)) != 0.0)
        zero = torch.tensor([0], dtype=terms.dtype, device=terms.device)
        terms = torch.where(self._aev_angular_valid, terms, zero)

        # Compute angular AEV
        terms = terms.reshape(num_atoms, num_atoms ** 2 * self.angular_sublength)
        angular_aev = torch.matmul(terms, self.angular_mapping)

        return angular_aev

    def compute_grad_angular_terms(self, grad_terms):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_terms.shape) == 4
        assert grad_terms.shape[0] == num_atoms
        assert grad_terms.shape[1] == num_atoms
        assert grad_terms.shape[2] == num_atoms
        assert grad_terms.shape[3] == self.angular_sublength

        grad_terms = grad_terms.reshape((num_atoms, num_atoms, num_atoms, 4, 8))
        grad_terms_factor1 = self._angular_factor2 * grad_terms
        grad_terms_factor2 = self._angular_factor1 * grad_terms

        # Compute the gradient of factor1
        Zeta = int(self.Zeta)
        grad_base_factor1 = Zeta * self._angular_factor1_base ** (Zeta - 1) * grad_terms_factor1
        grad_center_factor1 = -0.5 * torch.sin(self._angular_factor1_center) * grad_base_factor1
        grad_angles = grad_center_factor1.sum((3, 4))
        grad_vecs_factor1 = self.compute_grad_angles(grad_angles)

        # Compute the gradient of factor2
        grad_base_factor2 = self._angular_factor2 * grad_terms_factor2
        grad_center_factor2 = -2 * int(self.EtaA) * self._angular_factor2_center * grad_base_factor2
        grad_mean_factor2 = grad_center_factor2.sum((3, 4))

        # Compute the gradient of mean distances
        grad_dists_factor2 = 0.5 * (grad_mean_factor2.sum(1) + grad_mean_factor2.sum(2))
        grad_vecs_factor2 = torch.autograd.grad(self._distances, self._vectors, grad_dists_factor2, retain_graph=True)[0]

        grad_vecs = grad_vecs_factor1 + grad_vecs_factor2

        return grad_vecs

    def compute_grad_angular_scale(self, grad_terms):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_terms.shape) == 4
        assert grad_terms.shape[0] == num_atoms
        assert grad_terms.shape[1] == num_atoms
        assert grad_terms.shape[2] == num_atoms
        assert grad_terms.shape[3] == self.angular_sublength

        scale = self._aev_angular_scale.repeat_interleave(self.angular_sublength, dim=3)
        grad_dists = torch.autograd.grad(scale, self._distances, grad_terms, retain_graph=True)[0]

        return grad_dists

    def compute_grad_angular(self, grad_aev):

        num_atoms = int(self._coordinates.shape[1])

        assert len(grad_aev.shape) == 2
        assert grad_aev.shape[0] == num_atoms
        assert grad_aev.shape[1] == self.angular_length

        # Compute the gradient of radial AEV
        grad_terms = torch.matmul(grad_aev, self.angular_mapping.t())
        grad_terms = grad_terms.reshape((num_atoms, num_atoms, num_atoms, self.angular_sublength))

        # Filter the gradient of self-interaction terms
        zero = torch.tensor([0], dtype=grad_terms.dtype, device=grad_terms.device)
        grad_terms = torch.where(self._aev_angular_valid, grad_terms, zero)

        # Compte the gradient of scaling
        grad_vecs_terms = self.compute_grad_angular_terms(self._aev_angular_scale * grad_terms)
        grad_dists_scale = self.compute_grad_angular_scale(self._aev_angular_terms * grad_terms)
        grad_vecs_scale = torch.autograd.grad(self._distances, self._vectors, grad_dists_scale, retain_graph=True)[0]
        grad_vecs = grad_vecs_terms + grad_vecs_scale

        return grad_vecs

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
        self._vectors = vectors
        self._distances = distances

        # Compute AEV components
        radial_aev = self.compute_radial_aev(distances)
        angular_aev = self.compute_angular_aev(distances, vectors)

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

        # Split AEV components
        grad_aev_radial, grad_aev_angular = torch.split(grad_aevs[0], (self.radial_length, self.angular_length), dim=1)

        # Compute the gradient of AEV components
        grad_vecs = self.compute_grad_radial(grad_aev_radial) +\
                      self.compute_grad_angular(grad_aev_angular)

        grad_coords = torch.autograd.grad(self._vectors, self._coordinates, grad_vecs, retain_graph=True)[0]

        grad_coords.reshape((1, num_atoms, 3))

        return grad_coords
