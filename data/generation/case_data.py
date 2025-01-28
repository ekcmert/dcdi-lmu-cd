import os
import argparse
import csv
import numpy as np
import dag_generator as gen
from sklearn.preprocessing import StandardScaler

class dataset_generator:
    """ Generate datasets using dag_generator.py. If a custom adjacency matrix is provided,
    we use that DAG instead of generating a random one. """

    def __init__(self, mechanism, cause, intervention_type, struct_interv_distr, noise, noise_coeff,
                 nb_nodes, expected_degree, nb_points, suffix, rescale,
                 obs_data=True, nb_interventions=3,
                 min_nb_target=1, max_nb_target=3, conservative=False, cover=False,
                 verbose=True, adjacency_matrix=None):
        """
        Args:
            mechanism, cause, noise, noise_coeff, nb_nodes, expected_degree, nb_points, suffix, rescale:
                same as before.
            intervention_type (str): 'structural' or 'parametric'
            struct_interv_distr (str): distribution for interventions if structural
            obs_data (bool): if True, one observational setting first
            nb_interventions (int): number of interventional settings after observational
            min_nb_target (int), max_nb_target (int): range of number of targets per intervention
            conservative, cover (bool): constraints on interventions
            adjacency_matrix (np.ndarray): custom adjacency matrix if dag_type='custom'
        """

        self.mechanism = mechanism
        self.cause = cause
        self.noise = noise
        self.noise_coeff = noise_coeff
        self.nb_nodes = nb_nodes
        self.expected_degree = expected_degree
        self.i_dataset = 0
        self.nb_points = nb_points
        self.suffix = suffix
        self.rescale = rescale
        self.folder = f'data_p{nb_nodes}_e{nb_nodes * expected_degree}_n{nb_points}_{suffix}'
        self.verbose = verbose
        self.struct_interv_distr = struct_interv_distr
        self.intervention_type = intervention_type
        self.obs_data = obs_data
        self.nb_interventions = nb_interventions
        self.min_nb_target = min_nb_target
        self.max_nb_target = max_nb_target
        self.conservative = conservative
        self.cover = cover
        self.adjacency_matrix = adjacency_matrix

        # Create folder if not exists
        try:
            os.mkdir(self.folder)
        except OSError:
            pass

        # Check parameters
        self._checkup()

        # Initialize generator
        if self.adjacency_matrix is not None:
            self.generator = gen.DagGenerator(self.mechanism,
                                              noise=self.noise,
                                              noise_coeff=self.noise_coeff,
                                              cause=self.cause,
                                              npoints=self.nb_points,
                                              nodes=self.nb_nodes,
                                              expected_density=self.expected_degree,
                                              dag_type='custom',
                                              rescale=self.rescale,
                                              adjacency_matrix=self.adjacency_matrix)
        else:
            self.generator = gen.DagGenerator(self.mechanism,
                                              noise=self.noise,
                                              noise_coeff=self.noise_coeff,
                                              cause=self.cause,
                                              npoints=self.nb_points,
                                              nodes=self.nb_nodes,
                                              expected_density=self.expected_degree,
                                              dag_type='erdos',
                                              rescale=self.rescale)

    def _checkup(self):
        possible_mechanisms = ["linear","polynomial","sigmoid_add","sigmoid_mix","gp_add","gp_mix",
                               "anm","nn","nn_add","pnl_gp_mechanism","pnl_mult_mechanism","post_nonlinear","x","circle","adn"]
        possible_causes = ["gmm_cause","gaussian","variable_gaussian","uniform","uniform_positive"]
        possible_noises = ["gaussian","variable_gaussian","uniform","laplace","absolute_gaussian","nn"]

        assert self.mechanism in possible_mechanisms, \
                f"mechanism doesn't exist. It has to be in {possible_mechanisms}"
        assert self.cause in possible_causes, \
                f"initial cause doesn't exist. It has to be in {possible_causes}"
        assert self.noise in possible_noises, \
                f"noise doesn't exist. It has to be in {possible_noises}"

        if self.intervention_type not in ["structural", "parametric"]:
            raise ValueError("intervention_type must be 'structural' or 'parametric'")

        assert self.nb_interventions <= self.nb_points, \
                "nb_interventions should be smaller or equal to nb_points"
        assert self.min_nb_target <= self.max_nb_target, \
                "min_nb_target should be smaller or equal to max_nb_target"
        assert self.max_nb_target <= self.nb_nodes, \
                "max_nb_target should be smaller or equal to nb_nodes"
        if self.cover:
            assert self.max_nb_target * self.nb_interventions >= self.nb_nodes, \
                    "To cover all nodes, increase nb_interventions or max_nb_target"
        if self.conservative and not self.obs_data:
            assert (self.nb_nodes - self.max_nb_target) * self.nb_interventions >= self.nb_nodes

    def _is_conservative(self, elements, lists):
        for e in elements:
            conservative = False
            for l in lists:
                if e not in l:
                    conservative = True
                    break
            if not conservative:
                return False
        return True

    def _is_covering(self, elements, lists):
        return set(elements) == self._union(lists)

    def _union(self, lists):
        union_set = set()
        for l in lists:
            union_set = union_set.union(set(l))
        return union_set

    def _pick_targets(self, nb_max_iteration=100000):
        nodes = np.arange(self.nb_nodes)
        not_correct = True
        i = 0

        while(not_correct and i < nb_max_iteration):
            targets = []
            not_correct = False
            i += 1

            for _ in range(self.nb_interventions):
                nb_targets = np.random.randint(self.min_nb_target, self.max_nb_target+1, 1)
                intervention = np.random.choice(self.nb_nodes, nb_targets, replace=False)
                targets.append(intervention)

            if self.cover and not self._is_covering(nodes, targets):
                not_correct = True
            if self.conservative and not self.obs_data and not self._is_conservative(nodes, targets):
                not_correct = True

        if i == nb_max_iteration:
            raise ValueError("Could not generate appropriate targets after many attempts.")

        for j, t in enumerate(targets):
            targets[j] = np.sort(t)

        return targets

    def _save_data(self, i, data, regimes=None, mask=None):
        if mask is None:
            data_path = os.path.join(self.folder, f'data{i+1}.npy')
            np.save(data_path, data)
        else:
            data_path = os.path.join(self.folder, f'data_interv{i+1}.npy')
            np.save(data_path, data)

            data_path = os.path.join(self.folder, f'intervention{i+1}.csv')
            with open(data_path, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(mask)

        if regimes is not None:
            regime_path = os.path.join(self.folder, f'regime{i+1}.csv')
            with open(regime_path, 'w', newline="") as f:
                writer = csv.writer(f)
                for regime in regimes:
                    writer.writerow([regime])

    def generate(self, intervention=False):
        if self.generator is None:
            if self.verbose:
                print('Initializing generator...')
            self.generator = gen.DagGenerator(self.mechanism,
                                              noise=self.noise,
                                              noise_coeff=self.noise_coeff,
                                              cause=self.cause,
                                              npoints=self.nb_points,
                                              nodes=self.nb_nodes,
                                              expected_density=self.expected_degree,
                                              dag_type='custom' if self.adjacency_matrix is not None else 'erdos',
                                              rescale=self.rescale,
                                              adjacency_matrix=self.adjacency_matrix)

        if intervention:
            data = np.zeros((self.nb_points, self.nb_nodes))
            if self.obs_data:
                div = self.nb_interventions + 1
            else:
                div = self.nb_interventions

            points_per_interv = [self.nb_points // div + (1 if x < self.nb_points % div else 0) for x in range(div)]
            regimes = []
            mask_intervention = []

            # pick random targets
            target_list = self._pick_targets()

            # do interventions
            for j in range(self.nb_interventions):
                targets = target_list[j]
                dataset, _ = self.generator.intervene(self.intervention_type,
                                                      targets,
                                                      points_per_interv[j],
                                                      self.struct_interv_distr)
                self.generator.reinitialize()

                start = sum(points_per_interv[:j])
                end = start + points_per_interv[j]
                data[start:end, :] = dataset
                mask_intervention.extend([targets for _ in range(points_per_interv[j])])
                regimes.extend([j+1 for _ in range(points_per_interv[j])])

            if self.obs_data:
                j = self.nb_interventions
                self.generator.change_npoints(points_per_interv[j])
                dataset, _ = self.generator.generate()
                start = sum(points_per_interv[:j])
                end = start + points_per_interv[j]
                data[start:end, :] = dataset
                mask_intervention.extend([[] for _ in range(points_per_interv[j])])
                regimes.extend([0 for _ in range(points_per_interv[j])])

            if self.rescale:
                scaler = StandardScaler()
                data = scaler.fit_transform(data)

            self._save_data(self.i_dataset, data, regimes, mask_intervention)

        else:
            self.generator.change_npoints(self.nb_points)
            data, _ = self.generator.generate()
            if self.rescale:
                scaler = StandardScaler()
                data = scaler.fit_transform(data)
            self._save_data(self.i_dataset, data)

        # save DAG
        self.generator.save_dag_cpdag(self.folder, self.i_dataset+1)

if __name__ == "__main__":
    # Define a suitable DAG for our macroeconomic scenario with 10 nodes
    # Nodes:
    # 0: MonetaryPolicy
    # 1: FiscalPolicy
    # 2: Inflation
    # 3: Unemployment
    # 4: ConsumerConfidence
    # 5: Investment
    # 6: Exports
    # 7: Imports
    # 8: Productivity
    # 9: EconomicGrowth

    # Edges (directed):
    # MonetaryPolicy->Inflation (0->2)
    # MonetaryPolicy->Unemployment (0->3)
    # MonetaryPolicy->Investment (0->5)
    # FiscalPolicy->Inflation (1->2)
    # FiscalPolicy->ConsumerConfidence (1->4)
    # FiscalPolicy->Investment (1->5)
    # Inflation->ConsumerConfidence (2->4)
    # Unemployment->ConsumerConfidence (3->4)
    # ConsumerConfidence->Investment (4->5)
    # Investment->Productivity (5->8)
    # Productivity->Exports (8->6)
    # Productivity->Imports (8->7)
    # Productivity->EconomicGrowth (8->9)
    # Investment->Imports (5->7)
    # Exports->EconomicGrowth (6->9)
    # Imports->EconomicGrowth (7->9)

    adjacency_matrix = np.zeros((10, 10))
    adjacency_matrix[0,2] = 1
    adjacency_matrix[0,3] = 1
    adjacency_matrix[0,5] = 1
    adjacency_matrix[1,2] = 1
    adjacency_matrix[1,4] = 1
    adjacency_matrix[1,5] = 1
    adjacency_matrix[2,4] = 1
    adjacency_matrix[3,4] = 1
    adjacency_matrix[4,5] = 1
    adjacency_matrix[5,8] = 1
    adjacency_matrix[8,6] = 1
    adjacency_matrix[8,7] = 1
    adjacency_matrix[8,9] = 1
    adjacency_matrix[5,7] = 1
    adjacency_matrix[6,9] = 1
    adjacency_matrix[7,9] = 1

    # Choose parameters
    mechanism = 'linear'          # linear mechanisms for simplicity
    cause = 'gaussian'            # initial root distributions
    noise = 'gaussian'            # noise distribution
    noise_coeff = 0.4
    nb_nodes = 10
    expected_degree = 2  # Not relevant for custom DAG, just a placeholder
    nb_points = 10000
    suffix = 'macro_policy_scenario'
    rescale = True
    intervention_type = 'structural'
    struct_interv_distr = 'uniform'
    obs_data = True
    nb_interventions = 3
    min_nb_target = 1
    max_nb_target = 3
    conservative = False
    cover = False

    generator = dataset_generator(mechanism, cause, intervention_type,
                                  struct_interv_distr, noise, noise_coeff,
                                  nb_nodes, expected_degree, nb_points,
                                  suffix, rescale, obs_data, nb_interventions,
                                  min_nb_target, max_nb_target, conservative, cover,
                                  adjacency_matrix=adjacency_matrix)

    generator.generator.init_variables()

    # Generate observational data
    generator.generate(intervention=False)

    # Generate interventional data
    generator.generate(intervention=True)
