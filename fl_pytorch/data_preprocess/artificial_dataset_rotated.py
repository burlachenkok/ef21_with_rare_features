#!/usr/bin/env python3

from .read_file_cache import cacheItemThreadUnsafe, cacheMakeKey, cacheGetItem, cacheHasItem
from .fl_dataset import FLDataset
import numpy as np

# Import PyTorch root package import torch
import torch
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

def c_kn(k, n):
    return factorial(n) / (factorial(k) * factorial(n-k))

def generateSubsetsOfFixedCardinalityRecursively(result, cur_subset, k, d, start_from = 0):
    # Backtracking search for all subsets of cardinality "k" of set {1,2,3,...,d}
    if len(cur_subset) > k:
        # Terminate expansion - no futher solutions
        return
    elif len(cur_subset) == k:
        # Solution - add to solutions
        result.append(cur_subset.copy())
        return
    else:
        # For now we have potential solution "cur_subset" which has used some items {1,2,start_from-1}
        for i in range(start_from, d):
            cur_subset.add(i)
            generateSubsetsOfFixedCardinalityRecursively(result, cur_subset, k, d, i + 1)
            cur_subset.remove(i)

def generateSubsetsOfFixedCardinality(k, d):
    assert k < d
    cur_subset = set()
    result = list()
    generateSubsetsOfFixedCardinalityRecursively(result, cur_subset, k, d, 0)

    return result

class ArificialDatasetRotated(FLDataset):
    """
    Based FL class that loads H5 type data_preprocess.
    """
    def __init__(self, exec_ctx, args, train=None, client_id=None, transform=None, target_transform=None):
        """
        The constructor for a synthetic dataset.

        Args:
            exec_ctx: execution context from which random number generator should be use
            args: command-line argument with generation specification
            train(bool): True if we're in training mode
            client_id (int): make the view of the dataset as we work from the point of view of client client_id
            transform: input transformation applied to input attributes before feeding input into the compute of loss
            target_transform: output or label transformation applied to response variable before computation of loss
        """
        genSpecList = args.dataset_generation_spec.split(",")
        genSpec = {}
        for item in genSpecList:
            k, v = item.split(':')
            genSpec[k] = v

        optSpecList = args.algorithm_options.split(",")
        optSpec = {}
        for item in optSpecList:
            k, v = item.split(':')
            optSpec[k] = v

        self.transform = transform
        self.target_transform = target_transform

        self.b_perurbation = float(optSpec['b_perurbation'])

        # This num clients will be ignored
        self.num_clients = int(genSpec['clients'])
        self.n_client_samples = int(genSpec['samples_per_client'])

        d = int(genSpec['variables'])
        c_div_n = float(optSpec['c_div_n']) # In [0,1]
        c_value = math.ceil(self.num_clients * c_div_n)
        assert c_value > 0

        #k_div_d = c_div_n
        #k = math.ceil(k_div_d * d)
        #n = int(c_kn(k, d))
        #self.num_clients = n
        #c = math.ceil(c_div_n * self.num_clients)

        self.c_div_n = c_div_n

        L_plus_min_selector = float(optSpec['l_plus_min_selector'])   # In [0,1]

        attempts = 5

        while True:
            if attempts == 0:
                break
            #==============================================================================================================
            # Generate filling
            kSubSets = np.zeros( (self.num_clients, d) )
            for j in range(d):
                z = [i for i in range(self.num_clients)]
                exec_ctx.np_random.shuffle(z)

                for i in range(c_value):
                    kSubSets[z[i], j] = 1.0
            # ==============================================================================================================
            restart = False
            # Verify filling
            for c in range(self.num_clients):
                clientsCoordinatesLen = len(kSubSets[c, :].nonzero()[0])
                if clientsCoordinatesLen == 0:
                    restart = True
                    break

            if restart:
                attempts -= 1
                continue
            else:
                break
            # ==============================================================================================================
        assert attempts != 0

        rows = self.num_clients * self.n_client_samples
        cols = d

        if train is None or train == True:
            pass
        else:
            pass

        xSoltMultiplier = 0.5
        xSolution = np.ones(d) * xSoltMultiplier

        A = []
        B = []

        for c in range(self.num_clients):
        # ==============================================================================================================
            # Coordinates
            clientsCoordinates = kSubSets[c,:].nonzero()[0]
            clientsCoordinatesLen = len(clientsCoordinates)

            # Returns a anumpy array filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
            Ai = exec_ctx.np_random.rand(rows // self.num_clients, clientsCoordinatesLen)
            U, S, Vt = np.linalg.svd(Ai, full_matrices=True)
            L = float(genSpec['l'])
            mu = float(genSpec['mu'])
            S = np.zeros((U.shape[1], Vt.shape[0]))
            len_s = min(S.shape[0], S.shape[1])
            if len_s > 1:
                for i in range(len_s):
                    value_1 = math.sqrt((L - mu) * float(i) / (len(clientsCoordinates) - 1) + mu)
                    if c == 0:
                        value_2 = 10.0
                    else:
                        value_2 = 0.0

                    S[i][i] = value_1 * (1 - L_plus_min_selector) + value_2 * (L_plus_min_selector)
            else:
                S[0][0] = math.sqrt(L)

            Ai = U @ S @ Vt
            # Add fake columns
            Ai_sparse = np.zeros((rows // self.num_clients, d))
            clientsCoordinatesList = list(clientsCoordinates)
            clientsCoordinatesList.sort()

            for i in range(clientsCoordinatesLen):
                ai_column = Ai[:, i]
                dest_column = clientsCoordinatesList[i]
                Ai_sparse[:, dest_column] = ai_column

            Bi = Ai_sparse @ xSolution
            Bi = Bi.reshape(-1, 1)
            Bi += self.b_perurbation * ( 2 * exec_ctx.np_random.rand(*(Bi.shape)) - 1)             # random perturbation

            Ai_sparse *= math.sqrt(Ai_sparse.shape[0]/2.0)
            Bi *= math.sqrt(Ai_sparse.shape[0]/2.0)

            A.append(Ai_sparse)
            B.append(Bi)
        #===============================================================================================================
        A = np.vstack(A)
        B = np.vstack(B)

        # self.data = torch.from_numpy(A).float()
        # self.targets = torch.from_numpy(B).float()

        self.data = A
        self.targets = B

        self.targets = torch.Tensor(self.targets)
        self.data = torch.Tensor(self.data)
        self.kSubSets = kSubSets

        # ==============================================================================================================
        # Move data to GPU maybe
        # self.store_in_target_device = args.store_data_in_target_device
        # Move data to target device
        # if self.store_in_target_device:
        #    self.targets = self.targets.to(device = args.device)
        #    self.data = self.data.to(device = args.device)
        # ==============================================================================================================

        self.set_client(client_id)


    def compute_sigma_for_Z_prime(self, exec_ctx, args):
        D = self.data.shape[1]
        not_Z_prime = np.zeros((self.num_clients, D))
        eps = 1e-9

        for c in range(self.num_clients):
            self.set_client(c)
            subdata = self.data[int(self.client_id) * self.n_client_samples: (int(self.client_id) + 1) * self.n_client_samples, ...]
            a_tr_a__for_client = subdata.T @ subdata
            assert (D == a_tr_a__for_client.shape[0])
            assert (D == a_tr_a__for_client.shape[1])

            for j in range(D):
                j_feature_Linf_norm = a_tr_a__for_client[:, j].abs().max().item()

                if j_feature_Linf_norm < eps:
                    # For all samples j_features is close to Zero
                    # => (c,j) \in Z => (c,j) not in not(Z)
                    not_Z_prime[c, j] = 0.0
                else:
                    # For some samples j_features is not Zero
                    # We assume it's possibly imply that (c,j) is not in Z.
                    #  (c,j) \in not(Z)
                    not_Z_prime[c, j] = 1.0

        # Maximum cardinality by columns via looking into not Z'
        max_card = 0.0
        for j in range(D):
            cardinality_j = not_Z_prime[:, j].sum()
            if cardinality_j > max_card:
                max_card = cardinality_j

        # Sigma is quanity in [0, n]
        self.sigma_For_Zprime = max_card

        if False:
            import matplotlib.pyplot as plt

            figsize = (12, 12)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
            ax.spy(not_Z_prime, precision=0.1, markersize=5)

            fig.tight_layout()
            ax.set_title(args.dataset)
            ax.grid(True)
            fig.savefig(args.dataset + '.pdf', bbox_inches='tight', dpi=100)

    def compute_Li_for_linear_regression(self, args):
        # ==============================================================================================================
        genSpecList = args.dataset_generation_spec.split(",")
        genSpec = {}
        for item in genSpecList:
            k, v = item.split(':')
            genSpec[k] = v

        optSpecList = args.algorithm_options.split(",")
        optSpec = {}
        for item in optSpecList:
            k, v = item.split(':')
            optSpec[k] = v
        # ==============================================================================================================
        # Compute L, Li for linear regression
        # ==============================================================================================================
        A = self.data
        self.L = ((2/A.shape[0]) * torch.linalg.norm(A, 2)**2).item()
        self.Li_all_clients = []

        L_plus_matrices = 0.0
        L_plus_second = 0.0

        for c in range(self.num_clients):
            self.set_client(c)
            subdata = self.data[int(self.client_id) * self.n_client_samples:
                                (int(self.client_id) + 1) * self.n_client_samples, ...]

            Li = ((2/subdata.shape[0]) * torch.linalg.norm(subdata, 2) ** 2).item()

            L_plus_matrices = L_plus_matrices + (subdata.T @ subdata @ subdata.T @ subdata) * ( ( (2/subdata.shape[0])**2 ) * 1.0/ (self.num_clients) )

            self.Li_all_clients.append(Li)

        self.L_plus = torch.linalg.norm(L_plus_matrices, 2)
        self.L_plus = (self.L_plus)**0.5
        self.max_Li = max(self.Li_all_clients)

        assert max(self.Li_all_clients) + 1.0e+3 >= self.L
        assert max(self.Li_all_clients) - 1.0e+3 <= self.L * self.num_clients

    def set_client(self, index=None):
        """
        Set pointer to client's data_preprocess corresponding to index.
        If index is none complete dataset as union of all datapoint will be observable by higher level.

        Args:
            index(int): index of client.

        Returns:
            None
        """
        if index is None:
            self.client_id = None
            self.length = len(self.data)
        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.length = self.n_client_samples

    def load_data(self):
        """
        Explicit load all need datasets from the filesystem or cache for specific dataset instance.
        """
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.client_id is None:
            actual_index = index
        else:
            actual_index = int(self.client_id) * self.n_client_samples + index
        img, target = self.data[actual_index], self.targets[actual_index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # TODO: If __getitem__ will always fetch object from the CPU memory.
        # Suggestion use GPU memory or another GPU as a cache storage
        # return torch.from_numpy(img).float(), torch.from_numpy(target).float()
        # reference to objects from dataset (by reference)
        return img.detach(), target.detach()

    def __len__(self):
        return self.length
