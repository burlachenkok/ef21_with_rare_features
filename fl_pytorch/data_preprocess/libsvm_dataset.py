from .fl_dataset import FLDataset
from .read_file_cache import cacheItemThreadSafe, cacheMakeKey, cacheGetItem
from utils.logger import Logger
from utils import execution_context

import numpy as np
from torchvision.datasets.utils import download_url

import os
import math
import torch


#======================================================================================================================

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
#======================================================================================================================

# Train datasets URL's for trainset for several libsvm datasets
LIBSVM_DATASETS_TRAIN = {
    "a9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
    "a8a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a",
    "a7a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a7a",
    "a6a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a",
    "a5a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a",
    "a4a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a",
    "a3a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a",
    "a2a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a",
    "a1a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a",

    'w1a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w1a',
    'w2a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w2a',
    'w3a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w3a',
    'w4a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w4a',
    'w5a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w5a',
    'w6a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w6a',
    'w7a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a',
    'w8a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a',
    'w9a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w9a',

    "mushrooms": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms",
    "phishing": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing",
}

# Several datasets from libSVM are presented without split. The specific split in percentage for dataset split
LIBSVM_DATASETS_TRAIN_TRAIN_TEST_SPLIT = {
    "mushrooms": [90, 10],
    "phishing": [90, 10]
}

# Test datasets URL's for trainset for several libsvm datasets
LIBSVM_DATASETS_TEST = {
    "a9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t",
    "a8a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a8a.t",
    "a7a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a7a.t",
    "a6a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a.t",
    "a5a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a5a.t",
    "a4a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a4a.t",
    "a3a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a3a.t",
    "a2a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a2a.t",
    "a1a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t",

    'w1a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w1a.t',
    'w2a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w2a.t',
    'w3a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w3a.t',
    'w4a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w4a.t',
    'w5a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w5a.t',
    'w6a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w6a.t',
    'w7a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w7a.t',
    'w8a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t',
    'w9a': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w9a.t',

    'mushrooms': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms',
    'phishing':  'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing',
}


# ======================================================================================================================
# Utility functionality
# ======================================================================================================================
def analyzeDataset(inputfile, fieldSeparator='\t', featureNameValueSeparator=':'):
    """
    Analyze input file that contains examples and features in sparse format for a number of samples and features.

    Args:
        inputfile(str): Path for text file in which each example is presented with a single line
        fieldSeparator(str): Separator between fields that represents feature index, value pairs of each train example
        featureNameValueSeparator(str): Separator that used to split feature index from value

    Returns:
        (number of features, dictionary of various targets (key:target, value: number of examples with that target)).
    """
    f = 0
    targets = {}
    i = 0

    with open(inputfile, "r") as f_in:
        for line in f_in:
            line = line.strip("\r\n ").replace(fieldSeparator, ' ').split(' ')
            line = [item for item in line if len(item) > 0]

            # Ignore empty lines
            if len(line) == 0:
                continue

            # Positive and negative examples counters
            if float(line[0]) in targets:
                targets[float(line[0])] += 1
            else:
                targets[float(line[0])] = 1

            # With ignoring first letter
            features_pos = [int(i.split(featureNameValueSeparator)[0]) for i in line[1:]]

            # Maximum Feature number
            if len(features_pos) > 0:
                features_max_number = max(features_pos)
                f = max(features_max_number, f)
            i = i + 1

    return f, targets
# ======================================================================================================================
def readSparseInput(inputfile, features, examples, reinterpretate_labels,
                    fieldSeparator='\t', featureNameValueSeparator=':', includeBias = True):
    """
    Read input and construct training samples and target labels.

    Args:
        inputfile(str): Path for text file in which each example is presented with a single line
        features(int): Number of features. If you need bias term please add extra one.
        examples(int): Total number of examples
        fieldSeparator(str): Separator between fields that represents feature index, value pairs of each train example
        featureNameValueSeparator(str): Separator that used to split feature index from value
        includeBias(bool): Should we include bias term

    Returns:
        (number of features, dictionary of various targets).
    """

    # includeBias = False

    X = np.zeros((examples, features))
    Y = np.zeros((examples, 1))
    i = 0

    with open(inputfile, "r") as f_in:
        for line in f_in:
            line = line.strip("\r\n ").replace(fieldSeparator, ' ').split(' ')
            line = [item for item in line if len(item) > 0]

            if len(line) == 0:
                continue

            if reinterpretate_labels:
                Y[i, 0] = reinterpretate_labels[float(line[0])]
            else:
                Y[i, 0] = float(line[0])

            feature_pos = [int(i.split(featureNameValueSeparator)[0]) for i in line[1:]]
            feature_vals = [float(i.split(featureNameValueSeparator)[1]) for i in line[1:]]

            if includeBias:
                X[i, 0] = 1.0

            for j in range(len(feature_pos)):
                # https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q03:_Data_preparation
                X[i, feature_pos[j] + 1] = feature_vals[j]

            i = i + 1

    return X, Y
# ======================================================================================================================

class LibSVMDataset(FLDataset):

    @staticmethod
    def allowableDatasets():
        """Get a list of allowable dataset"""
        return LIBSVM_DATASETS_TRAIN.keys()

    """
    LibSVMDataset Dataset.
    """
    def __init__(self, exec_ctx, args, root, dataset, train=True, download=False, transform=None, target_transform=None,
                 client_id=None, num_clients=None):

        # Extra specs
        optSpecList = args.algorithm_options.split(",")
        optSpec = {}
        for item in optSpecList:
            k, v = item.split(':')
            optSpec[k] = v

        # Find URLs for data downloading
        train_ds_url = LIBSVM_DATASETS_TRAIN[dataset]
        test_ds_url = LIBSVM_DATASETS_TEST[dataset]
        self.root = root

        # Define file names into which save data
        fname_train = os.path.join(root, os.path.basename(train_ds_url))
        fname_test = os.path.join(root, os.path.basename(test_ds_url))

        # Download test and train datasets with global lock
        # (just prevent the case when two thread simultaneously download the same file)
        execution_context.torch_global_lock.acquire()

        if not os.path.isfile(fname_train):
            one_up = os.path.dirname(fname_train)
            if download:
                download_url(train_ds_url, one_up)

        if not os.path.isfile(fname_test):
            one_up = os.path.dirname(fname_test)
            if download:
                download_url(test_ds_url, one_up)

        execution_context.torch_global_lock.release()

        # Get keys for cache data
        cacheKeyExamples = cacheMakeKey(f"examples.sort={args.sort_dataset_by_class_before_split}.device={args.device}.reduction={optSpec['reduce_samples_by']}.remove_probability={optSpec['remove_probability']}",
                                        dataset)

        cacheKeyTargets = cacheMakeKey(f"targets.sort={args.sort_dataset_by_class_before_split}.device={args.device}.reduction={optSpec['reduce_samples_by']}.remove_probability={optSpec['remove_probability']}",
                                       dataset)

        cacheExamples = cacheGetItem(cacheKeyExamples)
        cacheTargets = cacheGetItem(cacheKeyTargets)

        # If dataset is not in cache
        # ==============================================================================================================
        if cacheExamples == None or cacheTargets == None:
            total_features_train, targets_train = analyzeDataset(fname_train)
            total_features_test, targets_test = analyzeDataset(fname_test)
            total_samples_train = sum(targets_train.values())
            total_samples_test = sum(targets_test.values())

            # Number of features in sparse representation of test/train datasets
            features = max(total_features_train, total_features_test)
            features += 1  # Extra offset for maybe features are indexed from #0
            features += 1  # And extra offset for bias term

            assert len(targets_train.keys()) == 2
            keys = list(targets_train.keys())
            reinterpretate_labels = {keys[0]: 0.0, keys[1]: 1.0}

            # Read data from files
            trainX, trainY = readSparseInput(fname_train, features, total_samples_train, reinterpretate_labels)
            testX, testY = readSparseInput(fname_test, features, total_samples_test, reinterpretate_labels)

            # Make splits
            if dataset in LIBSVM_DATASETS_TRAIN_TRAIN_TEST_SPLIT:
                total_samples_train_slice = math.floor(total_samples_train * LIBSVM_DATASETS_TRAIN_TRAIN_TEST_SPLIT[dataset][0] / 100.0)
                trainX = torch.Tensor(trainX[0:total_samples_train_slice, ...])
                trainY = torch.Tensor(trainY[0:total_samples_train_slice, ...])
                testX = torch.Tensor(testX[total_samples_train_slice:, ...])
                testY = torch.Tensor(testY[total_samples_train_slice:, ...])
            else:
                trainX = torch.Tensor(trainX)
                trainY = torch.Tensor(trainY)
                testX = torch.Tensor(testX)
                testY = torch.Tensor(testY)

            # Tunable parameter for experiments
            if args.sort_dataset_by_class_before_split:
                neg_samples = (trainY < 0.5).flatten()
                pos_samples = (trainY > 0.5).flatten()

                trainX_sorted = torch.cat( tuple((trainX[neg_samples], trainX[pos_samples])), axis=0)
                trainY_sorted = torch.cat( tuple((trainY[neg_samples], trainY[pos_samples])), axis=0)

                assert trainX.shape == trainX_sorted.shape
                assert trainY.shape == trainY_sorted.shape

                trainX = trainX_sorted
                trainY = trainY_sorted

                if 'reduce_samples_by' in optSpec:
                    total_length = ((len(trainX) // num_clients) * num_clients)
                    samples_per_client = total_length // num_clients
                    samples_per_client_new = math.ceil(samples_per_client * float(optSpec['reduce_samples_by']))
                    idx_new = []
                    for i in range(trainX.shape[0]):
                        sample_number = i % samples_per_client
                        if sample_number < samples_per_client_new:
                            idx_new.append(i)

                    trainX, trainY = trainX[idx_new], trainY[idx_new]

            else:
                # Do not shuffle data
                # Shuffle if we're not going work with sorted datasets
                shuffler = np.random.RandomState(seed=1234)
                idx = np.arange(trainX.shape[0])
                shuffler.shuffle(idx)
                trainX, trainY = trainX[idx], trainY[idx]

                if 'reduce_samples_by' in optSpec:
                    total_length = ((len(trainX) // num_clients) * num_clients)
                    samples_per_client = total_length // num_clients
                    samples_per_client_new = math.ceil(samples_per_client * float(optSpec['reduce_samples_by']))
                    idx_new = []
                    for i in range(trainX.shape[0]):
                        sample_number = i % samples_per_client
                        if sample_number < samples_per_client_new:
                            idx_new.append(i)

                    trainX, trainY = trainX[idx_new], trainY[idx_new]

            # Removing some dependencies
            remove_probability = 0.0
            if 'remove_probability' in optSpec:
                remove_probability = float(optSpec['remove_probability'])
            trainX = (torch.rand(trainX.shape) > remove_probability ).to(trainX.dtype) * trainX

            # Do not shuffle test data
            # Shuffle test data
            shuffler = np.random.RandomState(seed=123)
            idx = np.arange(testX.shape[0])
            shuffler.shuffle(idx)
            testX, testY = testX[idx], testY[idx]

            # Flag to to store in targets device data (if False data will store in CPU virtual memory)
            # self.store_in_target_device = args.store_data_in_target_device
            # Move data to target device
            # if self.store_in_target_device:
            #    trainX = trainX.to(device = args.device)
            #    trainY = trainY.to(device = args.device)
            #    testX  =  testX.to(device = args.device)
            #    testY  =  testY.to(device = args.device)

            # Cache train and test
            cacheItemThreadSafe(cacheKeyExamples, [trainX, testX])
            cacheItemThreadSafe(cacheKeyTargets, [trainY, testY])
            cacheExamples = cacheGetItem(cacheKeyExamples)
            cacheTargets = cacheGetItem(cacheKeyTargets)
        # ==============================================================================================================
        if train:
            self.data = cacheExamples[0]
            self.targets = cacheTargets[0]
            total_data = (len(self.data) // num_clients) * num_clients
            self.data = self.data[0:total_data, ...]
            self.targets = self.targets[0:total_data, ...]
        else:
            self.data = cacheExamples[1]
            self.targets = cacheTargets[1]

        # Setup extra information
        self.num_clients = num_clients
        self.n_client_samples = len(self.data) // self.num_clients

        logger = Logger.get(args.run_id)
        logger.info(f"Load {dataset} dataset for train({train}); number of clients: {self.num_clients}; clients has: {self.n_client_samples} samples; total samples: {len(self.data)}; label '1' classes: {(self.targets > 0.5).int().sum().item()}, label '0' classes: {(self.targets < 0.5).int().sum().item()}")

        # Setup extra transformations
        self.transform = transform
        self.target_transform = target_transform
        self.set_client(client_id)

    def spy(self, m, ax):
        from scipy.sparse import coo_matrix
        from matplotlib.patches import Rectangle
        if not isinstance(m, coo_matrix):
            m = coo_matrix(m)
        for (x, y) in zip(m.col, m.row):
            ax.add_artist(Rectangle(
                xy=(x - 0.5, y - 0.5), width=1, height=1))
        ax.set_xlim(-0.5, m.shape[1] - 0.5)
        ax.set_ylim(-0.5, m.shape[0] - 0.5)
        ax.invert_yaxis()
        ax.set_aspect(float(m.shape[0]) / float(m.shape[1]))

    def compute_sigma_for_Z_prime(self, exec_ctx, args):
        D = self.data.shape[1]
        not_Z_prime = np.zeros((self.num_clients, D))
        eps = 1e-9

        for c in range(self.num_clients):
            self.set_client(c)
            subdata = self.data[int(self.client_id) * self.n_client_samples: (int(self.client_id) + 1) * self.n_client_samples, ...]

            for j in range(D):
                j_feature_Linf_norm = subdata[:, j].abs().max().item()

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

        # Extra specs from options
        optSpecList = args.algorithm_options.split(",")
        optSpec = {}
        for item in optSpecList:
            k, v = item.split(':')
            optSpec[k] = v

        if "show_spy_with_color" in optSpec:
            import matplotlib.pyplot as plt

            figsize = (12, 12)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

            # https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors
            ax.set_xlabel("Coordinate of $x\\in \\mathbb{R}^d$")
            ax.set_ylabel("Clients {0,1,...,n-1}")
            ax.spy(not_Z_prime, precision=0.1, markersize=5, color=optSpec['show_spy_with_color'])

            fig.tight_layout()
            ax.set_title(args.dataset.upper())
            ax.grid(True)
            plt.show()
            #fig.savefig(args.dataset + '.pdf', bbox_inches='tight', dpi=100)

        pass

    def make_separable_experiment(self, exec_ctx, args):
        #==============================================================================================================
        optSpecList = args.algorithm_options.split(",")
        optSpec = {}
        for item in optSpecList:
            k, v = item.split(':')
            optSpec[k] = v
        #==============================================================================================================
        d = self.data[0, ...].numel()
        d = int(float(optSpec['d_multiplier']) * d)

        kSubSets = generateSubsetsOfFixedCardinality(int(optSpec['k_separable']), d)
        kSubSets = kSubSets * int(optSpec['n_multiplier'])
        exec_ctx.np_random.shuffle(kSubSets)
        n_sythetic_clients = len(kSubSets)

        self.num_clients = n_sythetic_clients
        self.n_sythetic_clients = n_sythetic_clients
        self.n_client_samples = len(self.data) // self.num_clients
        self.kSubSets = kSubSets

        # Prune some coordinates
        for c in range(self.num_clients):
            self.set_client(c)
            subdata = self.data[int(self.client_id) * self.n_client_samples: (int(self.client_id) + 1) * self.n_client_samples, ...]
            clientsCoordinates = kSubSets[c]

            for j in range(d):
                if j in clientsCoordinates:
                    continue

                subdata[:, j] = 0


    def compute_Li_for_logregression(self):
        # ==============================================================================================================
        # Compute L, Li for logistic regression
        # ==============================================================================================================
        examples_total = self.data.shape[0]

        # Bound for hessian for logistic regression
        # exp(x) / (1+exp(x))^2 <= 1/4

        # if False:
            # Suboptimal global step-size evaluation
        #    Z1 = torch.multiply(self.data, self.data)
        #    Z2 = torch.sum(Z1, axis = 1)
        #    Z3 = torch.sum(Z2, axis = 0)

            # Z3 is hessian for sum of losses() for all loss samples
            # Z3/examples_total is a final bound for Hessian for loss for f
        #    self.L = Z3.item() / examples_total * (1.0/4.0)
        # else:

        if True:
            self.L = np.linalg.eigvals(torch.matmul(self.data.T, self.data).numpy()).real.max() / examples_total * (1.0/4.0)

        self.Li_all_clients = []
        for c in range(self.num_clients):
            self.set_client(c)
            subdata = self.data[int(self.client_id) * self.n_client_samples: (int(self.client_id) + 1) * self.n_client_samples, ...]
            examples_client_i = subdata.shape[0]

            #if False:
            #    Z1 = torch.multiply(subdata, subdata)
            #    Z2 = torch.sum(Z1, axis=1)
            #    Z3 = torch.sum(Z2, axis=0)
            #    Li = Z3.item() / examples_client_i * (1.0/4.0)
                # Z3 is hessian for sum of losses() for all loss samples across client i
                # Z3/examples_client_i is a final bound for Hessian for loss for fi
            # else:
            if True:
                Li = np.linalg.eigvals(torch.matmul(subdata.T, subdata).numpy()).real.max() / examples_client_i * (1/4.)

            self.Li_all_clients.append(Li)
        # ==============================================================================================================

    def set_client(self, index=None):
        """ Set current client.

        Args:
            index(int): index of current client.
                        If index is None the partitioned dataset is considered as one single dataset

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
        return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of item that is fetched on behalf on current set client

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
        # TODO: Suggestion use GPU memory or another GPU as a cache storage

        #return torch.tensor(img, dtype=torch.float), torch.tensor(target, dtype=torch.float)
        return img.detach(), target.detach()      # Remove clone

    def __len__(self):
        """ Get length of dataset for a current client
        Returns:
            int: Number of train points for a current client
        """
        return self.length
