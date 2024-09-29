########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains the Lattice class. The Lattice class describes the design space and contains the information
# related to the explored configurations.
########################################################################################################################

import GA_tree
import numpy



class Lattice:

    def __init__(self, lattice_descriptor, max_distance):
        self.original_descriptor = lattice_descriptor
        self.discretized_descriptor = self.discretize_dataset(lattice_descriptor)
        self.lattice = GA_tree.Tree('lattice')
        # self.radii_struct = self.radii_vectors()
        self.max_distance = max_distance

    def discretize_dataset(self, lattice_descriptor):
        discretized_feature = []
        for feature in lattice_descriptor:
            # tmp = []
            # for x in self._frange(0, 1, 1. / (len(feature) - 1)):
            #     tmp.append(x)
            tmp = numpy.linspace(0, 1, len(feature))
            discretized_feature.append(tmp.tolist())
            # discretized_feature.append(tmp)

        return discretized_feature

    def _frange(self, start, stop, step):
        x = start
        output = []
        while x <= stop:
            output.append(x)
            x += step
        output[-1] = 1
        return output

    def revert_discretized_config(self, config):
        tmp = []
        for i in range(0, len(config)):
            for j in range(0, len(self.discretized_descriptor[i])):
                if numpy.isclose(self.discretized_descriptor[i][j], config[i], atol=0.000001):
                    tmp.append(self.original_descriptor[i][j])
                    break
        return tmp

    def beta_sampling(self, a, b, n_sample):
        samples = []
        for i in range(0, n_sample):
            s = []
            search = True
            while search:
                for d_set in self.discretized_descriptor:
                    r = numpy.random.beta(a, b, 1)[0]
                    s.append(self._find_nearest(d_set, r))

                if s in samples:
                    s = []
                    continue
                else:
                    samples.append(s)
                    break
        return samples



    def _find_nearest(self, array, value):
        idx = (numpy.abs(array-value)).argmin()
        return array[idx]
