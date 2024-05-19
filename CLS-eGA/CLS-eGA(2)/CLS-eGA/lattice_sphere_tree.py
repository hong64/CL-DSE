########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains the Sphere Tree class. This class defines the sphere used to perform the local search in the
# neighbourhood of the configuration to explore. A tree similar to the one for the explored configuration is used to
# keep track of the configuration to visit during the local exploration
########################################################################################################################

from lattice_tree import Node, Tree
import copy
import math
import numpy as np


class SphereTree:
    def __init__(self, configuration, lattice):
        self.root = Node(configuration)
        self.lattice = lattice
        self.radius = lattice.discretized_descriptor[0][1]
        self.min_radius = lattice.discretized_descriptor[0][1]
        self.min_increment = 0.05
        self.sphere_elements = []
        self.closest_distances, self.closest_elements_idx, self.n_of_children = self.get_closest_sphere_elements()
        if len(self.closest_distances) == 0:
            self.random_closest_element = None
        else:
            self.random_closest_element = self.get_closest_random_element().get_data()

    def get_closest_sphere_elements(self):
        visited_tree = Tree("visited")
        self.visit_config(self.root, visited_tree)
        while len(self.sphere_elements) == 0:
            # print(self.root.get_data())
            self.radius = self.radius + self.min_increment
            if self.radius > self.lattice.max_distance:
                break
            visited_tree = Tree("visited")            # print "Sphere elements: ", len(self.sphere_elements)

            self.visit_config(self.root, visited_tree)
            # if len(self.sphere_elements)!=0:
            #     print("post")
            #     print(self.root.get_data())
            # print "Number of children: ", visited_tree.get_n_of_children()
        closest_distances, closest_elements_idx = self.sort_sphere_elements()
        return closest_distances, closest_elements_idx, visited_tree.get_n_of_children()

    def visit_config(self, starting_config, visited_tree):
        children = []
        config = starting_config.get_data()
        visited_tree.add_config(config)
        for i in range(0, len(config)):
            delta = self.lattice.discretized_descriptor[i][1]
            cfg_plus = copy.copy(config)
            value_plus = cfg_plus[i] + delta
            cfg_plus[i] = self.lattice._find_nearest(self.lattice.discretized_descriptor[i], np.float64(value_plus))
            cfg_minus = copy.copy(config)
            value_minus = cfg_minus[i] - delta
            cfg_minus[i] = self.lattice._find_nearest(self.lattice.discretized_descriptor[i], np.float64(value_minus))

            # Generate the new config to add
            config_to_append_plus = self._add_config(cfg_plus)
            if config_to_append_plus is not None:
                children.append(config_to_append_plus)
            config_to_append_minus = self._add_config(cfg_minus)
            if config_to_append_minus is not None:
                children.append(config_to_append_minus)
            while len(children) > 0:
                c = children.pop(0)
                if not visited_tree.exists_config(visited_tree, c.get_data()):
                    self.visit_config(c, visited_tree)
                    if not self.lattice.lattice.exists_config(self.lattice.lattice, c.get_data()):
                        self.sphere_elements.append(c)

    def _add_config(self, config):
        distance = self._get_distance(config)
        if not np.isclose(distance, self.radius) and distance > self.radius:
            return None
        else:
            n = Node(config)
            return n

    def _get_distance(self, config):
        tmp = 0
        for i in range(len(config)):
            root_config = self.root.get_data()
            tmp += ((root_config[i]) - (config[i])) ** 2

        tmp = math.sqrt(tmp)
        return tmp

    def _is_in_sphere(self, config):
        for e in self.sphere_elements:
            if config == e.get_data():
                return True
        return False

    def sort_sphere_elements(self):
        distances = []
        for i in self.sphere_elements:
            c = i.get_data()
            distances.append(self._get_distance(c))

        return distances, sorted(range(len(distances)), key=lambda k: distances[k])

    def get_closest_random_element(self):
        tmp = []
        min_distance = min(self.closest_distances)
        for i in self.closest_elements_idx:
            if self.closest_distances[i] == min_distance:
                tmp.append(i)

        r = np.random.randint(0, len(tmp))
        return self.sphere_elements[r]

