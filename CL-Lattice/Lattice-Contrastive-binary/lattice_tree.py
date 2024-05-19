########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains the classes Node and Tree and the functions relevant for the classes.
########################################################################################################################

import copy
import os
import numpy


class Node:

    def __init__(self, data):
        """
        A node is characterized by the data it contains and the list of children
        """
        self.data = data
        self.children = []

    def add_child(self, node):
        """
        Add a child to the list of children
        """
        self.children.append(node)

    def has_children(self):
        """
        Checks if a node has children
        """
        if len(self.children) == 0:
            return False
        else:
            return True

    def has_child(self, data):
        """
        Checks if a node has a specific child
        """
        for c in self.children:
            if data == c.get_data():
                return True
            else:
                continue

        return False

    def get_child(self, data):
        """
        Check if the list of children contains a specific value and returns the corresponding node
        """
        for c in self.children:
            if data == c.get_data():
                return c

    def get_data(self):
        """
        Returns the data contained in a node
        """
        return self.data

    def get_depth(self):
        """
        Starting from a node, it returns the total number child generation it has.
        """
        counter = 0
        if not self.has_children():
            return counter
        else:
            child = self.children[-1]
            counter = child.go_in_depth(counter)
        return counter

    def go_in_depth(self, counter):
        """
        Starting from a node, it returns the total number child generation it has.
        """
        if not self.has_children():
            return counter + 1
        else:
            child = self.children[-1]
            counter = child.go_in_depth(counter)
        return counter + 1

    def search_configuration(self, config):
        """
        Starting from the calling node searches if the config configuration exists among all its possible children
        """
        exists = False
        tmp_config = copy.copy(config)
        if len(config) > 0:
            search = tmp_config.pop(0)
            for c in self.children:
                # print c.get_data()
                if search == c.get_data():
                    exists = exists or c.search_subconfiguration(tmp_config, exists)
                else:
                    continue
        else:
            return True
        return exists

    def search_subconfiguration(self, config, exists):
        """
        Same as before.
        """
        tmp_config = copy.copy(config)
        if len(config) > 0:
            search = tmp_config.pop(0)
            for c in self.children:
                # print c.get_data()
                if search == c.get_data():
                    exists = c.search_subconfiguration(tmp_config, exists)
                else:
                    continue
            return exists
        else:
            return True

    def add_configuration(self, config):
        """
        Given a not-known-a-priori configuration add it to the children if it do not already exists
        """
        if len(config) > 0:
            initial_config = copy.deepcopy(config)
            search = initial_config[0]
            subconfiguration = initial_config[1:]
            if self.has_children():
                for c in self.children:
                    if c.get_data() == search:
                        c.add_subconfiguration(subconfiguration)
                        return
            self.populate_subtree(initial_config)

    def add_subconfiguration(self, config):
        """
        Same as before
        """
        if len(config) > 0:
            initial_config = copy.deepcopy(config)
            search = initial_config[0]
            subconfiguration = initial_config[1:]
            if self.has_children():
                for c in self.children:
                    if c.get_data() == search:
                        c.add_subconfiguration(subconfiguration)
                        return
            self.populate_subtree(initial_config)

    def populate_subtree(self, config):
        """
        Add all config elements to the children list without checks
        """
        node = self
        for f in config:
            new_node = Node(f)
            node.add_child(new_node)
            node = new_node

    def n_of_children(self):
        """
        Return the total number of children of a node
        """
        n = 0
        if self.has_children():
            for c in self.children:
                n += c.n_of_children()
        else:
            n = 1

        return n


class Tree:
    def __init__(self, data):
        """
        Tree constructor. A tree starts for a root node containing its ID and all the configurations are the immediate
        child. To retrieve a configuration the tree has to be explored entirely in depth.
        """
        self.data = Node(data)
        # self.children = []

    def root(self):
        """
        returns the root of the tree
        """
        return self.data

    def populate_tree(self, data):
        """
        Given a set of configurations "data" generates a tree containing all such configurations
        """
        root = self.root()
        node = None
        for s in data:
            i = 0
            for f in s:
                if i == 0:
                    if root.has_child(f):
                        node = root.get_child(f)
                    else:
                        node = Node(f)
                        root.add_child(node)
                else:
                    if node.has_child(f):
                        node = node.get_child(f)
                    else:
                        new_node = Node(f)
                        node.add_child(new_node)
                        node = new_node
                i += 1

    def get_tree_depth(self):
        """
        Return the maximum depth of the tree
        """
        return self.root().get_depth()

    def exists_config(self, config):
        """
        Checks if a configuration is contained in the tree
        """
        root = self.root()
        return root.search_configuration(config)

    def exists_config(self, lattice, config):
        """
        Checks if a configuration is contained in the tree
        """
        root = lattice.root()
        return root.search_configuration(config)

    def add_config(self, config):
        """
        Add a configuration to the tree
        """
        root = self.root()
        return root.add_configuration(config)

    def print_tree(self, name):
        """
        Generate a dot file containing the tree
        """
        tree_script_dot = "digraph lattice {\n"
        indent = 1
        tree_script_dot += '\t'*indent + 'node [fontname="Courier"];\n'
        root = self.root()
        parent = "root"
        for f in root.children:
            child = f.get_data()
            child_name = parent + "_" + str(child)[:5].replace('.', '')
            tree_script_dot += '\t'*indent + parent + ' -> ' + child_name + ";\n"
            tree_script_dot += self.print_subtree(f, child_name, indent)

        tree_script_dot += "}\n"
        output_file = open(name+".gv", "w")
        output_file.write(tree_script_dot)
        output_file.close()
        os.system("dot -Tps -Gsize=9,15\! -Gdpi=100 tree.gv -o tree.ps")

    def print_subtree(self, node, name, indent):
        """
        Support recursive function function
        """
        tree_script_dot = ""
        if node.has_children():
            parent = name
            for f in node.children:
                child = f.get_data()
                child_name = parent + "_" + str(child)[:5].replace('.', '')
                tree_script_dot += '\t' * indent + parent + '->' + child_name + ";\n"
                tree_script_dot += self.print_subtree(f, child_name, indent)

        return tree_script_dot

    def get_n_of_children(self):
        """
        Return the total number of tree's children
        """
        root = self.root()
        return root.n_of_children()

    def _find_nearest(self, discrete_set, value):
        """
        Given a tree, and the set of discrete values returns the closer admissible value for the tree
        """
        array = numpy.asarray(discrete_set)
        idx = (numpy.abs(array - value)).argmin()
        return array[idx]


