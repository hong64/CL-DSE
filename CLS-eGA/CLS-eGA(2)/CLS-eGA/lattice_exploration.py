from lattice_data import Lattice
from lattice_synthesis import VivdoHLS_Synthesis
from lattice_ds_point import DSpoint
from lattice_sphere_tree import SphereTree as st
import lattice_utils
import datasets
# import matplotlib.pyplot as plt
import numpy as np
import copy

Autocorrelation_extended = datasets.Datasets("Autocorrelation_extended")
feature_sets = [i[1] for i in Autocorrelation_extended.autcorrelation_extended_directives_ordered]

# While used to run the experiments multiple times
n_of_runs = 1
if n_of_runs > 1:
    plot_chart = False
else:
    plot_chart = True

collected_run = []
for run in range(n_of_runs):
    # Create Lattice
    lattice = Lattice(feature_sets, 10)
    max_radius = 0

    # Probabilistic sample according to beta distribution
    # samples = lattice.beta_sampling(0.1, 0.1, 20)

    # Populate the tree with the initial sampled values
    # lattice.lattice.populate_tree(samples)
    # n_of_synthesis = len(samples)

    # Synthesise sampled configuration
    # hls = FakeSynthesis(entire_ds, lattice)
    prj_description = {"prj_name": "Autocorrelation_extended",
                       "test_bench_file": "gsm.c",
                       "source_folder": "<path_to_src_folder>",
                       "top_function": "Autocorrelation"}

    hls = VivdoHLS_Synthesis(lattice, Autocorrelation_extended.autcorrelation_extended,
                             Autocorrelation_extended.autcorrelation_extended_directives_ordered,
                             Autocorrelation_extended.autcorrelation_extended_bundling,
                             prj_description)

    sampled_configurations_synthesised = []
    # for s in samples:
    samples = []
    while len(samples) < 20:
        sample = lattice.beta_sampling(0.1, 0.1, 1).pop()
        latency, area = hls.synthesise_configuration(sample)
        # if latency is None:
        #     lattice.lattice.add_config(sample)
        #     continue
        samples.append(sample)
        synthesised_configuration = DSpoint(latency, area, sample)
        sampled_configurations_synthesised.append(synthesised_configuration)
        lattice.lattice.add_config(sample)

    n_of_synthesis = len(samples)
    print(samples)
    print(len(samples))
    print(len(sampled_configurations_synthesised))
    # Get pareto frontier from sampled configuration
    pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(sampled_configurations_synthesised)

    # Get exhaustive pareto frontier (known only if ground truth exists)
    # pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = lattice_utils.pareto_frontier2d(entire_ds)

    pareto_frontier_before_exploration = copy.deepcopy(pareto_frontier)
    intial_pareto_frontier_latency = []
    intial_pareto_frontier_area = []
    for pp in pareto_frontier_before_exploration:
        intial_pareto_frontier_latency.append(pp.latency)
        intial_pareto_frontier_area.append(pp.area)

    # # PLOT start
    # if plot_chart:
    #     for p in sampled_configurations_synthesised:
    #         plt.scatter(p.latency, p.area, color='b')
    #
    #     for pp in pareto_frontier_exhaustive:
    #         plt.scatter(pp.latency, pp.area, color='r')
    #
    #     for pp in pareto_frontier:
    #         plt.scatter(pp.latency, pp.area, color='g')
    #
    #     plt.grid()
    #     # plt.draw()
    #     # PLOT end

    # Calculate ADRS
    adrs_evolution = []
    # adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
    adrs = lattice_utils.adrs2d(pareto_frontier_before_exploration, pareto_frontier)
    adrs_evolution.append(adrs)

    # Select randomly a pareto configuration and find explore his neighbour
    r = np.random.randint(0, len(pareto_frontier))
    pareto_configurations = [samples[i] for i in pareto_frontier_idx]
    configuration_to_explore = pareto_configurations[r]

    # Search locally for the configuration to explore
    sphere = st(configuration_to_explore, lattice)
    new_configuration = sphere.random_closest_element

    # Until there are configurations to explore, try to explore these
    while new_configuration is not None:
        print("New iteration")
        # Synthesise configuration
        latency, area = hls.synthesise_configuration(new_configuration)
        # if latency is None:
        #     lattice.lattice.add_config(new_configuration)
        #     # Find new configuration to explore
        #     # Select randomly a pareto configuration
        #     r = np.random.randint(0, len(pareto_frontier))
        #     pareto_solution_to_explore = pareto_frontier[r].configuration
        #
        #     # Explore the closer element locally
        #     sphere = st(pareto_solution_to_explore, lattice)
        #     new_configuration = sphere.random_closest_element
        #     max_radius = max(max_radius, sphere.radius)
        #
        #     if new_configuration is None:
        #         print "Exploration terminated"
        #         break
        #     if max_radius > lattice.max_distance:
        #         print "Exploration terminated, max radius reached"
        #         break
        #     continue
        # Generate a new design point
        ds_point = DSpoint(latency, area, new_configuration)
        print("Lat:", latency, "\tArea:", area)

        # Update known synthesis values and configurations(only pareto + the new one)
        pareto_frontier.append(ds_point)

        # Add configuration to the tree
        lattice.lattice.add_config(ds_point.configuration)

        # Get pareto frontier
        pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(pareto_frontier)

        # Calculate ADRS
        # adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
        adrs = lattice_utils.adrs2d(pareto_frontier_before_exploration, pareto_frontier)
        adrs_evolution.append(adrs)
    #    if adrs == 0:
    #         break

        # Find new configuration to explore
        search_among_pareto = copy.copy(pareto_frontier)
        while len(search_among_pareto) > 0:
            r = np.random.randint(0, len(search_among_pareto))
            pareto_solution_to_explore = search_among_pareto[r].configuration

            # Explore the closer element locally
            sphere = st(pareto_solution_to_explore, lattice)
            new_configuration = sphere.random_closest_element
            if new_configuration is None:
                search_among_pareto.pop(r)
                continue

            max_radius = max(max_radius, sphere.radius)
            if max_radius > lattice.max_distance:
                search_among_pareto.pop(r)
                continue

            break

        exit_expl = False
        if len(search_among_pareto) == 0:
            print("Exploration terminated")
            exit_expl = True

        if max_radius > lattice.max_distance:
            print("Max radius reached")
            exit_expl = True

        if exit_expl:
            break

        n_of_synthesis += 1
        print(n_of_synthesis)

    final_pareto_frontier_latency = []
    final_pareto_frontier_area = []
    for pp in pareto_frontier:
        final_pareto_frontier_latency.append(pp.latency)
        final_pareto_frontier_area.append(pp.area)

    collected_run.append((n_of_synthesis, adrs_evolution, max_radius))
    n_of_synthesis = 0
    adrs_evolution = []
    max_radius = 0

    # if plot_chart:
    #     fig1 = plt.figure()
    #     for p in sampled_configurations_synthesised:
    #         plt.scatter(p.latency, p.area, color='b')
    #
    #     for pp in pareto_frontier_exhaustive:
    #         plt.scatter(pp.latency, pp.area, color='r', s=40)
    #
    #     for pp in pareto_frontier:
    #         plt.scatter(pp.latency, pp.area, color='g')
    #
    #     fig2 = plt.figure()
    #     plt.grid()
    #     pareto_frontier.sort(key=lambda x: x.latency)
    #     plt.step([i.latency for i in pareto_frontier], [i.area for i in pareto_frontier], where='post', color='r')
    #     pareto_frontier_before_exploration.sort(key=lambda x: x.latency)
    #     plt.step([i.latency for i in pareto_frontier_before_exploration], [i.area for i in pareto_frontier_before_exploration], where='post', color='b')
    #     # plt.draw()
    #
    #     fig3 = plt.figure()
    #     plt.grid()
    #     plt.plot(adrs_evolution)
    #     plt.show()

mean_adrs, radii, final_adrs_mean = lattice_utils.get_statistics(collected_run)
data_file = open("mean_adrs.txt", "w")
data_file.write(str(mean_adrs))
data_file.close()

data_file = open("radii.txt", "w")
data_file.write(str(radii))
data_file.close()

data_file = open("final_adrs_mean.txt", "w")
data_file.write(str(final_adrs_mean))
data_file.close()

data_file = open("inital_pareto.txt","w")
data_file.write(str(intial_pareto_frontier_latency))
data_file.write(str(intial_pareto_frontier_area))
data_file.close()

data_file = open("final_pareto.txt","w")
data_file.write(str(final_pareto_frontier_latency))
data_file.write(str(final_pareto_frontier_area))
data_file.close()
# print mean_adrs
# plt.plot(mean_adrs)
# plt.show()
