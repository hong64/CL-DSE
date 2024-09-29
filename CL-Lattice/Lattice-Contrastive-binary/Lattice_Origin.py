from lattice_data import Lattice
from lattice_synthesis import VivdoHLS_Synthesis
from lattice_synthesis import FakeSynthesis
from lattice_ds_point import DSpoint
from lattice_sphere_tree import SphereTree as st
import lattice_utils
import datasets
import numpy as np
import copy
import time

def Lattice_Origin(run, original_feature, radius,intial_sampling_size, entire_ds,timeEvolutions,max_n_of_synth,samples):
    print("---------------------------Origin----------------------------------")
    synthesize_ids_O = []
    time_start = time.perf_counter()
    goal_acheived = False
    # Collect stats
    online_statistics = {}
    online_statistics['adrs'] = []
    online_statistics['delta_adrs'] = []
    online_statistics['n_synthesis'] = []
    synthesised_history = []
    # run_stats = None

    print("Exploration n: " + str(run))
    sphere_elements_sizes = []
    max_radius = 0
    # Create Lattice
    lattice = Lattice(original_feature, radius)


    n_of_synthesis = len(samples)

    hls = FakeSynthesis(entire_ds, lattice)
    sampled_configurations_synthesised = []
    for s in samples:
        latency, area,id,status = hls.synthesise_configuration(s)
        synthesised_configuration = DSpoint(latency, area, s)
        sampled_configurations_synthesised.append(synthesised_configuration)
        lattice.lattice.add_config(s)
        synthesised_history.append(synthesised_configuration)
        synthesize_ids_O.append({id:status})
    # After the inital sampling, retrieve the pareto frontier
    pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(sampled_configurations_synthesised)

    # Get exhaustive pareto frontier (known only if ground truth exists)
    pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = lattice_utils.pareto_frontier2d(entire_ds)

    # Store a copy to save the pareto front before the exploration algorithm
    pareto_frontier_before_exploration = copy.deepcopy(pareto_frontier)

    # Calculate ADRS
    adrs_evolution = []
    adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
    # ADRS after initial sampling
    adrs_evolution.append(adrs)

    # Select randomly a pareto configuration and explore its neighbourhood
    r = np.random.randint(0, len(pareto_frontier))
    pareto_configurations = [samples[i] for i in pareto_frontier_idx]
    pareto_solution_to_explore = pareto_configurations[r]

    # 关键搜索步骤
    # Search locally for the configuration to explore
    # st __init__ 函数里边内置有get_closest_random_element函数用于获取最接近的元素集
    sphere = st(pareto_solution_to_explore, lattice)
    # 随机挑选pareto_solution_to_explore内最接近的元素之一进行综合
    new_configuration = sphere.random_closest_element
    run_stats = lattice_utils.collect_online_statis(online_statistics, adrs, n_of_synthesis)
    # Until there are configurations to explore, try to explore these
    count = intial_sampling_size
    while new_configuration is not None:

        latency, area,id,status = hls.synthesise_configuration(new_configuration)
        synthesize_ids_O.append({id:status})
        # Generate a new design point
        ds_point = DSpoint(latency, area, new_configuration)

        # Update known synthesis values and configurations(only pareto + the new one)
        pareto_frontier.append(ds_point)

        # Add configuration to the tree
        lattice.lattice.add_config(ds_point.configuration)

        # Get pareto frontier
        pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(pareto_frontier)

        # Calculate ADRS
        adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)

        # Find new configuration to explore
        # Select randomly a pareto configuration
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

        if max_n_of_synth < n_of_synthesis:
            max_n_of_synth = n_of_synthesis

        n_of_synthesis += 1
        run_stats = lattice_utils.collect_online_statis(online_statistics, adrs, n_of_synthesis)
        if exit_expl:
            time_end = time.perf_counter()
            # If the exploration is ending update the calculate final ADRS
            timeEvolutions.append(time_end - time_start)
            adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
            adrs_evolution.append(adrs)
            print("Number of synthesis:\t{:d}".format(n_of_synthesis))
            print("Max radius:\t{:0.4f}".format(max_radius))
            print("Final ADRS:\t{:0.4f}".format(adrs))
            print()
            # print(lattice.lattice.get_n_of_children())
            # goal_stats.append(n_of_synthesis)
            break
    return synthesize_ids_O,timeEvolutions,adrs_evolution,run_stats,max_n_of_synth
from Assistant import train_interval
def save_Origin(adrs_run_stats,n_of_runs, max_n_of_synth, goal_stats,intial_sampling_size,timeEvolutions,b,radius,synthesize_ids_O):
    collect_offline_stats = lattice_utils.collect_offline_stats(adrs_run_stats, n_of_runs, max_n_of_synth, goal_stats,intial_sampling_size)
    collect_offline_stats["timeEvolutions"] = timeEvolutions
    collect_offline_stats["synthesis_ids_status"] = synthesize_ids_O


    print (goal_stats)
    filepath = "results/origin/"
    np.save(filepath + b+"_parm_runs_"+str(n_of_runs)+"_radius_"+str(radius)+"_init_"+str(intial_sampling_size),collect_offline_stats)




