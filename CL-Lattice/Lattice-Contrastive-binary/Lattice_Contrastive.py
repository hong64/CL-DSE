import random

from lattice_data import Lattice
from lattice_synthesis import FakeSynthesis
from lattice_ds_point import DSpoint
from lattice_sphere_tree import SphereTree as st

import lattice_utils
import numpy as np
import copy
import time
from Assistant import Contrastive_Learning,TrainCondition

def Lattice_Contrastive(run,original_feature, radius,intial_sampling_size,entire_ds,timeEvolutions,max_n_of_synth):
    synthesize_ids_C = []
    print("---------------------------Contrastive----------------------------------")
    time_start = time.perf_counter()
    goal_acheived = False
    # Collect stats
    online_statistics = {}
    online_statistics['adrs'] = []
    online_statistics['delta_adrs'] = []
    online_statistics['n_synthesis'] = []
    history = []

    print("Exploration n: " + str(run))
    max_radius = 0
    lattice = Lattice(original_feature, radius)

    samples = lattice.beta_sampling(0.1, 0.1, intial_sampling_size)
    n_of_synthesis = len(samples)

    hls = FakeSynthesis(entire_ds, lattice)
    sampled_configurations_synthesised = []
    for s in samples:

        latency, area,id,status = hls.synthesise_configuration(s)
        synthesised_design = DSpoint(latency, area, s)
        sampled_configurations_synthesised.append(synthesised_design)
        lattice.lattice.add_config(s)


        if latency != 10000000 and area != 10000000:
            history.append(synthesised_design)
        synthesize_ids_C.append({id:status})
    pareto_frontier, pareto_frontier_idx = lattice_utils.pareto_frontier2d(sampled_configurations_synthesised)
    pareto_frontier_exhaustive, pareto_frontier_exhaustive_idx = lattice_utils.pareto_frontier2d(entire_ds)

    # Store a copy to save the pareto front before the exploration algorithm
    pareto_frontier_before_exploration = copy.deepcopy(pareto_frontier)

    adrs_evolution = []
    adrs = lattice_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
    adrs_evolution.append(adrs)

    r = np.random.randint(0, len(pareto_frontier))
    pareto_configurations = [samples[i] for i in pareto_frontier_idx]
    pareto_solution_to_explore = pareto_configurations[r]


    sphere = st(pareto_solution_to_explore, lattice)
    new_configuration = sphere.random_closest_element
    run_stats = lattice_utils.collect_online_statis(online_statistics, adrs, n_of_synthesis)
    train = TrainCondition(1,None)
    while new_configuration is not None:
        is_promising,train = Contrastive_Learning(history,new_configuration,pareto_frontier,train)

        if is_promising[0] :
            latency,area,id,status = hls.synthesise_configuration(new_configuration)
            is_synthesis = True
            synthesize_ids_C.append({id:status})
        else:
            latency, area = 10000000,10000000
            is_synthesis = False
        # test(is_promising,hls,new_configuration,pareto_frontier,history)


        # Generate a new design point
        ds_point = DSpoint(latency, area, new_configuration)
        # Add configuration to the tree
        lattice.lattice.add_config(ds_point.configuration)

        if is_synthesis:
            if (latency != 10000000) and (area != 10000000):
                history.append(ds_point)



            # Update known synthesis values and configurations(only pareto + the new one)
            pareto_frontier.append(ds_point)



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

        if is_synthesis:
            n_of_synthesis += 1
            run_stats = lattice_utils.collect_online_statis(online_statistics, adrs, n_of_synthesis)
        if max_n_of_synth < n_of_synthesis:
            max_n_of_synth = n_of_synthesis

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
            break

    return synthesize_ids_C,timeEvolutions,adrs_evolution,run_stats,samples,max_n_of_synth

from Assistant import train_interval
K = 0.5
def save_Contrastive(adrs_run_stats,n_of_runs, max_n_of_synth, goal_stats,intial_sampling_size,timeEvolutions,b,radius,synthesize_ids_C):
    collect_offline_stats = lattice_utils.collect_offline_stats(adrs_run_stats, n_of_runs, max_n_of_synth, goal_stats,intial_sampling_size)
    collect_offline_stats["timeEvolutions"] = timeEvolutions
    collect_offline_stats["synthesis_ids_status"] = synthesize_ids_C
    print (goal_stats)
    filepath = "results/contrastive/"
    np.save(filepath + b+"_parm_confidence_"+str(K)+"_runs_"+str(n_of_runs)+"_radius_"+str(radius)+"_init_"+str(intial_sampling_size)+"_trainInterval_"+str(train_interval),collect_offline_stats)



