import copy
from GA_ds_point import DSpoint
import numpy as np
import GA_utils
from NSGAII import *
import time
class EGA:
    def __init__(self,entire_ds):
        self.N_pare = 5
        self.N_terminate = 5
        self.latency_constraint = 100000000
        self.latency_min = min([design.latency for design in entire_ds])
        self.visited = [False for _ in range(len(entire_ds))]
        self.P_cros = 0.8
        self.max_stagnant = 5

def NSGAII_Selection(eGA,entire_ds,initial_sampling_size):
    latency = []
    area = []
    ids = []
    for i in range(len(eGA.visited)):
        if eGA.visited[i]:
            ids.append(i)
            latency.append(entire_ds[i].latency)
            area.append(entire_ds[i].area)
    fronts = fast_non_dominated_sort(latency,area)
    distance = []
    for i in range(0,len(fronts)):
        distance.append(crowding_distance(latency[:],area[:],fronts[i][:]))
    new_population = []
    for i in range(len(fronts)-1,-1,-1):
        if (len(new_population) + len(fronts[i])) <= initial_sampling_size:
            for no in fronts[i]:
                id = ids[no]
                new_population.append(entire_ds[id])
        else:
            sorted_id = sorted(range(len(distance[i])), key=lambda k: distance[i][k], reverse=True)
            for j in range(initial_sampling_size-len(new_population)):
                id = ids[sorted_id[j]]
                new_population.append(entire_ds[id])
            break
    return new_population



def Fake_Synthesis(configuration,entire_ds,EGA):
    for i,design in enumerate(entire_ds):
        if design.configuration == configuration:
            EGA.visited[i] = True
            if design.latency != 10000000 and design.area != 10000000:
                entire_ds[i].is_success = True
                return DSpoint(design.latency,design.area,configuration)
            entire_ds[i].is_success = False
            return DSpoint(design.latency,design.area,configuration,is_success=False)
def Crossover(d1,d2):
    config1,config2 = copy.deepcopy(d1.configuration),copy.deepcopy(d2.configuration)
    pos = random.randint(1,len(config1)-1)
    temp = config1[0:pos]
    config1[0:pos] = config2[0:pos]
    config2[0:pos] = temp
    return config1,config2

def Mutation(config1,config2,features):
    md1,md2 = np.random.choice(len(features),2,replace=True).tolist()
    config1[md1] = random.choice(features[md1])
    config2[md2] = random.choice(features[md2])
    return config1,config2

def Explore(d1,d2,features,entire_ds,eGA):
    for i in range(eGA.N_pare):
        if random.random() < eGA.P_cros:
            config1,config2 = Mutation(*Crossover(d1,d2),features)
            synthesis_design1,synthesis_design2 = Fake_Synthesis(config1,entire_ds,eGA),Fake_Synthesis(config2,entire_ds,eGA)
            if synthesis_design1.latency < eGA.latency_constraint and synthesis_design1.area < d1.area:
                d1 = synthesis_design1
            if synthesis_design2.latency < eGA.latency_constraint and synthesis_design2.area < d2.area:
                d2 = synthesis_design2
    return d1,d2
def get_synthesized_designs(eGA,entire_ds):
    synthesized_designs_characteristics = np.array([]).reshape(0, 2)
    for i, visited in enumerate(eGA.visited):
        if visited:
            synthesized_designs_characteristics = np.vstack(
                (synthesized_designs_characteristics, [entire_ds[i].latency, entire_ds[i].area]))
    return synthesized_designs_characteristics
def Global_Minimum_Solution(eGA,entire_ds):
    synthesized_designs_characteristics = get_synthesized_designs(eGA,entire_ds)
    synthesized_designs_characteristics = synthesized_designs_characteristics[(synthesized_designs_characteristics[:,0]<eGA.latency_constraint)]
    synthesized_designs_characteristics = synthesized_designs_characteristics[np.argsort(synthesized_designs_characteristics[:,1])]
    if synthesized_designs_characteristics.shape[0] == 0:
        return False,None,None
    else:
        return True,synthesized_designs_characteristics[0,1],synthesized_designs_characteristics[0,0]
def calculate_adrs(eGA,entire_ds,pareto_frontier_exhaustive):
    pareto_frontier = []
    for i, visited in enumerate(eGA.visited):
        if visited:
            pareto_frontier.append(entire_ds[i])
    pareto_frontier, pareto_frontier_idx = GA_utils.pareto_frontier2d(pareto_frontier)
    adrs = GA_utils.adrs2d(pareto_frontier_exhaustive, pareto_frontier)
    return adrs
def Origin(x,entire_ds,intial_sampling_size,original_feature,pareto_frontier_exhaustive,timeEvolutions_O,adrs_run_history_O,synthesize_evolutions_O,run):
    stagnant, adrs, time_start = 0, 100000000, time.perf_counter()
    eGA = EGA(entire_ds)
    ini_configs_ids = np.random.choice(x.shape[0], intial_sampling_size, replace=False)
    population = [Fake_Synthesis(x[id, :].tolist(), entire_ds, eGA) for id in ini_configs_ids]
    origin_initial_ids = copy.deepcopy(ini_configs_ids)
    while eGA.latency_constraint > eGA.latency_min:
        for i in range(eGA.N_terminate):
            selected_id1, selected_id2 = np.random.choice(len(population), 2, replace=False).tolist()
            explored_d1, explored_d2 = Explore(population[selected_id1], population[selected_id2], original_feature,
                                               entire_ds, eGA)
            population[selected_id1], population[selected_id2] = explored_d1, explored_d2

        adrs = calculate_adrs(eGA, entire_ds, pareto_frontier_exhaustive)
        population = NSGAII_Selection(eGA, entire_ds, intial_sampling_size)
        optimize, A_min, Li = Global_Minimum_Solution(eGA, entire_ds)
        if optimize:
            area_min, eGA.latency_constraint = A_min, Li
            stagnant = 0
        else:
            stagnant += 1
            if stagnant == eGA.max_stagnant:
                break
    time_end = time.perf_counter()
    timeEvolutions_O.append(time_end - time_start)
    adrs_run_history_O.append(adrs)
    visited_ids = np.where(np.array(eGA.visited) == True)[0].tolist()
    synthesize_evolutions_O.append(visited_ids)
    print("id:%d;adrs:%s;synthesis:%d" % (run, adrs, len(visited_ids)))
    return origin_initial_ids






