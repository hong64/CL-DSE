########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# This files contains some function, such metrics, support functions, etc. useful across different projects.
########################################################################################################################
# from itertools import zip_longest, map
import math
import numpy as np


def pareto_frontier2d(points):
    if len(points) ==1:
        return points,[0]
    """
    Function: pareto_frontier2d
    The function given in input a set of points return the 2 dimensional Pareto frontier elements and the corresponding
    indexes respect to the original set of point

    Input: A list of points.
    A list of object characterised by two attributes-area and latency.

    Output: A tuple composed of 2 lists.
    The first list contains the Pareto dominant objects. The second list contains the indexes of the pareto dominant
    objects, respect to the original set of points given in input
    """
    indexes = sorted(range(len(points)), key=lambda k: points[k].latency)
    p_idx = []
    p_front = []
    pivot_idx = indexes[0]
    for i in range(1, len(indexes)):
        # If are equal, I search for the minimum value until they become different
        data_pivot = points[pivot_idx]
        d = points[indexes[i]]
        if data_pivot.latency == d.latency:
            if d.area < data_pivot.area:
                pivot_idx = indexes[i]
        else:
            if d.area < data_pivot.area:
                p_idx.append(pivot_idx)
                p_front.append(data_pivot)
                pivot_idx = indexes[i]
        if i == len(indexes)-1 and len(p_idx) == 0:
            p_idx.append(pivot_idx)
            p_front.append(points[pivot_idx])
        if i == len(indexes)-1 and pivot_idx != p_idx[-1]:
            p_idx.append(pivot_idx)
            p_front.append(points[pivot_idx])
    return p_front, p_idx


def adrs2d(reference_set, approximate_set):
    """
    Function: adrs2d
    The function given in input a set of reference points and a different set of points calculates the Average Distance
    from Reference Set among the reference set and the approximate one.
    ADRS(Pr, Pa) = 1/|Pa| * sum_Pa( min_Pp( delta(Pr,Pa) ) )
    delta(Pr, Pa) = max(0, ( A(Pa) - A(Pr) ) / A(Pa), ( L(Pa) - L(Pr) ) / L(Pa) )

    Input: 2 list of points.
    A list points representing the reference set and a list of points representing the approximate one.

    Output: ADRS value.
    A value representing the ADRS distance among the two sets, the distance of the approximate one with respect to the
    reference set.
    """
    n_ref_set = len(reference_set)
    n_app_set = len(approximate_set)
    min_dist_sum = 0
    for i in range(0, n_ref_set):
        distances = []
        for j in range(0, n_app_set):
            distances.append(_p2p_distance_2d(reference_set[i], approximate_set[j]))
        min_dist = min(distances)
        min_dist_sum = min_dist_sum + min_dist

    avg_distance = min_dist_sum / n_ref_set
    return avg_distance


def _p2p_distance_2d(ref_pt, app_pt):
    """
    Function: _p2p_distance_2d
    Support function used in ADRS
    Point to point distance for a 2 dimensional ADRS calculation. Implements the delta function of ADRS

    Input: 2 points.
    The reference point and the approximate point. Both are objects characterised by area and latency attributes.

    Output: A float value
    The maximum distance among the 2 dimensions considered (in our case area and latency).
    """
    x = (float(app_pt.latency) - float(ref_pt.latency)) / float(ref_pt.latency)
    y = (float(app_pt.area) - float(ref_pt.area)) / float(ref_pt.area)
    to_find_max = [0, x, y]
    d = max(to_find_max)
    return d

def _mean(l):
    return sum(l) / len(l)

def avg(x):
    x = list(x)
    for i in range(0,len(x)):
        if x[i] is None:
            x[i] = 0
    x = [i for i in x if i is not None]
    return sum(x, 0.0) / len(x)


def get_euclidean_distance(a, b):
    tmp = 0
    for i in range(0, len(a)):
        tmp += ((a[i]) - (b[i])) ** 2

    tmp = math.sqrt(tmp)
    return tmp


def collect_online_statis(online_statistics, adrs, n_synthesis):
    adrs_stats = online_statistics['adrs']
    delta_adrs_stats = online_statistics['delta_adrs']
    n_synth_stats = online_statistics['n_synthesis']

    adrs_stats.append(adrs)
    online_statistics['adrs'] = adrs_stats
    if len(delta_adrs_stats) == 0:
        delta_adrs = 0
    else:
        delta_adrs = adrs_stats[-2] - adrs

    delta_adrs_stats.append(delta_adrs)
    online_statistics['delta_adrs'] = delta_adrs_stats
    n_synth_stats.append(n_synthesis)
    online_statistics['n_synthesis'] = n_synth_stats

    return online_statistics


def collect_offline_stats(online_statistics, n_of_run, max_n_of_synth, goal_stats,intial_sampling_size):
    all_adrs = np.empty((n_of_run, max_n_of_synth-intial_sampling_size+1))
    all_adrs[:] = np.nan

    all_delta_adrs = np.empty((n_of_run, max_n_of_synth-intial_sampling_size+1))
    all_delta_adrs[:] = np.nan

    all_max_n_of_synth = []
    all_final_adrs = []
    for run in range(0, len(online_statistics)):
        adrs_stats = online_statistics[run]['adrs']
        delta_adrs_stats = online_statistics[run]['delta_adrs']
        n_synth_stats = online_statistics[run]['n_synthesis']
        final_i = -1
        # print max_n_of_synth
        # print len(adrs_stats)
        for i in range(0, max_n_of_synth-intial_sampling_size+1):
            # print i
            if i > len(adrs_stats)-1:
                # print "Sono maggiore quindi entro"
                if final_i == -1:
                    # print "Setto final_i"
                    final_i = i-1
                all_adrs.itemset(run, i, adrs_stats[final_i])
                all_delta_adrs.itemset(run, i, 0)
            else:
                # print "Sono minore quindi copio"
                all_adrs.itemset(run, i, adrs_stats[i])
                all_delta_adrs.itemset(run, i, delta_adrs_stats[i])

        all_max_n_of_synth.append(n_synth_stats[-1])
        all_final_adrs.append(adrs_stats[-1])

    adrs_mean = np.nanmean(all_adrs, axis=0)
    delta_adrs_mean = np.nanmean(all_delta_adrs, axis=0)
    offline_stats = {}
    offline_stats['final_adrs'] = all_final_adrs
    offline_stats['final_adrs_outliers'] = __outliers_iqr(all_final_adrs)
    offline_stats['adrs_mean'] = adrs_mean
    offline_stats['adrs'] = all_adrs
    offline_stats['delta_adrs'] = all_delta_adrs
    offline_stats['delta_adrs_mean'] = delta_adrs_mean
    # offline_stats['adrs_mean_outliers'] = __outliers_iqr(adrs_mean)
    # offline_stats['delta_adrs_mean_outliers'] = __outliers_iqr(delta_adrs_mean)
    offline_stats['max_n_of_synth'] = all_max_n_of_synth
    offline_stats['max_n_of_synth_mean'] = _mean(all_max_n_of_synth)
    # offline_stats['max_n_of_synth_mean_outliers'] = __outliers_iqr(all_max_n_of_synth)
    return offline_stats


def __outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
