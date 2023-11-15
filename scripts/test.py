### Test uncertainty methods


import pypsa
import itertools
import numpy as np
import os
import pandas as pd

# pip install pypsa
# pip install pyDOE2
# pip install pyyaml
# pip install chaospy


def monte_carlo_sampling_pydoe2(N_FEATURES, SAMPLES, criterion=None, iteration=None, random_state=42, correlation_matrix=None):
    """
    Creates Latin Hypercube Sample (LHS) implementation from PyDOE2 with various options. Additionally all "corners" are simulated.

    Adapted from Disspaset: https://github.com/energy-modelling-toolkit/Dispa-SET/blob/master/scripts/build_and_run_hypercube.py
    Documentation on PyDOE2: https://github.com/clicumu/pyDOE2 (fixes latin_cube errors)
    """
    from pyDOE2 import lhs
    from scipy.stats import qmc 

    # Generate a Nfeatures-dimensional latin hypercube varying between 0 and 1:
    lh = lhs(N_FEATURES, samples=SAMPLES, criterion=criterion, iterations=iteration, random_state=random_state, correlation_matrix=correlation_matrix)
    discrepancy = qmc.discrepancy(lh)
    print("Discrepancy is:", discrepancy, " more details in function documentation.")

    return lh


def monte_carlo_sampling_chaospy(N_FEATURES, SAMPLES, DISTRIBUTION, DISTRIBUTION_PARAMS, rule="latin_hypercube", seed=42):
    """
    Creates Latin Hypercube Sample (LHS) implementation from chaospy.

    Documentation on Chaospy: https://github.com/clicumu/pyDOE2 (fixes latin_cube errors)
    Documentation on Chaospy latin-hyper cube (quasi-Monte Carlo method): https://chaospy.readthedocs.io/en/master/user_guide/fundamentals/quasi_random_samples.html#Quasi-random-samples

    """
    import chaospy
    from scipy.stats import qmc 
    from sklearn.preprocessing import MinMaxScaler
    params = tuple(DISTRIBUTION_PARAMS)
    # Generate a Nfeatures-dimensional latin hypercube varying between 0 and 1:
    N_FEATURES = f"chaospy.{DISTRIBUTION}{params}, "*N_FEATURES
    cube = eval(f"chaospy.J({N_FEATURES})")  # writes Nfeatures times the chaospy.uniform... command)
    lh = cube.sample(SAMPLES, rule=rule, seed=seed).T

    # rescale distribution to 0-1 if it is not a uniform distribution
    if DISTRIBUTION == "Uniform":
        pass
    elif DISTRIBUTION in ["Normal", "LogNormal", "Triangle", "Beta", "Gamma"]:
        mm = MinMaxScaler(feature_range=(0,1.))
        lh = mm.fit_transform(lh)
        print(lh.max())
        print(lh.min())
    else:
        raise ValueError(f"Distribution '{DISTRIBUTION}' not available. Please pick a valid one from possible options: 'Uniform', 'Normal', 'LogNormal', 'Triangle', 'Beta', 'Gamma'")

    discrepancy = qmc.discrepancy(lh)
    print("Discrepancy is:", discrepancy, " more details in function documentation.")

    return lh


def monte_carlo_sampling_scipy(N_FEATURES, SAMPLES, centered=False, strength=2, optimization=None, seed=42):
    """
    Creates Latin Hypercube Sample (LHS) implementation from SciPy with various options:
    - Center the point within the multi-dimensional grid, centered=True
    - optimization scheme, optimization="random-cd"
    - strength=1, classical LHS
    - strength=2, performant orthogonal LHS, requires the sample to be a prime or square of a prime e.g. sqr(121)=11

    Options could be combined to produce an optimized centered orthogonal array
    based LHS. After optimization, the result would not be guaranteed to be of strength 2.

    Documentation for Quasi-Monte Carlo approaches: https://docs.scipy.org/doc/scipy/reference/stats.qmc.html
    Documentation for Latin Hypercube: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html#scipy.stats.qmc.LatinHypercube
    Orthogonal LHS is better than basic LHS: https://github.com/scipy/scipy/pull/14546/files, https://en.wikipedia.org/wiki/Latin_hypercube_sampling
    """
    from scipy.stats import qmc    
    
    sampler = qmc.LatinHypercube(d=N_FEATURES, centered=centered, strength=strength, optimization=optimization, seed=seed)
    lh = sampler.random(n=SAMPLES)
    discrepancy = qmc.discrepancy(lh)
    print("Discrepancy is:", discrepancy, " more details in function documentation.")

    return lh


###
### MONTE-CARLO SCENARIO INPUTS
###

import yaml

path = os.path.join(os.getcwd(), f"config.yaml")
with open(path, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# remove all empty strings from config dictionary
MONTE_CARLO_PYPSA_FEATURES = {k: v for k, v in config["monte_carlo"]["pypsa_standard"].items() if v}  # removes key value pairs with empty value e.g. []
MONTE_CARLO_OPTIONS = config["monte_carlo"]["options"]
L_BOUNDS = [item[0] for item in MONTE_CARLO_PYPSA_FEATURES.values()]
U_BOUNDS = [item[1] for item in MONTE_CARLO_PYPSA_FEATURES.values()]
N_FEATURES = len(MONTE_CARLO_PYPSA_FEATURES)# only counts features when specified in config
SAMPLES = MONTE_CARLO_OPTIONS.get("samples")   # What is the optimal sampling? Probably depend on amount of features
SAMPLING_STRATEGY = MONTE_CARLO_OPTIONS.get("sampling_strategy")
DISTRIBUTION = MONTE_CARLO_OPTIONS.get("distribution") # Change the distribution var to title case
DISTRIBUTION_PARAMS = MONTE_CARLO_OPTIONS.get("distribution_params") # Change the distribution var to title case


###
### SCENARIO CREATION / SAMPLING STRATEGY
###
if SAMPLING_STRATEGY=="pydoe2":
    lh = monte_carlo_sampling_pydoe2(N_FEATURES, SAMPLES, criterion=None, iteration=None, random_state=42, correlation_matrix=None)
if SAMPLING_STRATEGY=="scipy":
    lh = monte_carlo_sampling_scipy(N_FEATURES, SAMPLES, centered=False, strength=2, optimization=None, seed=42)
if SAMPLING_STRATEGY=="chaospy":
    lh = monte_carlo_sampling_chaospy(N_FEATURES, SAMPLES, DISTRIBUTION, DISTRIBUTION_PARAMS, rule="latin_hypercube", seed=42)   


###
### SCENARIO ITERATION
###
from scipy.stats import qmc 

network = pypsa.examples.ac_dc_meshed(from_master=True)

lh_scaled = qmc.scale(lh, L_BOUNDS, U_BOUNDS)

# TODO parallize snakemake wildcard vs dask delayed
Nruns = lh.shape[0]
for i in range(Nruns):
    n = network.copy()
    j = 0
    for k, v in MONTE_CARLO_PYPSA_FEATURES.items():
        # this loop sets in one scenario each "i" feature assumption
        # i is the config input key "loads_t.p_set"
        # v is the lower and upper bound [0.8,1.3]
        # j is the sample interation number
        # Example: n.loads_t.p_set = network.loads_t.p_set = .loads_t.p_set * lh[0,0] * (1.3-0.8)
        exec(f"n.{k} = network.{k} * {lh_scaled[i,j]}")
        print(f"Scaled n.{k} by factor {lh_scaled[i,j]} in the {i} scenario")
        j = j + 1 

    # run optimization
    n.lopf(pyomo=False)
    # save each optimization result with a separate name
    n.monte_carlo = pd.DataFrame(lh_scaled).rename_axis('Nruns').add_suffix('_feature')
    directory_path = os.path.join(os.getcwd(), f"results")
    os.makedirs(directory_path, exist_ok=True)
    file_path = os.path.join(directory_path, f"result_{i}.nc")
    n.export_to_netcdf(file_path)
    print(f"Run {i}. Load_sum: {n.loads_t.p_set.sum().sum()} MW ")

