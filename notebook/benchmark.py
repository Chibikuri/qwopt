# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Optimization benchmarking

# Plotting benchmark result.

# +
import sys
import numpy as np

sys.path.append('../')
from qwopt.compiler import composer
from qwopt.benchmark import fidelity as fid
from qiskit import transpile
# -

# ## 1. Multi step of 4 node graph with one partition

# ### Check points
# - How to decrease the fidelity over the number of steps
# - The number of operations

# ### Count operations

# #### Without any optimizations

target_graph = np.array([[0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0]])
prob_dist = target_graph/2
for step in range(1, 10):
    qc = composer.CircuitComposer(target_graph, prob_dist, step).qw_circuit(optimization=False)
    nqc = transpile(qc, basis_gates=['cx', 'u3'])
    nc = nqc.count_ops()
    print('two qubit gates: ', nc.get('cx', 0), '/ one qubit gates: ', nc.get('u3', 0)+nc.get('u2', 0)+nc.get('u1', 0))
qc.draw(output='mpl')

# #### Fidelity decrese

fid_analysis = fid.FidelityAnalyzer(0, np.arange(0, 0.1, 0.01), [2, 3], extime=10, shots=10000)
fidelity = fid_analysis.fidelity_drop(qc)

print(fidelity)

# #### With optimizations

# ## 2. Multi step of 8 node graph with one partition

# ## 3. Multi step of 8 node graph with multi partition

# ## 4. Multi step of 1024 node graph with multi partition


