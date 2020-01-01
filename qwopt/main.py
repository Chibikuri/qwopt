from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Instruction
from compiler import composer
from optimizer import Optimizer

import numpy as np
import toml


def is_unitary(operator, tolerance=0.0001):
    h, w = operator.shape
    if not h == w:
        return False
    adjoint = np.conjugate(operator.transpose())
    product1 = np.dot(operator, adjoint)
    product2 = np.dot(adjoint, operator)
    ida = np.eye(h)
    return np.allclose(product1, ida) & np.allclose(product2, ida)


def prob_transition(graph, gtype='normal', alpha=0.85):
    if gtype == 'google':
        return google_matrix(alpha, graph)
    else:
        pmatrix = np.zeros(graph.shape)
        indegrees = np.sum(graph, axis=0)
        for ix, indeg in enumerate(indegrees):
            if indeg == 0:
                pmatrix[:, ix] = graph[:, ix]
            else:
                pmatrix[:, ix] = graph[:, ix]/indeg
        return pmatrix


def google_matrix(alpha, C):
    E = connect_to_E(C)
    N = len(C)
    G = alpha*E + (1-alpha)/N * np.ones((N, N), dtype=float)
    return G


def connect_to_E(C):
    '''
    C is conectivity matrix
    C: np.array
    output
    E: np.array
    '''
    N = len(C)
    C = np.array(C)
    E = np.zeros(C.shape)
    rowsum = np.sum(C, axis=0)
    for ind, val in enumerate(rowsum):
        if val == 0:
            for j in range(N):
                E[j][ind] = 1/N
        else:
            for j in range(N):
                E[j][ind] = C[j][ind]/val
    assert(np.sum(np.sum(E, axis=0)) == N)
    return E


graph = np.array([[0, 1, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 1],
                  [0, 1, 1, 0]])
pb = prob_transition(graph)
step = 1
qc = composer.CircuitComposer(graph, pb, step).qw_circuit(validation=False)
opt = Optimizer(graph, pb).optimize(qc, 3, open('ruleset.toml', 'r'))
