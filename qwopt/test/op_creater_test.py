import sys
sys.path.append('../../')
import unittest
import numpy as np
from qwopt.compiler.op_creater import OperationCreator
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit import Aer, execute, transpile


class OpcreatorTest(unittest.TestCase):

    def test_Kop(self):
        graph = np.array([[0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0]])
        pb = prob_transition(graph, gtype='google')
        # print(pb)
        op_creator = OperationCreator(graph, pb)
        nq = op_creator.q_size
        backend = Aer.get_backend('qasm_simulator')
        shots = 100000
        bins = [format(i, '0%sb' % str(nq//2)) for i in range(2**(nq//2))]

        cont = QuantumRegister(nq//2, 'cont')
        targ = QuantumRegister(nq//2, 'targ')
        anc = QuantumRegister(nq//2, 'ancillae')
        c = ClassicalRegister(nq//2)
        qc = QuantumCircuit(cont, targ, anc, c)
        qc.h(cont)
        Kop = op_creator.K_operation()
        qc.append(Kop, qargs=[*cont, *targ, *anc])
        qc.measure(targ, c)
        nqc = transpile(qc, basis_gates=['cx', 'u3', 'h', 'x'])
        # print(nqc)

        job = execute(qc, backend=backend, shots=shots)
        counts = job.result().get_counts(qc)
        # print(counts)
        prob = [counts.get(i, 0)/shots for i in bins]
        # print(prob)s

    def test_Top(self):
        graph = np.array([[0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0]])
        pb = prob_transition(graph, gtype='google')
        op_creator = OperationCreator(graph, pb)
        Tops = op_creator.T_operation()

        nq = op_creator.q_size
        qr = QuantumRegister(nq)
        c = ClassicalRegister(nq)
        anc = QuantumRegister(nq//2)
        qc = QuantumCircuit(qr, anc, c)
        for ts in Tops:
            qc.append(ts, qargs=[*qr, *anc])
        # print(qc)

    def test_bin_comverter(self):
        graph = np.array([[0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0]])
        pb = prob_transition(graph, gtype='google')
        op_creator = OperationCreator(graph, pb)
        top = op_creator.T_operation()


    def test_Dop(self):
        pass


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
    assert(np.sum(np.sum(E, axis=0))==N)
    return E


if __name__ == '__main__':
    unittest.main()
