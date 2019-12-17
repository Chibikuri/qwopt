import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from op_creater import OperationCreator
from qiskit import transpile


class CircuitComposer:

    def __init__(self, graph, prob_tran, optimize=True):
        self.operator = OperationCreator(graph, prob_tran)
        self.n_qubit = self.operator.q_size

    def qw_circuit(self, anc=True, name='quantumwalk'):
        cont = QuantumRegister(self.n_qubit//2, 'control')
        targ = QuantumRegister(self.n_qubit//2, 'target')
        anc = QuantumRegister(self.n_qubit//2, 'ancilla')
        qw = QuantumCircuit(cont, targ, anc, name=name)

        qw = self._circuit_composer(qw, cont, targ, anc)
        qw, correct = self._circuit_validation(qw)
        if correct:
            return qw
        else:
            raise Exception('Circuit validation failed')
    
    def _circuit_composer(self, circuit, cont, targ, anc):
        qubits = [*cont, *targ, *anc]
        Ts = self.operator.T_operation()
        Kdg = self.operator.K_operation(dagger=True)
        K = self.operator.K_operation(dagger=False)
        D = self.operator.D_operation()
        
        for t in Ts:
            circuit.append(t, qargs=qubits)
        circuit.append(Kdg, qargs=qubits)
        circuit.append(D, qargs=targ)
        circuit.append(K, qargs=qubits)
        for tdg in Ts:
            circuit.append(tdg, qargs=qubits)
        return circuit

    def _circuit_validation(self, circuit):
        flag = True
        return circuit, flag


def prob_transition(graph):
    pmatrix = np.zeros(graph.shape)
    indegrees = np.sum(graph, axis=0)
    for ix, indeg in enumerate(indegrees):
        if indeg == 0:
            pmatrix[:, ix] = graph[:, ix]
        else:
            pmatrix[:, ix] = graph[:, ix]/indeg
    return pmatrix


if __name__ == '__main__':
    graph = np.array([[0, 1, 0, 0, 1, 0, 0, 1],
                      [0, 0, 0, 1, 1, 0, 1, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1],
                      [0, 1, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1, 0, 1, 1],
                      [0, 1, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 1, 0, 1, 0]])
    pb = prob_transition(graph)
    comp = CircuitComposer(graph, pb)
    comp.qw_circuit()