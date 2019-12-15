import numpy as np
from parser import GraphParser
from qiskit import QuantumRegister, QuantumCircuit


class OperationCreator:
    '''
    Creating operations of szegedy walk.
    reference:
        Loke, T., and J. B. Wang. "Efficient quantum circuits
        for Szegedy quantum walks." Annals of Physics 382
        (2017): 64-84.
    
    all functions return instruction sets
    which can be added to circuit directly
    '''
    def __init__(self, graph, prob_tran, basis=0, optimize=True):
        self.parser = GraphParser(graph, prob_tran)
        self.graph = self.parser.graph
        self.dim = self.parser.dim()
        self.q_size = self._qubit_size(len(self.parser)) 
        self.basis_state = basis

    def T_operation(self):
        ref_states = self.parser.reference_state()
        print(ref_states)

    def Tdg_operation(self):
        pass

    def K_operation(self):
        pass

    def Kdg_operation(self):
        pass

    def D_operation(self, n_anilla=0, mode='basic', barrier=False):
        '''
        This operation is for flipping phase of specific basis state
        '''
        # describe basis state as a binary number
        nq = int(self.q_size/2)
        basis_bin = list(format(self.basis_state, '0%sb' % str(nq)))
        # create operation
        qr = QuantumRegister(nq)
        Dop = QuantumCircuit(qr, name='D')
        if n_anilla > 0:
            anc = QuantumRegister(n_anilla)
            Dop.add_register(anc)

        for q, b in zip(qr, basis_bin):
            if b == '0':
                Dop.x(q)
        # creating cz operation
        if barrier:
            Dop.barrier()
        Dop.h(qr[-1])
        Dop.mct(qr[:-1], qr[-1], None)
        Dop.h(qr[-1])
        if barrier:
            Dop.barrier()
        for q, b in zip(qr, basis_bin):
            if b == '0':
                Dop.x(q)
        D_instruction = Dop.to_instruction()
        return D_instruction
    
    @staticmethod
    def _qubit_size(dim):
        qsize = int(np.ceil(np.log2(dim)))
        return 2*qsize


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
    graph = np.array([[0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 1, 1, 0]])
    pb = prob_transition(graph)
    opcreator = OperationCreator(graph, pb)
    opcreator.D_operation()
    opcreator.T_operation()
