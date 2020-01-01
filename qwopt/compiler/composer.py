import numpy as np
import copy
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, execute
from .op_creater import OperationCreator
from qiskit import transpile
np.set_printoptions(linewidth=1000, precision=3)


class CircuitComposer:
    '''
    If you use google matrix as a probability distribution matrix,
    please use E as a graph
    '''

    def __init__(self, graph, prob_tran, step, optimize=True):
        self.operator = OperationCreator(graph, prob_tran, optimize=optimize)
        self.n_qubit = self.operator.q_size
        self.graph = self.operator.graph
        self.ptran = self.operator.ptran
        self.step = step
        if self.step < 0:
            raise ValueError('The number of steps must be 0 and over')

    def qw_circuit(self, anc=True, name='quantumwalk',
                   measurement=True, initialize=True, validation=True):

        cont = QuantumRegister(self.n_qubit//2, 'control')
        targ = QuantumRegister(self.n_qubit//2, 'target')
        anc = QuantumRegister(self.n_qubit//2, 'ancilla')
        qw = QuantumCircuit(cont, targ, anc, name=name)
        if initialize:
            self._initialize(qw, [*cont, *targ])
            # FIXME
            lp = len(self.ptran)**2
            init_state = [1/np.sqrt(lp) for i in range(lp)]
        else:
            # FIXME should be able to put arbitraly input initial
            init_state = [1] + [0 for i in range(lp-1)]

        qw = self._circuit_composer(qw, cont, targ, anc)
        # FIXME more efficient order
        if validation:
            qw, correct = self._circuit_validator(qw, [*cont, *targ],
                                                  init_state)
            if measurement:
                c = ClassicalRegister(self.n_qubit//2, 'classical')
                qw.add_register(c)
                # TODO is this correct order?
                qw.measure(cont[::-1], c)
            if correct:
                print('Circuit is validated!')
                return qw
            else:
                raise Exception('Circuit validation failed')
        else:
            if measurement:
                c = ClassicalRegister(self.n_qubit//2, 'classical')
                qw.add_register(c)
                # TODO is this correct order?
                qw.measure(cont[::-1], c)
            return qw

    def _circuit_composer(self, circuit, cont, targ, anc, measurement=True):
        qubits = [*cont, *targ, *anc]
        Ts = self.operator.T_operation()
        Kdg = self.operator.K_operation(dagger=True)
        K = self.operator.K_operation(dagger=False)
        D = self.operator.D_operation()
        # TODO remove commented out
        for t in Ts:
            circuit.append(t, qargs=qubits)
        circuit.barrier()
        circuit.append(Kdg, qargs=qubits)
        # nqc = transpile(circuit, basis_gates=['ccx', 'cx', 'x', 'h',
        #                                       'u2', 'u1', 'u3'])
        # circuit.append(D, qargs=targ)
        # circuit.append(K, qargs=qubits)
        # for tdg in Ts:
        #     circuit.append(tdg, qargs=qubits)
        # for i, j in zip(cont, targ):
        #     circuit.swap(i, j)
        return circuit

    def _circuit_validator(self, test_circuit, qregs, init_state,
                           remote=False, test_shots=100000, threshold=1e-4):
        # TODO check with unitary simulator(currently, ansilla is bothering...)
        circuit = copy.copy(test_circuit)
        nc = ClassicalRegister(self.n_qubit)
        circuit.add_register(nc)
        for q, c in zip(qregs[::-1], nc):
            circuit.measure(q, c)
        if remote:
            backend = IBMQ.get_backend('ibmq_qasm_simulator')
        else:
            backend = Aer.get_backend('qasm_simulator')
        theoretical_prob = self._theoretical_prob(init_state)

        vjob = execute(circuit, backend=backend, shots=test_shots)
        result = vjob.result().get_counts(circuit)
        rb = self.n_qubit
        bins = [format(i, '0%sb' % rb) for i in range(2**rb)]
        emp_states = []
        probs = np.array([result.get(bi, 0)/test_shots for bi in bins])
        # TODO check L2 norm is good enough to validate in larger case
        # and check threshold
        l2dist = self._L2_dist(theoretical_prob, probs)
        if l2dist < threshold:
            flag = True
        else:
            flag = False
        return test_circuit, flag

    def _initialize(self, circuit, qregs, state='super'):
        if isinstance(state, list or np.ndarray):
            circuit.initialize(state, qregs)
        else:
            for qr in qregs:
                circuit.h(qr)
        return circuit

    @staticmethod
    def _L2_dist(pa, pb):
        # FIXME using array
        ps = [(p1 - p2)**2 for p1, p2 in zip(pa, pb)]
        return sum(ps)

    def _theoretical_prob(self, init_state):
        Pi_op = self._Pi_operator()
        swap = self._swap_operator()
        operator = (2*Pi_op) - np.identity(len(Pi_op))
        Szegedy = np.dot(operator, swap)
        Szegedy_n = copy.copy(Szegedy)
        initial = init_state
        if self.step == 0:
            init_prob = np.array([abs(i)**2 for i in initial], dtype=np.float)
            return init_prob
        elif self.step == 1:
            prob = np.array([abs(i)**2 for i in np.dot(Szegedy, initial)],
                            dtype=np.float)
            return prob
        else:
            for n in range(self.step-1):
                Szegedy_n = np.dot(Szegedy_n, Szegedy)
            probs = np.array([abs(i)**2 for i in np.dot(Szegedy_n, initial)],
                             dtype=np.float)
            return probs

    def _swap_operator(self):
        q1 = QuantumRegister(self.n_qubit//2)
        q2 = QuantumRegister(self.n_qubit//2)
        qc = QuantumCircuit(q1, q2)
        for c, t in zip(q1, q2):
            qc.swap(c, t)
        # FIXME
        backend = Aer.get_backend('unitary_simulator')
        job = execute(qc, backend=backend)
        swap = job.result().get_unitary(qc)
        return swap

    def _Pi_operator(self):
        '''
        This is not a quantum operation,
        just returning matrix
        '''
        lg = len(self.ptran)
        psi_op = []
        count = 0
        for i in range(lg):
            psi_vec = [0 for _ in range(lg**2)]
            for j in range(lg):
                psi_vec[count] = np.sqrt(self.ptran[j][i])
                count += 1
            psi_op.append(np.kron(np.array(psi_vec).T,
                          np.conjugate(psi_vec)).reshape((lg**2, lg**2)))
        Pi = psi_op[0]
        for i in psi_op[1:]:
            Pi = np.add(Pi, i)
        return Pi


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


# def is_unitary(operator, tolerance=0.0001):
#     h, w = operator.shape
#     if not h == w:
#         return False
#     adjoint = np.conjugate(operator.transpose())
#     product1 = np.dot(operator, adjoint)
#     product2 = np.dot(adjoint, operator)
#     ida = np.eye(h)
#     return np.allclose(product1, ida) & np.allclose(product2, ida)


# def google_matrix(alpha, C):
#     E = connect_to_E(C)
#     N = len(C)
#     G = alpha*E + (1-alpha)/N * np.ones((N, N), dtype=float)
#     return G


# def connect_to_E(C):
#     '''
#     C is conectivity matrix
#     C: np.array
#     output
#     E: np.array
#     '''
#     N = len(C)
#     C = np.array(C)
#     E = np.zeros(C.shape)
#     rowsum = np.sum(C, axis=0)
#     for ind, val in enumerate(rowsum):
#         if val == 0:
#             for j in range(N):
#                 E[j][ind] = 1/N
#         else:
#             for j in range(N):
#                 E[j][ind] = C[j][ind]/val
#     assert(np.sum(np.sum(E, axis=0)) == N)
#     return E


# if __name__ == '__main__':
#     # graph = np.array([[0, 1, 0, 0, 1, 0, 0, 1],
#     #                   [0, 0, 0, 1, 1, 0, 1, 0],
#     #                   [0, 0, 0, 1, 0, 1, 0, 1],
#     #                   [0, 1, 0, 0, 0, 0, 1, 0],
#     #                   [0, 1, 0, 0, 0, 1, 0, 1],
#     #                   [0, 1, 0, 0, 1, 0, 1, 1],
#     #                   [0, 1, 0, 0, 1, 0, 0, 1],
#     #                   [0, 1, 0, 0, 1, 0, 1, 0]])
#     graph = np.array([[0, 1, 1, 0],
#                       [0, 0, 0, 1],
#                       [0, 0, 0, 1],
#                       [0, 1, 1, 0]])
#     # e = np.array([[1, 1, 1, 0],
#     #               [1, 0, 0, 1],
#     #               [1, 0, 0, 1],
#     #               [1, 1, 1, 0]])
#     pb = prob_transition(graph)
#     comp = CircuitComposer(graph, pb, 1)
#     comp.qw_circuit()
