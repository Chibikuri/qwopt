import numpy as np
from parser import GraphParser
from qiskit import QuantumRegister, QuantumCircuit
from numpy import pi


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
        ref_states, ref_index = self.parser.reference_state()
        ref_index.append(len(self.graph))
        print(self.graph)
        T_instructions = []
        for irf, rf in enumerate(ref_index[:-1]):
            temp = []
            for i in range(rf, ref_index[irf+1]):
                #  control from i and target from bins
                temp.append(self.graph[:, i])
            Ti_op = self._bin_converter(temp, rf)
            if Ti_op is not None:
                T_instructions.append(Ti_op)
        return T_instructions

    def _bin_converter(self, states, control):
        if len(states) == 1:
            # if the length of state is 1, we don't need to move
            # with T operation
            pass
        else:
            ref_state = states[0]
            for st in states[1:]:
                conv = self._take_bins(ref_state, st)
                # print(control, conv)
            target = [self._target_hm(cnv[0], cnv[1])[0] for cnv in conv]
            control = list(self._binary_formatter(control, self.q_size//2))
            addi_control = [self._target_hm(cnv[0], cnv[1])[1] for cnv in conv]
            # create instruction
            q_cont = QuantumRegister(self.q_size//2)
            q_targ = QuantumRegister(self.q_size//2)
            ancilla = QuantumRegister(self.q_size//2)
            qc = QuantumCircuit(q_cont, q_targ, ancilla, name='T%s'%control)
            for act, tgt in zip(addi_control, target):
                if act == []:
                    pass
                else:
                    for ic, cont in enumerate(control):
                        if cont == '1':
                            qc.x(q_cont[ic])
                    for tg in tgt:
                        qc.mct([*q_cont, *[q_targ[ac] for ac in act]], q_targ[tg], ancilla)
                    for ic, cont in enumerate(control):
                        if cont == '1':
                            qc.x(q_cont[ic])
                    return qc.to_instruction()

    def _target_hm(self, st1, st2):
        # print(st1, st2)
        hm = []
        # additional control operations
        ct_add = []
        if st1 != st2:
            for ind, s in enumerate(zip(st1, st2)):
                if s[0] != s[1]:
                    hm.append(ind)
                else:
                    ct_add.append(ind)
        return hm, ct_add

    def _take_bins(self, ref_state, other):
        converter = []
        for irf, rf in enumerate(ref_state):
            if rf == 1:
                converter.append([self._binary_formatter(irf, self.q_size//2)])
        ct = 0
        for iot, ot in enumerate(other):
            if ot == 1:
                converter[ct].append(self._binary_formatter(iot, self.q_size//2))
                ct += 1
        return converter

    @staticmethod
    def _binary_formatter(n, basis):
        return format(n, '0%sb' % str(basis))

    def K_operation(self, dagger=False, ancilla=True):
        '''
        create reference states from basis state
        K|b> = |phi_r> 
        '''
        refs, refid = self.parser.reference_state()
        rotations = self._get_rotaions(refs, dagger)
        # create Ki operations
        qcont = QuantumRegister(self.q_size//2)
        qtarg = QuantumRegister(self.q_size//2)
        if ancilla:
            anc = QuantumRegister(self.q_size//2)
            qc = QuantumCircuit(qcont, qtarg, anc, name='Kop_anc') 
        else:
            anc = None
            qc = QuantumCircuit(qcont, qtarg, name='Kop')
        ct = 0
        for i in range(self.dim[0]):
            ib = list(self._binary_formatter(i, self.q_size//2))
            if i in refid:
                rfrot = rotations[ct]
                ct += 1
            for ibx, bx in enumerate(ib):
                if bx == '0':
                    qc.x(qcont[ibx])
            qc = self._constructor(qc, rfrot, qcont, qtarg, anc)
            for ibx, bx in enumerate(ib):
                if bx == '0':
                    qc.x(qcont[ibx])
        K_instruction = qc.to_instruction()
        return K_instruction

    def _constructor(self, circuit, rotations, cont, targ, anc):
        # TODO check
        if len(rotations) == 1:
            circuit.mcry(rotations[0], cont, targ[0], anc)
            return circuit
        elif len(rotations) == 2:
            circuit.mcry(rotations[0], cont, targ[0], anc)
            circuit.x(targ[0])
            circuit.mcry(rotations[1], [*cont, targ[0]], targ[1], anc)
            circuit.x(targ[0])
            return circuit
        else:
            circuit.mcry(rotations[0], cont, targ[0], anc)
            circuit.x(targ[0])
            circuit.mcry(rotations[1], [*cont, targ[0]], targ[1], anc)
            circuit.x(targ[0])
            for irt, rt in enumerate(rotations[2:]):
                # insted of ccc...hhh...
                for tg in targ[irt+1:]:
                    circuit.mcry(pi/2, [*cont, *targ[:irt+1]], tg, anc)
                circuit.x(targ[:irt+2])
                circuit.mcry(rt, [*cont, *targ[:irt+2]], targ[irt+2], anc)
                circuit.x(targ[:irt+2])
            return circuit

    def _get_rotaions(self, state, dagger):
        rotations = []
        if self.basis_state != 0:
            raise Exception('Under construction')
        else:
            for st in state:
                rt = self._rotation(st, [], dagger)
                rotations.append(rt)
        return rotations

    def _rotation(self, state, rotations, dagger):
        lst = len(state)
        if lst == 2:
            if sum(state) != 0:
                rotations.append(2*np.arccos(np.sqrt(state[0]/sum(state))))
            else:
                rotations.append(0)
        else:
            if sum(state) != 0:
                rotations.append(2*np.arccos(np.sqrt(sum(state[:int(lst/2)])/sum(state))))
            else:
                rotations.append(0)
            self._rotation(state[:int(lst/2)], rotations, dagger)
        if dagger:
            rotations = [-i for i in rotations]
        return rotations

    def Kdg_operation(self):
        '''
        convert reference state to basis state
        '''
        pass

    def D_operation(self, n_anilla=0, mode='basic', barrier=False):
        '''
        This operation is for flipping phase of specific basis state
        '''
        # describe basis state as a binary number
        nq = int(self.q_size/2)
        basis_bin = list(self._binary_formatter(self.basis_state, nq))
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
    # graph = np.array([[0, 1, 0, 0],
    #                   [0, 0, 0, 1],
    #                   [0, 0, 0, 1],
    #                   [0, 1, 1, 0]])
    # graph = np.array([[0, 1, 0, 0, 1, 0],
    #                   [0, 0, 0, 1, 1, 0],
    #                   [0, 0, 0, 1, 1, 1],
    #                   [0, 1, 1, 0, 0, 0],
    #                   [0, 1, 0, 0, 0, 1],
    #                   [0, 1, 0, 0, 1, 0])
    graph = np.array([[0, 1, 0, 0, 1, 0, 0, 1],
                      [0, 0, 0, 1, 1, 0, 1, 0],
                      [0, 0, 0, 1, 0, 1, 0, 1],
                      [0, 1, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1, 0, 1],
                      [0, 1, 0, 0, 1, 0, 1, 1],
                      [0, 1, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 1, 0, 1, 0]])
    pb = prob_transition(graph)
    opcreator = OperationCreator(graph, pb)
    opcreator.D_operation()
    opcreator.T_operation()
    opcreator.K_operation()
