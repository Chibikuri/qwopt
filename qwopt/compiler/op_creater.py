import numpy as np
from .parser import GraphParser
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
    which can be added to a circuit directly
    '''
    def __init__(self, graph, prob_tran, basis=0, optimize=True):
        self.parser = GraphParser(graph, prob_tran)
        self.graph = self.parser.graph
        self.ptran = self.parser.ptrans
        self.dim = self.parser.dim()
        self.q_size = self._qubit_size(len(self.parser))
        self.basis_state = basis

    def T_operation(self):
        '''
        This is the operation called T operation.
        This operator is converting some state to its reference state
        by replacing binary order
        NOTE: I'm not sure if there is an optimized way to do this,
        but, thinking we can do this if we use some boolean calcuration.
        '''
        ref_states, ref_index = self.parser.reference_state()
        ref_index.append(len(self.graph))
        T_instructions = []
        for irf, rf in enumerate(ref_index[:-1]):
            temp = []
            for i in range(rf, ref_index[irf+1]):
                #  control from i and target from bins
                temp.append(self.graph[:, i])
            Ti_op = self._bin_converter(temp, range(rf, ref_index[irf+1]))
            if Ti_op is not None:
                T_instructions.append(Ti_op)
        return T_instructions

    # TODO more understandable name
    def _bin_converter(self, states, cont, ancilla=True):
        if len(states) == 1:
            # if the length of state is 1, we don't need to move
            # with T operation
            pass
        else:
            ref_state = states[0]
            # make correspondence table
            convs = [self._take_bins(ref_state, st) for st in states[1:]]
            if convs == [[]]:
                # if all table elements are the same value,
                # we don't have to apply
                return None
            else:
                # TODO
                # here we have to optimize additional target
                ctable = self._addi_analysis(convs)
                target = [self._target_hm(cnv) for cnv in ctable]

                control = [list(self._binary_formatter(ct, self.q_size//2))
                           for ct in cont]
                # create instruction
                q_cont = QuantumRegister(self.q_size//2)
                q_targ = QuantumRegister(self.q_size//2)
                if ancilla:
                    ancilla = QuantumRegister(self.q_size//2)
                    qc = QuantumCircuit(q_cont, q_targ,
                                        ancilla, name='T%s' % cont[0])
                else:
                    ancilla = None
                    qc = QuantumCircuit(q_cont, q_targ, name='T%s' % cont[0])
                for cts, tgt in zip(control[1:], target):
                    for ic, ct in enumerate(cts):
                        if ct == '0' and tgt != set():
                            qc.x(q_cont[ic])
                    for tg in tgt:
                        qc.mct(q_cont, q_targ[tg], ancilla)
                    for ic, ct in enumerate(cts):
                        if ct == '0' and tgt != set():
                            qc.x(q_cont[ic])
                return qc.to_instruction()

    def _target_hm(self, state):
        # the place we have to change
        hm = []
        for st in state:
            # FIXME this reverse operations must be done before here.
            # This reverse op is for making it applicable for qiskit.
            for ids, s in enumerate(zip(st[0][::-1], st[1][::-1])):
                if s[0] != s[1]:
                    hm.append(ids)
        return set(hm)

    def _take_bins(self, ref_state, other):
        converter = []
        for irf, rf in enumerate(ref_state):
            if rf == 1:
                converter.append([self._binary_formatter(irf, self.q_size//2)])
        ct = 0
        for iot, ot in enumerate(other):
            if ot == 1:
                converter[ct].append(self._binary_formatter(iot,
                                                            self.q_size//2))
                ct += 1
        return converter

    # more understandable name
    def _addi_analysis(self, conversions):
        '''
        remove duplications
        '''
        for icv, cv in enumerate(conversions):
            # FIXME are there any efficient way rather than this?
            if cv[0][0] == cv[0][1] and cv[1][0] == cv[1][1]:
                conversions[icv] = []
        conversion_table = conversions
        return conversion_table

    @staticmethod
    def _binary_formatter(n, basis):
        return format(n, '0%db' % basis)

    def K_operation(self, dagger=False, ancilla=True, optimization=True,
                    n_opt_ancilla=2, rccx=True):
        '''
        Args:
            dagger:
            ancilla:
            optimization:
            n_opt_ancilla:
            rccx
        create reference states from basis state
        or if this is Kdag, reverse operation
        TODO: should separate the creation part and optimization part
        '''
        refs, refid = self.parser.reference_state()
        rotations = self._get_rotaions(refs, dagger)
        # create Ki operations
        qcont = QuantumRegister(self.q_size//2, 'control')
        qtarg = QuantumRegister(self.q_size//2, 'target')
        # 1, ancilla mapping optimization
        # if the number of reference states is the same as
        # the length of matrix, we can't apply first optimization method.
        if optimization and len(refid) != self.dim[0]:
            # TODO surplus ancillary qubits should be used efficientry
            # separate the optimization part and creation part
            if n_opt_ancilla > len(qcont):
                # FIXME:
                raise ValueError('too much ancillary qubits')
            opt_anc = QuantumRegister(n_opt_ancilla, name='opt_ancilla')
            if ancilla:
                anc = QuantumRegister(self.q_size//2)
                qc = QuantumCircuit(qcont, qtarg, anc, opt_anc, name='opt_Kop')
            else:
                anc = None
                qc = QuantumCircuit(qcont, qtarg, opt_anc, name='opt_Kop_n')
            # HACK
            # Unlke to bottom one, we need to detect which i we apply or not
            qc = self._opt_K_operation(qc, qcont, qtarg, anc, opt_anc,
                                       refid, rotations, dagger, rccx)
            opt_K_instruction = qc.to_instruction()
            return opt_K_instruction
        else:
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
                qc.barrier()
            K_instruction = qc.to_instruction()
            return K_instruction

    def _opt_K_operation(self, qc, qcont, qtarg, anc, opt_anc,
                         refid, rotations, dagger, rcx=True):
        '''
        TODO: apply each optimizations separately.
        using ancilla, the structure is a litlle bit different from
        usual one. we need to care about it.
        '''
        # If you are using rccx, the phase is destroyed
        # you need to fix after one iteration
        # HACK
        # make the loop with rotations
        # we have to detect if we can apply ancilla optimization
        lv_table = self._level_detector(refid)
        if dagger:
            # mapping with rccx
            qc = self._map_ancilla_dag(qc, qcont, qtarg, anc, opt_anc,
                                       refid, rotations, lv_table, rcx)
        else:
            # fix phase of ancilla
            qc = self._map_ancilla(qc, qcont, qtarg, anc, opt_anc,
                                   refid, rotations, lv_table, rcx)
        return qc

    def _level_detector(self, refid):
        lv_table = []
        mx_bin = self.dim[0]
        for i in range(1, mx_bin):
            if i in refid:
                pre_i = list(self._binary_formatter(i-1, self.q_size//2))
                bin_i = list(self._binary_formatter(i, self.q_size//2))
                for ilv, st in enumerate(zip(pre_i, bin_i)):
                    if st[0] != st[1]:
                        lv_table.append(ilv)
                        break
        return lv_table

    # FIXME too much argument
    def _map_ancilla_dag(self, qc, cont, targ, anc, opt_anc,
                         refid, rotations, lv_table, rcx):
        '''
        applying dagger operation
        the number of rotaions is corresponding to the number of
        partitions of matrix.
        '''
        n_partitions = len(refid) - 1
        # FIXME debug settings!!!!!!!
        # opt_level = len(opt_anc)
        opt_level = 2
        start = min(lv_table)
        end = start + opt_level
        # mapping once from control to ancillary qubits
        # iopt is for pointing the index of ancillary qubits
        # procedure
        # In the case of the number of partitioin is one
        # 1. detect the position of partitions if the position is the
        # second most left or right, apply operations.
        # 2. else, we need to detect the level of optimization
        if n_partitions == 1:
            # HACK
            if refid[-1] == 1 or refid[-1] == self.dim[0] - 1:
                qc = self._mapping(qc, cont, opt_anc[0], False, anc)
                for rotation in rotations:
                    qc = self._constructor(qc, rotation, [opt_anc[0]],
                                           targ, anc)
                    qc.x(opt_anc[0])
                # no inverse is required here
            else:
                if opt_level == self.q_size//2 - 1:
                    print('hello')
                # general case
                print(refid)
                print(lv_table)
                for iopt, opt in enumerate(range(start, end)):
                    # TODO: do we have to take indices?
                    # qc.cx(cont[opt], opt_anc[iopt])
                    raise Warning('Under construction')
        else:
            raise Warning('the case of the number of partition is over \
                          1 is being under constructing.')
        print(qc)
        return qc

    @staticmethod
    def _mapping(qc, cont, targ, rcx, anc):
        if len(cont) == 1:
            qc.cx(cont, targ)
        elif len(cont) == 2:
            if rcx:
                qc.rccx(cont[0], cont[1], targ)
            else:
                qc.ccx(cont[0], cont[1], targ)
        else:
            qc.mct(cont, targ, anc)
        return qc

    def _map_ancilla(self, qc, cont, targ, anc, opt_anc,
                     refid, rotations, lv_table, rcx):
        '''
        '''
        n_partitions = len(refid) - 1
        # FIXME debug settings!!!!!!!
        # opt_level = len(opt_anc)
        opt_level = 2
        start = min(lv_table)
        end = start + opt_level
        if n_partitions == 1:
            # HACK
            if refid[-1] == 1 or refid[-1] == self.dim[0] - 1:
                for rotation in rotations:
                    qc = self._constructor(qc, rotation, [opt_anc[0]],
                                           targ, anc)
                    qc.x(opt_anc[0])
                # This corresponds to the reverse operation
                qc = self._mapping(qc, cont, opt_anc[0], False, anc)
                # no inverse is required here
            else:
                for iopt, opt in enumerate(range(start, end)):
                    # TODO: do we have to take indices?
                    # qc.cx(cont[opt], opt_anc[iopt])
                    raise Warning('Under construction')
        else:
            raise Warning('the case of the number of partition is over \
                          1 is being under constructing.')
        return qc

    def _qft_constructor(self):
        '''
        Thinking if we can reduce the number of operations with qft
        '''
        pass

    def _constructor(self, circuit, rotations, cont, targ, anc):
        # TODO check
        if len(rotations) == 1:
            circuit.mcry(rotations[0], cont, targ[0], anc)
            return circuit
        elif len(rotations) == 2:
            circuit.mcry(rotations[0], cont, targ[0], anc)
            circuit.mcry(rotations[1], cont, targ[1], anc)
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
                rotations.append(2*np.arccos(
                                 np.sqrt(sum(state[:int(lst/2)])/sum(state))))
            else:
                rotations.append(0)
            self._rotation(state[:int(lst/2)], rotations, dagger)
        if dagger:
            rotations = [-i for i in rotations]
        return rotations

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
    #                   [0, 0, 1, 1],
    #                   [0, 0, 0, 1],
    #                   [0, 1, 1, 0]])
    # graph = np.array([[0, 1, 0, 0, 1, 0],
    #                   [0, 0, 0, 1, 1, 0],
    #                   [0, 0, 0, 1, 1, 1],
    #                   [0, 1, 1, 0, 0, 0],
    #                   [0, 1, 0, 0, 0, 1],
    #                   [0, 1, 0, 0, 1, 0])
    # graph = np.array([[0, 0, 1, 0, 0, 0, 0, 1],
    #                   [0, 0, 0, 1, 0, 0, 1, 0],
    #                   [0, 0, 1, 0, 0, 1, 1, 1],
    #                   [0, 0, 0, 0, 0, 1, 1, 0],
    #                   [0, 0, 0, 1, 1, 1, 1, 1],
    #                   [0, 0, 0, 0, 1, 1, 1, 1],
    #                   [0, 0, 0, 0, 1, 0, 0, 1],
    #                   [0, 0, 0, 0, 1, 0, 1, 0]])
    graph = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 1, 0]])
    pb = prob_transition(graph)
    opcreator = OperationCreator(graph, pb)
    # opcreator.D_operation()
    # opcreator.T_operation()
    opcreator.K_operation(dagger=True)
