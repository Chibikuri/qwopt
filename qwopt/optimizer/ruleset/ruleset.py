from qiskit import QuantumRegister
from qiskit.transpiler.passes import Unroller
import toml
import sys

'''
FIXME This scheme might be unefficient.
TODO easier to write and make some parser
make rule description language with some language
Thinking Rust or Python or ...
'''


class Rules:
    '''
    In this script, making the rules of how to optimize the quantum circuit
    and what order.
    The order of appling rule is
    rule0, rule1, rule2, ...ruleN
    in the default.
    TODO make this executable in arbitrary order
    '''

    def __init__(self, config):
        '''
        HACK
        config: converted toml (opened toml)
        '''
        self.toml = toml.load(config)

    def apply_rules(self):
        pass

    def _rule0(self, dag):
        '''
        optimization about ancillary qubits
        In this optimization, once map the control operations to ancillae
        qubits, and then, reduce the number of operations.

        Input:
            dag: DAG
        Output:
            dag: DAG, success(bool)
        '''
        # FIXME if the position of partition is symmetry,
        # we can't apply this optimization because it's redundunt

        # procedure 1, add ancilal or find ancilla
        # FIXME

        p_rule0 = self.toml['rule0']
        n_ancilla = p_rule0['n_ancilla']

        # unrolling dag to basis compornents
        ndag = Unroller(['ccx', 'cx', 'x', 'h', 'u3']).run(dag)

        ndag_nodes = [i for i in ndag.nodes()]
        ndag_names = [i.name for i in ndag_nodes]

        # FIXME taking parameters dynamically
        dag_nodes = [i for i in dag.nodes()]

        if n_ancilla < 0:
            raise ValueError('The number of ancillary qubits \
                             must be 0 or over 0')
        # adding ancilla qubit
        q = QuantumRegister(n_ancilla, name='opt ancilla')
        dag.add_qreg(q)
        ndag.draw()
        return dag, False

    def _rule1(self, dag):
        '''
        Compressing neighbor controll operations
        In this optimization, search which control operations can be
        cancelled out, and then delete them.
        '''
        return dag, False

    def _rule2(self, dag):
        '''
        Using phase optimizations
        In this optimiation, replace K operation with QFT or other phasian
        methods
        '''
        return dag, True
