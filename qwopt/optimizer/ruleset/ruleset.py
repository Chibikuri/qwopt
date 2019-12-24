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

    def __init__(self):
        pass

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
