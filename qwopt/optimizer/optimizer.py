from .ruleset import ruleset
from .converter import Converter
from qiskit import transpile
import sys
# FIXME out of pep write __init__.py
sys.path.append('../')
from compiler import parser


class Optimizer:

    def __init__(self, graph, prob_dist):
        pass

    def optimize(self, qc, n_rules, config, **kwargs):
        '''
        qc: QuantumCircuit
        n_rules: the number of rules
        config: toml file
        kwargs: optimization config
        '''
        # nqc = transpile(qc, basis_gates=['cx', 'u3'])
        dag = Converter().converter(qc)
        rule = ruleset.Rules(config)
        errors = []
        # FIXME detect the number of rules
        for i in range(n_rules):
            dag, success = eval('rule._rule%s(dag)' % str(i))
            if not success:
                errors.append(i)
        if len(errors) == 0:
            print('All optimization rules are applied successfully!')
        else:
            ms = str(errors)[1:-1]
            print('Rules, %s are not applied for some reasons!' % ms)
