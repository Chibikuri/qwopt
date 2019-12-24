import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


class Converter:
    '''
    Converting quantum circuit to dag
    and extract information used for optimizations
    '''

    def __init__(self):
        pass

    def converter(self, qc):
        dag = circuit_to_dag(qc)
        print(dag.layers())
        print(dag.nodes())
        for i in dag.nodes():
            print(i)
        print(dag.ancestors(i))
        return dag

    def reverser(self, dag):
        pass


if __name__ == '__main__':
    q = QuantumRegister(4)
    qc = QuantumCircuit(q)
    

    dag = Converter().converter(qc)
    dag.draw()
