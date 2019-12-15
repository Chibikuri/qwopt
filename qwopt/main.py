from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import Instruction

import numpy as np

q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c, name='Had')

qc.h(q[0])
instruction = qc.to_instruction()

print(instruction)
qx = QuantumRegister(2)
cx = ClassicalRegister(1)
qcx = QuantumCircuit(qx, cx)
qcx.append(instruction, qargs=[qx[0]])
print(qcx)
