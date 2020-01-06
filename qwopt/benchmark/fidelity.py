import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange, tqdm
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit import Aer, execute
from qiskit.quantum_info import state_fidelity
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import depolarizing_error
from qiskit import transpile


QASM = Aer.get_backend('qasm_simulator')
STATEVEC = Aer.get_backend('statevector_simulator')


class FidelityAnalyzer:
    '''
    This analyzer is for analyzing the fidelity
    performance under the noisy situation.
    Errors are very naive.
    If you put quantum circuit, this module analyze the
    fidelity decreasing automatically.
    '''

    def __init__(self, one_error, two_error, measure_qubit,
                 extime=10, shots=5000):
        if not isinstance(one_error, (float, int, np.ndarray, list)):
            raise ValueError('one error is must be float, int, array or list.')
        else:
            # error rate of u3
            self.one_error = one_error

        if not isinstance(two_error, (float, int, np.ndarray, list)):
            raise ValueError('one error is must be float, int, array or list.')
        else:
            # error rate of cx
            self.two_error = two_error
        # TODO: make 3D plot when one error and two error are array
        self.shots = shots
        self.mes_qbt = measure_qubit
        self.extime = extime

    def fidelity_drop(self, qc, drawing=True, **kwargs):
        nqc = transpile(qc, optimization_level=0, basis_gates=['cx', 'u3'])
        # HACK efficient ways?
        if isinstance(self.one_error,
                      (float, int)) and isinstance(self.two_error,
                                                   (float, int)):
            fidelitis, std = self.fixed_fidelity(nqc)
            # FIXME more easy way
            self.pattern = 0
        # FIXME more seeable
        elif isinstance(self.one_error,
                        (float, int)) and isinstance(self.two_error,
                                                     (np.ndarray, list)):
            fidelities, std = self._u3fix(nqc)
            self.pattern = 1
        elif isinstance(self.two_error,
                        (float, int)) and isinstance(self.one_error,
                                                     (np.ndarray, list)):
            cxerror = depolarizing_error(self.two_error, 2)
            fidelities, std = self._cxfix(nqc)
            self.pattern = 2
        else:
            fidelities, std = self._nofix(nqc)
            self.pattern = 3
        if drawing:
            self._draw(fidelities, std, **kwargs)
        return fidelities

    def _u3fix(self, qc):
        print(qc.count_ops())
        nst = 2**len(self.mes_qbt)
        bins = [format(i, '0%db' % len(self.mes_qbt))
                for i in range(nst)]

        # ideal result of this circuit
        ideal = execute(qc, backend=QASM, shots=self.shots*10)
        idealcounts = ideal.result().get_counts()
        idealst = np.array([idealcounts.get(i, 0)/(self.shots*10)
                           for i in bins])
        # making noise model with error rate
        u3error = depolarizing_error(self.one_error, 1)
        # start simulations
        mean_fid = []
        std_fid = []
        for two_err in tqdm(self.two_error):
            mid = []
            noise_model = NoiseModel()
            cxerror = depolarizing_error(two_err, 2)
            noise_model.add_all_qubit_quantum_error(u3error, ['u3'])
            noise_model.add_all_qubit_quantum_error(cxerror, ['cx'])
            for t in range(self.extime):
                # execute!
                job = execute(qc, backend=QASM, noise_model=noise_model,
                              shots=self.shots)
                counts = job.result().get_counts()
                stvec = [counts.get(i, 0)/self.shots for i in bins]
                stf = state_fidelity(idealst, stvec)
                mid.append(stf)
            mean_fid.append(np.mean(mid))
            std_fid.append(np.std(mid))
        return mean_fid, std_fid

    def _cxfix(self, qc):
        nst = 2**len(self.mes_qbt)
        bins = [format(i, '0%db' % len(self.mes_qbt))
                for i in range(nst)]

        # ideal result of this circuit
        ideal = execute(qc, backend=QASM, shots=self.shots*10)
        idealcounts = ideal.result().get_counts()
        idealst = np.array([idealcounts.get(i, 0)/(self.shots*10)
                           for i in bins])
        # making noise model with error rate
        cxerror = depolarizing_error(self.two_error, 2)
        # start simulations
        mean_fid = []
        std_fid = []
        for one_er in tqdm(self.one_error):
            mid = []
            noise_model = NoiseModel()
            u3error = depolarizing_error(one_er, 1)
            noise_model.add_all_qubit_quantum_error(u3error, ['u3'])
            noise_model.add_all_qubit_quantum_error(cxerror, ['cx'])

            for t in range(self.extime):
                job = execute(qc, backend=QASM, noise_model=noise_model,
                              shots=self.shots)
                counts = job.result().get_counts()
                stvec = [counts.get(i, 0)/self.shots for i in bins]
                stf = state_fidelity(idealst, stvec)
                mid.append(stf)
            mean_fid.append(np.mean(mid))
            std_fid.append(np.std(mid))
        return mean_fid, std_fid

    def _nofix(self, qc):
        nst = 2**len(self.mes_qbt)
        bins = [format(i, '0%db' % len(self.mes_qbt))
                for i in range(nst)]

        # ideal result of this circuit
        ideal = execute(qc, backend=QASM, shots=self.shots*10)
        idealcounts = ideal.result().get_counts()
        idealst = np.array([idealcounts.get(i, 0)/(self.shots*10)
                           for i in bins])
        # start simulations
        if len(self.one_error) != len(self.two_error):
            raise ValueError('length of array of one error \
                              and two error must be the same.')

        mean_fid = []
        std_fid = []
        for one_er, two_er in tqdm(zip(self.one_error, self.two_error)):
            mid = []
            # HACK: might be efficient in top layer
            noise_model = NoiseModel()
            u3error = depolarizing_error(one_er, 1)
            cxerror = depolarizing_error(two_er, 2)
            noise_model.add_all_qubit_quantum_error(u3error, ['u3'])
            noise_model.add_all_qubit_quantum_error(cxerror, ['cx'])

            for t in range(self.extime):
                job = execute(qc, backend=QASM, noise_model=noise_model,
                              shots=self.shots)
                counts = job.result().get_counts()
                stvec = [counts.get(i, 0)/self.shots for i in bins]
                stf = state_fidelity(idealst, stvec)
                mid.append(stf)
            mean_fid.append(np.mean(mid))
            std_fid.append(np.std(mid))
        return mean_fid, std_fid

    def _fixed_fidelity(self, qc):
        fidelity = 0
        return fidelity

    def _draw(self, fidelities, std, errorbar=True, **kwargs):
        '''
        drawing fidelity dropping
        '''
        title = kwargs.get('title', 'Fidelity decrease')
        fontsize = kwargs.get('fontsize', 14)
        seaborn = kwargs.get('seaborn', True)
        plt.ylabel('Fidelity')
        if seaborn:
            sns.set()
        plt.title(title, fontsize=fontsize)
        if self.pattern == 0:
            raise Exception('No drawing is allowed in just \
                            one element(under construction)')
        elif self.pattern == 1:
            plt.xlabel('Two qubit gate error')
            if errorbar:
                plt.errorbar(self.two_error, fidelities, yerr=std)
            else:
                plt.plot(self.two_error, fidelities)
        elif self.pattern == 2:
            plt.xlabel('One qubit gate error')
            if errorbar:
                plt.errorbar(self.one_error, fidelities, yerr=std)
            else:
                plt.plot(self.one_error, fidelities)
        elif self.pattern == 3:
            raise Exception('under construction')
        else:
            pass
        plt.show()


if __name__ == '__main__':
    q = QuantumRegister(4)
    c = ClassicalRegister(2)
    qc = QuantumCircuit(q, c)
    qc.x(q[0])
    for i in range(3):
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[0])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])
    analyzer = FidelityAnalyzer(0.01, np.arange(0, 0.2, 0.001), [0, 1], extime=100)
    result = analyzer.fidelity_drop(qc)
    print(result)
