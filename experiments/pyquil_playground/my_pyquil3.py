import numpy as np
import matplotlib.pyplot as plt

from pyquil import get_qc
from pyquil.quil import Program, MEASURE, Pragma
from pyquil.gates import I, X, RX, H, CNOT
from pyquil.noise import (estimate_bitstring_probs, correct_bitstring_probs,
                          bitstring_probs_to_z_moments, estimate_assignment_probs)

DARK_TEAL = '#48737F'
FUSCHIA = '#D6619E'
BEIGE = '#EAE8C6'

qc = get_qc("1q-qvm")

for t in [5,20,50,200]:
    for prob in [0.5, 0.6, 0.62, 0.65, 0.7, 0.8, 0.9]:

        # number of angles
        # num_theta = 945*2 #random so it does not fit nicely
        # num_theta = 101 #random so it does not fit nicely
        num_theta = 501 #random so it does not fit nicely

        # number of program executions
        # trials = 10
        # trials = 50
        trials = t

        # length_in_half_cycles = 300
        # length_in_half_cycles = 15
        length_in_half_cycles = 15*5

        thetas = np.linspace(0, length_in_half_cycles*np.pi, num_theta)

        # p00s = [1., 0.95, 0.9, 0.8, 0.5]
        # p00s = [0.62]
        p00s = [prob]

        results_rabi = np.zeros((num_theta, len(p00s)))

        for jj, theta in enumerate(thetas):
            for kk, p00 in enumerate(p00s):
                # theta_mod = theta % 2*np.pi
                theta_mod=theta
                # qc.qam.random_seed = hash((jj, kk))
                qc.qam.random_seed = abs(hash((jj, kk)))
                p = Program(RX(theta_mod, 0)).wrap_in_numshots_loop(trials)
                # assume symmetric noise p11 = p00
                p.define_noisy_readout(0, p00=p00, p11=p00)
                ro = p.declare("ro", "BIT", 1)
                p.measure(0, ro[0])
                res = qc.run(p).readout_data.get("ro")
                results_rabi[jj, kk] = np.sum(res)

        f = open("out/" + str(t) + "_" + str(int(prob*100))+"_data.txt", "w")
        for measurement in results_rabi[:, 0]/trials:
            f.write(str(measurement) + "\n")
        f.close()

        f = open("out/" + str(t) + "_" + str(int(prob*100))+"_target.txt", "w")
        for measurement in np.sin(thetas-np.pi/2)/2+0.5:
            f.write(str(measurement) + "\n")
        f.close()
