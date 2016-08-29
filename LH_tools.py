from numpy import pi

from qutip import *
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def s_function(t, N=1024, epsilon=0.1):
    """Computes the rate function s(t)"""
    nomerator = np.sqrt(N - 1) * np.tan((2 * t * epsilon * np.sqrt(N - 1) - N * np.arctan(np.sqrt(N - 1))) / N)
    denominator = 1 - N
    return 1 / 2 * (1 - nomerator / denominator)


def find_s_one_numeric(s_function, N, epsilon):
    # We use hinge function to prevent negative solutions
    s_zero = lambda x: max(1 - s_function(x, N, epsilon), 2 - x)
    # Find closest solution to 0 for 1-s(t)=0
    return fsolve(s_zero, 0)[0]


# Or
def find_s_one(N, epsilon):
    return 1 / (2 * epsilon) * N / np.sqrt((N - 1)) * 2 * np.arctan(np.sqrt(N - 1))


def SendMail(subject, text, figure=False):
    From = 'rashbam.mailer@gmail.com'
    To = ['or.yonatan@gmail.com']
    if figure:
        figure.savefig("fig.png")
        img_data = open("fig.png", 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = 'or.yonatan@gmail.com'
    msg['To'] = 'or.yonatan@gmail.com'

    text = MIMEText(text)
    msg.attach(text)
    if figure:
        image = MIMEImage(img_data, name=os.path.basename("fig.png"))
        msg.attach(image)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("rashbam.mailer@gmail.com", "nv8ZqTi1oSUYSfN5DS$r")
    s.sendmail(From, To, msg.as_string())
    s.quit()


# Speed test

def benchmark(h_t, in_state, steps=100, T=100):
    """Run simulation for the given number of steps and returns P_mat, Evals_mat and time to compute
    """
    start_time = time.time()
    tlist = np.linspace(0, T, steps)
    args = {'t_max': max(tlist)}
    P_mat, evals_mat = simulate_adiabatic_process2(tlist, h_t, args, in_state, False)
    plot_PandEV(P_mat, evals_mat, tlist)
    print("Computation time %s seconds :" % (time.time() - start_time))
    return P_mat, evals_mat, time.time() - start_time


def plot_PandEV(P_mat, EV_mat, tlist, figsize=(15, 5)):
    fig = plt.figure(figsize=figsize)
    P_plt = fig.add_subplot(1, 2, 1)
    P_plt.set_title("Occupation probabilities")
    P_plt.plot(tlist, P_mat)
    ev_plt = fig.add_subplot(1, 2, 2)
    ev_plt.set_title("Eigenvalues")
    ev_plt.plot(tlist, EV_mat)
    return fig


"""

def plot_PandEV(P_mat, EV_mat, tlist, figsize=(15,5)):
    f,(P_plt,ev_plt) = plt.subplots(1, 2, figsize=figsize)
    P_plt.set_title("Occupation probabilities")
    P_plt.plot(tlist, P_mat)
    ev_plt.set_title("Eigenvalues")
    ev_plt.plot(tlist, EV_mat)
    
"""


def find_gap(evals_mat):
    """Finds the minimal gap in a list of eiganvalue pairs"""
    min_gap = float("inf")
    for pair in evals_mat:
        dist = abs(pair[0] - pair[1])
        min_gap = min(dist, min_gap)
    return min_gap


def gen_SAT_LH():
    zeromat = basis(2, 0) * basis(2, 0).trans()
    oneomat = basis(2, 1) * basis(2, 1).trans()
    zero_one = tensor(basis(2, 0), basis(2, 1))  # |01>
    zero_onemat = zero_one * zero_one.trans()
    hamiltonian_terms = []
    term1_hamiltonian = tensor(zero_onemat, qeye(2))
    hamiltonian_terms.append(term1_hamiltonian)
    hamiltonian_terms.append(tensor(qeye(2), zero_onemat))
    hamiltonian_terms.append(tensor(oneomat, qeye(2), zeromat))
    hamiltonian_terms.append(tensor(zeromat, qeye(2), qeye(2)))
    return sum(hamiltonian_terms)


def gen_simple_ham(n):
    zero3 = tensor([basis(2, 0)] * n)
    zero3_mat = zero3 * zero3.trans()
    id3 = tensor([qeye(2)] * n)
    simple_ham = id3 - zero3_mat
    return simple_ham, simple_ham.groundstate()[1]


def rotate_by_had(in_ham, groundstate):
    n = len(in_ham.dims[0])
    rot_ham = tensor([hadamard_transform()] * n) * in_ham * tensor([hadamard_transform()] * n)
    rot_gs = tensor([hadamard_transform()] * n) * groundstate
    return rot_ham, rot_gs


def simulate_adiabatic_process2(tlist, h_t, args, rho0, draw, options=Options()):
    #
    # callback function for each time-step
    #
    N = 2
    M = 2
    evals_mat = np.zeros((len(tlist), M))
    P_mat = np.zeros((len(tlist), M))
    psis = []
    idx = [0]
    def process_rho(tau, psi):

        # evaluate the Hamiltonian with gradually switched on interaction
        H = Qobj.evaluate(h_t, tau, args)

        evals, ekets = H.eigenstates(eigvals=M)
        evals_mat[idx[0], :] = np.real(evals)
        # print(psi)
        # find the overlap between the eigenstates and psi
        for n, eket in enumerate(ekets):
            P_mat[idx[0], n] = abs((eket.dag().data * psi.data)[0, 0]) ** 2
        psis.append(psi)
        idx[0] += 1

    output = qutip.sesolve(H=h_t,
                     rho0=rho0,
                     tlist=tlist,
                     e_ops=process_rho,
                     args=args,
                     options=options)
    # rc('font', family='serif')
    # rc('font', size='10')
    if draw:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        #
        # plot the energy eigenvalues
        #

        # first draw thin lines outlining the energy spectrum
        for n in range(len(evals_mat[0, :])):
            ls, lw = ('b', 1) if n == 0 else ('k', 0.25)
            axes[0].plot(tlist / max(tlist), evals_mat[:, n] / (2 * pi), ls, lw=lw)

        # second, draw line that encode the occupation probability of each state in
        # its linewidth. thicker line => high occupation probability.
        for idx in range(len(tlist) - 1):
            for n in range(len(P_mat[0, :])):
                lw = 0.5 + 4 * P_mat[idx, n]
                if lw > 0.55:
                    axes[0].plot(np.array([tlist[idx], tlist[idx + 1]]) / max(tlist),
                                 np.array([evals_mat[idx, n], evals_mat[idx + 1, n]]) / (2 * pi),
                                 'r', linewidth=lw)
        axes[0].set_xlabel(r'$\tau$')
        axes[0].set_ylabel('Eigenenergies')
        axes[0].set_title("Energyspectrum (%d lowest values) of a chain of %d spins.\n " % (M, N)
                          + "The occupation probabilities are encoded in the red line widths.")

        #
        # plot the occupation probabilities for the few lowest eigenstates
        #
        for n in range(len(P_mat[0, :])):
            if n == 0:
                axes[1].plot(tlist / max(tlist), 0 + P_mat[:, n], 'r', linewidth=2)
            else:
                axes[1].plot(tlist / max(tlist), 0 + P_mat[:, n])

        axes[1].set_xlabel(r'$\tau$')
        axes[1].set_ylabel('Occupation probability')
        axes[1].set_title("Occupation probability of the %d lowest " % M +
                          "eigenstates for a chain of %d spins" % N)
        axes[1].legend(("Ground state",))

    return P_mat, evals_mat,psis