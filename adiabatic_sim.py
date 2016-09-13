from qutip import *
import numpy as np
from scipy.linalg import expm


def sim_simple_adiabatic(tlist, H0, H1, s='linear'):
    """

    :param tlist: Time list
    :param H0: first hamiltonian
    :param H1: second hamiltonian
    :param s: function - relates time to coupling, default is linear dependecy
    :return:
    """
    if s == 'linear':
        s = lambda t: (t - tmin) / (tmax - tmin)
    duration = len(tlist)
    tmin = min(tlist)
    tmax = max(tlist)
    # start at H0 ground state
    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    P_mat = []
    eigvals_mat = []
    psi = eigvecs[0]
    psis = [psi]
    eigvals_mat.append(eigvals)

    P_mat.append(
        [abs(eigvecs[0].overlap(psi)) ** 2,
         abs(eigvecs[1].overlap(psi)) ** 2])
    oldt = tmin
    for t in tlist[1:]:
        dt = oldt - t
        Ht = H0 * (1 - s(t)) + H1 * (s(t))
        eigvals, eigvecs = Ht.eigenstates(eigvals=2)
        U = expm(-1j * Ht.data * dt)
        psi = Qobj(U * psi.data, dims=psi.dims)
        psis.append(psi)
        eigvals_mat.append(eigvals)
        P_mat.append(
            [abs(eigvecs[0].overlap(psi)) ** 2,
             abs(eigvecs[1].overlap(psi)) ** 2])
        oldt = t
    return P_mat, eigvals_mat, psis


def sim_noise_evolutoion(tlist, H0, H1, H3, noise_func):
    """
    Simulates a constant rate evolution with an addition a 'noise' H3 term that is controlled
    by an s function, s.t. 
    H(t) = H1 * (tmax-t)/(tmax-tmin)) + H2 * (t-tmin)/(tmax-tmin)) + H3*s(t)
    :param tlist: Time list
    :param H0: first hamiltonian
    :param H1: second hamiltonian
    :param H3: 'noise' hamiltonian
    :param noise_func: function - controls H3 power H3(t) = H3*noise_func(t)
    :return:
    """
    s = lambda t: (t - tmin) / (tmax - tmin)
    duration = len(tlist)
    tmin = min(tlist)
    tmax = max(tlist)
    # start at H0 ground state
    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    P_mat = []
    eigvals_mat = []
    psi = eigvecs[0]
    psis = [psi]
    eigvals_mat.append(eigvals)

    P_mat.append(
        [abs(eigvecs[0].overlap(psi)) ** 2,
         abs(eigvecs[1].overlap(psi)) ** 2])
    oldt = tmin
    for t in tlist[1:]:
        dt = oldt - t
        Ht = H0 * (1 - s(t)) + H1 * (s(t)) + H3 * noise_func(t)
        eigvals, eigvecs = Ht.eigenstates(eigvals=2)
        U = expm(-1j * Ht.data * dt)
        psi = Qobj(U * psi.data, dims=psi.dims)
        psis.append(psi)
        eigvals_mat.append(eigvals)
        P_mat.append(
            [abs(eigvecs[0].overlap(psi)) ** 2,
             abs(eigvecs[1].overlap(psi)) ** 2])
        oldt = t
    return P_mat, eigvals_mat, psis


def evolve(H0, H1, psi, step_size, t, max_time):
    s = (t + step_size) / max_time
    Ht = H0 * (1 - s) + H1 * s
    U = expm(-1j * Ht.data * step_size)
    psi = Qobj(U * psi.data, dims=psi.dims)
    return psi


def step_is_good(H0, H1, psi, step_size, t, max_time, target_p0):
    s_before = t / max_time
    Hbefore = H0 * (1 - s_before) + H1 * s_before
    eigstate_before = Hbefore.eigenstates(eigvals=1)[1][0]
    pbefore = np.abs(eigstate_before.overlap(psi)) ** 2
    psi_after = evolve(H0, H1, psi, step_size, t, max_time)
    s_after = (t + step_size) / max_time
    Hafter = H0 * (1 - s_after) + H1 * s_after
    eigstate_after = Hafter.eigenstates(eigvals=1)[1][0]
    pafter = np.abs(eigstate_after.overlap(psi_after)) ** 2
    # TODO: is this the right calculation to do ?
    pdiff = pbefore - pafter
    return (pdiff * (max_time - t) / step_size) <= target_p0


def sim_dynamic_evolution_binsearch(H0, H1, target_p0=0.3):
    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    tlist = []
    max_time = 39
    min_step = max_time / 5000
    max_step = max_time / 20
    step_size = min_step
    t = 0
    psi = eigvecs[0]
    import pydevd;
    # pydevd.settrace('localhost', port=4000, stdoutToServer=True, stderrToServer=True)
    while t <= max_time:
        if step_is_good(H0, H1, psi, step_size, t, max_time, target_p0):
            if t + step_size >= max_time:
                step_size = max_time - t
                if step_is_good(H0, H1, psi, step_size, t, max_time, target_p0):
                    tlist.append(max_time)
                    break
                else:
                    raise ValueError("something os wrong with step evaluation, last step should be good but is not")
                break
            double_step = 2 * step_size
            if step_size >= max_step:
                step_size = max_step
                psi = evolve(H0, H1, psi, step_size, t, max_time)
                t += step_size
                tlist.append(t)
            if step_is_good(H0, H1, psi, double_step, t, max_time, target_p0):
                step_size = double_step
            else:
                psi = evolve(H0, H1, psi, step_size, t, max_time)
                t += step_size
                tlist.append(t)

        else:  # step is too big
            if step_size <= min_step:
                step_size = min_step
                psi = evolve(H0, H1, psi, step_size, t, max_time)
                t += step_size
                tlist.append(t)
            else:
                half_step = step_size / 2
                if step_is_good(H0, H1, psi, half_step, t, max_time, target_p0):
                    psi = evolve(H0, H1, psi, half_step, t, max_time)
                    t += half_step
                    tlist.append(t)
                step_size = half_step
    return tlist

#
# def sim_dynamic_evolution_pc(H0, H1, target_p0=0.4):
#     time_eps = 0.15  # assume the 15% in the end of the process can be jumped over
#     eigvals, eigvecs = H0.eigenstates(eigvals=2)
#     tlist = []
#     end_time = 0
#     steps = 10  # sets a initial estimation for the step size
#     min_step = 0.001
#     step_size = 0
#     s = 0
#     while (step_size + s) <= (s - time_eps):
#         if step_is_good(step_size):  # we can add bypass optimization later, however performance gain will be negligible
#             for rate in range(10, 0, -1):  # try multiplying by up to 10
#                 step = rate * step_size
#                 if step_is_good(step):
#                     if rate == 10:
#                         step_size = step
#                         break
#                     else:
#                         psi = evolve(psi, step, s)
#                         s = s + step_size
#                         step_size = step
#                         stepped = True
#                         break
#         else:  # step is too big
#             for rate in range(2, 11):  # try dividing by up to 10
#                 step = 1 / rate * step_size
#                 if step_is_good(step):
#                     psi = evolve(psi, step, s)
#                     s = s + step_size
#                     step_size = step
#                     stepped = True
#                     break
#                 else:
#                     if rate == 10:
#                         step_size = 1 / rate * step_size
#
#
#
#                         # try bigger steps
#                         # find biggest good
#                         # if biggest good is the biggest one ->
#                         #   set step size = biggest good
#                         #   goto line 100 ( try increasing size again)
#                         # else :
#                         #   do biggest good
#                         #   save step size
#                         #   save step time
#         else:
#             pass
#         # try smaller steps
#         # find biggest good
#         # if even smallest is bad:
#         #   set step size = smallest
#         #   goto line 100

#
# def sim_dynamic_evolution_full_dynamic(H0, H1, target_p0=0.4):
#     """
#     Simulates an evolution that dynamically changes it's rate as a response to gap closure
#     :param H0: First hamiltonian
#     :param H1: Second hamiltonian
#     :return: tlist
#     """
#     time_eps = 0.15  # assume the 15% in the end of the process can be jumped over
#     eigvals, eigvecs = H0.eigenstates(eigvals=2)
#     tlist = []
#     end_time = 0
#     steps = 10  # sets a initial estimation for the step size
#     s = 0
#     while s <= (1 - time_eps):
#         good_jump_found = False
#         end_time = 1 - time_eps / 2
#         linear_tlist = np.linspace(s, end_time, steps)
#         while not good_jump_found:
#             for t_try in linear_tlist[::-1]:
#                 # evaluate goodness of jump
#                 good_jump, psi_after = test_jump(psi, s, t_try)
#                 if good_jump:
#                     tlist.append(t_try)
#                     psi = psi_after
#                     good_jump_found = True
#                     s = t_try
#                     break
#             if not good_jump_found:
#                 # If even the smallest step is too big - divide it to similar sized steps
#                 linear_tlist = np.linspace(s, linear_tlist[0], steps)
#
#     psi = evolve(psi, H_0, H_1, t)
#     return tlist, psi
