from qutip import *
import numpy as np
from scipy.linalg import expm
from tqdm import tnrange, tqdm_notebook
import LH_tools as LHT
import scipy.sparse
import multiprocessing
import concurrent.futures

import ctypes
mkl_rt = ctypes.CDLL('libmkl_rt.so')
mkl_get_max_threads = mkl_rt.mkl_get_max_threads
mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(48)))
import os
import time

def sim_simple_adiabatic(tlist, H0, H1, s='linear'):
    """

    :param tlist: Time list
    :param H0: first hamiltonian
    :param H1: second hamiltonian
    :param s: function - relates time to coupling, default is linear dependecy
    :return: P_mat, eigvals_mat, psis
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


def sim_dynamic_evolution_tight_deriviative(H0, H1, epsilon=0.3):
    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    slist = []
    max_time = 39
    step_size = 1
    t = 0
    psi = eigvecs[0]
    psis = [psi]
    s = 0
    for t in range(max_time):
        slist.append(s)
        eigvals, eigvecs = H0.eigenstates(eigvals=2)
        gap = eigvals[1] - eigvals[0]
        if gap < 0:
            raise ValueError("EV0 should be lower than EV1")
        overlap_01 = eigvecs[0].overlap(eigvecs[1])
        ds = epsilon * gap ** 2 / np.abs(overlap_01)
        s = s + ds
        # evolve psi
        Ht = H0 * (1 - s) + H1 * s
        U = expm(-1j * Ht.data * step_size)
        psi = Qobj(U * psi.data, dims=psi.dims)
        psis.append(psi)
    slist.append(s)
    return slist, psis


    # while t < max_time:
    #     s = t / max_time
    #     Ht = H0 * (1 - s) + H1 * s
    #     eigvals, eigvecs = H0.eigenstates(eigvals=2)
    #     gap = eigvals[1] - eigvals[0]
    #     if gap < 0:
    #         raise ValueError("EV0 should be lower than EV1")
    #     overlap_01 = eigvecs[0].overlap(eigvecs[1])
    #     dt = epsilon * gap ** 2 / np.abs(overlap_01)
    #     psi = evolve(H0, H1, psi, ds, s, max_time)
    #     psis.append(psi)
    #     t = t + dt
    #     tlist.append(s)
    # return tlist, psis


def find_ds_and_evolve_psi(psi, H0, H1, target_p0, dt, s, process_precentage, linearization_factor=1):
    # Binary search for maximal ds
    max_search_depth = 10
    search_depth = 1
    first_legal_s = s
    end_of_search_space = 1
    while search_depth <= max_search_depth:
        search_space = end_of_search_space - first_legal_s
        ds = search_space / 2
        if s_step_is_good(psi, H0, H1, first_legal_s, ds, dt, target_p0, process_precentage):
            search_depth += 1
            first_legal_s += ds
        else:
            search_depth += 1
            end_of_search_space = first_legal_s + ds
    ds = first_legal_s - s
    # TODO: check if  s_evolve_linear_approximation behaves exactly a s_evolve when  linearization_factor=1
    if linearization_factor > 1:
        return s_evolve_linear_approximation(psi, H0, H1, s, ds, dt, linearization_factor)
    return s_evolve(psi, H0, H1, s, ds, dt)


def s_step_is_good(psi, H0, H1, s, ds, dt, target_p0, process_precentage):
    _, pr_psi, _, _, _, _ = s_evolve(psi, H0, H1, s, ds, dt)
    if (1 - pr_psi[0]) / process_precentage <= target_p0:
        return True
    else:
        return False


def s_evolve(psi, H0, H1, s, ds, dt):
    s_new = s + ds
    Ht = H0 * (1 - s_new) + H1 * s_new
    eigvals, eigvecs = Ht.eigenstates(eigvals=2)
    # Evolve
    U = expm(-1j * Ht.data * dt)
    psi = Qobj(U * psi.data, dims=psi.dims)
    # Compute projection
    pr_psi = [np.abs(eigvecs[0].overlap(psi)) ** 2,
              np.abs(eigvecs[1].overlap(psi)) ** 2]
    return psi, pr_psi, Ht, ds, eigvals, eigvecs


def s_evolve2(psi, H0, H1, s, ds, dt, eigvals_to_show=2) -> (Qobj, float, Qobj, [Qobj], [Qobj]):
    """

    :param psi:
    :param H0:
    :param H1:
    :param s:
    :param ds:
    :param dt:
    :return: psi, pr_psi, Ht, eigvals, eigvecs
    """
    s_new = s + ds
    Ht = H0 * (1 - s_new) + H1 * s_new
    eigvals, eigvecs = Ht.eigenstates(eigvals=eigvals_to_show)
    # Evolve
    U = expm(-1j * Ht.data * dt)
    psi = Qobj(U * psi.data, dims=psi.dims)
    # Compute projection
    pr_psi = np.abs(eigvecs[0].overlap(psi)) ** 2
    return psi, pr_psi, Ht, eigvals, eigvecs


def s_evolve_linear_approximation(psi, H0, H1, s_init, ds, dt, linearization_factor):
    s_new = s_init + ds

    list_of_s = np.linspace(s_init, s_new, linearization_factor)
    for s in list_of_s:
        Ht = H0 * (1 - s) + H1 * s
        # Evolve
        U = expm(-1j * Ht.data * dt / linearization_factor)
        psi = Qobj(U * psi.data, dims=psi.dims)

    # Compute projection on the last hamiltonian
    eigvals, eigvecs = Ht.eigenstates(eigvals=2)
    pr_psi = [np.abs(eigvecs[0].overlap(psi)) ** 2,
              np.abs(eigvecs[1].overlap(psi)) ** 2]
    return psi, pr_psi, Ht, ds, eigvals, eigvecs


def sim_dynamic_evolution_binsearch2(H0, H1, target_p0, max_time, dt, linearization_factor=1):
    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    psi = eigvecs[0]
    psis = [psi]
    slist = [0]
    pr_list = [(1, 0)]
    ev_list = [eigvals]
    s = 0
    # Define per step error bound , this might also work with a global error bound, however this is simpler
    max_error_per_step = 1 / 2 * target_p0 / (max_time / dt)
    total_number_of_steps = len(range(0, max_time, dt))
    for step_count, t in enumerate(tnrange(0, max_time, dt)):
        # find appropriate ds and evolve with ds
        if s >= 1 - 1 / (total_number_of_steps * 100):
            slist.append(1)
            print("Saturated at t = %s , stopping evolution" % t)
            psi, pr_psi, Ht, ds, eigvals, evecs = s_evolve(psi, H0, H1, s, 1 - s, dt)
        else:
            psi, pr_psi, Ht, ds, eigvals, evecs = \
                find_ds_and_evolve_psi(psi,
                                       H0, H1,
                                       target_p0,
                                       dt,
                                       s,
                                       (step_count + 1) / total_number_of_steps,
                                       linearization_factor)
            s = s + ds
            psis.append(psi)
            slist.append(s)
            pr_list.append(pr_psi)
            ev_list.append(eigvals)
    return psis, slist, pr_list, ev_list


def sim_dynamic_evolution_binsearch3(H0, H1, target_p0, dt, linearization_factor=1, max_depth=10, eigvals_to_show=2) \
        -> ([Qobj], [float], [float], [float]):
    """

    :param H0:
    :param H1:
    :param target_p0:
    :param dt:
    :param linearization_factor:
    :param max_depth:
    :param eigvals_to_show:
    :return:
    """
    eigvals, eigvecs = H0.eigenstates(eigvals=eigvals_to_show)
    psi = eigvecs[0]
    psis = [psi]
    slist = [0]
    pr_list = [1]
    ev_list = [eigvals]
    s = 0
    time_rounding = 0.01
    first_step = True
    with tqdm_notebook(total=100, desc="progress of s") as progress_bar:
        while (s < 1 - time_rounding):
            search_range = 1 - s
            s0 = s
            for i in range(max_depth):
                d2s = search_range / (2 ** (i + 1))
                phi, pr_phi, _, _, _ = s_evolve2(psi, H0, H1, s0, s - s0 + d2s, dt)
                if (((1 - pr_phi) / (s + d2s)) * (1 + first_step)) < target_p0:
                    s += d2s
                else:
                    continue
            first_step = False
            if s == s0:
                s += d2s
            progress_bar.update(int(100 * (s - s0)))
            psi, pr_psi, _, eigvals, _ = s_evolve2(psi, H0, H1, s0, s - s0 + d2s, dt, eigvals_to_show)
            psis.append(psi)
            slist.append(s)
            pr_list.append(pr_psi)
            ev_list.append(eigvals)
        ds = 1 - s
        psi, pr_psi, _, eigvals, _ = s_evolve2(psi, H0, H1, s, ds, dt, eigvals_to_show)
        psis.append(psi)
        slist.append(1)
        pr_list.append(pr_psi)
        ev_list.append(eigvals)

    return psis, slist, pr_list, ev_list


def sim_dynamic_evolution_baf3(H0, H1, target_p0, dt, linearization_factor=1, max_depth=10, eigvals_to_show=2) \
        -> ([Qobj], [float], [float], [float]):
    """

    :param H0:
    :param H1:
    :param target_p0:
    :param dt:
    :param linearization_factor:
    :param max_depth:
    :param eigvals_to_show:
    :return:
    """
    eigvals, eigvecs = H0.eigenstates(eigvals=eigvals_to_show)
    psi = eigvecs[0]
    psis = [psi]
    slist = [0]
    pr_list = [1]
    ev_list = [eigvals]
    s = 0
    time_rounding = 0.01
    forward_prop = qeye(H0.dims[0]).data
    back_prop = qeye(H0.dims[0]).data
    first_step = True

    with tqdm_notebook(total=100, desc="progress of s") as progress_bar:
        while (s < 1 - time_rounding):
            search_range = 1 - s
            s0 = s
            for i in range(max_depth):
                d2s = search_range / (2 ** (i + 1))
                phi, pr_phi, _, _, _, _, _ = \
                    s_evolve_baf3(psi, H0, H1, eigvecs, s0, s - s0 + d2s, dt, forward_prop, back_prop)

                if (((1 - pr_phi) / (s + d2s)) * (1 + first_step)) < target_p0:
                    s += d2s
                else:
                    continue
            first_step = False

            if s == s0:
                s += d2s

            progress_bar.update(int(100 * (s - s0)))
            psi, pr_psi, _, eigvals, _, forward_prop, back_prop = \
                s_evolve_baf3(psi, H0, H1, eigvecs, s0, s - s0 + d2s, dt, forward_prop, back_prop, eigvals_to_show)
            psis.append(psi)
            slist.append(s)
            pr_list.append(pr_psi)
            ev_list.append(eigvals)
        ds = 1 - s
        psi, pr_psi, _, eigvals, _ = s_evolve2(psi, H0, H1, s, ds, dt, eigvals_to_show)
        psis.append(psi)
        slist.append(1)
        pr_list.append(pr_psi)
        ev_list.append(eigvals)
        print(s)

    return psis, slist, pr_list, ev_list


def sim_dynamic_evolution_binsearch(H0, H1, target_p0=0.1):
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



def sim_degenerateGS_adiabatic(tlist, H0, H1, s='linear'):
    """
    :param tlist: Time list
    :param H0: first hamiltonian
    :param H1: second hamiltonian
    :param s: function - relates time to coupling, default is linear dependecy
    :return:
    """
    print("Assuming degenrate GS, running this function on an Hamiltonian without degeneracy will result in wrong"
          "probabilities.")
    if s == 'linear':
        s = lambda t: (t - tmin) / (tmax - tmin)
    duration = len(tlist)
    tmin = min(tlist)
    tmax = max(tlist)
    # start at H0 ground state
    eigvals, eigvecs = H0.eigenstates(eigvals=10)
    print(eigvals)
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
        # Assume 2-degenerate-GS , we should maximize overlap on the first two eigeinstates.
        eigvals, eigvecs = Ht.eigenstates(eigvals=10)
        print(eigvals)

        U = expm(-1j * Ht.data * dt)
        psi = Qobj(U * psi.data, dims=psi.dims)
        psis.append(psi)
        eigvals_mat.append(eigvals)
        # Take max on the degenrate GS
        P_mat.append(
            [max(abs(psi.overlap(eigvecs[0])) ** 2,
                 abs(psi.overlap(eigvecs[1])) ** 2),
             abs(eigvecs[2].overlap(psi)) ** 2])
        oldt = t
    return P_mat, eigvals_mat, psis


def sim_dynamic_evolution_binsearch_back_and_forth(H0, H1, target_p0, max_time, dt, linearization_factor=1):
    import sys

    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    psi = eigvecs[0]
    psi0 = eigvecs[0]
    psis = [psi]
    slist = [0]
    pr_list = [(1, 0)]
    ev_list = [eigvals]
    s = 0
    forward_prop = qeye(H0.dims[0]).data
    back_prop = qeye(H0.dims[0]).data

    # Define per step error bound , this might also work with a global error bound, however this is simpler
    max_error_per_step = 1 / 2 * target_p0 / (max_time / dt)
    tlist = np.linspace(0, max_time, max_time / dt)
    total_number_of_steps = len(tlist)
    for step_count, t in enumerate(tlist):
        # find appropriate ds and evolve with ds
        if s >= 1 - 1 / (total_number_of_steps * 100):
            slist.append(1)
            print("Saturated at t = %s , stopping evolution" % t)
            psi, pr_psi, Ht, ds, eigvals, evecs = s_evolve_linear_approximation(psi, H0, H1, s, 1 - s, dt,
                                                                                linearization_factor)
        else:
            psi, pr_psi, Ht, ds, eigvals, evecs, forward_prop, back_prop = \
                find_ds_and_evolve_back_and_forth(forward_prop, back_prop,
                                                  psi0, psi,
                                                  H0, H1, eigvecs,
                                                  target_p0,
                                                  dt,
                                                  s,
                                                  (step_count + 1) / total_number_of_steps,
                                                  linearization_factor)
            s = s + ds
            psis.append(psi)
            slist.append(s)
            pr_list.append(pr_psi)
            ev_list.append(eigvals)
    return psis, slist, pr_list, ev_list


def find_ds_and_evolve_back_and_forth(forward_prop, back_prop, psi0, psi, H0, H1, H0_eigvecs, target_p0, dt, s,
                                      process_precentage,
                                      linearization_factor=1):
    """

    :param forward_prop:
    :param back_prop:
    :param psi0:
    :param psi:
    :param H0:
    :param H1:
    :param H0_eigvecs:
    :param target_p0:
    :param dt:
    :param s:
    :param process_precentage:
    :param linearization_factor:
    :return:
    """
    # Binary search for maximal ds
    max_search_depth = 10
    search_depth = 1
    first_legal_s = s
    end_of_search_space = 1
    while search_depth <= max_search_depth:
        search_space = end_of_search_space - first_legal_s
        ds = search_space / 2

        if s_step_is_good_baf(forward_prop, back_prop, psi0, psi, H0, H1, H0_eigvecs, first_legal_s, ds, dt, target_p0,
                              process_precentage):
            search_depth += 1
            first_legal_s += ds
        else:
            search_depth += 1
            end_of_search_space = first_legal_s + ds
    ds = first_legal_s - s
    # TODO: check if  s_evolve_linear_approximation behaves exactly a s_evolve when  linearization_factor=1
    # if linearization_factor > 1:
    #     return s_evolve_linear_approximation(psi, H0, H1, s, ds, dt, linearization_factor)

    return s_evolve_baf(psi, H0, H1, H0_eigvecs, s, ds, dt, forward_prop, back_prop)


def s_step_is_good_baf(forward_prop, back_prop, psi0, psi, H0, H1, H0_eigvecs, s, ds, dt, target_p0,
                       process_precentage):
    _, pr_psi, _, _, _, _, forward_prop, back_prop = s_evolve_baf(psi, H0, H1, H0_eigvecs, s, ds, dt, forward_prop,
                                                                  back_prop)

    if (1 - pr_psi[0]) / process_precentage <= target_p0:
        return True
    else:
        return False


def s_evolve_baf(psi, H0, H1, H0_eigvecs, s, ds, dt, forward_prop, back_prop):
    """

    :param psi:
    :param H0:
    :param H1:
    :param H0_eigvecs:
    :param s:
    :param ds:
    :param dt:
    :param forward_prop:
    :param back_prop:
    :return:
    """
    s_new = s + ds
    Ht = H0 * (1 - s_new) + H1 * s_new
    eigvals, eigvecs = Ht.eigenstates(eigvals=2)
    # Evolve
    U = expm(-1j * Ht.data * dt)
    forward_prop = U * forward_prop
    # TODO: is this the correct back-propegator?
    # TODO: should I use np.matrix(U).getH() ?
    back_prop = back_prop * U
    psi = Qobj(U * psi.data, dims=psi.dims)
    # Compute projection after back-and-forth process
    reversed_psi = Qobj(back_prop * psi.data, dims=psi.dims)
    pr_psi = [np.abs(H0_eigvecs[0].overlap(reversed_psi)) ** 2,
              np.abs(H0_eigvecs[1].overlap(reversed_psi)) ** 2]
    import pydevd;
    # pydevd.settrace('localhost', port=4000, stdoutToServer=True, stderrToServer=True)
    return psi, pr_psi, Ht, ds, eigvals, eigvecs, forward_prop, back_prop


def s_evolve_baf3(psi, H0, H1, H0_eigvecs, s, ds, dt, forward_prop, back_prop, eigvals_to_show=2):
    """

    :param psi:
    :param H0:
    :param H1:
    :param H0_eigvecs:
    :param s:
    :param ds:
    :param dt:
    :param forward_prop:
    :param back_prop:
    :return:
    """
    s_new = s + ds
    Ht = H0 * (1 - s_new) + H1 * s_new
    eigvals, eigvecs = Ht.eigenstates(eigvals=eigvals_to_show)
    # Evolve
    U = expm(-1j * Ht.data * dt)
    forward_prop = U * forward_prop
    back_prop = back_prop * U
    psi = Qobj(U * psi.data, dims=psi.dims)
    # Compute projection after back-and-forth process
    reversed_psi = Qobj(back_prop * psi.data, dims=psi.dims)
    pr_psi = np.abs(H0_eigvecs[0].overlap(reversed_psi)) ** 2
    import pydevd;
    return psi, pr_psi, Ht, eigvals, eigvecs, forward_prop, back_prop


"""Gaussian noise

Ill work with the gaussian 2.5*e^(-(x-5)^2/2) /(sqrt(2*pi) ), it is centered around 5 with std of 1 and achieves ~0.997
at 5, thus making it easy to work with on 10 steps process
"""


def sim_dynamic_evolution_gauss_noise(H0, H1, target_p0, max_time, dt):
    eigvals, eigvecs = H0.eigenstates(eigvals=2)
    psi = eigvecs[0]
    psis = [psi]
    pr_list = [(1, 0)]
    ev_list = [eigvals]
    total_number_of_steps = len(range(0, max_time, dt))
    Hnoise = rand_herm(H0.shape[0], dims=H0.dims)

    # # TODO: remove debug
    # #
    # import pydevd
    # from  importlib import reload
    # reload(pydevd)
    # pydevd.settrace('localhost', port=4000, stdoutToServer=True, stderrToServer=True)

    for step_count, t in enumerate(tnrange(0, max_time, dt)):
        psi, pr_psi, eigvals, evecs = \
            find_rate_and_evolve_psi_with_noise(psi,
                                                H0, H1,
                                                target_p0,
                                                t, dt, max_time,
                                                (step_count + 1) / total_number_of_steps,
                                                Hnoise)
        psis.append(psi)
        pr_list.append(pr_psi)
        ev_list.append(eigvals)

    return psis, pr_list, ev_list


def find_rate_and_evolve_psi_with_noise(psi, H0, H1, target_p0, t, dt, tmax, process_precentage, Hnoise):
    """
     Binary search for maximal ds, we ignore the found ds and actually evolve with dt
     the ds factor is only used to compute the rate ds/dt, which will be used as an indicator to the wanted noise volume
    :param psi:
    :param H0:
    :param H1:
    :param target_p0:
    :param t:
    :param dt:
    :param tmax:
    :param process_precentage:
    :param Hnoise:
    :return:
    """
    s = t / tmax
    max_search_depth = 10
    search_depth = 1
    first_legal_s = s
    end_of_search_space = 1
    while search_depth <= max_search_depth:
        search_space = end_of_search_space - first_legal_s
        ds = search_space / 2
        if s_step_is_good(psi, H0, H1, first_legal_s, ds, dt, target_p0, process_precentage):
            search_depth += 1
            first_legal_s += ds
        else:
            search_depth += 1
            end_of_search_space = first_legal_s + ds

    # TODO: add in all the implementations - we need to solve the case where ds was not found, and set minimal ds to be
    # 1/2^max_search_depth

    ds = max(first_legal_s - s, 1 / 2 ** max_search_depth)
    noise_power_constant = 1 / 250

    rate = min(dt / tmax / ds * noise_power_constant, 1000)
    # print("ds:%s rate:%s" % (ds, rate))
    return evolve_gaussian_noise(psi, H0, H1, t, dt, tmax, Hnoise, noise_volume=rate)


def evolve_gaussian_noise(psi, H0, H1, t_init, dt, tmax, Hnoise, noise_volume):
    t_new = t_init + dt

    gauss = lambda x: 2.5 * np.exp(-(x - 5) ** 2 / 2) / (np.sqrt(2 * np.pi))
    steps = 10
    list_of_s = np.linspace(t_init, t_new, steps) / tmax

    for x, s in enumerate(list_of_s):
        Ht = H0 * (1 - s) + H1 * s + gauss(x) * Hnoise * noise_volume
        # Evolve
        U = expm(-1j * Ht.data * dt / steps)
        psi = Qobj(U * psi.data, dims=psi.dims)
    # Compute projection
    eigvals, eigvecs = Ht.eigenstates(eigvals=2)

    pr_psi = [np.abs(eigvecs[0].overlap(psi)) ** 2,
              np.abs(eigvecs[1].overlap(psi)) ** 2]
    return psi, pr_psi, eigvals, eigvecs


"""
Degeneracy evolution
"""
PRECISION = 2 ** -40


def sim_degenerate_adiabatic(tlist, H0: qobj, H1: qobj, psi0: qobj, max_degen=False):
    """
    Simulates evolution under hamiltonians with degenerate GS
    :param tlist: Time list
    :param H0: first hamiltonian
    :param H1: second hamiltonian
    :param s: function - relates time to coupling, default is linear dependecy
    :return:
    """

    
    tmin = min(tlist)
    tmax = max(tlist)
    s = lambda t: (t - tmin) / (tmax - tmin)
    H0_energies, H0_ev = H0.eigenstates(eigvals=max_degen)
    H0_degeneracy = sum(abs(H0_energies - H0_energies.min()) < PRECISION)
    groundspace = H0_ev[0:H0_degeneracy]
    P_mat = []
    eigvals_mat = []
    psi = psi0
    psis = [psi]
    eigvals_mat.append(H0_energies)
    # gs_projection = sum([abs(groundstate.overlap(psi)) ** 2 for groundstate in groundspace])
    gs_projection = LHT.get_total_projection_size(groundspace, psi)[0]
    P_mat.append(
        [gs_projection])
    oldt = tmin
    
    
    # # TODO: remove debug
    # #
    # import pydevd
    # from  importlib import reload
    # reload(pydevd)
    # pydevd.settrace('localhost', port=4000, stdoutToServer=True, stderrToServer=True)

    for t in tlist[1:]:
        dt = t - oldt
        Ht = H0 * (1 - s(t)) + H1 * (s(t))
        Ht_energies, HT_ev = Ht.eigenstates(eigvals=max_degen)
        Ht_degeneracy = sum(abs(Ht_energies - Ht_energies.min()) < PRECISION)
        groundspace = HT_ev[0:Ht_degeneracy]
        start_time = time.time()
        #U = expm(-1j * Ht.data * dt) use qutip expm for speedup
        #psi = Qobj(U * psi.data, dims=psi.dims)
        U = (Ht*-1j*dt).expm()
        print ("Finished expm in %f" % (time.time() - start_time )); start_time = time.time()
        psi = U * psi
        psis.append(psi)
        eigvals_mat.append(Ht_energies)
        # gs_projection = sum([abs(groundstate.overlap(psi)) ** 2 for groundstate in groundspace])
        # if len(groundspace) > 1:
        #     print("old method ", gs_projection, " new method", LHT.get_total_projection_size(groundspace, psi))
        gs_projection = LHT.get_total_projection_size(groundspace, psi)
        P_mat.append(
            [gs_projection])
        oldt = t
    return P_mat, eigvals_mat, psis


#
# def sim_degenerate_adiabatic2(tlist, H0: qobj, H1: qobj, psi0: qobj):
#     """
#     Simulates evolution under hamiltonians with degenerate GS
#     :param tlist: Time list
#     :param H0: first hamiltonian
#     :param H1: second hamiltonian
#     :param s: function - relates time to coupling, default is linear dependecy
#     :return:
#     """
#     eigsh = scipy.sparse.linalg.eigsh
#     tmin = min(tlist)
#     tmax = max(tlist)
#     s = lambda t: (t - tmin) / (tmax - tmin)
#
#     # TODO: remove debug
#     #
#     import pydevd
#     from  importlib import reload
#     reload(pydevd)
#     pydevd.settrace('localhost', port=4000, stdoutToServer=True, stderrToServer=True)
#
#     evals, evecs = eigsh(H0.data)
#     idx = evals.argsort()
#     evals = evals[idx]
#     evecs = evecs[:, idx]
#     H0_degeneracy =sum(abs(evals - evals.min()) < PRECISION)
#     groundspace = [Qobj(evecs[:,i]) for i in range(H0_degeneracy)]
#     P_mat = []
#     eigvals_mat = []
#     psi = psi0
#     psis = [psi]
#     eigvals_mat.append(evals)
#     gs_projection = sum([abs(groundstate.overlap(psi)) ** 2 for groundstate in groundspace])
#     P_mat.append(
#         [gs_projection])
#     oldt = tmin
#
#
#
#     for t in tlist[1:]:
#         dt = t - oldt
#         Ht = H0 * (1 - s(t)) + H1 * (s(t))
#         ## TODO: using scipy.linalg.eigsh gives great performance boost
#
#         Ht_energies, evecs = eigsh(H0.data)
#         Ht_degeneracy = sum(abs(Ht_energies - Ht_energies.min()) < PRECISION)
#         _, groundspace = [Qobj(evecs[:i]) for i in range(H0_degeneracy)]
#
#
#         U = expm(-1j * Ht.data * dt)
#         psi = Qobj(U * psi.data, dims=psi.dims)
#         psis.append(psi)
#         eigvals_mat.append(Ht_energies)
#         #gs_projection = sum([abs(groundstate.overlap(psi)) ** 2 for groundstate in groundspace])
#         # if len(groundspace) > 1:
#         #     print("old method ", gs_projection, " new method", LHT.get_total_projection_size(groundspace, psi))
#         gs_projection = LHT.get_total_projection_size(groundspace, psi)
#         P_mat.append(
#             [gs_projection])
#         oldt = t
#     return P_mat, eigvals_mat, psis
#
#


def find_min_gap(H0: Qobj, H1: Qobj, low=0, high=1, epsilon=2 ** (-20), initial_resolution=50,max_threads = None) -> float:
    """
    Finds the minimal gap on the convex H0*(1-s)+H1*s between the energies indexed in low and high
    default - find the gap between the lowest and second lowest energies
    this might fail to local minima
    :param H0: First Hamiltoniabn
    :param H1: Second Hamiltoniabn
    :param low: index of low energy
    :param high:index of higi energy
    :param epsilon: convergence
    :return: the lowest energy found
    """
    # random sample over the convex initial_resolution points
    slist = [np.random.uniform() for i in range(initial_resolution)]
    local_searches = []
    executor = concurrent.futures.ThreadPoolExecutor(max_threads)
    for s in slist:
        local_searches.append(
            executor.submit(__find_local_gap_minimum, s, H0, H1, low, high, epsilon, initial_resolution))
    results = [search.result() for search in concurrent.futures.as_completed(local_searches)]
    return min(results)


def __find_local_gap_minimum(s: float, H0: Qobj, H1: Qobj, low=0, high=1, epsilon=2 ** (-20), initial_resolution=50):
    """
    Finds the local minimal gap in the vicinity of s
    :param s: Point on the convex
    :param H0: First Hamiltoniabn
    :param H1: Second Hamiltoniabn
    :param low: index of low energy
    :param high:index of high energy
    :param epsilon: accuracy of search
    :param initial_resolution:
    :return: local_minima, s
    """
    ds = 1 / (initial_resolution * 2)
    gap_s = __get_gap(s, H0, H1, low, high)
    gap_s_ds = __get_gap(s + ds, H0, H1, low, high)
    delta_gap = gap_s - gap_s_ds
    while abs(delta_gap) > epsilon:
        if gap_s < gap_s_ds:
            # change direction, walk slower
            ds = -ds / 2
        s = s + ds
        # handle overflows
        if s > 1 or s < 0 :
            s = round(s)
            gap_s = __get_gap(s, H0, H1, low, high)
            gap_s_ds = gap_s
            ds = 0
        gap_s = __get_gap(s, H0, H1, low, high)
        gap_s_ds = __get_gap(s + ds, H0, H1, low, high)
        delta_gap = gap_s - gap_s_ds
    return min((gap_s, s), (gap_s_ds, s + ds))


def __get_gap(s, H0, H1, low, high) -> float:
    """
    Find the gap in the convex at parameter s
    :param s: Point on the convex
    :param H0: First Hamiltoniabn
    :param H1: Second Hamiltoniabn
    :param low: index of low energy
    :param high:index of high energy
    :return: gap
    """
    Hs = H0 * (1 - s) + H1 * s
    Hs_en = Hs.eigenenergies(eigvals=high + 1)
    return Hs_en[high] - Hs_en[low]
