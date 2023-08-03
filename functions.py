# ----------------- Helper Functions --------------------------------#

import warnings
import copy
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from my_constants import *
import os
import shutil
import numba
from numba import jit
# import vtk
import numpy as np
# import vtk.util.numpy_support as numpy_support
# from scipy.ndimage.filters import laplace
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from plots import *
# save global constant for dimensionless conversions
# global data
data = np.genfromtxt('reference.csv', delimiter=",")

p_dim = data[0]
u_dim = data[1]
v_dim = data[2]
Ut_dim = data[3]
rho_dim = data[4]
e_dim = data[5]
T_dim = data[6]
Ts_dim = data[7]
de_dim = data[8]
m_out_dim = data[9]
dx_dim = data[10]
dr_dim = data[11]
dt_dim = data[12]
Re_inf = data[13]
M_inf = data[14]
Pr_inf = data[15]


np.set_printoptions(threshold=sys.maxsize)

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# u_in_x = np.sqrt(7./5.*R*T_in/M_n)*1.0  # Inlet velocity, m/s (gamma*RT)


def no_slip_no_mdot(p, rho, tg, u, v, Ut, e):
    # no mass deposition
    u[:, Nr] = 0.
    v[:, Nr] = 0.
    Ut[:, Nr] = 0.
    # energy assumed constant
    tg = (e - 1./2.*Ut**2) * 2./5./R*M_n
    p = rho*R/M_n*tg
    return p, tg, u, v, Ut, e

# simple time integration


def simple_time(p, q, tg, u, v, Ut, e, p_in, rho_in, T_in, e_in, u_in, v_in, rho_0, dt):
    N = n_matrix()

    qq = copy.deepcopy(q)  # density
    uxx = copy.deepcopy(q)  # velocity axial
    urr = copy.deepcopy(q)  # velocity radial
    uu = copy.deepcopy(q)  # total velocity
    ee = copy.deepcopy(q)  # energy
    tt = copy.deepcopy(q)  # temperature
    # if np.any(tg < 0):
    #     print("Temp before no slip simple_time has at least one negative value")
    #     exit()

    # p, tg, u, v, Ut, e = no_slip_no_mdot(p, q, tg, u, v, Ut, e)
    # negative temp check
    # if np.any(tg < 0):
    #     print("Temp no slip has at least one negative value")
    #     exit()

    # plot_imshow(p, u, tg, q, e)
    u, v, Ut, p, q, tg, e = inlet_BC(
        u, v, Ut, p, q, tg, e, p_in, u_in, rho_in, T_in, e_in)
    # negative temp check
    # if np.any(tg < 0):
    #     print("Temp inlet_BC has at least one negative value")
    #     exit()
    # plot_imshow(p, u, tg, q, e)
    p, q, tg, u, Ut, e = outlet_BC(p, e, q, u, v, Ut, rho_0)
    # u, Ut, e = parabolic_velocity(q, tg, u, u_in, Ut, e)

    # negative temp check
    # if np.any(tg < 0):
    #     print("Temp outlet_BC has at least one negative value")
    #     exit()
    # print(tg)
    # print("plotting")
    # plot_imshow(p, Ut, tg, q, e)


# # NOTE: Convert dimensionless
#     p, q, tg, e, u, v = get_dimless(p, q, tg, e, u, v)
#     Ut = np.sqrt(u**2. + v**2.)
# # Time dimensionless
#     dt = dt / (D_hyd / Ut_dim)

    rho, rhoU, rhoV, rhoE = flux_construct(q, u, v, e)


# Calculating gradients (first and second)

# Calculate RHS all equations
# these are all properties
    r = RHS_rho(u, v, q, N)
    r_u = RHS_rhoU(u, v, p, q, tg, N)
    r_v = RHS_rhoV(u, v, p, q, tg, N)
    r_e = RHS_rhoE(u, v, p, q, tg, e, N)

# first LHS calculations
    rho2 = rho + dt*r
    rhoU2 = rhoU + dt*r_u
    rhoV2 = rhoV + dt*r_v

# apply no slip later

    rhoE2 = rhoE + dt*r_e

#     p, q, tg, e, u, v = recover_physical(p, q, tg, e, u, v)
#     Ut = np.sqrt(u**2. + v**2.)
# # time physical
#     dt = dt * (D_hyd / Ut_dim)

# deconstruction
    qq, uxx, urr, ee = flux_deconstruct(
        rho2, rhoU2, rhoV2, rhoE2)


# ensure no division by zero
    qq = no_division_zero(qq)

# Velocity
    uu = np.sqrt(uxx**2. + urr**2.)
# pressure defining
    tt = (ee - 1./2.*uu**2.) * 2./5. / R*M_n
    pp = qq*R/M_n*T_in

# no slip condition - pressure and temp recalculated within
    # pp, tt, uxx, urr, uu, ee = no_slip_no_mdot(pp, qq, tt, uxx, urr, uu, ee)
    # print("plotting no slip")
    # plot_imshow(pp, uu, tt, qq, ee)
    # uxx, uu, ee = parabolic_velocity(qq, tg, uxx, u_in, Ut, ee)
    uxx, urr, uu, pp, qq, tt, ee = inlet_BC(
        uxx, urr, uu, pp, qq, tt, ee, p_in, u_in, rho_in, T_in, e_in)
    # negative temp check
    # if np.any(tg < 0):
    #     print("Temp inlet_BC has at least one negative value")
    #     exit()
    # plot_imshow(p, u, tg, q, e)
    pp, qq, tt, uxx, uu, ee = outlet_BC(pp, ee, qq, uxx, urr, uu, rho_0)

    return pp, qq, tt, uxx, urr, uu, ee


def dt2nd_w_matrix(Tw, T_in):
    dt2nd_w_m = np.zeros((Nx+1), dtype=(np.float64))

    for m in np.arange(Nx+1):
        if m == 0:
            dt2nd_w_m[m] = (T_in - 2 * Tw[m] +
                            Tw[m+1])/(dx**2)  # 3-point CD
    #       dt2nd = Tw1[m+1]-Tw1[m]-Tw1[m-1]+T_in
        elif m == Nx:
            # print("m=Nx", m)
            dt2nd_w_m[m] = (-Tw[m-3] + 4*Tw[m-2] - 5*Tw[m-1] +
                            2*Tw[m]) / (dx**2)  # Four point BWD
        else:
            dt2nd_w_m[m] = Tw[m-1]-2*Tw[m]+Tw[m+1]/(dx**2)
    return dt2nd_w_m

# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8)')


def initialize_grid(p_0, rho_0, e_0, T_0, T_s):

    # rho12 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
    p1 = np.full((Nx+1, Nr+1), p_0, dtype=(np.float64, np.float64))  # Pressure
    rho1 = np.full((Nx+1, Nr+1), rho_0,
                   dtype=(np.float64, np.float64))  # Density
    ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  # velocity -x
    ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  # velocity -r
    u1 = np.sqrt(np.square(ux1) + np.square(ur1))  # total velocity
    # Internal energy
    e1 = np.full((Nx+1, Nr+1), e_0, dtype=(np.float64, np.float64))
    # CHECK TODO: calculate using equation velocity.
    # TODO: calculate using equation velocity.

    T1 = np.full((Nx+1, Nr+1), T_0,
                 dtype=(np.float64, np.float64))  # Temperature

    rho2 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))
    ux2 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ur2 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    u2 = np.sqrt(np.square(ux2) + np.square(ur2))  # total velocity
    e2 = np.full((Nx+1, Nr+1), e_0, dtype=(np.float64, np.float64))
    T2 = np.full((Nx+1, Nr+1), T_0, dtype=(np.float64, np.float64))
    p2 = np.full((Nx+1, Nr+1), p_0, dtype=(np.float64, np.float64))  # Pressure

    Tw1 = np.full((Nx+1), T_s, dtype=(np.float64))  # Wall temperature
    Tw2 = np.full((Nx+1), T_s, dtype=(np.float64))
    # Temperature of SN2 surface
    Ts1 = np.full((Nx+1), T_0, dtype=(np.float64))
    Ts2 = np.full((Nx+1), T_0, dtype=(np.float64))

    # Average temperature of SN2 layer
    Tc1 = np.full((Nx+1), T_s, dtype=(np.float64))
    Tc2 = np.full((Nx+1), T_s, dtype=(np.float64))
    de0 = np.zeros((Nx+1), dtype=(np.float64))  # Deposition mass, kg/m
    de1 = np.full((Nx+1), 0., dtype=(np.float64))  # Deposition rate
    de2 = np.full((Nx+1), 0., dtype=(np.float64))  # Deposition rate
    qhe = np.zeros_like(de0, dtype=np.float64)  # heat transfer
    q_dep = np.zeros_like(de0, dtype=np.float64)  # heat transfer

    # These matrices are just place holder. These will be overwritten and saved. (remove r=0)
    rho3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    ux3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    ur3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    u3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    e3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    T3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))
    p3 = np.full((Nx+1, Nr), T_s, dtype=(np.float64, np.float64))

    # Dimensionless number in grid:
    Pe1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                                        np.float64))  # Peclet number
    Pe2 = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                   np.float64))  # Peclet number
    out = [p1, rho1, ux1, ur1, u1, e1, T1, rho2, ux2, ur2, u2, e2, T2, p2, Tw1, Tw2,
           Ts1, Ts2, Tc1, Tc2, de0, de1, qhe, q_dep, rho3, ux3, ur3, u3, e3, T3, p3, Pe1, Pe2]
    return out


def tau_diff(u, v):
    # create gradients arrays.
    du_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dv_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    du_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dv_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for i in np.arange(Nx+1):
        for j in np.arange(1, Nr+1):
            if i == Nx:
                du_dx[i, j] = (u[i, j]-u[i-1, j])/dx_dim
                dv_dx[i, j] = (v[i, j]-v[i-1, j])/dx_dim
            # if i == 1:
            #     m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i, j]*ux[i-1, j])/dx

            # elif i == Nx:
            #     m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]*ux[i-1, j])/dx

            else:
                du_dx[i, j] = (u[i+1, j]-u[i, j])/dx_dim
                dv_dx[i, j] = (v[i+1, j]-v[i, j])/dx_dim

            if j == 1:
                # NOTE: SYMMETRY BC
                du_dr[i, j] = (u[i, j+2]-u[i, j])/(4*dr_dim)
                dv_dr[i, j] = (v[i, j+2]-v[i, j])/(4*dr_dim)

            else:
                du_dr[i, j] = (u[i, j]-u[i, j-1])/dr_dim
                dv_dr[i, j] = (v[i, j]-v[i, j-1])/dr_dim

    return du_dx, dv_dx, du_dr, dv_dr

# This takes properties


def tau(mu, u, v, N):
    # define velocity spatial derivatives
    du_dx, dv_dx, du_dr, dv_dr = tau_diff(u, v)
    # print(Re)
    tau_xx = 2*mu/3 * (2*du_dx - dv_dr - 1/(N*dr_dim)*(v))

    tau_rr = 2*mu/3 * (-du_dx + 2*dv_dr - 1/(N*dr_dim)*(v))

    tau_rx = mu * (du_dr + dv_dx)

    tau_xx = tau_xx / Re_inf
    tau_rr = tau_rr / Re_inf
    tau_rx = tau_rx / Re_inf
    return tau_xx, tau_rr, tau_rx

# This takes properties


def flux_rho(u, v, rho):
    # print("stopped here")
    # Matrices
    A = rho*u
    B = rho*v
    D = rho*v
    return A, B, D


# This takes properties
def flux_rhoU(u, v, p, rho, tg, N):
    mu = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

# Viscosity calc
# revert physical
    tg = tg * T_dim
    p = p * p_dim

    mu = mu_matrix(tg, p)
    assert np.isfinite(mu).all()
    if np.any(mu == 0):
        print("The viscous matrix has zeros")
        exit()

# convert dimensionless
    mu = mu / (rho_dim*Ut_dim*D_hyd)

# Shear stress calc
    tau_xx, tau_rr, tau_rx = tau(mu, u, v, N)
    A = rho*u*u + p - tau_xx
    B = rho*u*v - tau_rx
    D = rho*u*v - tau_rx
    return A, B, D


def flux_rhoV(u, v, p, rho, tg, N):
    mu = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

# viscosity calc

# Viscosity calc
# revert physical
    tg = tg * T_dim
    p = p * p_dim

    mu = mu_matrix(tg, p)
    assert np.isfinite(mu).all()
    if np.any(mu == 0):
        print("The viscous matrix has zeros")
        exit()

# convert dimensionless
    mu = mu / (rho_dim*Ut_dim*D_hyd)

# Shear stress calc

    tau_xx, tau_rr, tau_rx = tau(mu, u, v, N)

    A = rho*u*v - tau_rx
    B = rho*v*v + p - tau_rr
    D = rho*v*v - tau_rr
    return A, B, D


def flux_rhoE(u, v, p, rho, tg, e, N):
    M_local = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    M_local[:, :] = np.sqrt(u[:, :]**2. + v[:, :]**2,) / \
        np.sqrt(gamma_n*R/M_n*tg[:, :])

    # total enthalpy
    Ut = np.sqrt(u**2. + v**2.)
    E_tot = tg/(gamma_n*(gamma_n-1)*M_local**2) + 1./2.*Ut**2

    H_tot = E_tot + p/rho
    mu = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
# Matrices

# Viscosity calc
# revert physical
    tg = tg * T_dim
    p = p * p_dim

    mu = mu_matrix(tg, p)
    assert np.isfinite(mu).all()
    if np.any(mu == 0):
        print("The viscous matrix has zeros")
        exit()

# convert dimensionless
    mu = mu / (rho_dim*Ut_dim*D_hyd)

# Shear stress calc
    tau_xx, tau_rr, tau_rx = tau(mu, u, v, N)

# heat flux calc
    qr, qx = heatflux(tg, p)

    A = rho*u*H_tot + qx - u*tau_xx - v*tau_rx
    B = rho*v*H_tot + qr - u*tau_rx - v*tau_rr
    D = rho*v*H_tot + qr - u*tau_rx - v*tau_rr
    return A, B, D

# returns variables in the RhoE equation in the matrices


def heatflux(tg, p):
    dt_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dt_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    mu = mu_matrix(tg, p)

    for i in np.arange(Nx+1):
        for j in np.arange(Nr+1):
            if i == 0:
                dt_dx[i, j] = (tg[i+1, j]-tg[i, j])/dx_dim

            else:
                dt_dx[i, j] = (tg[i, j]-tg[i-1, j])/dx_dim

            if j == 1:
                # NOTE: SYMMETRY BC
                dt_dr[i, j] = (tg[i, j+2]-tg[i, j])/(4*dr_dim)
            else:
                dt_dr[i, j] = (tg[i, j]-tg[i, j-1])/dr_dim  # BWD

    qr = -mu / (Pr_inf * (gamma_n-1)*M_inf**2*Re_inf) * dt_dr
    qx = -mu / (Pr_inf * (gamma_n-1)*M_inf**2*Re_inf) * dt_dx
    return qr, qx


def diff_rhs(A, B, D, N):
    # print("stopped  here")
    dA_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dB_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    D_r = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    # print("stopped here")
    # NOTE: Can be written better?
    for i in np.arange(Nx+1):
        for j in np.arange(1, Nr+1):
            if i == Nx:
                dA_dx[i, j] = (A[i, j]-A[i-1, j])/dx_dim

            else:
                dA_dx[i, j] = (A[i+1, j]-A[i, j])/dx_dim  # upwind

            if j == 1:
                # NOTE: SYMMETRY BC
                dB_dr[i, j] = (B[i, j+2]-B[i, j])/(4*dr_dim)

            else:
                dB_dr[i, j] = (B[i, j]-B[i, j-1])/dr_dim  # BWD

    D_r = D/(N*dr_dim)
    # print("stopped  here")

    return dA_dx, dB_dr, D_r


def RHS_rho(u, v, q, N):
    A = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    B = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    D = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    A, B, D = flux_rho(u, v, q)
    convective_term = convective_term()

    dA_dx, dB_dr, D_r = diff_rhs(A, B, D, N)
    # print("stopped xxxxxxxxx  here")

    rhs_rho = - dA_dx - dB_dr - D_r
    return rhs_rho


def RHS_rhoU(u, v, p, rho, tg, N):
    # print(tg)
    A = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    B = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    D = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    U, A, B, D = flux_rhoU(u, v, p, rho, tg, N)
    dA_dx, dB_dr, D_r = diff_rhs(A, B, D, N)

    rhs_rhoU = - dA_dx - dB_dr - D_r
    return rhs_rhoU


def RHS_rhoV(u, v, p, rho, tg, N):
    A, B, D = flux_rhoV(u, v, p, rho, tg, N)
    dA_dx, dB_dr, D_r = diff_rhs(A, B, D, N)
    rhs_rhoV = - dA_dx - dB_dr - D_r
    return rhs_rhoV


def RHS_rhoE(u, v, p, rho, tg, e, N):
    A, B, D = flux_rhoE(u, v, p, rho, tg, e, N)
    dA_dx, dB_dr, D_r = diff_rhs(A, B, D, N)
    rhs_rhoE = - dA_dx - dB_dr - D_r
    return rhs_rhoE


def n_matrix():
    # Initialized once when starting main
    n = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for i in np.arange(np.int64(0), np.int64(Nx+1)):
        for j in np.arange(np.int64(1), np.int64(Nr+1)):
            n[i, j] = j
    n[:, 0] = 1
    print(n)
    return n

# returns viscosity matrix


def mu_matrix(tg, p):
    # print("T", tg, "P", p)
    visc_matrix = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            visc_matrix[m, n] = mu_n(tg[m, n], p[m, n])
    # save_visc(i, dt, visc_matrix)

# perform NAN value matrix checks:
    # print("performing finite check on visc_matrix")
    # # print(visc_matrix)
    for x in np.arange(len(visc_matrix)):
        assert np.isfinite(visc_matrix).all()

# negative viscosity check
    if np.any(visc_matrix < 0):
        print("The viscous matrix has at least one negative value")
        exit()

    if np.any(visc_matrix == 0):
        print("The viscous matrix has zero value")
        exit()
    return visc_matrix


# return S term matrix, returns next mdot. input previous de
def source_mass_depo_matrix(p, q, e, rho_0, tg, Ts1, u, v, de, N, visc, htransfer):
    # -4/D* mdot
    dm = np.zeros((Nx+1), dtype=(np.float64))
    dm_r = np.zeros((Nx+1), dtype=(np.float64))
    de3 = np.zeros((Nx+1), dtype=(np.float64))
    S = np.zeros((Nx+1), dtype=(np.float64))
    if htransfer == 1:
        # chosen at gridpoint Nr-1 because at Nr, ux =0
        for m in np.arange(Nx+1):
            if m == Nx:
                dm[m] = rho[m, Nr-1]*u[m, Nr-1] - rho[m-1, Nr-1]*u[m-1, Nr-1]
            else:
                dm[m] = rho[m+1, Nr-1]*u[m+1, Nr-1] - rho[m, Nr-1]*u[m, Nr-1]
        for m in np.arange(Nx+1):
            if m == Nx:
                dm_r[m] = rho[m, Nr] * Nr*dr*v[m, Nr] - \
                    rho[m, Nr-1]*(Nr-1)*dr*v[m, Nr-1]
            else:
                dm_r[m] = rho[m, Nr] * Nr*dr*v[m, Nr] - \
                    rho[m, Nr-1]*(Nr-1)*dr*v[m, Nr-1]
    # skip m=0, not needed
        # dm_r[m], ur[m, Nr], N)  # used BWD

        de3 = m_de(p, rho, tg, u, v, e, Ts1, de, dm, visc)

        # print(T)
        for m in np.arange(np.int64(1), np.int64(Nx+1)):
            if rho[m, Nr] > 2.*rho_0:
                de3[m] = 0.
    S = -4./D * de3     # 1d array
    S_mdot = [de3, S]
    return S_mdot


def no_division_zero(array):
    # ensure no division by zero
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if array[m, n] == 0:
                array[m, n] = 0.0001
    return array


def energy_difference_dt(e1, e2):
    # sum energy grid at t1
    sum_e1 = np.sum(e1)
# sum energy grid at t2
    sum_e2 = np.sum(e2)

    d_e = sum_e1 - sum_e2

    return d_e

# abb = [dp_dx, ux_dx, ur_dx, grad_x, dt2x_ux,dt2r_ux, ux_dr, ur_dr, dt2x_ur, dt2r_ur]


def tvdrk3(N, rho, rhou, rhov, rhoe, p_in, u_in, rho_in, T_in, e_in, rho_0, v_in, tw, ts, tc, de, Rks, visc, htransfer):
    # This iterates RK3 for all equations
    # q = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))  #
    de2 = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    de_var = np.zeros((Nx+1), dtype=(np.float64))  # place holder

    q, u, v, e = flux_deconstruct(
        rho, rhou, rhov, rhoe)

# calculate pressure, temp
    tg = 2./5.*(e - 1./2.*q * v**2.)*M_n/q/R
    p = q*R/M_n*tg
    Ut = np.sqrt(u**2. + v**2.)

    if visc == 1:
        u, v, Ut, p, q, e = no_slip(p, q, tg, u, v, Ut, e)
        p, q, tg, e, u, v, Ut = outlet_BC(
            p, q, tg, e, u, v, Ut, rho_0)
        u, v, Ut, p, q, tg, e = inlet_BC(
            u, v, Ut, p, q, tg, e, p_in, u_in, v_in, rho_in, T_in, e_in)
        l = [u, v, Ut, p, q, tg, e]

# ALL ITERATION CHECKS
# ensure no division by zero
        q = no_division_zero(q)
# l array ready - with dimensions
    if htransfer == 1:

        # NOTE: Write this in conservative form
        # NOTE: ADD MASS DEPO SOURCE TERM
        # S_mdot = [de3, S]
        S_mdot = source_mass_depo_matrix(
            p, q, e, rho_0, tg, ts, u, v, de, N, visc, htransfer)
        # This de1 is returned again, first calculation is the right one for this iteration. it takes last values.
        de_var = S_mdot[0]

    else:
        S_mdot = source_mass_depo_matrix(
            p, q, e, rho_0, tg, ts, u, v, de, N, visc, htransfer)
        de_var[:] = S_mdot[0]
        de_var[:] = 0

# save variable de
        de2 = de_var

# radial velocity as function of heat transfer
    if htransfer == 0:  # ur
        v[:, Nr] = 0
    else:
        v[1][:, Nr] = de_var[:, Nr]/q[:, Nr]

# radial velocity on surface is function of mass deposition
    u, v, Ut, p, q, e = htransfer_wall_bc(
        u, v, Ut, tg, p, q, e, S_mdot[0], htransfer)


# get_dimless
    # l[3], l[4], l[5], l[6], l[0], l[1], l[2] = get_dimless(l[3], l[4], l[5], l[6], l[0], l[1])
# NOTE: CHECK
# de3
# if htransfer == 0:
    # S_mdot[0] = S_mdot[0] / m_out_dim
    # S_mdot[1] = S_mdot[0] * -4  # NOTE CHECK THIS

    # S_mdot[1] = -4./D * de     # 1d array

# Calculate RHS all equations
    r = RHS_rho(u, v, q, S_mdot[1], N, visc, htransfer)
    r_u = RHS_rhoU(u, v, p, q, tg, N, visc)
    r_v = RHS_rhoV(u, v, p, q, tg, N, visc)
    r_e = RHS_rhoE(u, v, p, q, tg, e, S_mdot[1], N, visc, htransfer)
    return r, r_u, r_v, r_e

#             # dx, dr, dt = dx_dr_dt_physical(dx, dr, dt)


# # recover Physical
#         rho_final = rho_final * (rho_dim*v_dim)
#         rhoU_final = rhoU_final * (rho_dim*v_dim)
#         rhoV_final = rhoV_final * (rho_dim*v_dim)
#         rhoE_final = rhoE_final * (rho_dim*v_dim)

def f_ps(ts):
    #   Calculate saturated vapor pressure (Pa)
    ts = float(ts)
    if ts < 10.:
        p_sat = 12.4-807.4*10**(-1)-3926.*10**(-2)+62970. * \
            10**(-3)-463300.*10**(-4)+1325000.*10**(-5)
    elif ts < 35.6:
        p_sat = 12.4-807.4*ts**(-1)-3926.*ts**(-2)+62970. * \
            ts**(-3)-463300.*ts**(-4)+1325000.*ts**(-5)
    else:
        p_sat = 8.514-458.4*ts**(-1)-19870.*ts**(-2) + \
            480000.*ts**(-3)-4524000.*ts**(-4)
    p_sat = np.exp(p_sat)*100000.
    return p_sat

# @jit(nopython=True)


def f_ts(ps):
    #   Calculate saturated vapor temperature (K)
    # print("Ps for f_ts calc: ", ps)
    ps1 = np.log(ps/100000.0)
    t_sat = 74.87701+6.47033*ps1+0.45695*ps1**2+0.02276*ps1**3+7.72942E-4*ps1**4+1.77899E-5 * \
        ps1**5+2.72918E-7*ps1**6+2.67042E-9*ps1**7+1.50555E-11*ps1**8+3.71554E-14*ps1**9
    return t_sat

# @jit(nopython=True)


def delta_h(tg, ts):
    #   Calculate sublimation heat of nitrogen (J/kg)  ## needed for thermal resistance of SN2 layer when thickness is larger than reset value.
    # print("Tg, Ts for delta_h calc: ", tg, ts)
    delta_h62 = 6775.0/0.028
    if ts > 35.6:
        h_s = 4696.25245*62.0-393.92323*62.0**2/2+17.11194*62.0**3/3-0.35784*62.0**4/4+0.00371*62.0**5/5-1.52168E-5*62.0**6/6 -\
            (4696.25245*ts-393.92323*ts**2/2+17.11194*ts**3/3 -
             0.35784*ts**4/4+0.00371*ts**5/5-1.52168E-5*ts**6/6)
    else:
        h_s = 4696.25245*62.0-393.92323*62.0**2/2+17.11194*62.0**3/3-0.35784*62.0**4/4+0.00371*62.0**5/5-1.52168E-5*62.0**6/6 -\
            (4696.25245*35.6-393.92323*35.6**2/2+17.11194*35.6**3/3-0.35784*35.6**4/4+0.00371*35.6**5/5-1.52168E-5*35.6**6/6) +\
            (-0.02633*35.6+4.72107*35.6**2/2-5.13485*35.6**3/3+1.53391*35.6**4/4-0.13279*35.6**5/5+0.00557*35.6**6/6-1.16225E-4*35.6**7/7+9.67937E-7*35.6**8/8) -\
            (-0.02633*ts+4.72107*ts**2/2-5.13485*ts**3/3+1.53391*ts**4/4 -
             0.13279*ts**5/5+0.00557*ts**6/6-1.16225E-4*ts**7/7+9.67937E-7*ts**8/8)
    h_g = (tg-62.0)*7.0/2.0*R/M_n
    dH = delta_h62+h_s+h_g
    return dH


# @jit(nopython=True)
def c_n(ts):
    #   Calculate specific heat of solid nitrogen (J/(kg*K))
    # print("Ts for c_n specific heat SN2 calc: ", ts)
    if ts > 35.6:
        cn = (4696.25245-393.92323*ts+17.11194*ts**2 -
              0.35784*ts**3+0.00371*ts**4-1.52168E-5*ts**5)
    else:
        cn = (-0.02633+4.72107*ts-5.13485*ts**2+1.53391*ts**3-0.13279 *
              ts**4+0.00557*ts**5-1.16225E-4*ts**6+9.67937E-7*ts**7)
    return cn

# @jit(nopython=True)


def v_m(tg):
    #   Calculate arithmetic mean speed of gas molecules (m/s)
    print("Tg for v_m gas: ", tg)
    v_mean = np.sqrt(8.*R*tg/np.pi/M_n)
    # ipdb.set_trace()
    return v_mean

# @jit(nopython=True)


def c_c(ts):
    #   Calculate the heat capacity of copper (J/(kg*K))
    # print("Ts for c_c (specific heat copper) calc: ", ts)
    #  print("ts",ts)
    c_copper = 1.22717-10.74168*np.log10(ts)**1+15.07169*np.log10(
        ts)**2-6.69438*np.log10(ts)**3+1.00666*np.log10(ts)**4-0.00673*np.log10(ts)**5
    c_copper = 10.**c_copper
    return c_copper


# @jit(nopython=True)
def k_cu(T):
    #   Calculate the coefficient of thermal conductivity of copper (RRR=10) (W/(m*K)) (for pde governing copper wall, heat conducted in the x axis.)
    # print("Tw for k_cu copper: ", T)
    k1 = 3.00849+11.34338*T+1.20937*T**2-0.044*T**3+3.81667E-4 * \
        T**4+2.98945E-6*T**5-6.47042E-8*T**6+2.80913E-10*T**7
    k2 = 1217.49161-13.76657*T-0.01295*T**2+0.00188*T**3-1.77578E-5 * \
        T**4+7.58474E-8*T**5-1.58409E-10*T**6+1.31219E-13*T**7
    k3 = k2+(k1-k2)/(1+np.exp((T-70)/1))
    return k3

# @jit(nopython=True)


def D_nn(T_g, P_g):
    #   Calculate self mass diffusivity of nitrogen (m^2/s)
    if T_g > 63:
        D_n_1atm = -0.01675+4.51061e-5*T_g**1.5
    else:
        D_n_1atm = (-0.01675+4.51061e-5*63**1.5)/63**1.5*T_g**1.5
    D_n_p = D_n_1atm*101325/P_g
    D_n_p = D_n_p/1e4
    return D_n_p

# @jit(nopython=True)


def mu_n(T, P):
    #   Calculate viscosity of nitrogen (Pa*s)
    # print("viscosity temp and pressure", T, P)
    if T == 0:
        T = 0.0001
    tao = 126.192/T
    if tao == 0:
        tao = 0.0001
    if P == 0:
        P = 0.0001
    delta = 1/(3395800/P/tao)
    if T == 98.94:
        T = 98.95
#    print("T value in viscosity function", T)
    omega = np.exp(0.431*np.log(T/98.94)**0-0.4623*np.log(T/98.94)**1 +
                   0.08406*np.log(T/98.94)**2+0.005341*np.log(T/98.94)**3 -
                   0.00331*np.log(T/98.94)**4)
#    print("omega", omega)
    mu_n_2 = 0.0266958*np.sqrt(28.01348*T)/0.3656**2/omega
    mu_n_1 = 10.72*tao**0.1*delta**2*np.exp(-0*delta**0) +\
        0.03989*tao**0.25*delta**10*np.exp(-1*delta**1) +\
        0.001208*tao**3.2*delta**12*np.exp(-1*delta**1) -\
        7.402*tao**0.9*delta**2*np.exp(-1*delta**2) +\
        4.620*tao**0.3*delta**1*np.exp(-1*delta**3)
    mu_t = mu_n_1 + mu_n_2
    # print(mu_t)
    if np.any(mu_t == 0):
        print("The viscous matrix has zeros")
        exit()

    return mu_t/1e6


@jit(nopython=True)
def gamma(a):
    #   Calculate the correction factor of mass flux
    gam1 = np.exp(-np.power(a, 2.))+a*np.sqrt(np.pi)*(1+math.erf(a))
    return gam1


@jit(nopython=True)
def exp_smooth(grid, hv, lv, order, tran):  # Q: from where did we get this?
    #   Exponential smooth from hv to lv
    coe = ((hv+lv)/2-lv)/(np.exp(order*tran)-1)
    c = lv-coe
    if grid < 0:
        s_result = hv
    elif grid < (tran+1):
        s_result = -coe*np.exp(order*(grid))-c+hv+lv
    elif grid < (2*tran+1):
        s_result = coe*np.exp(order*(2*tran-grid))+c
    else:
        s_result = lv
    return s_result

# @jit(nopython=True)


def bulk_test1(T_s):
    T_0 = T_s
    rho_0 = 1e-5  # An arbitrary small initial density in pipe, kg/m3
    # p_0 = rho_0/M_n*R*T_0  # Initial pressure, Pa
    p_0 = rho_0*R/M_n*T_0
    e_0 = 5./2./M_n*R*T_0  # Initial internal energy
    ux_0 = 0
    bulk = [T_0, rho_0, p_0, e_0, ux_0]
    return bulk

# @jit(nopython=True)


def bulk_values(T_s):
    T_0 = T_s
    rho_0 = 1e-5  # An arbitrary small initial density in pipe, kg/m3
    p_0 = rho_0/M_n*R*T_0  # Initial pressure, Pa
    e_0 = 5./2.*p_0  # Initial internal energy
    u_0 = 0
    v_0 = 0
    Ut_0 = np.sqrt(u_0**2. + v_0**2.)
    bulk = [T_0, rho_0, p_0, e_0, Ut_0, u_0, v_0]
    print("p_0: ", p_0, "T_0:", T_0, "rho_0: ", rho_0, "e_0: ", e_0)
    return bulk


# @jit(nopython=True)


def integral_mass_delSN(de):
    # del_SN = np.zeros((Nx+1), dtype=(np.float64))
    de0 = np.zeros((Nx+1), dtype=(np.float64))
    # Integrate deposited mass
    for m in np.arange(Nx+1):
        de0[m] += dt*np.pi*D*de[m]

# Calculate the SN2 layer thickness
    del_SN = de0/np.pi/D/rho_sn
    # print("del_SN: ", del_SN)

    return de0, del_SN  # the de0 is incremented and never restarted

# recalculates Tg to be equal to Ts.
# NOTE: does this affect the velocities? does mde change? and if yes, does it mean ur changes?


# @jit(nopython=True)
def gas_surface_temp_check(tg, ts, v, e, u, rho):
    # print("starting Tg> Ts check")
    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        if tg[m, Nr] < ts[m]:
            e[m, Nr] = 5./2.*rho[m, Nr]*R*ts[m] / \
                M_n + 1./2.*rho[m, Nr]*v[m, Nr]**2

            # # print("THIS IS T2 < Ts")
            # # print("e2 surface", e2[m, n])
            # check_negative(e2[m, n], n)

    tg = 2./5.*(e - 1./2.*rho * v**2.)*M_n/rho/R

    #     print(
    #         "T2 surface recalculated to make it equal to wall temperature (BC)", T2[m, n])
    #     check_negative(T2[m, n], n)

    # NOTE: Energy is changed assuming density and radial velocity constant. Is this correct?
    # print("balancing energies")
    p = rho*R/M_n*tg
    return tg, e, p, rho, u


# @jit(nopython=True)
def Cu_Wall_function(urx, Tx, Twx, Tcx, Tsx, T_in, delSN, de1, ex, ux, rhox, px, T2, p2, e2, rho2, u2, ur2):
    # define wall second derivative
    dt2nd_w_m = dt2nd_w_matrix(Twx, T_in)
    qi = np.zeros((Nx+1), dtype=(np.float64))
    q_dep = np.zeros((Nx+1), dtype=(np.float64))
    Tw2 = np.zeros((Nx+1), dtype=(np.float64))
    Tc2 = np.zeros((Nx+1), dtype=(np.float64))
    Ts2 = np.zeros((Nx+1), dtype=(np.float64))
    qhe = np.zeros((Nx+1), dtype=(np.float64))

# Initial calculations:

# Only consider thermal resistance in SN2 layer when del_SN > 1e-5:
# # NOTE: CHECK THIS LOGIC TREE

    print("calculating Tw, Tc, Ts, qdep")

    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        if delSN[m] > 1e-5:
            print(
                "This is del_SN > 1e-5 condition, conduction across SN2 layer considered")

            # heatflux into copper wall from frost layer
            qi[m] = k_sn*(Tsx[m]-Twx[m])/delSN[m]
            # print("qi: ", qi)
            # check_negative(qi, n)
        else:
            # no heatflux into copper wall
            qi[m] = 0
            # print("qi: ", qi)
            # check_negative(qi, n)


# pipe wall equation
        Tw2[m] = Twx[m] + dt/(w_coe*c_c(Twx[m]))*(qi[m]-q_h(Twx[m])
                                                  * Do/D) + dt/(rho_cu*c_c(Twx[m]))*k_cu(Twx[m])*dt2nd_w_m[m]

# q deposited into frost layer. Nusselt convection neglected
        q_dep[m] = de1[m]*(1/2*(urx[m, Nr])**2 + delta_h(Tx[m, Nr], Tsx[m]))

# SN2 Tc equation
# SN2 surface temperature calculation
        if delSN[m] < 1e-5:
            Tc2[m] = Tw2[m]
            Ts2[m] = 2*Tc2[m] - Tw2[m]
        else:
            Tc2[m] = Tcx[m] + dt * (q_dep[m]-qi[m]) / \
                (rho_sn * c_n(Tsx[m])*delSN[m])
            Ts2[m] = 2*Tc2[m] - Tw2[m]

    # NOTE: delta_h will change if T =Ts and will be zero.
    # NOTE: Check this logic, very important
    # print("Forcing Tg >= Ts")
    # for j in np.arange(np.int64(0), np.int64(Nx+1)):
    #     if T2[j, Nr] < Ts2[j]:
    #         # NOTE: Should i use the old mass deposition? import mdot of last matrix?
    #         # NOTE: Is this the current or updated velocity?
    #         T2[j, Nr] = Ts2[j]

# q deposited into frost layer. Nusselt convection neglected

        print("recalculating energies")
        # T2, e2, p2, rho2, u2 = gas_surface_temp_check(
        #     T2, Ts2, ur2, e2, u2, rho2)

    print("calculating qhe")
    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        qhe[m] = q_h(Twx[m])

# NOTE: Check qhe values larger than e2 of Tg
# NOTE: Put a limiting factor for qhe
    # s = ex[:, Nr]
    # ID = s < qhe
    # print("Checking qhe > e1", np.any(ID))

    # print("saving qhe")
    # save_qhe(i, dt, qhe)
    # print("saving q_dep")
    # save_qdep(i,dt,q_dep)
    print("Wall function complete")
    w_out = [Tw2, Ts2, Tc2, qhe, q_dep]
    return w_out


# # @jit(nopython=True)
# def no_slip_poisson(ux, ur, u, p, rho, T):
#     # Define convergence criteria
#     max_iter = 1000  # Maximum number of iterations
#     tolerance = 1e-6  # Convergence tolerance

#     # Iterate to solve the pressure equation
#     for iteration in range(max_iter):
#         p_old = p.copy()  # Store the old pressure field

#         # Update pressure at interior points
#         for i in range(1, Nx-1):
#             for j in range(1, Nr-1):
#                 p[i, j] = 0.25 * (p_old[i+1, j] + p_old[i-1, j] +
#                                   p_old[i, j+1] + p_old[i, j-1])

#         # Apply boundary conditions
#         p[0, :] = p[1, :]  # Dirichlet boundary condition at left boundary
#         p[-1, :] = p[-2, :]  # Dirichlet boundary condition at right boundary
#         p[:, 0] = p[:, 1]  # Dirichlet boundary condition at bottom boundary
#         p[:, -1] = p[:, -2]  # Dirichlet boundary condition at top boundary

#         # Check convergence
#         residual = np.abs(p - p_old).max()
#         if residual < tolerance:
#             print(
#                 f"Converged in {iteration+1} iterations with a residual of {residual}")
#             break
#     return p

def save_data(tx, dt, rho1, ux1, ur1, u1, e1, T1, Tw1, Ts1, de0, p1):
    increment = (tx+1)*dt

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(increment) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("rho.csv", rho1, delimiter=",")
    np.savetxt("Tg.csv", T1, delimiter=",")
    np.savetxt("u.csv", u1, delimiter=",")
    np.savetxt("ux.csv", ux1, delimiter=",")
    np.savetxt("ur.csv", ur1, delimiter=",")
    np.savetxt("e.csv", e1, delimiter=",")
    np.savetxt("tw.csv", Tw1, delimiter=",")
    np.savetxt("ts.csv", Ts1, delimiter=",")
    np.savetxt("de_mass.csv", de0, delimiter=",")
    # np.savetxt("de_rate.csv", de2, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    # np.savetxt("peclet.csv", pe, delimiter=",")
    # np.savetxt("qhe.csv", qhe, delimiter=",")
    # np.savetxt("qdep.csv", qdep, delimiter=",")
    # np.savetxt("visc.csv", visc, delimiter=",")


def parabolic_velocity(rho, tg, u, u_in, Ut, e):
    # for i in np.arange(n_trans):
    # diatomic gas gamma = 7/5   WE USED ANY POINT, since this preparation area is constant along R direction.
    # any temperature works, they are equl in the radial direction
    v_max = np.sqrt(7./5.*R*tg[0, 1]/M_n)
    for y in np.arange(Nr+1):
        # a = v_max
        # a = u_in
        u[0, y] = v_max*(1.0 - ((y*dr)/R_cyl)**2)
        # print("parabolic y", y)
        Ut[0, y] = u[0, y]
    u[:, Nr] = 0
    Ut[:, Nr] = 0
    e = 5./2.*R/M_n*tg + 1./2. * Ut**2.
    out = u, Ut, e
    return out


# @jit(nopython=True)


def smoothing_inlet(p, rho, T, p_in, p_0, rho_in, rho_0, n_trans):
    for i in range(0, Nx+1):
        p[i, :] = exp_smooth(i+n_trans, p_in*2.-p_0, p_0, 0.4, n_trans)
    # print("P1 smoothing values", p1[i,:])
        rho[i, :] = exp_smooth(i + n_trans, rho_in*2 -
                               rho_0, rho_0, 0.4, n_trans)
    #    T1[i, :] = T_neck(i)
        # if i<51: T1[i]=T_in
        T[i, :] = p[i, :]/rho[i, :]/R*M_n
        # v_max = np.sqrt(7./5.*R*T/M_n)  # diatomic gas gamma = 7/5
    #    u1[i, :] = exp_smooth(i + n_trans, ux_in*2, 0, 0.4, n_trans)

        # if i < n_trans+1:
        #     e1[i, :] = 5./2.*p1[i, :]+1./2.*rho1[i, :]*u1[i, :]**2

    #        rho1[i, :] = p1[i, :]*M_n/R/T1[i, :]  # IDEAL GAS LAW

        # print("p1 matrix after smoothing", p1)
        # else:
        #     e1[i, :] = 5/2*rho1[i, :]/M_n*R*T_in+1/2**rho1[i, :]*u1[i, :]**2
    # for i in range(0, Nx+1):
    out = p, rho, T
    return out


# remove timestepping folder


def remove_timestepping():
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "timestepping"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/RK3/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "RK3"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/second_gradients_surface'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "second_gradients_surface"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/mdot/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "mdot"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/rhs/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "rhs"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/gradients'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "gradients"
        path = os.path.join(location, dir)
        shutil.rmtree(path)


def htransfer_wall_bc(u, v, Ut, tg, p, rho, e, de, htransfer):
    if htransfer == 1:
        v[:, Nr] = de/rho[:, Nr]
    else:
        v[:, Nr] = 0
    Ut = np.sqrt(u**2 + v**2)
# Balance energies
    e = 5./2. * rho*R/M_n*tg + 1./2. * Ut**2.
    return u, v, Ut, p, rho, e

# @jit(nopython=True)


def no_slip(p, q, tg, u, v, Ut, e):
    u[:, Nr] = 0
    v = v
    Ut[:, Nr] = v[:, Nr]
    q = q
    p = p
# Balance energies
    e = tg/(gamma_n * (gamma_n-1)**M_inf**2) + 1./2.*Ut**2
    return p, tg, u, v, Ut, e

# @jit(nopython=True)


# @jit(nopython=True)
def inlet_BC(u, v, Ut, p, rho, T, e, p_inl, u_inl, rho_inl, T_inl, e_inl):
    p[0, :] = p_inl
    rho[0, :] = rho_inl
    T[0, :] = T_inl
    e[0, :] = e_inl
    u[0, :] = u_inl

# no slip
    u[:, Nr] = 0
    v[:, Nr] = 0
    Ut[:, Nr] = 0
    Ut = np.sqrt(u**2. + v**2.)
    e = 5./2. * p/rho + 1./2 * Ut**2
    return [u, v, Ut, p, rho, T, e]


def outlet_BC(p, e, rho, u, v, Ut, rho_0):
    for n in np.arange(Nr):
        p[Nx, n] = 2./5.*rho[Nx, n]*(e[Nx, n]-1/2 ** Ut[Nx, n]**2)  # Pressure
        rho[Nx, n] = max(2*rho[Nx-1, n]-rho[Nx-2, n], rho_0)  # Free outflow
        u[Nx, n] = max(2*rho[Nx-1, n]*u[Nx-1, n] -
                       rho[Nx-2, n]*u[Nx-2, n], 0) / rho[Nx, n]
        u = np.sqrt(u**2. + v**2.)
    u[:, Nr] = 0  # no slip
    v[:, Nr] = 0
    Ut = np.sqrt(u**2. + v**2.)
    # e[Nx, n] = 2*e[Nx-1, n]-e[Nx-2, n]
    e = 5./2. * p/rho + 1./2 * Ut**2
    tg = p/rho/R*M_n
    bc = [p, rho, tg,  u, Ut, e]
    return bc


def wall_BC_no_freeze(tw, ts, tc, T_in):
    tw[0] = T_in
    ts[0] = T_in
    tc[0] = T_in
    return tw, ts, tc


def flux_deconstruct(Rho, RhoU, RhoV, RhoE):
    Rho = Rho
    U = RhoU/Rho
    V = RhoV/Rho
    E = RhoE/Rho
    return Rho, U, V, E


def flux_construct(rho, u, v, e):
    rho = rho
    rhoU = rho*u
    rhoV = rho*v
    rhoE = rho*e
    return rho, rhoU, rhoV, rhoE


# # ideal gas law
# def recalculate_pressure_ideal(q, tg):
#     p = q*R/M_n*tg
#     return p


# recalculating energy from p, rho, and velocity
# @jit(nopython=True)
# def balance_energy_tg(q, tg, u, v):
#     Ut = np.sqrt(u**2 + v**2)
#     e = 5./2. * q*R/M_n*tg + 1./2.*Ut**2
#     # e = tg/(gamma_n * (gamma_n-1)**M_inf**2) + 1./2.*Ut**2
#     return e

# recalculating velocity from energy, T, rho
# def recalculate_velocity_e_p(e, q, p):
#     Ut = np.sqrt((e-5./2.*p)*2./q)
#     return Ut

# recalculating energy from rho, T, and velocity
# def dx_dr_dt_physical(dx, dr, dt):
#     dx = dx * D_hyd
#     dr = dr * D_hyd
#     dt = dt * (D_hyd / Ut_dim)
#     return dx, dr, dt


# def dx_dr_dt_dimless(dx, dr, dt):
#     dx = dx / D_hyd
#     dr = dr / D_hyd
#     dt = dt / (D_hyd / Ut_dim)
#     return dx, dr, dt

# NOTE: CHECK CONVERSIONS

# NOTE: adaptive timestep, to integrate, must include as an argument in all functions using dt, and then dimless must convert all the time.


def calc_dt(cfl, gamma_n, q, nx, nr, dx, dr):
    a = 30.0
    a = np.max([a, 0.0])
    for j in np.arange(nr):
        for i in np.arange(nx):
            rho, ma_x, ma_r, ma_energy = q[:, i, j]
            ux, ur, e = ma_x/rho, ma_r/rho, ma_energy/rho
            p = rho*(gamma_n-1)*(e-0.5*(ux ^ 2+ur ^ 2))
            c = np.sqrt(gamma_n*p/rho)
            a = np.max([a, abs(ux), abs(ux+c), abs(ux-c),
                       abs(ur), abs(ur+c), abs(ur-c)])
    dt = cfl*np.min([dx, dr])/a
    return dt

# recovers physical quantities from dimensionless


def get_dimless(p, q, tg, e, u, v):
    # global rho_in_dim, u_in_dim, M_inf
    u = u/u_dim
    v = v/u_dim
    q = q / rho_dim
    p = p / (rho_dim*u_dim**2.)
    e = e / u_dim**2.
    tg = (e - 1./2.*(u**2.)) * gamma_n * (gamma_n-1)**M_inf**2.
    return p, q, tg, e, u, v


# def get_dimless_visc(mu):
#     mu = mu / (rho_dim*D_hyd*Ut_dim)
#     return mu


# def physical_visc(mu):
#     mu = mu * (rho_dim*D_hyd*Ut_dim)
#     return mu


# returns the physical scales
def recover_physical(p, q, tg, e, u, v):
    # recovers physical quantities from dimensionless
    q = q * rho_dim
    p = p * (rho_dim*u_dim**2)
    u = u * u_dim
    v = v * v_dim
    e = e * (rho_dim*u_dim**2)
    tg = (e - 1./2.*(u**2)) * gamma_n * (gamma_n-1)**M_inf**2
    # mdot = mdot / (rho_dim * Ut_dim)
    return p, q, tg, e, u, v

# @jit(nopython=True)


# @jit(nopython=True)
def val_in_constant(p_in, T_in, u_in):
    #   Calculate instant flow rate (kg/s)
    p_in = 600.
    T_in = 298.
    rho_in = p_in / T_in/R*M_n
    u_in = np.sqrt(gamma_n*R/M_n*T_in)
    v_in = 0.
    Ut_in = np.sqrt(u_in**2 + v_in**2)
    e_in = 5./2./M_n*R*T_in + 1./2.*Ut_in**2
    return p_in, u_in, v_in, rho_in, e_in, T_in

# @jit(nopython=True)


def val_in(n):
    #   Calculate instant flow rate (kg/s)
    # Fitting results
    #    A1 = 0.00277; C = 49995.15263  # 50 kPa
    T_in = 298.
    A1 = 0.00261
    C = 100902.5175  # 100 kPa
   # A1 = 0.00277; C = 10000.15263  # 50 kPa
    ux_in = np.sqrt(7./5.*R*T_in/M_n)*1.0  # Inlet velocity, m/s
    # ux_in = 351.
    P_in_fit = np.power(A1*n*dt+np.power(C, -1./7.), -7.)
    # P_in_fit = 1000.
    # P_in_fit = 1./2.*P_in_fit

    dP_in_fit = -7.*A1*np.power(A1*n*dt+np.power(C, -1./7.), -8.)

    q_in = -(np.power(C, 2./7.)*0.230735318/1.4/297./T_in) * \
        (np.power(P_in_fit, -2./7.)*dP_in_fit)
    ma_in_x = q_in/A
    rho_in = ma_in_x/ux_in
    p_in = rho_in/M_n*R*T_in
    print("p_in", p_in)
    ur_in = 0.
    e_in = 5./2./M_n*R*T_in + 1./2.*ux_in**2
    # print("u_in_x", u_in_x)
    out = np.array([p_in, ux_in, ur_in, rho_in, e_in, T_in])
    return out


# @jit(nopython=True)
def DN(tg, p, u, tw, rho):
    #   Calculate dimensionless numbers
    rho = p*M_n/R/tg
    # rho_w=f_ps(T_w)*M_n/R/T #_w
    mu = mu_n(tg, p)
    neu = mu/rho
    # print("mu", mu)
    Re = rho*(u)*D/mu  # Reynolds number
    D_n = D_nn(tg, p)
    Sc = mu/rho/D_n  # Schmidt number
    Kn = 2*mu/p/np.sqrt(8*M_n/np.pi/R/tg)/D
    mu_w = mu_n(tw, f_ps(tw))
    Sh = 0.027*Re**0.8*Sc**(1/3)*(mu/mu_w)**0.14  # Sherwood number
    Nu = 0.027*Re**0.8*Pr_n**(1/3)*(mu/mu_w)**0.14  # Nusselt number
    Cou = u*dt/dx  # Courant Number
    Pe = u * L/neu
    DN_all = np.array([Re, Sc, Kn, Sh, Nu, Cou, Pe])
    # print(DN_all)
    # print("Courant Number is: ", Cou)
    return DN_all


def Peclet_grid(u, p, tg):
    pe = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    mu = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for m in np.arange(np.int64(0), np.int64(Nx+1)):
        for n in np.arange(np.int64(1), np.int64(Nr+1)):
            mu[m, n] = mu_n(tg[m, n], p[m, n])
    pe = u*D_hyd / mu
    return pe

# de1[m] = m_de(T1[m, n], p1[m, n], Tw1[m], de1[m], rho1[m, n]*ur1[m, n]-rho1[m, n-1]*ur1[m, n-1])

# NOTE: I am getting wrong mass deposition values... from 1d it is in the order of e-6
# returns mass deposition rate to put in de1 matrix
# @numba.jit('f8(f8,f8,f8,f8,f8)')
# @jit(nopython=True)
# dm_r, ur, N):


def m_de(p, q, tg, u, v, e, ts, de, dm, htransfer):
    m_out = np.zeros((Nx+1), dtype=(np.float64))
    if htransfer == 0:
        m_out[:] = 0
    else:
        # print("T,P,ur, Ts1, de, dm", [T, P, ur, Ts1, de, dm])
        q = np.zeros((Nx+1, Nr+1), dtype=(np.float64,
                                          np.float64))  # place holder
        v_m1 = np.zeros((Nx+1), dtype=(np.float64))
        u_mean1 = np.zeros((Nx+1), dtype=(np.float64))
        gam1 = np.zeros((Nx+1), dtype=(np.float64))
        P_s = np.zeros((Nx+1), dtype=(np.float64))
        q_min = np.zeros((Nx+1), dtype=(np.float64))
        beta = np.zeros((Nx+1), dtype=(np.float64))
        m_max = np.zeros((Nx+1), dtype=(np.float64))

        p_0 = bulk_values(T_s)[2]

        for m in np.arange(np.int64(0), np.int64(Nx+1)):

            if tg[m, Nr] == 0:
                tg[m, Nr] = 0.00001
            q[m, Nr] = p[m, Nr]*M_n/R/tg[m, Nr]
            q_min[m] = p_0*M_n/R/tg[m, Nr]
        # # no division by zero
        #     if rho[m,Nr] == 0:
        #         rho[m,Nr] = 0.00001
            # thermal velocity of molecules
            v_m1[m] = np.sqrt(2*R*tg[m, Nr]/M_n)
            # mean flow velocity towards the wall.
            u_mean1[m] = de[m]/q[m, Nr]
            beta[m] = u_mean1[m]/v_m1[m]  # this is Beta from Hertz Knudson
            gam1[m] = gamma(beta[m])  # deviation from Maxwellian velocity.
            P_s[m] = f_ps(ts1[m])

            if (p[m, Nr] > P_s[m] and p[m, Nr] > p_0):
                # Correlated Hertz-Knudsen Relation #####
                m_out[m] = np.sqrt(M_n/2/np.pi/R)*Sc_PP * \
                    (gam1[m]*P[m, Nr]/np.sqrt(tg[m, Nr])-P_s[m]/np.sqrt(ts1[m]))
                # print("m_out calc", m_out)

                if ts1[m] > 25:
                    # print("P>P0, P>Ps")
                    # Arbitrary smooth the transition to steady deposition
                    # NOTE: Check this smoothing function.
                    m_out[m] = m_out[m]*exp_smooth(ts1[m]-25., 1., 0.05,
                                                   0.03, (f_ts(p[m, Nr]*np.sqrt(ts1[m]/tg[m, Nr]))-25.)/2.)

                # Speed of sound limit for the condensation flux
                # sqrt(7./5.*R*T/M_n)*rho
                # Used Conti in X-direction, since its absolute flux.
                # m_max[:] = D/4./dt*(rho[:, Nr-1]-rho_min)-D/4./dx*dm - D/4. * \
                #     1/N[:, Nr-1]/dr*(rho[:, Nr-1]*N[:, Nr-1]*dr*ur[:,
                #                      Nr-1] - rho[:, Nr-2]*N[:, Nr-2]*dr*ur[:, Nr-2])

                # using conti surface
                # m_max = D/4./dt*(rho-rho_min)-D/4. * (1/Nr/dr*dm_r)
                # dm is a matrix
                m_max[m] = D/4./dt*(q[m, Nr]-rho_min[m]) - \
                    D/4./dx*dm[m]  # sqrt(7./5.*R*T/M_n)*rho
                m_max[m] = 2.0e-30

                if m_out[m] > m_max[m]:
                    m_out[m] = m_max[m]
                    # print("mout = mmax")
            else:
                m_out[m] = 0
            # m_out = 0  # NO HEAT TRANSFER/ MASS DEPOSITION CASE
            # print("de2: ", m_out)

        # print("m_out", m_out)

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/m_dot/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "m_dot"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
        # os.rmdir('C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/')
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    os.chdir(pathname)
    np.savetxt("m_out.csv", m_out, delimiter=",")
    if visc == 1:
        np.savetxt("m_max.csv", m_max, delimiter=",")

    return m_out  # Output: mass deposition flux, no convective heat flux MATRIX

# @jit(nopython=True)


def q_h(tw, BW_coe=0.017):  # (W/(m^2*K)
    # Boiling heat transfer rate of helium (W/(m^2*K))
    # delT = ts-4.2
    delT = tw-4.2

    q_con = 0.375*1000.*delT  # Convection
    q_nu = 58.*1000.*(delT**2.5)  # Nucleate boiling
    q_tr = 7500.  # Transition to film boiling
    # print("qcond: ", q_con, "q_nu: ", q_nu, "q_tr: ", q_tr)
    tt = np.power(q_tr/10000./BW_coe, 1./1.25)
    b2 = tt-1.
    # Breen and Westwater Correlation with tuning parameter
    # print("delT", delT)
    q_fi = (BW_coe*np.power(delT, 1.25))*10000.  # Film boiling
    if q_con > q_nu or delT < 0.01:
        q_he = q_con
    elif q_nu <= q_tr:
        q_he = q_nu
    elif q_nu > q_tr and q_fi < q_tr:
        q_he = q_tr
    else:
        q_he = q_fi
    # Smooth the turning point
    if delT >= tt-b2 and delT <= tt+b2:
        q_min = 1.25/2/b2*10000*BW_coe * \
            (1/2.25*np.power(tt-b2, 2.25)-(tt-b2)/1.25*np.power(tt-b2, 1.25))
        q_max = 1.25/2/b2*10000*BW_coe * \
            (1/2.25*np.power(tt+b2, 2.25)-(tt-b2)/1.25*np.power(tt+b2, 1.25))
        q_max1 = (BW_coe*np.power(tt+b2, 1.25))*10000
        q_he = (1.25/2/b2*10000*BW_coe*(1/2.25*np.power(delT, 2.25)-(tt-b2) /
                1.25*np.power(delT, 1.25))-q_min)/(q_max-q_min)*(q_max1-q_tr)+q_tr
    # print("rate of heat transfer to helium:", q_he)
    # print("q_h calc: ", q_he, "Tw: ", tw)
    # q_he = 0  # NO HEAT TRANSFER CASE
    return q_he

# Initialization


if __name__ == '__main__':
    t_sat = 70
    # p_test = f_ps(t_sat)
    # print("p_test", p_test)

    p_sat = 10000
    # t_test = f_ts(p_sat)
    # print("t_test", t_test)

    l = np.linspace(1, 300, 1000)
    k = k_cu(l)

    # plt.figure()
    # plt.plot(l, k)
    # plt.show()

#    p_before = np.log(-2)

    # Nx = 20; Nr =20
    # rho1 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
    # T1 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
    # T1[2,5] = -2
    # eps = 5./2.*rho1[2, 2]/M_n*R * \
    #                     T1[2, 5]
    # check_negative(eps, eps, 1)

    # p_before = -3
    # check_negative(p_before, p_before, 1)

#   Nx = 10; Nr =10
#    rho12 = np.full((Nx+1, Nr+1), rho_0, dtype=(np.float64, np.float64))  # Density
 #    check_negative(rho12[1,2],rho12, 1)

#  Sublimation heat  #

    tg = 4.2
    ts = 4.2
    print("delta_h ", delta_h(tg, ts))

# specific heat of solid nitrogen (J/(kg*K))  #

    # print("c_n ", c_n(ts))
# thermal velocity #

    # print("vm ", v_m(tg))

# heat capacity of copper (J/(kg*K))  #

    # print("c_c ", c_c(ts))

# thermal conductivity of copper (RRR=10) (W/(m*K)) #
    T = 4.2
    print("k_cu", k_cu(4))

# self mass diffusivity of nitrogen (m^2/s) #

    T_g = 298
    P_g = 1000
    # print("D_nn", D_nn(T_g, P_g))

#  Viscosity  #

    T = 273.15
    P = 9806649
    # print("mu_n ", mu_n(T, P))

# Error function #

#    a = umean/vm1
    a = 0.5
    # print("gamma ", gamma(a))


# ------------------------- Inlet values ------------------------------- #

    # print("val_in ", val_in(0))

# ------------------------- Dimensionless numbers ------------------------------- #

    T = 30
    P = 3000
    u = 100
    T_w = 15
    # print("DN ", DN(T, P, u, T_w))

# ------------------------- Mass Deposition ------------------------------- #

    #   Time and spatial step
    L = 6.45
    Nx = 120.  # Total length & spatial step - x direction 6.45
    R_cyl = 1.27e-2
    Nr = 5.  # Total length & spatial step - r direction
    T = 3.
    Nt = 70000.  # Total time & time step
    dt = T/Nt
    dx = L/Nx
    dr = R_cyl/Nr
    ur = 3

    T = 100
    P = 4000
    T_s = 30
    de = 20
    dm = -10

#  Heat transferred ------------------------------- #

    tw = 298

    k = np.zeros(30000, dtype=np.float64())
    l = np.linspace(4.2, 100, 30000)
    for i in range(len(l)):
        k[i] = q_h(l[i])

    # k = np.zeros(30000, dtype=np.float64())
    # k[l] = q_he(l[:])
    # for x, i in zip(l, range(len(l))):
    #     k[i] = q_h(x)

    #  in l:
        # k[x] = q_h(x, BW_coe=0.017)

    plt.figure()
    plt.plot(l, k)
    plt.show()
