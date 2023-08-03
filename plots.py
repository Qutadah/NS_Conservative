import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt


def save_initial_conditions(rho1, u1, v1, Ut1, e1, T1, Tw1, Ts1, de0, p1, de1, pe, Tc1):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/'
    if os.path.exists(pathname):
        location = "C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/"
        dir = "initial_conditions"
        path = os.path.join(location, dir)
        shutil.rmtree(path)
        # os.rmdir('C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/initial_conditions/')
    if not os.path.exists(pathname):
        os.makedirs(pathname)
    os.chdir(pathname)
    np.savetxt("rho.csv", rho1, delimiter=",")
    np.savetxt("Tg.csv", T1, delimiter=",")
    np.savetxt("u.csv", u1, delimiter=",")
    np.savetxt("v.csv", v1, delimiter=",")
    np.savetxt("Ut.csv", Ut1, delimiter=",")
    np.savetxt("e.csv", e1, delimiter=",")
    np.savetxt("tw.csv", Tw1, delimiter=",")
    np.savetxt("ts.csv", Ts1, delimiter=",")
    np.savetxt("tc.csv", Tc1, delimiter=",")
    np.savetxt("de.csv", de0, delimiter=",")
    np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    np.savetxt("pe.csv", pe, delimiter=",")


def save_data(tx, dt, rho1, u1, v1, Ut1, e1, tg1, tw1, ts1, de0, p1, de2, pe, qhe, qdep, invisc, htransfer):
    increments = (tx+1)*dt

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(increments) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("rho.csv", rho1, delimiter=",")
    np.savetxt("Tg.csv", tg1, delimiter=",")
    np.savetxt("u.csv", u1, delimiter=",")
    np.savetxt("v.csv", u1, delimiter=",")
    np.savetxt("Ut.csv", Ut1, delimiter=",")
    np.savetxt("e.csv", e1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")
    np.savetxt("peclet.csv", pe, delimiter=",")

    if htransfer == 1:
        np.savetxt("tw.csv", tw1, delimiter=",")
        np.savetxt("ts.csv", ts1, delimiter=",")
        np.savetxt("de_mass.csv", de0, delimiter=",")
        np.savetxt("de_rate.csv", de2, delimiter=",")
        np.savetxt("qhe.csv", qhe, delimiter=",")
        np.savetxt("qdep.csv", qdep, delimiter=",")
    # if invisc == 1:
        # np.savetxt("visc.csv", visc, delimiter=",")


def save_gradients(x, tx, dt, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11):
    increment = (tx+1)*dt
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/gradients/' + \
        '/' + "{:.4f}".format(increment) + '/' + "{:.1f}".format(x) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    # np.savetxt("a.csv", array1, delimiter=",")
    np.savetxt("d_dr.csv", array2, delimiter=",")
    np.savetxt("m_dx.csv", array3, delimiter=",")
    np.savetxt("dp_dx.csv", array4, delimiter=",")
    np.savetxt("ux_dx.csv", array5, delimiter=",")
    np.savetxt("ux_dr.csv", array6, delimiter=",")
    np.savetxt("dp_dr.csv", array7, delimiter=",")
    np.savetxt("ur_dx.csv", array8, delimiter=",")
    np.savetxt("ur_dr.csv", array9, delimiter=",")
    np.savetxt("grad_x.csv", array10, delimiter=",")
    np.savetxt("grad_r.csv", array11, delimiter=",")


def save_rhs(x, tx, dt, array1, array2, array3, array4):
    increment = (tx+1)*dt
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/rhs/' + \
        '/' + "{:.4f}".format(increment) + '/' + "{:.1f}".format(x) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    # np.savetxt("a.csv", array1, delimiter=",")
    np.savetxt("rhs_rho.csv", array1, delimiter=",")
    np.savetxt("rhs_mx.csv", array2, delimiter=",")
    np.savetxt("rhs_mr.csv", array2, delimiter=",")
    np.savetxt("rhs_e.csv", array4, delimiter=",")


def save_RK3(x, tx, dt, rho1, ux1, ur1, u1, e1, T1, p1, de1):
    increment = (tx+1)*dt

    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/RK3/' + \
        "{:.4f}".format(increment) + '/' + "{:.1f}".format(x) + '/'
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
    np.savetxt("de_rate.csv", de1, delimiter=",")
    np.savetxt("p.csv", p1, delimiter=",")


def delete_r0_point(rho2, ux2, ur2, u2, e2, T2, p1, pe):
    rho3 = np.delete(rho2, 0, axis=1)
    ux3 = np.delete(ux2, 0, axis=1)
    ur3 = np.delete(ur2, 0, axis=1)
    u3 = np.delete(u2, 0, axis=1)
    e3 = np.delete(e2, 0, axis=1)
    T3 = np.delete(T2, 0, axis=1)
    p3 = np.delete(p1, 0, axis=1)
    pe3 = np.delete(pe, 0, axis=1)
    return [rho3, ux3, ur3, u3, e3, T3, p3, pe3]


def plot_gradients(abb):

    dpdx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    uxdx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    urdx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    gradx = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    dt2xux = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    dt2rux = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    uxdr = np.zeros((Nx+1), dtype=(np.float64))  # place holder
    urdr = np.zeros((Nx+1), dtype=(np.float64))  # place holder

# plotting and saving gradients along surface

# Momentum X
    dpdx[:] = abb[0][:, Nr]
    uxdx[:] = abb[1][:, Nr]
    uxdr[:] = abb[6][:, Nr]

# Momentum R
# NOTE: plot dp/dr
    urdx[:] = abb[2][:, Nr]
    urdr[:] = abb[7][:, Nr]

# viscosity

    dt2xux[:] = abb[4][:, Nr]
    dt2rux[:] = abb[5][:, Nr]

# energy
    gradx[:] = abb[3][:, Nr]

    aa = 40
    plt.figure()
    x = np.linspace(0, aa, aa+1)
    y1 = dpdx[0:aa+1]
    y2 = uxdx[0:aa+1]
    y3 = urdx[0:aa+1]
    y4 = gradx[0:aa+1]
    y5 = dt2xux[0:aa+1]
    y6 = dt2rux[0:aa+1]
    y7 = uxdr[0:aa+1]
    y8 = urdr[0:aa+1]

    plt.plot(x, y1, color="black", label="dp_dx")
    plt.plot(x, y2, color="blue", label="ux_dx")
    # plt.plot(x, y3, color="brown", label="ur_dx")
    plt.plot(x, y4, color="yellow", label="grad_x")
    # plt.plot(x, y5, color="green", label="dt2x_ux")
    # plt.plot(x, y6, color="red", label="dt2r_ux")
    plt.plot(x, y7, color="c", label="ux_dr")
    # plt.plot(x, y8, color="m", label="ur_dr")
    # plt.legend()
    # legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # plt.legend(["dp_dx", "ux_dx", "ur_dx", "grad_x", "dt2x_ux",
    #            "dt2r_ux", "ux_dr", "ur_dr"], loc="lower right")
    plt.legend(["dp_dx", "ux_dx", "grad_x", "ux_dr"], loc="lower right")

    plt.show()

# plotting and saving gradients along n = Nr/2
 # Momentum X
    dpdx[:] = abb[0][:, Nr//2]
    uxdx[:] = abb[1][:, Nr//2]
    uxdr[:] = abb[6][:, Nr//2]

# Momentum R
# NOTE: plot dp/dr
    urdx[:] = abb[2][:, Nr//2]
    urdr[:] = abb[7][:, Nr//2]

# viscosity
    dt2xux[:] = abb[4][:, Nr//2]
    dt2rux[:] = abb[5][:, Nr//2]

# energy
    gradx[:] = abb[3][:, Nr//2]

    aa = 40
    plt.figure()
    x = np.linspace(0, aa, aa+1)
    y1 = dpdx[0:aa+1]
    y2 = uxdx[0:aa+1]
    y3 = urdx[0:aa+1]
    y4 = gradx[0:aa+1]
    y5 = dt2xux[0:aa+1]
    y6 = dt2rux[0:aa+1]
    y7 = uxdr[0:aa+1]
    y8 = urdr[0:aa+1]

    plt.plot(x, y1, color="black", label="dp_dx")
    plt.plot(x, y2, color="blue", label="ux_dx")
    # plt.plot(x, y3, color="brown", label="ur_dx")
    plt.plot(x, y4, color="yellow", label="grad_x")
    # plt.plot(x, y5, color="green", label="dt2x_ux")
    # plt.plot(x, y6, color="red", label="dt2r_ux")
    plt.plot(x, y7, color="c", label="ux_dr")
    # plt.plot(x, y8, color="m", label="ur_dr")
    # plt.legend()
    # legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    # plt.legend(["dp_dx", "ux_dx", "ur_dx", "grad_x", "dt2x_ux",
    #            "dt2r_ux", "ux_dr", "ur_dr"], loc="lower right")
    plt.legend(["dp_dx", "ux_dx", "grad_x", "ux_dr"], loc="lower right")
    plt.show()

    return


def plot_imshow(p, ux, T, rho, e):
    fig, axs = plt.subplots(5)
    fig.suptitle('Fields along tube for all R')

    # PRESSURE DISTRIBUTION
    im = axs[0].imshow(p.transpose())
    plt.colorbar(im, ax=axs[0])
    # plt.colorbar(im, ax=ax[0])
    axs[0].set(ylabel='Pressure [Pa]')
    # plt.title("Pressure smoothing")

    # VELOCITY DISTRIBUTION
    # axs[1].imshow()
    im = axs[1].imshow(ux.transpose())
    plt.colorbar(im, ax=axs[1])
    # axs[1].colorbars(location="bottom")
    axs[1].set(ylabel='Ux [m/s]')
    # plt.title("velocity parabolic smoothing")

    # Temperature DISTRIBUTION
    im = axs[2].imshow(T.transpose())
    plt.colorbar(im, ax=axs[2])
    axs[2].set(ylabel='Tg [K]')

    # axs[1].colorbars(location="bottom")
    # axs[2].set(ylabel='temperature [K]')

    im = axs[3].imshow(rho.transpose())
    plt.colorbar(im, ax=axs[3])
    axs[3].set(ylabel='Density [kg/m3]')

    im = axs[4].imshow(e.transpose())
    plt.colorbar(im, ax=axs[4])
    axs[4].set(ylabel='energy [kg/m3]')

    plt.xlabel("L(x)")
    plt.show()
