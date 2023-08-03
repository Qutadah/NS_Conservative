# THIS SOLVER WILL USE THE CONSERVATIVE FORM OF THE NS EQUATIONS IN CYLINDRICAL COORDINATES/

from inspect import currentframe
from my_constants import *
from functions import *
from plots import *
from initialization import *

# calculate Peclet initial field
Pe1 = Peclet_grid(Ut1, p1, T1)

# NOTE: Check de0, de1 arguments.

# def main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2, ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3):


def main_cal(p1, rho1, T1, u1, v1, Ut1, e1, p2, rho2, T2, u2, v2, Ut2, e2, p3, rho3, T3, u3, v3, Ut3, e3, Pe1, Pe2, p_in, u_in, v_in, rho_in, T_in, e_in):

    print("Main loop started")

    # NO SLIP BC
    if np.any(rho1 == 0):
        print("The density rho1 has zeros")
        exit()

    # p, tg, u, v, Ut, e = no_slip_no_mdot(p1, rho1, T1, u1, v1, Ut1, e1)

    # print("Plotting no slip main fields")
    # plot_imshow(p1, u1, T1, rho1, e1)

# inlet BCs
    u1, v1, Ut1, p1, rho1, T1, e1 = inlet_BC(
        u1, v1, Ut1, p1, rho1, T1, e1, p_in, u_in, rho_in, T_in, e_in)

    print("Plotting inlet BC fields")
    plot_imshow(p1, u1, T1, rho1, e1)

# negative temp check
    if np.any(T1 < 0):
        print("Temp inlet_BC after smoothing has at least one negative value")
        exit()

    p1, rho1, T1, u1, Ut1, e1 = outlet_BC(p1, e1, rho1, u1, v1, Ut1, rho_0)

    print("Plotting outlet BC fields")
    plot_imshow(p1, u1, T1, rho1, e1)

# PARABOLIC VELOCITY PROFILE - inlet prepping area

    # u1, Ut1, e1 = parabolic_velocity(rho1, T1, u1, u_in, Ut1, e1)

    # print("Plotting parabolic BC fields")
    # plot_imshow(p1, u1, T1, rho1, e1)

# PREPPING AREA - smoothing
    p1, rho1, T1 = smoothing_inlet(
        p1, rho1, T1, p_in, p_0, rho_in, rho_0, n_trans)


# recalculate energy
    for j in range(0, Nx+1):
        u1[j, :] = exp_smooth(j + n_trans, u1[j, :]*2, 0, 0.4, n_trans)

    Ut1 = np.sqrt(u1**2. + v1**2.)

    # print("Plotting smoothing BC fields")
    # plot_imshow(p1, u1, T1, rho1, e1)
    # print("BC fields")
    # plot_imshow(p1, u1, T1, rho1, e1)

# negative temp check
    if np.any(T1 < 0):
        print("Temp smoothing has at least one negative value")
        exit()

    # SAVING INITIAL FIELDS
    print("Saving initial fields")
    save_initial_conditions(rho1, u1, v1, Ut1, e1, T1,
                            Tw1, Ts1, de0, p1, de1, Pe1, Tc1)

    print("Plotting initial fields")
    plot_imshow(p1, u1, T1, rho1, e1)

# TIME LOOPING
    for i in np.arange(np.int64(1), np.int64(Nt+1)):
        print("Starting Time Looping")
        print("Iteration: #", i)

# constant inlet

        # p_in, q_in, ux_in, ur_in, rho_in, e_in = val_in(
        #     i, ux_in)  # define inlet values

        # p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(i)

        print("Assigning inlet values")
    # out1 = val_in(i)
# out1 = val_in_constant()
        out1 = val_in_constant(p_in, T_in, u_in)

        p_in = out1[0]
        u_in = out1[1]
        v_in = out1[2]
        rho_in = out1[3]
        e_in = out1[4]
        T_in = out1[5]
        Ut_in = np.sqrt(u_in**2 + v_in**2)

    # simple time integration
        p2, rho2, T2, u2, v2, Ut2, e2 = simple_time(
            p1, rho1, T1, u1, v1, Ut1, e1, p_in, rho_in, T_in, e_in, u_in, v_in, rho_0, dt)

        print("NAN check next")

    # perform NAN value matrix checks:
        # for x in np.arange(len(out)):
        #     assert np.isfinite(out[x]).all()

        # negative density check
        if np.any(rho2 < 0):
            print("The Density Array has at least one negative value")
            exit()

            # negative energy check
        if np.any(e2 < 0):
            print("The energy has at least one negative value")
            exit()
    # Returning results of current time step for i++
        print("Returning results for the next time iteration")

        rho1[:, :] = rho2
        u1[:, :] = u2
        u1[:, :] = u2
        v1[:, :] = v2
        e1[:, :] = e2
        p1[:, :] = p2
        T1[:, :] = T2
        Ut2[:, :] = Ut1
    # DELETE R=0 Point/Column
    # The 3 index indicates matrices with no r=0, deleted column..
        print("Deleting the r=0 for plotting and saving purposes")
        rho3, u3, v3, u3, e3, T3, p3 = delete_r0_point(
            rho1, u1, v1, u1, e1, T1, p1)

    # SAVING DATA
        save_data(i, dt, rho3, u3, v3, Ut3, e3, p3)

    # PLOTTING FIELDS
        # if i > 10:
        plot_imshow(p3, u3, T3, rho3, e3)
        # # animate
        #         anim = animation.FuncAnimation(
        #             fig, animate_func(p3, ux3, T3, rho3, e3), frames=20, interval=1000 / 20,)  # in ms
        # plt.show()
        # anim.save('test_anim.mp4', fps=20,
        #   extra_args=['-vcodec', 'libx264'])


if __name__ == "__main__":
    # main_cal(rho1, ux1, ur1, T1, e1, Tw1, Ts1, Tc1, de0, rho2, ux2,
    #          ur2, T2, e2, Tw2, Ts2, Tc2, de1, T3)

    main_cal(p1, rho1, T1, u1, v1, u1, e1, p2, rho2, T2, u2, v2,
             Ut2, e2, p3, rho3, T3, u3, v3, Ut3, e3, Pe1, Pe2, p_in, u_in, v_in, rho_in, T_in, e_in)


# END OF PROGRAM

# Plotting values after BCs

#         # fig, axs = plt.subplots(2, 2)
# #        print("Radius", R_cyl)
#         r1 = np.linspace(0, R_cyl, Nr+1)  # r = 0 plotted
#         r = np.delete(r1, 0, axis=0)  # r = 0 point removed from array
# #        print("array", r)
#         X = np.linspace(0, L, Nx+1)

# #        print("linspace", R)
#        # print("shape r", np.shape(r))
#         # RADIAL DIRECTION
#         # a = rho1[0,:]
#         b = u3[20, :]
#         c = T3[20, :]
#         # d = Ts1[:]
#         # e = Tw1[0,:]
#         f = p3[20, :]
#         g = ur3[20, :]

    # AXIAL DIRECTION
    # a = rho3[:,Nr]
    # b = u3[:, Nr]
    # c = T1[:, Nr]
    # d = Ts1[:]
    # e = Tw1[:]
    # f = p3[:, Nr]
    # g= de1[:]
    # h= de0[:]


# ----------------------- start plot radius ---------------------------------------- #
    """
    # NOTE:
            fig, axs = plt.subplots(4)
            fig.tight_layout()
            fig.suptitle('Properties along radial axis @ m=20')
            axs[0].scatter(r, b, label="Velocity", color='red')
            axs[0].set(ylabel='U [m/s]')
            # plt.ylabel("Velocity [m/s]")
            axs[1].scatter(r, c, label="Temperature", color='blue')
            axs[1].set(ylabel='Temperature [K]')
            # plt.ylabel("Temperature [K]")
            axs[2].scatter(r, f, label="Pressure", color='green')
            axs[2].set(ylabel='Pressure [Pa]')
            # plt.ylabel("Pressure [Pa]")
            axs[3].scatter(r, g, label="Ur", color='yellow')
            axs[3].set(ylabel='Ur [m/s]')
            plt.xlabel("radius (m)")
            plt.show()

    """

# end plot radius
    # plt.figure()
    # plt.subplot(210)
    # plt.scatter(r, b, label="Velocity", color='red')
    # plt.title("Velocity - Radial axis")
    # plt.xlabel("radius (m)")
    # plt.ylabel("Velocity [m/s]")
    # # ax.set_xlabel("Radius (r)", fontsize=14)
    # # ax.set_ylabel("Velocity",
    # #       color="black",
    # #       fontsize=14)

    # plt.subplot(211)
    # plt.scatter(r, c, label="Temperature", color='blue')
    # plt.subplot(212)
    # plt.title("Tg- Radial axis")
    # plt.xlabel("radius (m)")
    # plt.ylabel("Temperature [K]")

    # plt.scatter(r, f, label="Pressure", color='green')
    # plt.title("Pressure - Radial axis")
    # plt.xlabel("radius (m)")
    # plt.ylabel("P [Pa]")

    # plt.scatter(r,d)
    # plt.scatter(r,e)
 #       plt.scatter(r,f)
    # plt.plot(r,b)
    # plt.plot(r,c)
    # plt.plot(r,d)
    # plt.plot(r,e)
#        plt.plot(X,g)
    # plt.plot(X,h)
    # plt.plot(X,b)
    # plt.title("Axial velocity along X axis")
    # plt.xlabel("Length (m)")
    # plt.ylabel("Ux (m/s)")

#        plt.ylim((0, 0.05))   # set the ylim to bottom, top
    # axs[0, 0].scatter(r, a)
    # axs[0, 0].set_title('density along R')
    # axs[0, 1].plot(r, b, 'tab:orange')
    # axs[0, 1].set_title('Velocity along R')
    # axs[1, 0].plot(r, c, 'tab:green')
    # axs[1, 0].set_title('Tg along R')
    # axs[1, 1].plot(X, d, 'tab:red')
    # axs[1, 1].set_title('Ts along R')

    # axs[0, 0].scatter(X, a)
    # axs[0, 0].set_title('density along R')
    # axs[0, 1].plot(X, b, 'tab:orange')
    # axs[0, 1].set_title('Velocity along R')
    # axs[1, 0].plot(X, c, 'tab:green')
    # axs[1, 0].set_title('Tg along R')
    # axs[1, 1].plot(X, d, 'tab:red')
    # axs[1, 1].set_title('Ts along R')
#        plt.title("Pressure along inlet in the r-direction")
 #       plt.legend()


# define global tx to save in worksheets.

#        tx = t


# VTK CONVERSION - not working - not important
    # vtk_convert(rho3, ux3, ur3, u3, e3, T3, Tw2, Ts2, de0, p3, de1, Pe3)
    # numpyToVTK(rho3)
    # numpyToVTK(ux3)
    # numpyToVTK(ur3)
