# extra functions
@jit(nopython=True)
def dt2nd_wall(m, Tw1, T_in):

    if m == 0:
        dt2nd = (T_in - 2 * Tw1[m] +
                 Tw1[m+1])/(dx**2)  # 3-point CD
#       dt2nd = Tw1[m+1]-Tw1[m]-Tw1[m-1]+T_in
    elif m == Nx:
        # print("m=Nx", m)
        dt2nd = (-Tw1[m-3] + 4*Tw1[m-2] - 5*Tw1[m-1] +
                 2*Tw1[m]) / (dx**2)  # Four point BWD
    else:
        dt2nd = Tw1[m-1]-2*Tw1[m]+Tw1[m+1]/(dx**2)
    return dt2nd

# @jit(nopython=True)

################
# def initialize_ghost():
#     # This is for the ghost cells for the WENO reconstruction.

#     return ro_rec, ux_rec,

    # x-direction
    if (iflx==1)
        qLx,qRx = weno5(nx,ny,nz,q,1)
    elseif (iflx==2)
        qLx,qRx = upwind5(nx,ny,nz,q,1)
    end
    FLx = flux(nx,ny,nz,qLx,1)
    FRx = flux(nx,ny,nz,qRx,1)
    Fx  = rusanov_3d(q,qLx,FLx,qRx,FRx,nx,ny,nz,1)

    #y-direction
    if (iflx==1)
        qLy,qRy = weno5(nx,ny,nz,q,2)
    elseif (iflx==2)
        qLy,qRy = upwind5(nx,ny,nz,q,2)
    end
    FLy = flux(nx,ny,nz,qLy,2)
    FRy = flux(nx,ny,nz,qRy,2)
    Fy  = rusanov_3d(q,qLy,FLy,qRy,FRy,nx,ny,nz,2)

def rhs_rho(m, n, dr, dx, ur, ux, rho, a, ux_in, rho_in):

    NOTE: PLOTTING GRADIENTS

    # Plot gradients along tube.
            abb = [dp_dx, ux_dx, ur_dx, grad_x, dt2x_ux,
                   dt2r_ux, ux_dr, ur_dr, dt2x_ur, dt2r_ur]
            # plot_gradients(abb)

            save_gradients(n, Rks, dt, d_dr, m_dx, dp_dx, ux_dx, ux_dr,
                           dp_dr, ur_dx, ur_dr, grad_x, grad_r)


# adaptive timestep
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


def vtk_convert(rho3, ux3, ur3, u3, e3, T3, Tw3, Ts2, de0, p3, de1, Pe3):

def numpyToVTK(data):
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=flat_data_array, deep=True, array_type=data_type)

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    return img


@jit(nopython=True)
def grad_rho2(m, n, ux_in, rho_in, ur, ux, rho):
    # if m == 0:
    #     a = rho_in
    #     m_dx = (rho[m, n]*ux[m, n]-rho_in*ux_in)/dx

    if m == 1:
        m_dx = (rho[m, n]*ux[m, n]-rho_in*ux_in)/dx

    elif m == Nx:
        m_dx = (rho[m, n]*ux[m, n]-rho[m-1, n]*ux[m-1, n])/dx

    # elif (m <= n_trans+2 and m >= n_trans+2):
    #     # NOTE Use four point CD at transition point.
    #     a = rho1[m, n]
    #     m_dx = (rho1[m-2, n] - 8*rho1[m-1, n] + 8 *
    #             rho1[m+1, n] - rho1[m+2, n])/(12*dx)

    else:
        m_dx = (rho[m, n]*ux[m, n]-rho[m-1, n]*ux[m-1, n])/(dx)

    if n == 1:
        # NOTE: SYMMETRY BC
        d_dr = (rho[m, n+2]*(n+2)*dr*ur[m, n+2] -
                rho[m, n] * n*dr*ur[m, n]) / (4*dr)

    else:
        d_dr = (rho[m, n]*n*dr*ur[m, n] -
                rho[m, n-1] * (n-1)*dr*ur[m, n-1])/dr

    # else:
    #     d_dr = (rho[m, n+1]*(n+1)*dr*ur[m, n+1] -
    #             rho[m, n-1] * (n-1)*dr*ur[m, n-1])/(2*dr)

    return d_dr, m_dx


# @jit(nopython=True)
def grad_rho_matrix(ux_in, rho_in, ur, ux, rho):
    # create gradients arrays.
    m_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    d_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for i in np.arange(Nx+1):
        for j in np.arange(1, Nr+1):
            if i == 0:
                m_dx[i, j] = (rho[i+1, j]*ux[i+1, j]-rho[i, j]*ux[i, j])/dx
            # if i == 1:
            #     m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i, j]*ux[i-1, j])/dx

            # elif i == Nx:
            #     m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]*ux[i-1, j])/dx

            else:
                m_dx[i, j] = (rho[i, j]*ux[i, j]-rho[i-1, j]*ux[i-1, j])/(dx)

            if j == 1:
                # NOTE: SYMMETRY BC
                d_dr[i, j] = (rho[i, j+2]*(j+2)*dr*ur[i, j+2] -
                              rho[i, j] * j*dr*ur[i, j]) / (4*dr)

            # elif j == Nr-1 or j == Nr:
            #     d_dr[i, j] = (rho[i, j]*j*dr*ur[i, j] -
            #                   rho[i, j-1] * (j-1)*dr*ur[i, j-1])/dr

            # else:
            #     d_dr[i, j] = (rho[i, j+1]*(i+1)*dr*ur[i, j+1] -
            #                   rho[i, j-1] * (j-1)*dr*ur[i, j-1])/(2*dr)
            else:
                d_dr[i, j] = (rho[i, j]*j*dr*ur[i, j] -
                              rho[i, j-1] * (j-1)*dr*ur[i, j-1])/dr
    return d_dr, m_dx


@numba.jit('f8(f8,f8,f8,f8,f8,f8)')

@jit(nopython=True)
def grad_ux2(p_in, p, ux_in, ux, m, n):  # bulk

    if n == 1:
        # NOTE: SYMMETRY CONDITION HERE done
        ux_dr = (ux[m, n+2] - ux[m, n])/(4*dr)

    elif n == Nr-1:
        ux_dr = (ux[m, n] - ux[m, n-1])/dr  # BWD

    else:
        # upwind 1st order  - positive flow - advection
        ux_dr = (ux[m, n] - ux[m, n-1])/(dr)  # CD

    # if m == 0:
    #     # upwind 1st order  - positive flow - advection
    #     dp_dx = (p[m, n] - p_in)/dx
    #     ux_dx = (ux[m, n] - ux_in)/dx
        # 4-point CD
        # dp_dx = (p_in - 8*p_in + 8 *
        #          p1[m+1, n] - p1[m+2, n])/(12*dx)
        # ux_dx = (ux1[m+1, n] - ux_in)/(2*dx)

    # elif (m <= n_trans+2 and m >= n_trans-2):
    #     # NOTE Use four point CD at transition point.
    #     dp_dx = (p1[m-2, n] - 8*p1[m-1, n] + 8 *
    #              p1[m+1, n] - p1[m+2, n])/(12*dx)
    #     ux_dx = (ux1[m-2, n] - 8*ux1[m-1, n] + 8 *
    #              ux1[m+1, n] - ux1[m+2, n])/(12*dx)

    if m == 0:
        dp_dx = (p[m+1, n] - p[m, n])/dx  # BWD
        ux_dx = (ux[m+1, n] - ux[m, n])/dx  # BWD

    elif m == 1:
        dp_dx = (p[m, n] - p[m-1, n])/dx  # BWD
        ux_dx = (ux[m, n] - ux[m-1, n])/dx  # BWD
        # dp_dx = (p[m, n] - p_in)/dx  # BWD
        # ux_dx = (ux[m, n] - ux_in)/dx  # BWD

    elif m == Nx:
        dp_dx = (p[m, n] - p[m-1, n])/dx  # BWD
        ux_dx = (ux[m, n] - ux[m-1, n])/dx  # BWD

    # elif (m >= 1 and m <= Nx - 2):

    else:
        # upwind 1st order  - positive flow - advection
        dp_dx = (p[m, n] - p[m-1, n])/dx
        ux_dx = (ux[m, n] - ux[m-1, n])/dx
        # dp_dx = (3*p1[m, n] - 4*p1[m-1, n] + p1[m-2, n]) / \
        #     dx  # # upwind, captures shocks
        # ux_dx = (3*ux1[m, n] - 4*ux1[m-1, n] + ux1[m-2, n]) / \
        #     dx  # # upwind, captures shocks

    # else:
    #     dp_dx = (p[m+1, n] - p[m-1, n])/(2*dx)
    #     ux_dx = (ux[m+1, n] - ux[m-1, n])/(2*dx)

    return dp_dx, ux_dx, ux_dr


# @jit(nopython=True)
def grad_ux2_matrix(p_in, p, ux_in, ux):  # bulk
    ux_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dp_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ux_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for m in np.arange(Nx+1):
        for n in np.arange(1, Nr+1):
            if n == 1:
                # NOTE: SYMMETRY CONDITION HERE done
                ux_dr[m, n] = (ux[m, n+2] - ux[m, n])/(4*dr)

            # elif n == Nr-1 or n == Nr:
            #     ux_dr[m, n] = (ux[m, n] - ux[m, n-1])/dr  # BWD

            else:
                # upwind 1st order  - positive flow - advection
                ux_dr[m, n] = (ux[m, n] - ux[m, n-1])/(dr)  # CD

            if m == 0:
                dp_dx[m, n] = (p[m+1, n] - p[m, n])/dx  # BWD
                ux_dx[m, n] = (ux[m+1, n] - ux[m, n])/dx  # BWD

            # elif m == Nx:
            #     dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx  # BWD
            #     ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx  # BWD

            else:
                # upwind 1st order  - positive flow - advection
                dp_dx[m, n] = (p[m, n] - p[m-1, n])/dx
                ux_dx[m, n] = (ux[m, n] - ux[m-1, n])/dx

    return dp_dx, ux_dx, ux_dr

@numba.jit('f8(f8,f8,f8,f8,f8)')


@jit(nopython=True)
def grad_ur2(m, n, p, ur, ur_in):  # first derivatives BULK

    if n == 1:
        # NOTE: Symmetry BC done
        dp_dr = (p[m, n+2] - p[m, n])/(4*dr)
        ur_dr = (ur[m, n+2]-ur[m, n])/(4*dr)  # increased to 2dx

# n == Nr-1

    else:
        dp_dr = (p[m, n] - p[m, n-1])/dr  # BWD
        ur_dr = (ur[m, n] - ur[m, n-1])/dr

    # elif (n != 1 and n != Nr-1):
    #     dp_dr = (p[m, n+1] - p[m, n-1])/(2*dr)  # CD
    #     ur_dr = (ur[m, n+1] - ur[m, n-1])/(2*dr)

    # if m == 0:
    #     ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

    if m == 0:
        ur_dx = (ur[m+1, n] - ur[m, n])/(dx)  # upwind 1st order

    # if m == 1:
    #     ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

    # elif (m <= n_trans+2 and m >= n_trans-2):
    #     ur_dx = (ur1[m-2, n] - 8*ur1[m-1, n] + 8 *
    #              ur1[m+1, n] - ur1[m+2, n])/(12*dx)  # 4 point CD

    elif m == Nx:
        ur_dx = (ur[m, n] - ur[m-1, n])/dx

    elif (m > 1 and m <= Nx - 2):
        # upwind 1st order  - positive flow - advection
        ur_dx = (ur[m, n] - ur[m-1, n])/dx

    else:
        # upwind 1st order  - positive flow - advection
        ur_dx = (ur[m, n] - ur[m-1, n])/(dx)  # CD

    return dp_dr, ur_dx, ur_dr


# @jit(nopython=True)
def grad_ur2_matrix(p, ur, ur_in):  # first derivatives BULK
    dp_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ur_dr = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    ur_dx = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for m in np.arange(Nx+1):
        for n in np.arange(1, Nr+1):

            if n == 1:
                # NOTE: Symmetry BC done
                dp_dr[m, n] = (p[m, n+2] - p[m, n])/(4*dr)
                ur_dr[m, n] = (ur[m, n+2]-ur[m, n])/(4*dr)  # increased to 2dx

        # n == Nr-1

            else:
                dp_dr[m, n] = (p[m, n] - p[m, n-1])/dr  # BWD
                ur_dr[m, n] = (ur[m, n] - ur[m, n-1])/dr

            # elif (n != 1 and n != Nr-1):
            #     dp_dr = (p[m, n+1] - p[m, n-1])/(2*dr)  # CD
            #     ur_dr = (ur[m, n+1] - ur[m, n-1])/(2*dr)

            # if m == 0:
            #     ur_dx = (ur[m+1, n] - ur_in)/(dx)  # upwind 1st order

            if m == 1:
                ur_dx[m, n] = (ur[m+1, n] - ur[m, n])/(dx)  # upwind 1st order

            # elif (m <= n_trans+2 and m >= n_trans-2):
            #     ur_dx = (ur1[m-2, n] - 8*ur1[m-1, n] + 8 *
            #              ur1[m+1, n] - ur1[m+2, n])/(12*dx)  # 4 point CD

            # elif m == Nx:
            #     ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/dx

            # elif (m > 1 and m <= Nx - 2):
            #     # upwind 1st order  - positive flow - advection
            #     ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/dx

            else:
                # upwind 1st order  - positive flow - advection
                ur_dx[m, n] = (ur[m, n] - ur[m-1, n])/(dx)  # CD

            return dp_dr, ur_dx, ur_dr


# @numba.jit('f8(f8,f8,f8,f8,f8,f8,f8)')
@jit(nopython=True)
def grad_e2(m, n, ur1, ux1, ux_in, e_in, e1):     # use upwind for Pe > 2

    if n == 1:
        # NOTE: Symmetry BC done
        grad_r = ((n+2)*dr*ur1[m, n+2]*e1[m, n+2] - n *
                  dr*ur1[m, n]*e1[m, n])/(4*dr)  # ur=0 @ r=0 #CD

    elif n == Nr:
        grad_r = (n*dr*ur1[m, n]*e1[m, n] -
                  (n-1)*dr*ur1[m, n-1]*e1[m, n-1])/dr  # BWD

    # We dont need the surface case, this is the bulk...

# n == Nr-1:
    else:
        grad_r = ((n)*dr*ur1[m, n]*e1[m, n] - (n-1)
                  * dr*ur1[m, n-1]*e1[m, n-1])/(dr)  # BWD

    if m == 0:
        grad_x = (e1[m+1, n]*ux1[m+1, n]-e1[m, n]*ux1[m, n])/(dx)

    if m == Nx:
        # print("e1[m, n]*ux1[m, n]: ", e1[m, n]*ux1[m, n],
        #       "-e1[m-1, n]*ux1[m-1, n]: ", -e1[m-1, n]*ux1[m-1, n])
        grad_x = (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])/dx  # BWD

    # elif (m <= n_trans+2 and m >= n_trans-2):
    #     grad_x = (e1[m-2, n]*ux1[m-2, n] - 8*e1[m-1, n]*ux1[m-1, n] + 8 *
    #               e1[m+1, n]*ux1[m+1, n] - e1[m+2, n]*ux1[m+2, n])/(12*dx)
    elif (m >= 1 and m <= Nx - 2):
        # upwind 1st order  - positive flow - advection
        grad_x = (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])/dx
        # grad_x = 3*(e1[m, n]*ux1[m, n]) - 4*(e1[m-1, n]
        #                                      * ux1[m-1, n]) + (e1[m-2, n]
        #                                                        * ux1[m-2, n]) / dx  # upwind, captures shocks
    else:  # 0 < m < Nx,  1 < n < Nr
        grad_x = (e1[m, n]*ux1[m, n]-e1[m-1, n]
                  * ux1[m-1, n])/dx  # upwind

    return grad_x, grad_r


# @jit(nopython=True)
def grad_e2_matrix(ur1, ux1, ux_in, e_in, e1):     # use upwind for Pe > 2
    grad_r = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    grad_x = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))

    for m in np.arange(Nx+1):
        for n in np.arange(1, Nr+1):
            # We dont need the surface case, this is the bulk...

            if n == 1:
                # NOTE: Symmetry BC done
                grad_r[m, n] = ((n+2)*dr*ur1[m, n+2]*e1[m, n+2] - n *
                                dr*ur1[m, n]*e1[m, n])/(4*dr)  # ur=0 @ r=0 #CD

            # surface case
            # elif n == Nr:
            #     grad_r[m, n] = (n*dr*ur1[m, n]*e1[m, n] -
            #                     (n-1)*dr*ur1[m, n-1]*e1[m, n-1])/dr  # BWD

        # n == Nr-1:
            else:
                grad_r[m, n] = ((n)*dr*ur1[m, n]*e1[m, n] - (n-1)
                                * dr*ur1[m, n-1]*e1[m, n-1])/(dr)  # BWD

            # if m == 0:
            #     grad_x = (e1[m+1, n]*ux1[m+1, n]-e_in*ux_in)/(dx)
            if m == 0:
                # print("e1[m, n]*ux1[m, n]: ", e1[m, n]*ux1[m, n],
                #       "-e1[m-1, n]*ux1[m-1, n]: ", -e1[m-1, n]*ux1[m-1, n])
                grad_x[m, n] = (e1[m+1, n]*ux1[m+1, n]-e1[m, n]
                                * ux1[m, n])/dx  # BWD

            elif m == Nx:
                # print("e1[m, n]*ux1[m, n]: ", e1[m, n]*ux1[m, n],
                #       "-e1[m-1, n]*ux1[m-1, n]: ", -e1[m-1, n]*ux1[m-1, n])
                grad_x[m, n] = (e1[m, n]*ux1[m, n]-e1[m-1, n]
                                * ux1[m-1, n])/dx  # BWD

            # elif (m <= n_trans+2 and m >= n_trans-2):
            #     grad_x = (e1[m-2, n]*ux1[m-2, n] - 8*e1[m-1, n]*ux1[m-1, n] + 8 *
            #               e1[m+1, n]*ux1[m+1, n] - e1[m+2, n]*ux1[m+2, n])/(12*dx)
            elif (m >= 1 and m <= Nx - 2):
                # upwind 1st order  - positive flow - advection
                grad_x[m, n] = (e1[m, n]*ux1[m, n]-e1[m-1, n]*ux1[m-1, n])/dx
                # grad_x = 3*(e1[m, n]*ux1[m, n]) - 4*(e1[m-1, n]
                #                                      * ux1[m-1, n]) + (e1[m-2, n]
                #                                                        * ux1[m-2, n]) / dx  # upwind, captures shocks
            else:  # 0 < m < Nx,  1 < n < Nr
                grad_x[m, n] = (e1[m, n]*ux1[m, n]-e1[m-1, n]
                                * ux1[m-1, n])/dx  # upwind

    return grad_x, grad_r


# @numba.jit('f8(f8,f8,f8,f8)')
# @jit(nopython=True)
def dt2nd_radial(ux1, ur1, m, n):
    if n == 1:
        # NOTE: Symmetry Boundary Condition assumed for ur1 radial derivative along x axis..
        # --------------------------- dt2nd radial ux1 ---------------------------------#
        dt2nd_radial_ux1 = (ux1[m, n+2] - ux1[m, n]) / (4*dr**2)

        # --------------------------- dt2nd radial ur1 ---------------------------------#
        dt2nd_radial_ur1 = (ur1[m, n+2] - ur1[m, n]) / (4*dr**2)

        # print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
        # print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

    else:  # (n is between 1 and Nr)

        # --------------------------- dt2nd radial ux1 ---------------------------------#
        dt2nd_radial_ux1 = (ux1[m, n+1] + ux1[m, n-1] -
                            2*ux1[m, n])/(dr**2)  # CD
    # --------------------------- dt2nd radial ur1 ---------------------------------#
        dt2nd_radial_ur1 = (ur1[m, n+1] + ur1[m, n-1] -
                            2*ur1[m, n])/(dr**2)  # CD
        # print("dt2nd_radial_ur1:", dt2nd_radial_ur1)
    return dt2nd_radial_ux1, dt2nd_radial_ur1


def save_dt2x_matrix(array1, array2):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/second_gradients_surface/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("dt2x_ux1.csv", array1, delimiter=",")
    np.savetxt("dt2x_ur1.csv", array2, delimiter=",")
    return


def save_dt2r_matrix(array1, array2):
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/second_gradients_surface/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("dt2r_ux1.csv", array1, delimiter=",")
    np.savetxt("dt2r_ur1.csv", array2, delimiter=",")
    return


# @jit(nopython=True)


def dt2r_matrix(ux1, ur1):
    dt2r_ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dt2r_ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):

            if n == 1:
                # NOTE: Symmetry Boundary Condition assumed for ur1 radial derivative along x axis..
                # --------------------------- dt2nd radial ux1 ---------------------------------#
                dt2r_ux1[m, n] = (ux1[m, n+2] - ux1[m, n]) / (4*dr**2)

                # --------------------------- dt2nd radial ur1 ---------------------------------#
                dt2r_ur1[m, n] = (ur1[m, n+2] - ur1[m, n]) / (4*dr**2)

                # print("dt2nd_radial_ux1_n1:", dt2nd_radial_ux1)
                # print("dt2nd_radial_ur1_n1:", dt2nd_radial_ur1)

            elif n == Nr:
                # --------------------------- dt2nd radial ux1 ---------------------------------#
                # NOTE: CHECK
                dt2r_ux1[m, n] = (2*ux1[m, n] - 5*ux1[m, n-1] +
                                  4*ux1[m, n-2] - ux1[m, n-3]) / (dr**2)

                # --------------------------- dt2nd radial ur1 ---------------------------------#
                dt2r_ur1[m, n] = (2*ur1[m, n] - 5*ur1[m, n-1] +
                                  4*ur1[m, n-2] - ur1[m, n-3]) / (dr**2)
            else:  # (n is between 1 and Nr)

                # --------------------------- dt2nd radial ux1 ---------------------------------#
                dt2r_ux1[m, n] = (ux1[m, n+1] + ux1[m, n-1] -
                                  2*ux1[m, n])/(dr**2)  # CD
            # --------------------------- dt2nd radial ur1 ---------------------------------#
                dt2r_ur1[m, n] = (ur1[m, n+1] + ur1[m, n-1] -
                                  2*ur1[m, n])/(dr**2)  # CD
    save_dt2r_matrix(dt2r_ux1, dt2r_ur1)
    return dt2r_ux1, dt2r_ur1

# @numba.jit('f8(f8,f8,f8,f8,f8,f8)')


@jit(nopython=True)
def dt2nd_axial(ux_in, ur_in, ux1, ur1, m, n):
    if m == 0:
        # --------------------------- dt2nd axial ux1 ---------------------------------#
        dt2nd_axial_ux1 = (ux_in - 2*ux1[m, n] + ux1[m+1, n]) / (dx**2)
        # dt2nd_axial_ux1 = (ux1[m+2,n] -2*ux1[m+1,n] + ux1[m,n])/(dx**2) #FWD

    # --------------------------- dt2nd axial ur1 ---------------------------------#
        #                        dt2nd_axial_ur1 = (ur1[m+2,n] -2*ur1[m+1,n] + ur1[m,n])/(dx**2) #FWD
        # FWD
        dt2nd_axial_ur1 = (-ur_in + ur_in - 30 *
                           ur1[m, n] + 16*ur1[m+1, n] - ur1[m+2, n])/(12*dx**2)
        # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)
 #                        dt2nd_axial_ur1 = (2*ur1[m,n] - 5*ur1[m+1,n] + 4*ur1[m+2,n] -ur1[m+3,n])/(dx**3)  # FWD

    elif m == Nx:
        # --------------------------- dt2nd axial ux1 ---------------------------------#

        dt2nd_axial_ux1 = (ux1[m-2, n] - 2*ux1[m-1, n] +
                           ux1[m, n])/(dx**2)  # BWD
    # dt2nd_axial_ux1 = (2*ux1[m,n] - 5*ux1[m-1,n] + 4*ux1[m-2,n] -ux1[m-3,n])/(dx**3) # BWD
        # --------------------------- dt2nd axial ur1 ---------------------------------#
    # Three-point BWD
        dt2nd_axial_ur1 = (ur1[m-2, n] - 2*ur1[m-1, n] + ur1[m, n])/(dx**2)
        # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

    else:
        # --------------------------- dt2nd axial ux1 ---------------------------------#
        dt2nd_axial_ux1 = (ux1[m+1, n] + ux1[m-1, n] -
                           2*ux1[m, n])/(dx**2)  # CD

    # --------------------------- dt2nd axial ur1 ---------------------------------#
        dt2nd_axial_ur1 = (ur1[m+1, n] + ur1[m-1, n] -
                           2*ur1[m, n])/(dx**2)  # CD
        # print("dt2nd_axial_ur1:", dt2nd_axial_ur1)

    return dt2nd_axial_ux1, dt2nd_axial_ur1

#
# @jit(nopython=True)


def dt2x_matrix(ux_in, ur_in, ux1, ur1):
    dt2x_ux1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    dt2x_ur1 = np.zeros((Nx+1, Nr+1), dtype=(np.float64, np.float64))
    for m in np.arange(Nx+1):
        for n in np.arange(Nr+1):
            if m == 0:
                # --------------------------- dt2nd axial ux1 ---------------------------------#
                dt2x_ux1[m, n] = (ux_in - 2*ux1[m, n] + ux1[m+1, n]) / (dx**2)

            # --------------------------- dt2nd axial ur1 ---------------------------------#
                dt2x_ur1[m, n] = (-ur_in + ur_in - 30 *
                                  ur1[m, n] + 16*ur1[m+1, n] - ur1[m+2, n])/(12*dx**2)

            elif m == Nx:
                # --------------------------- dt2nd axial ux1 ---------------------------------#

                dt2x_ux1[m, n] = (ux1[m-2, n] - 2*ux1[m-1, n] +
                                  ux1[m, n])/(dx**2)  # BWD
                # --------------------------- dt2nd axial ur1 ---------------------------------#
            # Three-point BWD
                dt2x_ur1[m, n] = (
                    ur1[m-2, n] - 2*ur1[m-1, n] + ur1[m, n])/(dx**2)

            else:
                # --------------------------- dt2nd axial ux1 ---------------------------------#
                dt2x_ux1[m, n] = (ux1[m+1, n] + ux1[m-1, n] -
                                  2*ux1[m, n])/(dx**2)  # CD

            # --------------------------- dt2nd axial ur1 ---------------------------------#
                dt2x_ur1[m, n] = (ur1[m+1, n] + ur1[m-1, n] -
                                  2*ur1[m, n])/(dx**2)  # CD
    save_dt2x_matrix(dt2x_ux1, dt2x_ur1)
    return dt2x_ux1, dt2x_ur1


def save_qhe(tx, dt, qhe):
    incrementx = (tx+1)*dt
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(incrementx) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("qhe.csv", qhe, delimiter=",")


def save_visc(tx, dt, array):
    increment = (tx+1)*dt
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(increment) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("visc.csv", array, delimiter=",")


def save_qdep(tx, dt, qdep):
    incrementy = (tx+1)*dt
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(incrementy) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("q_dep1.csv", qdep, delimiter=",")


def save_mdot():
    pathname = 'C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/timestepping/' + \
        "{:.4f}".format(increment) + '/'
    newpath = pathname
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(pathname)
    np.savetxt("de1.csv", de1, delimiter=",")
    return

@numba.jit('f8(f8)')


@jit(nopython=True)
def val_in_bulk_constant_T_P():
    #   Calculate instant flow rate (kg/s)
    T_0, rho_0, p_0, e_0, ux_0 = bulk_values()
    p_in = p_0
    T_in = T_0
    rho_in = p_in / T_in/R*M_n
    ux_in = 30.
    ur_in = 0.
    e_in = 5./2.*rho_in/M_n*R*T_in + 1./2.*rho_in*ux_in**2
    e_in_x = e_in
    out = np.array([p_in, ux_in, ur_in, rho_in, e_in, e_in_x, T_in])
    return out


@jit(nopython=True)
def surface_BC(ux, ur, p):
    ux[:, Nr] = 0
    ux, ur , p = balance_energy_p(ux, ur ,p)
    return ux


def m_de(T, P, Ts1, de, dm, dm_r, ur, N):
    p_0 = bulk_values(T_s)[2]
    # print("mdot calc: ", "Tg: ", T, " P: ",
    #       P, "Ts: ", T_s, "de: ", de, "dm: ", dm)
    #   Calculate deposition rate (kg/(m^2*s))
    if T == 0:
        T = 0.00001
    rho = P*M_n/R/T
# no division by zero
    if rho == 0:
        rho = 0.00001
    v_m1 = np.sqrt(2*R*T/M_n)  # thermal velocity of molecules
    u_mean1 = de/rho  # mean flow velocity towards the wall.
    beta = u_mean1/v_m1  # this is Beta from Hertz Knudson
    gam1 = gamma(beta)  # deviation from Maxwellian velocity.
    P_s = f_ps(Ts1)

    if P > P_s and P > p_0:
        # Correlated Hertz-Knudsen Relation #####
        m_out = np.sqrt(M_n/2/np.pi/R)*Sc_PP * \
            (gam1*P/np.sqrt(T)-P_s/np.sqrt(Ts1))
        print("m_out calc", m_out)

        if Ts1 > 25:
            print("P>P0, P>Ps")
            # Arbitrary smooth the transition to steady deposition
            # NOTE: Check this smoothing function.
            m_out = m_out*exp_smooth(Ts1-25., 1., 0.05,
                                     0.03, (f_ts(P*np.sqrt(Ts1/T))-25.)/2.)

        # Speed of sound limit for the condensation flux
        rho_min = p_0*M_n/R/T
        # sqrt(7./5.*R*T/M_n)*rho
        # Used Conti in X-direction, since its absolute flux.
        m_max[:] = D/4./dt*(rho[:, Nr-1]-rho_min)-D/4./dx*dm - D/4. * \
            1/N[:, Nr-1]/dr*(rho[:, Nr-1]*N[:, Nr-1]*dr*ur[:,
                             Nr-1] - rho[:, Nr-2]*N[:, Nr-2]*dr*ur[:, Nr-2])

        # using conti surface
        # m_max = D/4./dt*(rho-rho_min)-D/4. * (1/Nr/dr*dm_r)
        print("m_max", m_max, "dm", dm)
        if m_out > m_max:
            m_out = m_max
            # print("mout = mmax")
    else:
        m_out = 0
    rho_min = p_0*M_n/R/T
    # m_out = 0  # NO HEAT TRANSFER/ MASS DEPOSITION CASE
    # print("de2: ", m_out)
    return m_out  # Output: mass deposition flux, no convective heat flux


def delete_surface_inviscid(rho, u, v, Ut, e, tg, p, pe):
    rho = np.delete(rho, Nr-1, axis=1)
    u = np.delete(u, Nr-1, axis=1)
    v = np.delete(v, Nr-1, axis=1)
    Ut = np.delete(Ut, Nr-1, axis=1)
    e = np.delete(e, Nr-1, axis=1)
    T = np.delete(tg, Nr-1, axis=1)
    p = np.delete(p, Nr-1, axis=1)
    pe = np.delete(pe, Nr-1, axis=1)
    return [rho, u, v, Ut, e, tg, p, pe]


def namestr(obj, namespace):  # Returns variable name for check_negative function
    return [name for name in namespace if namespace[name] is obj]


def check_array(array_in):
    if np.any(array_in < 0):
        array_name = namestr(array_in, globals())[0]
        print(array_name + " has at least one negative value.")
        exit()

def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno

# @numba.jit()
def check_negative(var_in, n):  # CHECKS CALCULATIONS FOR NEGATIVE OR NAN VALUES
    # at surface
    if n == Nr:

        if var_in < 0:
            print("negative Surface", var_in)
            exit()
        if math.isnan(var_in):
            print("NAN Surface ", var_in)
            assert not math.isnan(var_in)

    # at BULK
    else:

        if var_in < 0:
            print("negative Bulk ", var_in)
            exit()
        if math.isnan(var_in):
            print("NAN Bulk ", var_in)
            assert not math.isnan(var_in)



# NOTE: ANIMATION

    # fps = 30
    # nSeconds = 5

    # im1 = plt.imshow(p3, interpolation='none')
    # im2 = plt.imshow(ux3, interpolation='none')
    # im3 = plt.imshow(T3, interpolation='none')
    # im4 = plt.imshow(rho3, interpolation='none')
    # im5 = plt.imshow(e3, interpolation='none')

    # def init():
    #     im1.set_data()
    #     im2.set_data()
    #     im3.set_data()
    #     im4.set_data()
    #     im5.set_data()
    #     return im1, im2, im3, im4, im5
    # fig, axs = plt.subplots(5)  # figsize=(18, 18)
    # fig.suptitle('Fields along tube for all R')

    # def animate_func(p, ux, T, rho, e):

    #     # PRESSURE DISTRIBUTION
    #     im1 = axs[0].imshow(p.transpose())
    #     plt.colorbar(im1, ax=axs[0])
    #     # plt.colorbar(im, ax=ax[0])
    #     axs[0].set(ylabel='Pressure [Pa]')
    #     # plt.title("Pressure smoothing")

    #     # VELOCITY DISTRIBUTION
    #     # axs[1].imshow()
    #     im2 = axs[1].imshow(ux.transpose())
    #     plt.colorbar(im2, ax=axs[1])
    #     # axs[1].colorbars(location="bottom")
    #     axs[1].set(ylabel='Ux [m/s]')
    #     # plt.title("velocity parabolic smoothing")

    #     # Temperature DISTRIBUTION
    #     im3 = axs[2].imshow(T.transpose())
    #     plt.colorbar(im3, ax=axs[2])
    #     axs[2].set(ylabel='Tg [K]')

    #     # axs[1].colorbars(location="bottom")
    #     # axs[2].set(ylabel='temperature [K]')

    #     im4 = axs[3].imshow(rho.transpose())
    #     plt.colorbar(im4, ax=axs[3])
    #     axs[3].set(ylabel='Density [kg/m3]')

    #     im5 = axs[4].imshow(e.transpose())
    #     plt.colorbar(im5, ax=axs[4])
    #     axs[4].set(ylabel='energy [kg/m3]')

    #     plt.xlabel("L(x)")

    #     if i % fps == 0:
    #         print('.', end='')
    #     im1.set_array(p3)
    #     im2.set_array(ux3)
    #     im3.set_array(T3)
    #     im4.set_array(rho3)
    #     im5.set_array(e3)
    #     return im1, im2, im3, im4, im5
    # print("rho", rho1)


#rho included, energy density
def balance_energy_T(q, tg, Ut):
    e = 5./2. * q*R*tg/M_n + 1./2 * q*Ut**2
    p = q*R*tg/M_n
    return e, p


# calculate initial gradients matrix:
    print("Calculating initial gradients")
    d_dr, m_dx = grad_rho_matrix(ux_in, rho_in, ur1, ux1, rho1)
    dp_dx, ux_dx, ux_dr = grad_ux2_matrix(p_in, p1, ux_in, ux1)
    dp_dr, ur_dx, ur_dr = grad_ur2_matrix(p1, ur1, ur_in)
    grad_x, grad_r = grad_e2_matrix(ur1, ux1, ux_in, e_in, e1)

    save_gradients(1, 1, 1, d_dr, m_dx, dp_dx, ux_dx, ux_dr,
                   dp_dr, ur_dx, ur_dr, grad_x, grad_r)
