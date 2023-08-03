
from my_constants import *
from functions import *
# import numpy
from numpy import asarray, savetxt

# Calculate initial values
# T_0, rho_0, p_0, e_0, ux_0 = bulk_values(T_s)
T_0, rho_0, p_0, e_0, ux_0 = bulk_test1(T_s)

# Array initialization
p1, rho1, u1, v1, Ut1, e1, T1, rho2, u2, v2, Ut2, e2, T2, p2, Tw1, Tw2, Ts1, Ts2, Tc1, Tc2, de0, de1, qhe, q_dep, rho3, u3, v3, Ut3, e3, T3, p3, Pe1, Pe2 = initialize_grid(
    p_0, rho_0, e_0, T_0, T_s)

print("initialization fields")
plot_imshow(p1, u1, T1, rho1, e1)

# constant inlet
p_in, u_in, v_in, rho_in, e_in, T_in = val_in_constant(p_0, T_0, u_0)
Ut_in = np.sqrt(u_in**2 + v_in**2)

# print("val in constant applied")
# plot_imshow(p1, u1, T1, rho1, e1)

# setting wall and frost layer initial conditions
# p_in, q_in, ux_in, ur_in, rho_in, e_in, T_in = val_in(0)
print("p_in: ", p_in, "u_in: ", u_in, "v_in: ", v_in,
      "rho_in: ", rho_in, "e_in: ", e_in, "T_in: ", T_in)

# PREPPING AREA - smoothing
p1, rho1, T1, u1, Ut1, e1 = smoothing_inlet(
    p1, rho1, T1, e1, u1, v1, u_in, Ut1, p_in, p_0, rho_in, rho_0, n_trans)

print("Plotting smoothing inlet initialization")
plot_imshow(p1, u1, T1, rho1, e1)


# print("smoothing initially applied")
# plot_imshow(p1, u1, T1, rho1, e1)

# PARABOLIC VELOCITY PROFILE - inlet prepping area
# ux, u = parabolic_velocity(ux1, ux_in, T1)

# remove timestepping folder
print("Deleting old data")
remove_timestepping()

# Define reference

# p_dim = p_in
# u_dim = u_in
# v_dim = v_in
# Ut_dim = u_in

# rho_dim = rho_in
# e_dim = e_in
# T_dim = T_in

# Ts_dim = 4.2
# de_dim = rho_dim * Ut_dim
# # NOTE: Check

# dx_dim = dx * D_hyd
# dr_dim = dr * D_hyd
# dt_dim = dt * (D_hyd / Ut_dim)


# m_out_dim = rho_dim * Ut_dim ** 2.
# # Convert dimensionless
# mu_dim = mu_n(T_dim, p_dim)


def dim_inf():
    # Inlet used as reference
    cp = 1.04  # Reference 300K KJ/Kg.K
    k_inf = 25.97  # 300K mW/m K
    u_inf = u_dim
    c_inf = np.sqrt(gamma_n*R*T_dim)
    mu_inf = mu_n(T_dim, p_dim)
    Re_inf = rho_dim * Ut_dim*D_hyd / mu_inf
    M_inf = u_inf / c_inf
    Pr_inf = mu_inf * cp * k_inf
    return Re_inf, M_inf, Pr_inf


# Re_inf, M_inf, Pr_inf = dim_inf()


# Save to csv
# data = asarray([[p_dim, u_dim, v_dim, Ut_dim, rho_dim, e_dim, T_dim,
#                  Ts_dim, de_dim, m_out_dim, dx_dim, dr_dim, dt_dim, Re_inf, M_inf, Pr_inf]])

# print(data)
# with open("C:/Users/rababqjt/Documents/programming/git-repos/2d-vacuumbreak-explicit-V1-func-calc/reference.csv", "w") as f:
#     savetxt(f, data, delimiter=',')

# CONTINUED IN MAIN
