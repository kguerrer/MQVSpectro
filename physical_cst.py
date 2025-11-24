# Physical constants
h_bar = 0.654  # (meV*ps)
c = 2.9979 * 1e2  # (um/ps)
eV_to_J = 1.60218 * 1e-19
h_bar_SI = 1.05457182 * 1e-34

# Microcavity parameters
rabi = 5.07 / 2 / h_bar  # (meV/h_bar) linear coupling (Rabi split)
g0 = (
    1e-2
) / h_bar  # (frequency/density) (meV/hbar)/(1/um^2) nonlinear coupling constant
gamma_exc, gamma_cav = (
    0.0015 / h_bar,
    0.07 / 0.6571 / h_bar,
)  # (meV/h_bar) exc and ph linewidth 1microeV 69microeV  original value 0.07/h_bar
gamma_res = 0.0015 / h_bar  # (meV/h_bar) reservoir linewidth
omega_exc = (
    1484.44 / h_bar
)  # (meV/h_bar) exciton energy measured from the cavity energy #-0.5
omega_cav = (
    1482.76 / h_bar
)  # (meV/h_bar) cavity energy at k=0  original value: 1482.76 /h_bar
n_cav = 3.54
k_z = 27  # (1/µm) n_cav*omega_cav/c
