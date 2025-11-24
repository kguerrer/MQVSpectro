import numpy as np
import cupy as cp
from ddGPE import field_creation_functions as fc
from ddGPE import ddgpe2d
from ddGPE.analysis_functions import polariton_fields
import physical_cst as cte
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

cp.cuda.Device(0).use()

font = 11

matplotlib.rcParams["figure.figsize"] = [1.5, 1.5]
matplotlib.rcParams["axes.labelsize"] = font
matplotlib.rcParams["axes.titlesize"] = font
matplotlib.rc("xtick", labelsize=font)
matplotlib.rc("ytick", labelsize=font)
matplotlib.rc("legend", fontsize=font)
matplotlib.rc("lines", linewidth=1)
plt.rc("lines", markersize=3)

params = {"text.usetex": True, "font.size": 11, "font.family": "lmodern"}

plt.rcParams.update(params)


def save_raw_data(folder, parameters):

    # Save experimets parameters
    with open(folder + "/parameters.txt", "w") as f:
        f.write("[parameters]\n")
        f.write("folder: " + folder)
        f.write("\n")
        f.write("\n".join("{}: {}".format(x[0], x[1]) for x in parameters))

    # Import from simu arrays to be saved
    mean_cav_t_x_y = simu.mean_cav_t_x_y
    mean_exc_t_x_y = simu.mean_exc_t_x_y
    mean_den_reservoir_t_x_y = simu.mean_den_reservoir_t_x_y
    stationary_cav_x_y = simu.mean_cav_x_y_stat
    stationary_exc_x_y = simu.mean_exc_x_y_stat
    stationary_den_reservoir_x_y = simu.mean_den_reservoir_x_y_stat
    hopfield_coefs = simu.hopfield_coefs
    F_t = simu.F_t
    F_pump_r = simu.F_pump_r

    # Save data as numpy arrays
    cp.save(folder + "/raw_arrays/mean_cav_t_x_y", mean_cav_t_x_y)
    cp.save(folder + "/raw_arrays/mean_exc_t_x_y", mean_exc_t_x_y)
    cp.save(folder + "/raw_arrays/mean_den_reservoir_t_x_y", mean_den_reservoir_t_x_y)
    if stationary_cav_x_y is not None:
        cp.save(folder + "/raw_arrays/stationary_cav_x_y", stationary_cav_x_y)
    if stationary_exc_x_y is not None:
        cp.save(folder + "/raw_arrays/stationary_exc_x_y", stationary_exc_x_y)
    if stationary_den_reservoir_x_y is not None:
        cp.save(
            folder + "/raw_arrays/stationary_den_reservoir_x_y",
            stationary_den_reservoir_x_y,
        )
    cp.save(folder + "/raw_arrays/hopfield_coefs", hopfield_coefs)
    cp.save(folder + "/raw_arrays/F_t", F_t)
    cp.save(folder + "/raw_arrays/F_pump_r", F_pump_r)


# Time parameters
cst = 4
t_min = 0  # (ps) initial time of evolution
t_max = 2000  # (ps) final time of evolution
t_stationary = 990
t_noise = 1e9  # (ps) time from which the noise starts
t_probe = 1e9  # (ps) time from which the probe starts
t_obs = 0  # (ps) time from which the observation starts
dt_frame = 1 / (
    0.1
)  # cst/delta_E avec delta_E la résolution/omega_max en énergie en meV // delta_E fixes the window you will see without aliasing in frequencies, delta_E*2pi/2 = nyquist frequency
n_frame = int((t_max - t_obs) / dt_frame) + 1
print("dt_frame is %s" % (dt_frame))
print("n_frame is %s" % (n_frame))

# Laser parameters
hdetuning = 0.13  # (meV) detuning between the pump and the LP energy
F_pump = 1
C = 4  # vortex charge
ppump = 0.0
rad_pump = 75  # (µm) pump radius
rad_crop = 1e9  # (µm) pump radius
inner_waist = 0  # (µm) inner waist of the vortex
F_probe = 0  # (meV) probe amplitude
apply_reservoir = False  # (bool) True if you want to apply the reservoir
name = "gaussian"

try:
    parser = argparse.ArgumentParser(description="Vortex steady State parameters")
    parser.add_argument(
        "--hdetuning",
        type=float,
        default=hdetuning,
        help="detuning between the pump and the LP energy",
    )
    parser.add_argument("--F_pump", type=float, default=F_pump, help="pump amplitude")
    parser.add_argument("--C", type=int, default=C, help="vortex charge")
    parser.add_argument("--ppump", type=float, default=ppump, help="pump phase")
    parser.add_argument("--rad_pump", type=float, default=rad_pump, help="pump radius")
    parser.add_argument("--name", type=str, default=name, help="name of the simulation")
    args, unknown = parser.parse_known_args()
    print("args are %s" % (args))
    hdetuning = args.hdetuning
    F_pump = args.F_pump
    C = args.C
    ppump = args.ppump
    rad_pump = args.rad_pump
    name = args.name
except Exception as e:
    print(e)

detuning = (
    hdetuning / cte.h_bar
)  # (meV/hbar) detuning between the pump and the LP energy

# Grid parameters
Lx, Ly = 256, 256
Nx, Ny = 512, 512

if (Lx / Nx) ** 2 < cte.g0 / cte.gamma_cav or (Ly / Ny) ** 2 < cte.g0 / cte.gamma_cav:
    print("WARNING: TWA NOT VALID")

# Opening directories for saving data and simulation parameters: -------------------------

folder_DATA = os.getcwd() + "/simulations_vortex_bistability"

string_name = "_%s_reservoir_vortex_on_tophat_real_C%s_ppump%s_F%s_hdet%s_N%s" % (
    name,
    str(C),
    str(ppump),
    str(F_pump),
    str(hdetuning),
    str(Nx),
)

try:
    os.mkdir(folder_DATA)
except:
    print("folder already created")

folder_DATA += "/data_set" + string_name
print("/data_set" + string_name)

try:
    os.mkdir(folder_DATA)
except:
    print("folder already created")

try:
    os.mkdir(folder_DATA + "/raw_arrays")
except:
    print("folder already created")


# Load class with the simulation parameters
simu = ddgpe2d.ggpe(
    cte.omega_exc,
    cte.omega_cav,
    cte.gamma_exc,
    cte.gamma_cav,
    cte.gamma_res,
    apply_reservoir,
    cte.g0,
    cte.rabi,
    cte.k_z,
    detuning,
    F_pump,
    F_probe,
    cst,
    t_max,
    t_stationary,
    t_obs,
    dt_frame,
    t_noise,
    Lx,
    Ly,
    Nx,
    Ny,
)

simu.v_gamma = simu.v_gamma * 0
simu.pump_spatial_profile = fc.gaussian(simu.F_pump_r, simu.R, radius=rad_pump)
simu.pump_spatial_profile = fc.vortex_beam(
    simu.F_pump_r, simu.R, simu.THETA, inner_waist=inner_waist, C=C
)

# Apply a sharp mask in Fourier space to simu.F_pump_r (cp array)
F_pump_r_fft = cp.fft.fftshift(cp.fft.fft2(simu.F_pump_r))
# Define a circular mask in k-space
kx = 2 * np.pi * cp.fft.fftshift(cp.fft.fftfreq(Nx, d=Lx / Nx))
ky = 2 * np.pi * cp.fft.fftshift(cp.fft.fftfreq(Ny, d=Ly / Ny))
KX, KY = cp.meshgrid(kx, ky, indexing="ij")
K = cp.sqrt(KX**2 + KY**2)
k_cutoff = 1.5  # adjust cutoff as needed (in 1/um)
mask = K < k_cutoff
F_pump_r_fft_filtered = F_pump_r_fft * mask
simu.F_pump_r = cp.fft.ifft2(cp.fft.ifftshift(F_pump_r_fft_filtered))

# Plot the amplitude and phase of the filtered pump field after Fourier mask
F_pump_r_filtered = simu.F_pump_r.get()
fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
zoom = 80

# Amplitude
im0 = ax[0].imshow(
    np.abs(F_pump_r_filtered),
    cmap="gray",
    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
)
ax[0].set_xlabel(r"$x~(\mu \mathrm{m})$")
ax[0].set_ylabel(r"$y~(\mu \mathrm{m})$")
ax[0].set_xlim(-zoom, zoom)
ax[0].set_ylim(-zoom, zoom)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im0, cax=cax0, orientation="vertical")

# Phase
im1 = ax[1].imshow(
    np.angle(F_pump_r_filtered),
    cmap="twilight_shifted",
    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
    interpolation="none",
)
ax[1].set_xlim(-zoom, zoom)
ax[1].set_ylim(-zoom, zoom)
ax[1].set_yticklabels([])
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1, orientation="vertical")
plt.tight_layout()
#
simu.pump_temporal_profile = fc.to_turning_point(
    simu.F_pump_t, simu.time, t_up=100, t_down=100
)

omega_probe = 0

# Run the simulation and save the raw data: ----------------------------------------------
parameters = [
    ("h_bar", cte.h_bar),
    ("h_bar_SI", cte.h_bar_SI),
    ("c", cte.c),
    ("eV_to_J", cte.eV_to_J),
    ("n_cav", cte.n_cav),
    ("omega_exc (div by hbar)", cte.omega_exc),
    ("omega_cav (div by hbar)", cte.omega_cav),
    ("gamma_exc (div by hbar)", cte.gamma_exc),
    ("gamma_cav (div by hbar)", cte.gamma_cav),
    ("g0 (div by hbar)", cte.g0),
    ("rabi", cte.rabi),
    ("k_z", cte.k_z),
    ("detuning (div by hbar)", detuning),
    ("F_pump", F_pump),
    ("F_probe", F_probe),
    ("t_min", t_min),
    ("t_max", t_max),
    ("t_stationary", t_stationary),
    ("t_obs", t_obs),
    ("dt_frame", dt_frame),
    ("t_noise", t_noise),
    ("t_probe", t_probe),
    ("Nx", Nx),
    ("Ny", Ny),
    ("Lx", Lx),
    ("Ly", Ly),
    ("omega_probe", omega_probe),
    ("Pump_spatial_profile", simu.pump_spatial_profile),
    ("Pump_temporal_profile", simu.pump_temporal_profile),
    ("Probe_spatial_profile", simu.probe_spatial_profile),
    ("Probe_temporal_profile", simu.probe_temporal_profile),
    ("Potential_profile", simu.potential_profile),
]
print("dt is %s" % (simu.dt))
print("omega_max is %s" % (1 / (simu.cst * simu.dt)))
simu.evolution()
save_raw_data(folder_DATA, parameters)
# Analysis pump and mean field

psi = simu.mean_cav_t_x_y[-2, :, :].get()
F = simu.F_pump_r[:, :].get()


fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
zoom = 80
psi_renorm = psi / np.max(np.abs(psi))  # Renormalize amplitude

# Plot magnitude
im0 = ax[0].imshow(
    np.abs(psi_renorm),
    cmap="gray",
    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
)
ax[0].set_xlabel(r"$x~(\mu \mathrm{m})$")
ax[0].set_ylabel(r"$y~(\mu \mathrm{m})$")
ax[0].set_xlim(-zoom, zoom)
ax[0].set_ylim(-zoom, zoom)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
cbar0 = fig.colorbar(im0, cax=cax0, orientation="vertical")
cbar0.set_ticks([0, 1])
cbar0.set_ticklabels([r"$0$", r"$1$"])

# Plot phase
im1 = ax[1].imshow(
    np.angle(psi_renorm),
    cmap="twilight_shifted",
    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
    interpolation="none",
)
ax[1].set_xlim(-zoom, zoom)
ax[1].set_ylim(-zoom, zoom)
ax[1].set_yticklabels([])
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im1, cax=cax1, orientation="vertical")
cbar.set_ticks([-np.pi, 0, np.pi])
cbar.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
plt.subplots_adjust(left=0.15, right=0.9, top=1, bottom=0.1)
plt.savefig(
    folder_DATA + "/mean_field.pdf",
    dpi=300,
)

# same for the pump
F_renorm = F / np.max(np.abs(F))  # Renormalize amplitude
fig, ax = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
# Plot magnitude
im0 = ax[0].imshow(
    np.abs(F_renorm),
    cmap="gray",
    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
)
ax[0].set_xlabel(r"$x~(\mu \mathrm{m})$")
ax[0].set_ylabel(r"$y~(\mu \mathrm{m})$")
ax[0].set_xlim(-zoom, zoom)
ax[0].set_ylim(-zoom, zoom)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
cbar0 = fig.colorbar(im0, cax=cax0, orientation="vertical")
cbar0.set_ticks([0, 1])
cbar0.set_ticklabels([r"$0$", r"$1$"])
# Plot phase
im1 = ax[1].imshow(
    np.angle(F_renorm),
    cmap="twilight_shifted",
    extent=(-Lx / 2, Lx / 2, -Ly / 2, Ly / 2),
    interpolation="none",
)
ax[1].set_xlim(-zoom, zoom)
ax[1].set_ylim(-zoom, zoom)
ax[1].set_yticklabels([])
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im1, cax=cax1, orientation="vertical")
cbar.set_ticks([-np.pi, 0, np.pi])
cbar.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
plt.subplots_adjust(left=0.15, right=0.9, top=1, bottom=0.1)
plt.savefig(
    folder_DATA + "/pump_field.pdf",
    dpi=300,
)
# Bistability
string_name = "_%s_reservoir_vortex_on_tophat_C%s_ppump%s_F%s_hdet%s_N%s" % (
    name,
    str(C),
    str(ppump),
    str(F_pump),
    str(hdetuning),
    str(Nx),
)

try:
    os.mkdir(folder_DATA)
except:
    print("folder already created")

folder_DATA += "/data_set" + string_name
print("/data_set" + string_name)

try:
    os.mkdir(folder_DATA)
except:
    print("folder already created")

try:
    os.mkdir(folder_DATA + "/raw_arrays")
except:
    print("folder already created")

t_max = 3500
# Load class with the simulation parameters
simu = ddgpe2d.ggpe(
    cte.omega_exc,
    cte.omega_cav,
    cte.gamma_exc,
    cte.gamma_cav,
    cte.gamma_res,
    apply_reservoir,
    cte.g0,
    cte.rabi,
    cte.k_z,
    detuning,
    F_pump,
    F_probe,
    cst,
    t_max,
    t_stationary,
    t_obs,
    dt_frame,
    t_noise,
    Lx,
    Ly,
    Nx,
    Ny,
)

simu.v_gamma = simu.v_gamma * 0
simu.pump_spatial_profile = fc.gaussian(simu.F_pump_r, simu.R, radius=rad_pump)
simu.pump_spatial_profile = fc.vortex_beam(
    simu.F_pump_r, simu.R, simu.THETA, inner_waist=inner_waist, C=C
)

simu.pump_temporal_profile = fc.bistab_cycle(simu.F_pump_t, time=simu.time, t_max=t_max)

simu.F_pump_t = 0.4 * simu.F_pump_t  # reduce the pump amplitude to avoid saturation

omega_probe = 0

# Run the simulation and save the raw data: ----------------------------------------------
parameters = [
    ("h_bar", cte.h_bar),
    ("h_bar_SI", cte.h_bar_SI),
    ("c", cte.c),
    ("eV_to_J", cte.eV_to_J),
    ("n_cav", cte.n_cav),
    ("omega_exc (div by hbar)", cte.omega_exc),
    ("omega_cav (div by hbar)", cte.omega_cav),
    ("gamma_exc (div by hbar)", cte.gamma_exc),
    ("gamma_cav (div by hbar)", cte.gamma_cav),
    ("g0 (div by hbar)", cte.g0),
    ("rabi", cte.rabi),
    ("k_z", cte.k_z),
    ("detuning (div by hbar)", detuning),
    ("F_pump", F_pump),
    ("F_probe", F_probe),
    ("t_min", t_min),
    ("t_max", t_max),
    ("t_stationary", t_stationary),
    ("t_obs", t_obs),
    ("dt_frame", dt_frame),
    ("t_noise", t_noise),
    ("t_probe", t_probe),
    ("Nx", Nx),
    ("Ny", Ny),
    ("Lx", Lx),
    ("Ly", Ly),
    ("omega_probe", omega_probe),
    ("Pump_spatial_profile", simu.pump_spatial_profile),
    ("Pump_temporal_profile", simu.pump_temporal_profile),
    ("Probe_spatial_profile", simu.probe_spatial_profile),
    ("Probe_temporal_profile", simu.probe_temporal_profile),
    ("Potential_profile", simu.potential_profile),
]
print("dt is %s" % (simu.dt))
print("omega_max is %s" % (1 / (simu.cst * simu.dt)))
simu.evolution()
save_raw_data(folder_DATA, parameters)
# Analysis bistability

psi = (
    simu.mean_cav_t_x_y[:, :, :].get() * simu.hopfield_coefs[0, 0, 0].get()
)  # Get the cavity field

psi = polariton_fields(
    simu.mean_cav_t_x_y, simu.mean_exc_t_x_y, simu.hopfield_coefs, only_LP=True
)[0]
psi = psi.get()  # Convert to numpy array
F = simu.F_t.get()

sum_psi_2 = np.sum(np.abs(psi) ** 2, axis=(-1, -2)) / np.max(np.sum(psi, axis=(-1, -2)))
sum_F_2 = np.abs(F) ** 2

fig, ax = plt.subplots(3, 1, figsize=(6, 12))

# Plot cavity field
ax[0].plot(sum_psi_2, label="Cavity field")
ax[0].set_title("Cavity Field")
ax[0].legend()

# Plot pump amplitude
ax[1].plot(sum_F_2, label="Pump amplitude")
ax[1].set_title("Pump Amplitude")
ax[1].legend()

# Plot cavity field against pump amplitude
ax[2].plot(sum_F_2, sum_psi_2, label="Cavity field vs Pump amplitude")
ax[2].set_title("Cavity Field vs Pump Amplitude")
ax[2].set_xlabel("Pump Amplitude")
ax[2].set_ylabel("Cavity Field")
ax[2].legend()

plt.tight_layout()
# Define radial grid
dr = 25  # radial step size

radii = np.array([25])

# Create radial masks
X, Y = np.meshgrid(np.linspace(-Lx / 2, Lx / 2, Nx), np.linspace(-Ly / 2, Ly / 2, Ny))
R = np.sqrt(X**2 + Y**2)

# Initialize arrays to store results
sum_psi_2_radii = np.zeros((len(radii), psi.shape[0]))
sum_F_2_radii = np.zeros((len(radii), simu.F_t.shape[0]))

# Compute sum_psi_2 and sum_F_2 for each radius
for i, r in enumerate(radii):
    mask = (R >= r - dr) & (R < r + dr)
    sum_psi_2_radii[i, :] = np.mean(np.abs(psi[:, mask]) ** 2, axis=-1)
    sum_F_2_radii[i, :] = (
        np.mean(np.abs(simu.F_pump_r.get()[mask])) * np.abs(simu.F_t.get())
    ) ** 2

# Plot results
fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
for i, r in enumerate(radii):
    ax.plot(sum_F_2_radii[i, :], sum_psi_2_radii[i, :], label=f"r={r:.1f}" + r"$\mu m$")
ax.set_xlabel(r"$|F_{\mathrm{p}}(r)|^2(\mathrm{a.u})$")
ax.set_ylabel(r"$n(r)(\mathrm{a.u})$")
plt.tight_layout()
plt.savefig(folder_DATA + "/radius_analysis_bistability.pdf", dpi=300)

m = 0.32
hbar = 0.654


def compute_Ftp_squared(n, delta, k, g, gamma, g_r=0, n_r=0):
    """
    Compute |F_tp|^2 for a given set of parameters.

    Parameters:
        n (float or np.ndarray): polariton density
        delta (float): pump-cavity detuning
        k (float): in-plane wavevector
        g (float): interaction constant for polaritons
        gamma (float): decay rate
        g_r (float): interaction constant with reservoir (default: 0)
        n_r (float): reservoir density (default: 0)

    Returns:
        float or np.ndarray: |F_tp|^2
    """
    delta_v = delta - hbar * k**2 / (2 * m)
    real_part = -hbar * delta_v + hbar * g_r * n_r + hbar * g * n
    imag_part = (hbar * gamma) / 2
    return n * (real_part**2 + imag_part**2) / 0.65727115**2


# Example values
g = cte.g0 * simu.X02.get() ** 2  # set your value
gamma = 0.07  # set your value
delta = 0.13
k = 0

N = np.linspace(0, 103, 200)
I = [compute_Ftp_squared(n=n, delta=delta, k=k, g=g, gamma=gamma) for n in N]

# select N points where dN/dI is negative
N = np.array(N)
I = np.array(I)

negative_slope_indices = np.where((np.diff(N) / np.diff(I)) < 0)[0]
N_neg = N[negative_slope_indices]
I_neg = I[negative_slope_indices]

N_pos = np.delete(N, negative_slope_indices)
I_pos = np.delete(I, negative_slope_indices)

fig, ax = plt.subplots(figsize=(2, 2), dpi=300)
ax.plot(I_pos, N_pos, label=r"$|F_{\mathrm{tp}}|^2$ (Positive Slope)", color="blue")
ax.plot(I_neg, N_neg, label=r"$|F_{\mathrm{tp}}|^2$ (Negative Slope)", color="red")
ax.set_ylabel(r"$n(\mathrm{a.u})$")
ax.set_xlabel(r"$|F_{\mathrm{p}}|^2(\mathrm{a.u})$")
plt.tight_layout()
plt.savefig(
    folder_DATA + "/analytical_bistability_curve.pdf",
    dpi=300,
)
