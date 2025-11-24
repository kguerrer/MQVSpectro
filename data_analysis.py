# %% mass and reservoir analysis
import os
from ast import literal_eval

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import polarTransform
from polarTransform import convertToPolarImage
import pyfftw
import scipy.constants as const
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.restoration import unwrap_phase
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Wedge

from contrast import im_osc_fast
from dict_utilities import open_dict_all_str
from polar_projection import reproject_image_into_polar
from velocity import velocity, vortex_detection, az_avg

font = 11

matplotlib.rcParams["figure.figsize"] = [3.45, 1]
matplotlib.rcParams["axes.labelsize"] = font
matplotlib.rcParams["axes.titlesize"] = font
matplotlib.rc("xtick", labelsize=font)
matplotlib.rc("ytick", labelsize=font)
matplotlib.rc("legend", fontsize=font)
matplotlib.rc("lines", linewidth=1)
plt.rc("lines", markersize=8)

params = {"text.usetex": True, "font.size": 11, "font.family": "lmodern"}

plt.rcParams.update(params)

h = const.Planck
c = const.speed_of_light
e = const.elementary_charge
n_air = 1.000273  # air index
hbar = 0.654  # hbar value in meV*ps


def ntm(wvl, n_medium=n_air):
    return h * c / wvl / n_medium / e


def idx(arr, val):
    try:
        c = np.argwhere(arr == val)[0][0]
    except:  # find the closest value
        c = np.argmin(np.abs(arr - val))
    return c


def lorentzian(x, x0, gamma, A):
    return A * gamma**2 / ((x - x0) ** 2 + gamma**2)


def retrieve_probe_field(
    folder,
    date,
    serie,
    m,
    p,
    r,
    dr,
    hw,
    vmax=None,
    roi_far_field=100,
    norm=None,
    plot=True,
    save=False,
    show=False,
    svg=False,
):

    # set default matplotlib params

    p = P[idx(P, p)]
    if show == False:
        plt.switch_backend("agg")
    hdf5_filename = os.path.join(
        folder,
        "Interferograms",
        f"{date}_serie{serie}",
        f"interferograms_m={m}_p={p}_r={r}_dr={dr}.h5",
    )

    # Open the HDF5 file
    with h5py.File(hdf5_filename, "r") as hdf:
        # Retrieve the dataset and load the specific interferogram
        interferograms = hdf["interferograms"]  # Reference to the dataset
        im = interferograms[idx(energies, hw)]  # Load only the specific slice

    field = im_osc_fast(im[:, :], center=(3 * im.shape[-2] // 4, im.shape[-1] // 4))
    amplitude = np.abs(field)
    phase = np.angle(field)
    extent = [
        -field.shape[1] // 2 * pixel_cam,
        field.shape[1] // 2 * pixel_cam,
        -field.shape[0] // 2 * pixel_cam,
        field.shape[0] // 2 * pixel_cam,
    ]
    # phase, amplitude = fast_field(interferogram[:,::-1])
    if plot:
        fig, ax = plt.subplots(1, 3)
        fig.tight_layout(pad=5)
        divider0 = make_axes_locatable(ax[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        divider1 = make_axes_locatable(ax[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        divider2 = make_axes_locatable(ax[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        im0 = ax[0].imshow(
            phase,
            "twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
            extent=extent,
            interpolation="None",
        )
        ax[0].set_xlabel("x (µm)")
        ax[0].set_ylabel("y (µm)")
        ax[1].set_xlabel("x (µm)")
        if vmax == None:
            im1 = ax[1].imshow(amplitude, "inferno", extent=extent)
        else:
            im1 = ax[1].imshow(amplitude, "inferno", extent=extent, vmax=vmax)
        im_fft = pyfftw.interfaces.numpy_fft.rfft2(im)
        ky = (
            2
            * np.pi
            * np.fft.rfftfreq(im.shape[0], pixel_cam)
            * roi_far_field
            / (im.shape[0] / 2)
        )
        kx = (
            2
            * np.pi
            * np.fft.rfftfreq(im.shape[1], pixel_cam)
            * roi_far_field
            / (im.shape[1] / 2)
        )
        extent = [-kx[-1] / 2, kx[-1] / 2, -ky[-1] / 2, ky[-1] / 2]
        far_field = im_fft[
            3 * im_fft.shape[0] // 4
            - roi_far_field // 2 : 3 * im_fft.shape[0] // 4
            + roi_far_field // 2,
            im_fft.shape[1] // 2
            - roi_far_field // 2 : im_fft.shape[1] // 2
            + roi_far_field // 2,
        ]
        im2 = ax[2].imshow(np.abs(far_field), "inferno", norm=norm, extent=extent)
        ax[2].set_xlabel(r"$k_x(µm^{-1})$")
        ax[2].set_ylabel(r"$k_y(µm^{-1})$")
        fig.colorbar(im0, cax=cax0)
        fig.colorbar(im1, cax=cax1)
        fig.colorbar(im2, cax=cax2)
        fig.suptitle(
            "Probe for m = %s, p = %s µm^-1, r = %sµm, dr = %sµm, hw = %s"
            % (str(m), str(p), str(r), str(dr), str(hw)),
            fontsize="x-large",
        )
        if save:
            plt.savefig(
                folder
                + "/Interferograms"
                + "/%s_serie%s" % (date, str(serie))
                + "/field_m=%s_p=%s_r=%s_dr=%s_hw=%s.png"
                % (str(m), str(p), str(r), str(dr), str(hw))
            )
        if svg:
            plt.savefig(
                folder
                + "/Interferograms"
                + "/%s_serie%s" % (date, str(serie))
                + "/field_m=%s_p=%s_r=%s_dr=%s_hw=%s.svg"
                % (str(m), str(p), str(r), str(dr), str(hw)),
                format="svg",
            )
        if show:
            plt.show()
        else:
            plt.close("all")
        plt.close("all")

    return phase, amplitude


def mean_azim(image, dimage=None, std=False):
    """
    Azimuthal averaging with conservative uncertainty propagation.

    Parameters:
    - image : 2D array
        Input image (e.g. intensity).
    - dimage : 2D array or None
        Pixel-wise uncertainty (same shape as image).
        If provided, the output includes the propagated uncertainty per radius.
    - std : bool
        If True, return standard deviation over θ at each radius.

    Returns:
    - mean_profile : 1D array
    - std_profile : 1D array (if std=True)
    - err_profile : 1D array (if dimage is provided)
    """
    image_polar = polarTransform.convertToPolarImage(
        image, center=[512, 512], finalRadius=512, radiusSize=512, angleSize=1024
    )[0].T

    mean_profile = np.mean(image_polar, axis=1)
    outputs = [mean_profile]

    if std or dimage is not None:
        std_profile = np.std(image_polar, axis=1)
        if std:
            outputs.append(std_profile)

    if dimage is not None:
        dimage_polar = polarTransform.convertToPolarImage(
            dimage, center=[512, 512], finalRadius=512, radiusSize=512, angleSize=1024
        )[0].T

        N_theta = image_polar.shape[1]

        # Term 1: sum of uncertainties (not divided)
        sum_dimage2 = np.mean(dimage_polar**2, axis=1)

        # Term 2: statistical scatter / N
        var_image = np.sum((image_polar - mean_profile[:, None]) ** 2, axis=1) / N_theta

        err_profile = np.sqrt(sum_dimage2 + var_image)
        outputs.append(err_profile)

    return tuple(outputs) if len(outputs) > 1 else mean_profile


def Lower_pol(ky, m, E_LP_0, ky_0):
    bogo = E_LP_0 + hbar**2 * (ky - ky_0) ** 2 / 2 / m
    return bogo


def Bogo_sonic(ky, gn, m, E_pump, ky_pump, kx):
    bogo = (
        E_pump
        + hbar**2 * ky_pump * (ky - ky_pump) / m
        + np.real(
            np.emath.sqrt(
                (hbar**2 * (kx**2 + (ky - ky_pump) ** 2) / (2 * m))
                * (hbar**2 * (kx**2 + (ky - ky_pump) ** 2) / (2 * m) + 2 * gn)
            )
        )
    )
    return bogo


def Bogo_r(ky, gn, grnr, m, E_pump, delta, vy, kx):
    bogo = (
        E_pump
        + hbar * vy * ky
        + np.real(
            np.emath.sqrt(
                (hbar**2 * (kx**2 + ky**2) / (2 * m) + 2 * gn + grnr - delta) ** 2
                - ((gn) ** 2)
            )
        )
    )
    return bogo


def Bogo(ky, gn, alpha_sq, m_LP, E_pump, Delta, vy, vx, kx):
    """Compute the Bogolioubov energy for a given mean field and probe mode
    Args:
        ky (_type_): µm^-1 (lab frame)
        gn (_type_): meV
        alpha_sq (_type_): such that grnr = (1-alpha_sq)(gn+grnr)
        grnr = gn * (1 - alpha_sq) / alpha_sq
        m_LP (_type_): meVps^2µm^-2
        E_pump (_type_): meV
        delta (_type_): meV
        vy (_type_): µm/ps (lab frame)
        vx (_type_): µm/ps (lab frame)
        kx (_type_): µm^-1 (lab frame)

    Returns:
        _type_: bogolioubov energy
    """
    kx_pump = m_LP * vx / hbar
    ky_pump = m_LP * vy / hbar
    grnr = gn * (1 - alpha_sq) / alpha_sq
    doppler = hbar * (vy * (ky - ky_pump) + vx * (kx - kx_pump))
    delta = Delta - 0.5 * m_LP * (vy**2 + vx**2)
    sq_root_term = np.real(
        np.emath.sqrt(
            (
                hbar**2 * ((kx - kx_pump) ** 2 + (ky - ky_pump) ** 2) / (2 * m_LP)
                + 2 * gn
                + grnr
                - delta
            )
            ** 2
            - ((gn) ** 2)
        )
    )
    bogo = E_pump + doppler + sq_root_term
    return bogo


def Bogo_2d(kx, ky, gn, alpha_sq, m_LP, E_pump, detuning, vx, vy):
    """Compute the Bogolioubov energy for a given mean field and probe mode
    Args:
        ky (_type_): µm^-1 (lab frame)
        gn (_type_): meV
        alpha_sq (_type_): such that grnr = (1-alpha_sq)(gn+grnr)
        grnr = gn * (1 - alpha_sq) / alpha_sq
        m_LP (_type_): meVps^2µm^-2
        E_pump (_type_): meV
        delta (_type_): meV
        vy (_type_): µm/ps (lab frame)
        vx (_type_): µm/ps (lab frame)
        kx (_type_): µm^-1 (lab frame)

    Returns:
        _type_: bogolioubov energy
    """
    kx_pump = m_LP * vx / hbar
    ky_pump = m_LP * vy / hbar
    grnr = gn * (1 - alpha_sq) / alpha_sq
    doppler = hbar * (vy * (ky - ky_pump) + vx * (kx - kx_pump))
    delta = detuning - 0.5 * m_LP * (vy**2 + vx**2)
    sq_root_term = np.real(
        np.emath.sqrt(
            (
                hbar**2 * ((kx - kx_pump) ** 2 + (ky - ky_pump) ** 2) / (2 * m_LP)
                + 2 * gn
                + grnr
                - delta
            )
            ** 2
            - ((gn) ** 2)
        )
    )
    bogo = E_pump + doppler + sq_root_term
    return bogo


def Bogo_WKB_m(m, r, coeff, alpha_val):
    idx_r = idx(r_axis, r)
    n = np.argmin(np.abs(r_axis - 3))
    gn = coeff * gn_m[idx_r]
    vazim = vazim_m[idx_r]
    vr = vr_m[idx_r]
    return Bogo(
        m / r,
        gn=coeff * gn_m[idx_r],
        alpha_sq=alpha_val,
        m_LP=m_LP,
        E_pump=hw_pump,
        Delta=detuning,
        vy=vazim,
        vx=vr,
        kx=0,
    )


# %% Probe Transmission analysis in linear regime to obtain the mass
folder = os.getcwd() + "/Reservoir"
date = "20250405"
serie = 1

params = open_dict_all_str(
    folder + "/Parameters" + "/params_%s_%s.csv" % (date, str(serie))
)
KY = np.load(folder + "/Parameters" + "/%s_serie%s_KY.npy" % (date, str(serie)))
KX = np.load(folder + "/Parameters" + "/%s_serie%s_KX.npy" % (date, str(serie)))
dKY = KY[1] - KY[0]
R = np.load(folder + "/Parameters" + "/%s_serie%s_R.npy" % (date, str(serie)))
pixel_cam = literal_eval(params["pixel_cam"])
# pixel_slm = literal_eval(params["pixel_slm"])

# wavelength axis definition
wavelength = np.load(
    folder + "/Wavelengths" + "/%s_serie%s_wavelengths_raw.npy" % (date, serie)
)
wavelength_fitted = np.load(
    folder + "/Wavelengths" + "/%s_serie%s_wavelengths_fitted.npy" % (date, serie)
)
wavelength_axis = np.mean(wavelength_fitted, axis=0)
energies = 1e12 * h * c / wavelength_axis / n_air / e

r = R[idx(R, 30)]
kx = KX[idx(KX, 0)]
ky = KY[idx(KY, 0)]

wvl_err_c = 0.005  # systematic error in nm
energy_err = np.mean(1e12 * h * c / (wavelength_axis**2) * wvl_err_c / n_air / e)

dr = 0

transmission = np.load(
    folder + "/Transmission" + "/transmission_20250405_serie1_r=55.0_kx=0.0.npy"
)

E_res = np.zeros(transmission.shape[1])
E_res_err = np.ones(transmission.shape[1])

n_mc = 500  # Number of Monte Carlo samples
x_sigma = 1 * energy_err  # Example uncertainty on the x-axis (energies)

for i, m in enumerate(KY):
    slice_data = transmission[:, i]
    E_res_samples = []

    try:
        for _ in range(n_mc):
            # Perturb x values (energies) according to assumed uncertainty
            energies_perturbed = energies + np.random.normal(
                0, x_sigma, size=energies.shape
            )

            # Fit the Lorentzian to the perturbed x data
            popt, _ = curve_fit(
                lorentzian,
                energies_perturbed,
                slice_data,
                p0=[energies[np.argmax(slice_data)], 0.1, np.max(slice_data)],
            )
            E_res_samples.append(popt[0])  # x0 is the resonance energy
        # Compute mean and std deviation of x0 values
        E_res[i] = np.mean(E_res_samples)
        E_res_err[i] = np.std(E_res_samples)

    except Exception as e:
        print(f"Fit failed at KY[{i}]: {e}")
        E_res[i] = energies[np.argmax(slice_data)]
        E_res_err[i] = np.nan

from scipy.optimize import curve_fit

m = 0.32
E_LP_0 = 1480

p0 = [m, E_LP_0, 0]

ky_max = 0.65
idx_max = idx(KY, ky_max) + 1
idx_min = idx(KY, -ky_max)
KY_fit = KY[idx_min:idx_max]
E_res_fit = E_res[idx_min:idx_max]
E_res_err_fit = E_res_err[idx_min:idx_max]

popt, pcov = curve_fit(
    Lower_pol, KY_fit, E_res_fit, sigma=E_res_err_fit, p0=p0, absolute_sigma=True
)

m_LP = popt[0]
E_LP_0 = popt[1]
ky_0 = popt[2]

dm_LP = np.sqrt(pcov[0][0])
dE_LP_0 = np.sqrt(pcov[1][1])
dky_0 = np.sqrt(pcov[2][2])

# %% Probe Transmission analysis in resting fluid regime to obtain the reservoir
folder = os.getcwd() + "/Reservoir"
date = "20250405"
serie = 2

params = open_dict_all_str(
    folder + "/Parameters" + "/params_%s_%s.csv" % (date, str(serie))
)
KY = np.load(folder + "/Parameters" + "/%s_serie%s_KY.npy" % (date, str(serie)))
KX = np.load(folder + "/Parameters" + "/%s_serie%s_KX.npy" % (date, str(serie)))
dKY = KY[1] - KY[0]
R = np.load(folder + "/Parameters" + "/%s_serie%s_R.npy" % (date, str(serie)))
pixel_cam = literal_eval(params["pixel_cam"])
pump_wvl = literal_eval(params["pump_wvl"])
nb_shot = literal_eval(params["nb_shot"])

# wavelength axis definition
wavelength = np.load(
    folder + "/Wavelengths" + "/%s_serie%s_wavelengths_raw.npy" % (date, serie)
)
wavelength_fitted = np.load(
    folder + "/Wavelengths" + "/%s_serie%s_wavelengths_fitted.npy" % (date, serie)
)
wavelength_axis = np.mean(wavelength_fitted, axis=0)
energies = 1e12 * h * c / wavelength_axis / n_air / e

hw_pump = 1e12 * h * c / pump_wvl / n_air / e
energy_err = np.mean(1e12 * h * c / (wavelength_axis**2) * wvl_err_c / n_air / e)

# Probe parameters
r = 55.0  # diameter
kx = 0.0

transmission = np.load(
    folder + "/Transmission" + "/transmission_20250405_serie2_r=55.0_kx=0.0.npy"
)

E_res = np.zeros(transmission.shape[1])
E_res_err = np.zeros(transmission.shape[1])

n_mc = 500  # Number of Monte Carlo samples
x_sigma = energy_err  # Example uncertainty on the x-axis (energies)

for i, m in enumerate(KY):
    slice_data = transmission[:, i]
    E_res_samples = []

    try:
        for _ in range(n_mc):
            # Perturb x values (energies) according to assumed uncertainty
            energies_perturbed = energies + np.random.normal(
                0, x_sigma, size=energies.shape
            )

            # Fit the Lorentzian to the perturbed x data
            popt, _ = curve_fit(
                lorentzian,
                energies_perturbed,
                slice_data,
                p0=[energies[np.argmax(slice_data)], 0.1, np.max(slice_data)],
            )
            E_res_samples.append(popt[0])  # x0 is the resonance energy

        # Compute mean and std deviation of x0 values
        E_res[i] = np.mean(E_res_samples)
        E_res_err[i] = np.std(E_res_samples)

    except Exception as e:
        print(f"Fit failed at KY[{i}]: {e}")
        E_res[i] = energies[np.argmax(slice_data)]
        E_res_err[i] = np.nan

from scipy.optimize import curve_fit

ky_max = 0.65
ky_min = -ky_max
idx_max = idx(KY, ky_max) + 1
idx_min = idx(KY, ky_min)
KY_fit = KY[idx_min:idx_max]
E_res_fit = E_res[idx_min:idx_max]

detuning = hw_pump - E_LP_0


def Bogo_fit(ky, gn, grnr):
    return Bogo_r(
        ky - ky_0, gn, grnr, m=m_LP, E_pump=hw_pump, delta=detuning, vy=0, kx=0
    )


alpha_sq = 0.5

p0 = [alpha_sq * detuning, (1 - alpha_sq) * detuning]
popt, pcov = curve_fit(
    Bogo_fit,
    KY_fit,
    E_res_fit,
    p0=p0,
    sigma=E_res_err[idx_min:idx_max],
    absolute_sigma=True,
)
perr = np.sqrt(np.diag(pcov))

gn, grnr = popt
dgn, dgrnr = perr

# Include cross-correlation in the error estimation
cov_gn_grnr = pcov[0, 1]
alpha_sq = gn / (gn + grnr)
dalpha_sq = (
    (grnr * dgn) ** 2 + (gn * dgrnr) ** 2 + 2 * gn * grnr * cov_gn_grnr
) ** 0.5 / (gn + grnr) ** 2

print(f"fitted alpha_sq = {alpha_sq:.6f} ± {dalpha_sq:.6f}")
print(f"popt = {popt}")
if (popt.sum() < detuning) or popt.any() < 0:
    print("fit failed on gn + grnr > detuning")
print(f"gn = {gn:.6f} ± {dgn:.6f}")
print(f"grnr = {grnr:.6f} ± {dgrnr:.6f}")


extent = [
    KY[0] - dKY / 2 - ky_0,
    KY[-1] + dKY / 2 - ky_0,
    energies[-1] - hw_pump,
    energies[0] - hw_pump,
]
aspect_ratio = (0.64 + 0.615) / (energies[0] - energies[-1])
plt.figure(figsize=(3, 1.8))
plt.imshow(
    transmission / np.max(transmission),
    aspect=aspect_ratio,
    extent=[
        KY[0] - dKY / 2 - ky_0,
        KY[-1] + dKY / 2 - ky_0,
        energies[-1] - hw_pump,
        energies[0] - hw_pump,
    ],
    cmap="inferno",
)

plt.plot(
    KY_fit - ky_0,
    Bogo_fit(KY_fit, *popt) - hw_pump,
    label="Fit",
    linewidth=1,
    color="gray",
)
plt.errorbar(
    KY - ky_0,
    E_res - hw_pump,
    yerr=E_res_err,
    fmt="o",
    markersize=3,
    color="gray",
    markerfacecolor="lightgray",
    markeredgecolor="gray",
    alpha=1,
)
plt.ylabel(r"$\hbar\omega (\mathrm{meV})$")
plt.xlabel(r"$k (\mu \mathrm{m}^{-1})$")
plt.xlim(-0.64, 0.615)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.getcwd() + "/reservoir_fit.pdf")
plt.close("all")

# %% Vortex transmission analysis

folder = os.getcwd()

date = "20250410"
serie = 3

params = open_dict_all_str(
    folder + "/Parameters" + "/params_%s_%s.csv" % (date, str(serie))
)
M = np.load(folder + "/Parameters" + "/%s_serie%s_M.npy" % (date, str(serie)))
P = np.load(folder + "/Parameters" + "/%s_serie%s_P.npy" % (date, str(serie)))
R = np.load(folder + "/Parameters" + "/%s_serie%s_R.npy" % (date, str(serie)))
DR = np.load(folder + "/Parameters" + "/%s_serie%s_DR.npy" % (date, str(serie)))
pixel_cam = literal_eval(params["pixel_cam"])
C = literal_eval(params["pump_circulation"])
n_shot = literal_eval(params["nb_shot"])

# wavelength axis definition
wavelength = np.load(
    folder + "/Wavelengths" + "/%s_serie%s_wavelengths_raw.npy" % (date, serie)
)
wavelength_fitted = np.load(
    folder + "/Wavelengths" + "/%s_serie%s_wavelengths_fitted.npy" % (date, serie)
)
wavelength_axis = np.mean(wavelength_fitted, axis=0)
energies = 1e12 * h * c / wavelength_axis / n_air / e
energy_axis = ntm(wavelength_axis * 1e-12)
pump_wvl = literal_eval(params["pump_wvl"])
hw_pump = 1e3 * h * c / pump_wvl / n_air / e


# %% (MainFig1(b)(d)) Get mean field maps
def load_tiff_as_numpy(file_path):
    """
    Load a .tiff file and convert it to a numpy array.
    """
    with Image.open(file_path) as img:
        return np.array(img)


interferogram_pump = load_tiff_as_numpy(
    folder
    + "/Interferograms_pump"
    + f"/{date}_serie{serie}"
    + "/interferogram_pump.tiff"
)
field = im_osc_fast(interferogram_pump)
ampl, phase = np.abs(field), np.angle(field)

X, Y = np.meshgrid(
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
)
R_grid = np.sqrt(X**2 + Y**2)
THETA = np.angle(X + 1j * Y)

field = im_osc_fast(interferogram_pump)
field = gaussian_filter(field, 1.5 / pixel_cam)
center = (512 - 520, 512 - 504)
field = np.roll(field, center, axis=(0, 1))

ampl, phase = np.abs(field), np.angle(field)
extent = [-512 * pixel_cam, 512 * pixel_cam, -512 * pixel_cam, 512 * pixel_cam]

fig, ax = plt.subplots(1, 3, figsize=(4, 2))

# Plot interferogram
im_main = ax[0].imshow(
    interferogram_pump / np.max(interferogram_pump), cmap="gray", extent=extent
)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_main, cax=cax0)
ax[0].set_ylabel(r"$y (\mu \mathrm{m})$")
ax[0].set_xlabel(r"$x (\mu \mathrm{m})$")

# Plot amplitude
im_ampl = ax[1].imshow(ampl / np.max(ampl), cmap="gray", extent=extent)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_ampl, cax=cax1)
ax[1].set_xticks([])
ax[1].set_yticks([])

# Plot phase
im_phase = ax[2].imshow(
    phase, cmap="twilight_shifted", interpolation="None", extent=extent
)
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im_phase, cax=cax2, ticks=[-np.pi + 0.001, 0, np.pi - 0.0001])
cbar.ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax[2].set_xticks([])
ax[2].set_yticks([])
plt.tight_layout()
plt.savefig("mean_field_maps_%s_serie%s.pdf" % (date, serie), bbox_inches="tight")
plt.close("all")

# %% Compute velocities
kx = velocity(phase, dx=pixel_cam)[0]
ky = velocity(phase, dx=pixel_cam)[1]

vx, vy = hbar * kx / m_LP, hbar * ky / m_LP

vazim = -vy * np.cos(THETA) + vx * np.sin(THETA)
vr = vx * np.cos(THETA) + vy * np.sin(THETA)

vazim_m = mean_azim(vazim)
vr_m = mean_azim(vr)
ampl_m = mean_azim(ampl)
r_axis = mean_azim(R_grid)
# %% Density calibration:from processed data to get patch value for last m value
r_fit = 36
delta_r = 3
dr = 50
r = 25
p = 0

wvl_err_p = 0.010  # systematic error in nm
energy_err = np.mean(1e12 * h * c / (wavelength_axis**2) * wvl_err_p / n_air / e)

# Grid and radial axis
Nx = Ny = 1024
X, Y = np.meshgrid(
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
)
R_grid = np.sqrt(X**2 + Y**2)
r_axis = mean_azim(R_grid)
theta_size = Nx

# Load θ-resolved transmission (shape: θ, m, energy)
transmission_theta_w_m = np.load(
    folder
    + "/Transmission"
    + "/transmission_avg_theta_w_m_%s_serie%s_r=%s_dr=%s_p=%s.npy"
    % (date, serie, str(r_fit), str(dr), str(p))
)

theta_size, M_size, E_size = transmission_theta_w_m.shape
M_fit = np.array(M)
m_over_r = M_fit / r_fit

vr_polar = convertToPolarImage(
    vr,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=theta_size,
)[0].T

vazim_polar = convertToPolarImage(
    vazim,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=theta_size,
)[0].T

r_idx_min = idx(r_axis, r_fit - delta_r)
r_idx_max = idx(r_axis, r_fit + delta_r)
vr_theta = np.mean(vr_polar[r_idx_min:r_idx_max, :], axis=0)
vazim_theta = np.mean(vazim_polar[r_idx_min:r_idx_max, :], axis=0)

E_res_all = np.zeros((theta_size, M_size)) + np.nan
E_res_err_all = np.zeros((theta_size, M_size)) + np.nan
for m_idx in tqdm(range(M_size), desc="Lorentzian fits"):
    for theta_idx in range(theta_size):
        spectrum = transmission_theta_w_m[theta_idx, m_idx, :]
        try:
            popt, pcov = curve_fit(
                lorentzian,
                energies,
                spectrum,
                p0=[energies[np.argmax(spectrum)], 0.1, np.max(spectrum)],
                maxfev=10000,
            )
            E_res_all[theta_idx, m_idx] = popt[0]
            E_res_err_all[theta_idx, m_idx] = np.sqrt(pcov[0, 0])
        except Exception:
            pass  # leave as NaN

detuning = hw_pump - E_LP_0
gn_list = []
gn_err_list = []

for theta_idx in tqdm(range(theta_size), desc="Bogo fits"):
    E_res_theta = E_res_all[theta_idx, :]
    E_res_err_theta = E_res_err_all[theta_idx, :]
    if np.any(np.isnan(E_res_theta)):
        continue

    def Bogo_fit_theta(m_over_r, gn):
        return Bogo(
            ky=m_over_r,
            gn=gn,
            alpha_sq=alpha_sq,
            m_LP=m_LP,
            E_pump=hw_pump,
            Delta=detuning,
            vy=vazim_theta[theta_idx],
            vx=vr_theta[theta_idx],
            kx=0,
        )

    try:
        popt, pcov = curve_fit(
            Bogo_fit_theta,
            m_over_r,
            E_res_theta,
            sigma=E_res_err_theta,
            absolute_sigma=True,
        )
        gn_list.append(popt[0])
        gn_err_list.append(np.sqrt(pcov[0, 0]))
    except Exception:
        continue


gn_array = np.array(gn_list)
gn_err_array = np.array(gn_err_list)

gn_mean = np.mean(gn_array)
gn_std = np.std(gn_array)

# -- PLOT --
# Evaluate Bogo curve
fit_curve = Bogo(
    ky=m_over_r,
    gn=gn_mean,
    alpha_sq=alpha_sq,
    m_LP=m_LP,
    E_pump=hw_pump,
    Delta=detuning,
    vy=np.mean(vazim_theta),
    vx=np.mean(vr_theta),
    kx=0,
)

# Derivative for 1σ band
delta = 1e-6
fit_plus = Bogo(
    m_over_r,
    gn_mean + delta,
    alpha_sq,
    m_LP,
    hw_pump,
    detuning,
    np.mean(vazim_theta),
    np.mean(vr_theta),
    kx=0,
)
fit_minus = Bogo(
    m_over_r,
    gn_mean - delta,
    alpha_sq,
    m_LP,
    hw_pump,
    detuning,
    np.mean(vazim_theta),
    np.mean(vr_theta),
    kx=0,
)
df_dgn = (fit_plus - fit_minus) / (2 * delta)
fit_err = np.abs(df_dgn) * gn_std

E_res_avg = np.nanmean(E_res_all, axis=0)
E_res_err_avg = np.nanstd(E_res_all, axis=0)

# take last m value
E_res_patch = E_res_avg[-1]
E_res_err_patch = E_res_err_avg[-1]

plt.figure(figsize=(3, 3))
# Normalized data (column-wise)
data = np.mean(transmission_theta_w_m, axis=0).T
data_norm = data / np.max(data, axis=0, keepdims=True)

im = plt.imshow(
    data_norm,
    extent=[
        (M[0] - C - 0.5),
        (M[-1] - C + 0.5),
        energies[-1] - hw_pump,
        energies[0] - hw_pump,
    ],
    aspect=(M[-1] - M[0]) / (energies[0] - energies[-1]),
    cmap="inferno",
    origin="upper",
)

# Add transmission fit curve and experimental points
plt.plot(
    M_fit - C, fit_curve - hw_pump, label=r"Bogo Fit (local $\theta$)", color="gray"
)
plt.errorbar(
    M_fit - C,
    E_res_avg - hw_pump,
    yerr=E_res_err_avg,
    fmt="o",
    markersize=3,
    color="gray",
    markerfacecolor="lightgray",
    markeredgecolor="gray",
    alpha=1,
)

# Labels and ticks (small for tight figure)
plt.xlabel(r"$m$")
plt.ylabel(r"$\hbar\omega\ (\mathrm{meV})$")

plt.colorbar()
plt.tight_layout()
plt.savefig(
    f"transmission_fit_localflow_r={r_fit}_%s_serie%s.pdf" % (date, serie), dpi=300
)

plt.close("all")


# %% (SMFig4) Density calibration from raw data with patch for last m value (compute transmission)

r_fit = 36

dr = 50
r = 25
p = 0

# Setup grid and mask parameters
Nx, Ny = 1024, 1024
X, Y = np.meshgrid(
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
)
R_grid = np.sqrt(X**2 + Y**2)
r_axis = mean_azim(R_grid)
M_list = M
theta_size = Nx
transmission_theta_w_m = np.zeros((theta_size, len(M_list), len(energies)))

# Prepare transmission arrays per r_mask
nb_shot_for_plot = 80
shot_indices = np.arange(0, n_shot, n_shot // nb_shot_for_plot)

# Compute extent and aspect

for m_idx, m_val in enumerate(M_list):
    try:
        hdf5_filename = os.path.join(
            folder,
            "Interferograms",
            f"{date}_serie{serie}",
            f"interferograms_m={m_val}_p={p}_r={r}_dr={dr}.h5",
        )
        with h5py.File(hdf5_filename, "r") as hdf:
            interferograms = hdf["interferograms"]
            for col_idx, n in enumerate(tqdm(shot_indices)):
                im = interferograms[n]
                field = im_osc_fast(
                    im[:, :], center=(3 * im.shape[-2] // 4, im.shape[-1] // 4)
                )
                field = np.roll(field, center, axis=(0, 1))
                field = gaussian_filter(field, 1.5 / pixel_cam)

                field_polar = polarTransform.convertToPolarImage(
                    field,
                    center=[Nx // 2, Ny // 2],
                    finalRadius=Nx // 2,
                    radiusSize=Nx // 2,
                    angleSize=Nx,
                )[0].T

                den_polar_theta = np.sum(
                    np.abs(field_polar)[
                        idx(r_axis, r_fit - 3) : idx(r_axis, r_fit + 3), :
                    ]
                    ** 2,
                    axis=0,
                )

                # field_polar_ft = np.fft.fft(field_polar_theta)
                # field_polar_m = np.zeros(field_polar_ft.shape, dtype=complex)
                # field_polar_m[m_idx] = field_polar_ft[m_idx]
                # field_polar = np.fft.ifft(field_polar_m)

                transmission_theta_w_m[:, m_idx, col_idx] = den_polar_theta
    except Exception as error:
        print(f"Error processing m={m_val}: {error}")
np.save(
    os.path.join(
        folder,
        "Transmission",
        f"transmission_avg_theta_w_m_%s_serie%s_r=%s_dr=%s_p=%s_fromraw.npy"
        % (date, serie, str(r_fit), str(dr), str(p)),
    ),
    transmission_theta_w_m,
)

# %% (SMFig4) Density calibration from raw data with patch for last m value (fit Bogo to get gn pixel value calibration)
# Load θ-resolved transmission (shape: θ, m, energy)
transmission_theta_w_m = np.load(
    folder
    + "/Transmission"
    + "/transmission_avg_theta_w_m_%s_serie%s_r=%s_dr=%s_p=%s_fromraw.npy"
    % (date, serie, str(r_fit), str(dr), str(p))
)

theta_size, M_size, E_size = transmission_theta_w_m.shape
M_fit = np.array(M)
m_over_r = M_fit / r_fit

vr_polar = convertToPolarImage(
    vr,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=theta_size,
)[0].T

vazim_polar = convertToPolarImage(
    vazim,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=theta_size,
)[0].T

r_idx_min = idx(r_axis, r_fit - delta_r)
r_idx_max = idx(r_axis, r_fit + delta_r)
vr_theta = np.mean(vr_polar[r_idx_min:r_idx_max, :], axis=0)
vazim_theta = np.mean(vazim_polar[r_idx_min:r_idx_max, :], axis=0)

E_res_all = np.zeros((theta_size, M_size)) + np.nan
E_res_err_all = np.zeros((theta_size, M_size)) + np.nan
for m_idx in tqdm(range(M_size), desc="Lorentzian fits"):
    # exception for last m value replace by patch from previous cell
    if m_idx == M_size - 1:
        E_res_all[:, m_idx] = E_res_patch
        E_res_err_all[:, m_idx] = E_res_err_patch
        print(
            f"Using patch value for m = {M_fit[m_idx]}: E_res = {E_res_patch} ± {E_res_err_patch} meV"
        )
        continue
    for theta_idx in range(theta_size):
        spectrum = transmission_theta_w_m[theta_idx, m_idx, :]
        try:
            popt, pcov = curve_fit(
                lorentzian,
                energies,
                spectrum,
                p0=[energies[np.argmax(spectrum)], 0.1, np.max(spectrum)],
                maxfev=10000,
            )
            E_res_all[theta_idx, m_idx] = popt[0]
            E_res_err_all[theta_idx, m_idx] = np.sqrt(pcov[0, 0])
        except Exception:
            pass  # leave as NaN

detuning = hw_pump - E_LP_0
gn_list = []
gn_err_list = []

for theta_idx in tqdm(range(theta_size), desc="Bogo fits"):
    E_res_theta = E_res_all[theta_idx, :]
    E_res_err_theta = E_res_err_all[theta_idx, :]
    if np.any(np.isnan(E_res_theta)):
        continue

    def Bogo_fit_theta(m_over_r, gn):
        return Bogo(
            ky=m_over_r,
            gn=gn,
            alpha_sq=alpha_sq,
            m_LP=m_LP,
            E_pump=hw_pump,
            Delta=detuning,
            vy=vazim_theta[theta_idx],
            vx=vr_theta[theta_idx],
            kx=0,
        )

    try:
        popt, pcov = curve_fit(
            Bogo_fit_theta,
            m_over_r,
            E_res_theta,
            sigma=E_res_err_theta,
            absolute_sigma=True,
        )
        gn_list.append(popt[0])
        gn_err_list.append(np.sqrt(pcov[0, 0]))
    except Exception:
        continue


gn_array = np.array(gn_list)
gn_err_array = np.array(gn_err_list)

gn_mean = np.mean(gn_array)
gn_std = np.std(gn_array)


# -- PLOT --
# Evaluate Bogo curve
fit_curve = Bogo(
    ky=m_over_r,
    gn=gn_mean,
    alpha_sq=alpha_sq,
    m_LP=m_LP,
    E_pump=hw_pump,
    Delta=detuning,
    vy=np.mean(vazim_theta),
    vx=np.mean(vr_theta),
    kx=0,
)

# Derivative for 1σ band
delta = 1e-6
fit_plus = Bogo(
    m_over_r,
    gn_mean + delta,
    alpha_sq,
    m_LP,
    hw_pump,
    detuning,
    np.mean(vazim_theta),
    np.mean(vr_theta),
    kx=0,
)
fit_minus = Bogo(
    m_over_r,
    gn_mean - delta,
    alpha_sq,
    m_LP,
    hw_pump,
    detuning,
    np.mean(vazim_theta),
    np.mean(vr_theta),
    kx=0,
)
df_dgn = (fit_plus - fit_minus) / (2 * delta)
fit_err = np.abs(df_dgn) * gn_std

# Mean and std of E_res over θ
E_res = np.nanmean(E_res_all, axis=0)
E_res_err = np.nanstd(E_res_all, axis=0)

plt.figure(figsize=(3, 3))
# Normalized data (column-wise)
data = np.mean(transmission_theta_w_m, axis=0).T
data_norm = data / np.max(data, axis=0, keepdims=True)

im = plt.imshow(
    data_norm,
    extent=[
        (M[0] - C - 0.5),
        (M[-1] - C + 0.5),
        energies[-1] - hw_pump,
        energies[0] - hw_pump,
    ],
    aspect=(M[-1] - M[0]) / (energies[0] - energies[-1]),
    cmap="inferno",
    origin="upper",
)

# Add transmission fit curve and experimental points
plt.plot(
    M_fit - C, fit_curve - hw_pump, label=r"Bogo Fit (local $\theta$)", color="gray"
)
plt.errorbar(
    M_fit - C,
    E_res - hw_pump,
    yerr=E_res_err,
    fmt="o",
    markersize=3,
    color="gray",
    markerfacecolor="lightgray",
    markeredgecolor="gray",
    alpha=1,
)

# Labels and ticks (small for tight figure)
plt.xlabel(r"$m$")
plt.ylabel(r"$\hbar\omega\ (\mathrm{meV})$")

plt.colorbar()
plt.tight_layout()
plt.savefig(
    f"transmission_fit_localflow_r={r_fit}_%s_serie%s_patched.pdf" % (date, serie),
    dpi=300,
)
plt.close("all")

extent = [-512 * pixel_cam, 512 * pixel_cam, -512 * pixel_cam, 512 * pixel_cam]
plt.figure(figsize=(3, 3))
levels = [r_fit - 3, r_fit + 3]
plt.imshow(ampl / np.max(ampl), cmap="gray", extent=extent)
plt.colorbar()
plt.contour(
    R_grid,
    levels=levels,
    colors="tab:blue",
    linewidth=1,
    extent=extent,
    linestyles="--",
)
contour = plt.contourf(
    R_grid, levels=levels, colors="lightblue", alpha=0.8, extent=extent
)
plt.xlabel(r"$x (\mu \mathrm{m})$")
plt.ylabel(r"$y (\mu \mathrm{m})$")
plt.xticks([-50, 0, 50])
plt.yticks([-50, 0, 50])
plt.tight_layout()
plt.savefig("den_pump_SM_%s_serie%s.pdf" % (date, serie), dpi=300)
plt.close("all")

# %% Define azimuthal profiles of all quantities for velocities
filter_rad = 1.5  # avg over approximatively the healing length

den_pump = load_tiff_as_numpy(
    folder + "/Interferograms_pump" + f"/{date}_serie{serie}" + "/den_pump.tiff"
)
den_pump = gaussian_filter(den_pump, filter_rad / pixel_cam)
den_pump = np.roll(den_pump, center, axis=(0, 1))

den_pump_polar = polarTransform.convertToPolarImage(
    den_pump,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=theta_size,
)[0].T

den_polar_theta = np.mean(den_pump_polar[r_idx_min:r_idx_max, :], axis=0)

calib_array = gn_array / den_polar_theta

calib_mean = np.mean(calib_array)
calib_std = np.std(calib_array)

gn = den_pump * calib_mean
dgn = den_pump * calib_std

eff_det = detuning - 0.5 * m_LP * (vazim**2 + vr**2)
eff_det_m = mean_azim(eff_det)
grnr = gn * (1 - alpha_sq) / alpha_sq
dgrnr = dgn * (1 - alpha_sq) / alpha_sq
cs = np.real(np.sqrt(gn / m_LP))
dcs = dgn / (2 * cs)
dcs = np.nan_to_num(dgn / (2 * cs))
cs_g = np.real(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP))
dcs_g = np.nan_to_num(
    np.real(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP)) * dgn / (2 * cs_g)
)
cs_g_imag = np.nan_to_num(np.imag(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP)))
dcs_g_imag = (
    np.imag(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP)) * dgn / (2 * cs_g_imag)
)

gn_m, gn_err = mean_azim(gn, dimage=dgn)
grnr_m, grnr = mean_azim(grnr, dimage=dgrnr)
eff_det_m, eff_det_err = mean_azim(eff_det, dimage=eff_det * 0)
vazim_m, vazim_err = mean_azim(vazim, dimage=vazim * 0)
vr_m, vr_err = mean_azim(vr, dimage=vr * 0)
cs_m, cs_err = mean_azim(cs, dimage=dcs)
cs_g_m, cs_g_std, cs_g_err = mean_azim(cs_g, dimage=dcs_g, std=True)
cs_g_imag_m, cs_g_imag_err = mean_azim(cs_g_imag, dimage=dcs_g_imag)
r_axis = mean_azim(R_grid)

# %% (MainFig1(c)) Plot velocities
r_max = 60
r_min = 0

r_lim = 5

# Create masks
mask_low = r_axis < r_min
mask_high = r_axis >= r_min

plt.figure(figsize=(3, 1.7))

plt.vlines(r_min, -0.5, 1.5, color="k", linewidth=1)

# Plot lines with dummy legend handles to get colors and legend right
# line1, = plt.plot([], [], label=r"$c_s$")
# color1 = line1.get_color()
(line1,) = plt.plot([], [], label=r"$c_{B}$")
color1 = line1.get_color()
(line2,) = plt.plot([], [], label=r"$v_\theta$")
color2 = line2.get_color()
(line3,) = plt.plot([], [], label=r"$v_r$")
color3 = line3.get_color()

# Plot actual data with split transparency
plt.vlines(r_lim, -0.25, 1.3, color="k", linestyle="--", linewidth=1)
plt.plot(r_axis[mask_low], cs_m[mask_low], color=color1, alpha=0.2)
plt.plot(r_axis[mask_high], cs_m[mask_high], color=color1, alpha=0.8)
plt.fill_between(
    r_axis[mask_low],
    (cs_m - cs_err)[mask_low],
    (cs_m + cs_err)[mask_low],
    color=color1,
    alpha=0.1,
)
plt.fill_between(
    r_axis[mask_high],
    (cs_m - cs_err)[mask_high],
    (cs_m + cs_err)[mask_high],
    color=color1,
    alpha=0.4,
)

plt.plot(r_axis[mask_low], vazim_m[mask_low], color=color2, alpha=0.2)
plt.plot(r_axis[mask_high], vazim_m[mask_high], color=color2, alpha=0.8)
plt.fill_between(
    r_axis[mask_low],
    (vazim_m - vazim_err)[mask_low],
    (vazim_m + vazim_err)[mask_low],
    color=color2,
    alpha=0.1,
)
plt.fill_between(
    r_axis[mask_high],
    (vazim_m - vazim_err)[mask_high],
    (vazim_m + vazim_err)[mask_high],
    color=color2,
    alpha=0.4,
)

plt.plot(r_axis[mask_low], vr_m[mask_low], color=color3, alpha=0.2)
plt.plot(r_axis[mask_high], vr_m[mask_high], color=color3, alpha=0.8)
plt.fill_between(
    r_axis[mask_low],
    (vr_m - vr_err)[mask_low],
    (vr_m + vr_err)[mask_low],
    color=color3,
    alpha=0.1,
)
plt.fill_between(
    r_axis[mask_high],
    (vr_m - vr_err)[mask_high],
    (vr_m + vr_err)[mask_high],
    color=color3,
    alpha=0.4,
)

plt.ylim(-0.25, 1.3)
plt.xlim(0, r_max)
plt.xlabel(r"$r (\mu \mathrm{m})$")
plt.ylabel(r"$\mathrm{velocities}(\mu \mathrm{m/ps})$")
plt.yticks(np.arange(-0, 1 + 0.5, 0.5))
plt.tight_layout()
plt.savefig("velocities_%s_serie%s.pdf" % (date, serie), dpi=300)
plt.close("all")
# %% Plot filtered transmissions W_R(M) (last column empty due to data deterioration at m=M[-1])

radii = [6, 9, 18, 27]

dr = 50
r = 25
p = 0

# Setup grid and mask parameters
X, Y = np.meshgrid(
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
    np.arange(-512 * pixel_cam, 512 * pixel_cam, pixel_cam),
)
R_grid = np.sqrt(X**2 + Y**2)

R_mask_values = radii
dr_mask = 6

# Build all circular masks once
masks = {
    r_mask: (R_grid > r_mask + dr_mask / 2) | (R_grid < r_mask - dr_mask / 2)
    for r_mask in R_mask_values
}

# Prepare transmission arrays per r_mask
nb_shot_for_plot = 80
shot_indices = np.arange(0, n_shot, n_shot // nb_shot_for_plot)
transmissions = {
    r_mask: np.zeros((len(M), nb_shot_for_plot)) for r_mask in R_mask_values
}

# Compute extent and aspect
try:
    extent = [M[0] - C, M[-1] - C, energies[-1] - hw_pump, energies[0] - hw_pump]
    aspect_ratio = (M[-1] - M[0]) / (energies[0] - energies[-1])
except Exception:
    extent = [M[0] - C, M[-1] - C, energies[-1], energies[0]]
    aspect_ratio = (M[-1] - M[0]) / (energies[0] - energies[-1])

for m_idx, m_val in enumerate(M):
    try:
        hdf5_filename = os.path.join(
            folder,
            "Interferograms",
            f"{date}_serie{serie}",
            f"interferograms_m={m_val}_p={p}_r={r}_dr={dr}.h5",
        )
        with h5py.File(hdf5_filename, "r") as hdf:
            interferograms = hdf["interferograms"]
            for col_idx, n in enumerate(tqdm(shot_indices)):
                im = interferograms[n]
                field = im_osc_fast(
                    im[:, :], center=(3 * im.shape[-2] // 4, im.shape[-1] // 4)
                )
                field = np.roll(field, center, axis=(0, 1))
                field = gaussian_filter(field, 1.5 / pixel_cam)
                for r_mask, mask in masks.items():
                    field_masked = field.copy()
                    field_masked[mask] = 0
                    transmissions[r_mask][m_idx, col_idx] = np.sum(
                        np.abs(field_masked) ** 2
                    )
    except Exception as error:
        print(f"Error processing m={m_val}: {error}")

for r_mask, filtered_transmission in transmissions.items():
    filename_base = f"filtered_transmission_{date}_serie{serie}_r={r_mask}_dr={dr_mask}_p={p}_fromraw"
    np.save(
        os.path.join(folder, "Transmission", filename_base + ".npy"),
        filtered_transmission,
    )

hw_0 = 1480.0127
hw_1 = 1480.1168
hw_2 = 1480.1987

m = -4
idx_energy_0 = idx(energies, hw_0)  # Example index for energy
idx_energy_1 = idx(energies, hw_1)  # Example index for energy
idx_energy_2 = idx(energies, hw_2)  # Example index for energy

transmissions_fromraw = []
for rad in radii:
    try:
        dr = 6
        transmission_fromraw = np.load(
            folder
            + "/Transmission"
            + "/filtered_transmission_%s_serie%s_r=%s_dr=%s_p=%s_fromraw.npy"
            % (date, serie, str(rad), str(dr), str(p))
        )
        transmissions_fromraw.append(transmission_fromraw)
    except Exception as error:
        print(f"Error loading transmission for r={rad}: {error}")

fig, axs = plt.subplots(1, 4, figsize=(6, 2.5))

for i, rad in enumerate(radii):
    im = axs[i].imshow(
        transmissions_fromraw[i].transpose(),
        extent=[
            (M[0] - C - 0.5),
            (M[-1] - C + 0.5),
            energies[-1] - hw_pump,
            energies[0] - hw_pump,
        ],
        aspect=(M[-1] - M[0]) / (energies[0] - energies[-1]),
        cmap="inferno",
    )

    # get auto aspect and print it

    axs[i].set_title(r"$r=%s \mu \mathrm{m}$" % rad)
    axs[i].vlines(
        0,
        energies[-1] - hw_pump,
        energies[0] - hw_pump,
        color="w",
        linestyle="-",
        linewidth=1.5,
    )
    axs[i].hlines(
        0, M[0] - C - 0.5, M[-1] - C + 0.5, color="w", linestyle="-", linewidth=1.5
    )
    axs[i].set_xlabel(r"$m$")
    if axs[i] == axs[0]:
        axs[i].set_ylabel(r"$\hbar\omega_{probe} (\mathrm{meV})$")
        axs[i].set_yticks([-0.2, 0, 0.2])  # Set yticks at -12, -4, and 4
        axs[i].set_xticks([-12, -4, 4])  # Set xticks at -12, -4, and 4

    if axs[i] != axs[0]:
        axs[i].set_yticklabels([])
        axs[i].set_xticks([-12, -4, 4])  # Set xticks at -12, -4, and 4
        # axs[i].set_xlabel("")

    # uncomment to add Fig2 inset
    # if i==1:
    #     #add inset
    #     xl, xr = -6, -2
    #     hw_l, hw_h = -0.15, 0.05
    #     axins = inset_axes(axs[i], width="100%", height="100%", loc="upper right", borderpad=2)
    #     axins.imshow(transmissions_fromraw[i].transpose(),
    #          extent=[(M[0]-C-0.5), (M[-1]-C+0.5), energies[-1]-hw_pump, energies[0]-hw_pump],
    #          aspect=(M[-1]-M[0]) / (energies[0]-energies[-1]),
    #          cmap="inferno")
    #     axins.scatter(m-C, energies[idx_energy_0]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2, label="Pump")
    #     axins.scatter(m-C, energies[idx_energy_1]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.scatter(m-C, energies[idx_energy_2]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.set_xlim(xl, xr)
    #     axins.set_ylim(hw_l, hw_h)
    #     axins.set_xticks([-4])  # Set xticks to -6 and -2
    #     mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="red")
    # if i==2:
    #     #add inset
    #     xl, xr = -11, -5
    #     hw_l, hw_h = -0.05, np.max(energies) - hw_pump
    #     axins = inset_axes(axs[i], width="100%", height="100%", loc="upper right", borderpad=2)
    #     axins.imshow(transmissions_fromraw[i].transpose(),
    #              extent=[(M[0]-C-0.5), (M[-1]-C+0.5), energies[-1]-hw_pump, energies[0]-hw_pump],
    #              aspect=(M[-1]-M[0]) / (energies[0]-energies[-1]),
    #              cmap="inferno")
    #     axins.scatter(m-C, energies[idx_energy_0]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2, label="Pump")
    #     axins.scatter(m-C, energies[idx_energy_1]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.scatter(m-C, energies[idx_energy_2]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.set_xlim(xl, xr)
    #     axins.set_ylim(hw_l, hw_h)
    #     mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="red")

    # plt.subplots_adjust(top=1, bottom=0, left=0.1, right=0.9, hspace=0, wspace=0.05)
# Add a colorbar for the last subplot

# # divider = make_axes_locatable(ax[-1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, ax=axs, pad=0.05)
custom_ticks = [
    1e-9 * max([np.max(transmissions_fromraw[i]) for i in range(len(radii))]),
    max([np.max(transmissions_fromraw[i]) for i in range(len(radii))]),
]
cbar.set_ticks(custom_ticks)
cbar.set_ticklabels([r"$0$", r"$1$"])
plt.savefig("transmission_w_m_%s_serie%s_fromraw.pdf" % (date, serie), dpi=300)
plt.close("all")
# %% (MainFig2(e)-(h)) Plot filtered transmissions W_R(M)
transmissions = []
for rad in radii:
    try:
        dr = 6
        transmission = np.load(
            folder
            + "/Transmission"
            + "/filtered_transmission_%s_serie%s_r=%s_dr=%s_p=%s.npy"
            % (date, serie, str(rad), str(dr), str(p))
        )
        transmissions.append(transmission)
    except Exception as error:
        print(f"Error loading transmission for r={rad}: {error}")


fig, axs = plt.subplots(1, 4, figsize=(6, 2.5))

for i, rad in enumerate(radii):
    im = axs[i].imshow(
        transmissions[i].transpose(),
        extent=[
            (M[0] - C - 0.5),
            (M[-1] - C + 0.5),
            energies[-1] - hw_pump,
            energies[0] - hw_pump,
        ],
        aspect=(M[-1] - M[0]) / (energies[0] - energies[-1]),
        cmap="inferno",
    )

    axs[i].set_title(r"$r=%s \mu \mathrm{m}$" % rad)
    axs[i].vlines(
        0,
        energies[-1] - hw_pump,
        energies[0] - hw_pump,
        color="w",
        linestyle="-",
        linewidth=1.5,
    )
    axs[i].hlines(
        0, M[0] - C - 0.5, M[-1] - C + 0.5, color="w", linestyle="-", linewidth=1.5
    )
    axs[i].set_xlabel(r"$m$")
    if axs[i] == axs[0]:
        axs[i].set_ylabel(r"$\hbar\omega_{probe} (\mathrm{meV})$")
        axs[i].set_yticks([-0.2, 0, 0.2])  # Set yticks at -12, -4, and 4
        axs[i].set_xticks([-12, -4, 4])  # Set xticks at -12, -4, and 4

    if axs[i] != axs[0]:
        axs[i].set_yticklabels([])
        axs[i].set_xticks([-12, -4, 4])  # Set xticks at -12, -4, and 4
        # axs[i].set_xlabel("")

    # uncomment to add Fig2 inset
    # if i==1:
    #     #add inset
    #     xl, xr = -6, -2
    #     hw_l, hw_h = -0.15, 0.05
    #     axins = inset_axes(axs[i], width="100%", height="100%", loc="upper right", borderpad=2)
    #     axins.imshow(transmissions[i].transpose(),
    #          extent=[(M[0]-C-0.5), (M[-1]-C+0.5), energies[-1]-hw_pump, energies[0]-hw_pump],
    #          aspect=(M[-1]-M[0]) / (energies[0]-energies[-1]),
    #          cmap="inferno")
    #     axins.scatter(m-C, energies[idx_energy_0]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2, label="Pump")
    #     axins.scatter(m-C, energies[idx_energy_1]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.scatter(m-C, energies[idx_energy_2]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.set_xlim(xl, xr)
    #     axins.set_ylim(hw_l, hw_h)
    #     axins.set_xticks([-4])  # Set xticks to -6 and -2
    #     mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="red")
    # if i==2:
    #     #add inset
    #     xl, xr = -11, -5
    #     hw_l, hw_h = -0.05, np.max(energies) - hw_pump
    #     axins = inset_axes(axs[i], width="100%", height="100%", loc="upper right", borderpad=2)
    #     axins.imshow(transmissions[i].transpose(),
    #              extent=[(M[0]-C-0.5), (M[-1]-C+0.5), energies[-1]-hw_pump, energies[0]-hw_pump],
    #              aspect=(M[-1]-M[0]) / (energies[0]-energies[-1]),
    #              cmap="inferno")
    #     axins.scatter(m-C, energies[idx_energy_0]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2, label="Pump")
    #     axins.scatter(m-C, energies[idx_energy_1]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.scatter(m-C, energies[idx_energy_2]-hw_pump, color="white", edgecolor="black", marker="*", s=100, linewidth=1.2)
    #     axins.set_xlim(xl, xr)
    #     axins.set_ylim(hw_l, hw_h)
    #     mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="red")

    # plt.subplots_adjust(top=1, bottom=0, left=0.1, right=0.9, hspace=0, wspace=0.05)
# Add a colorbar for the last subplot

# # divider = make_axes_locatable(ax[-1])
# cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im, ax=axs, pad=0.05)
custom_ticks = [
    1e-9 * max([np.max(transmissions[i]) for i in range(len(radii))]),
    max([np.max(transmissions[i]) for i in range(len(radii))]),
]
cbar.set_ticks(custom_ticks)
cbar.set_ticklabels([r"$0$", r"$1$"])

plt.savefig("transmission_w_m_%s_serie%s.pdf" % (date, serie), dpi=300)
plt.close("all")
# %% (MainFig2(a)-(d)) Plot WKB for same R cuts

filter_rad_wkb = 3.0  # matching dr=6
den_pump = load_tiff_as_numpy(
    folder + "/Interferograms_pump" + f"/{date}_serie{serie}" + "/den_pump.tiff"
)
den_pump = gaussian_filter(den_pump, filter_rad_wkb / pixel_cam)
den_pump = np.roll(den_pump, center, axis=(0, 1))

den_pump_polar = polarTransform.convertToPolarImage(
    den_pump,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=theta_size,
)[0].T

den_polar_theta = np.mean(den_pump_polar[r_idx_min:r_idx_max, :], axis=0)

calib_array = gn_array / den_polar_theta

calib_mean = np.mean(calib_array)
calib_std = np.std(calib_array)

gn = den_pump * calib_mean
dgn = den_pump * calib_std

eff_det = detuning - 0.5 * m_LP * (vazim**2 + vr**2)
eff_det_m = mean_azim(eff_det)
grnr = gn * (1 - alpha_sq) / alpha_sq
dgrnr = dgn * (1 - alpha_sq) / alpha_sq
cs = np.real(np.sqrt(gn / m_LP))
dcs = dgn / (2 * cs)
dcs = np.nan_to_num(dgn / (2 * cs))
cs_g = np.real(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP))
dcs_g = np.nan_to_num(
    np.real(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP)) * dgn / (2 * cs_g)
)
cs_g_imag = np.nan_to_num(np.imag(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP)))
dcs_g_imag = (
    np.imag(np.emath.sqrt((2 * gn + grnr - eff_det) / m_LP)) * dgn / (2 * cs_g_imag)
)

gn_m, gn_err = mean_azim(gn, dimage=dgn)
grnr_m, grnr = mean_azim(grnr, dimage=dgrnr)
eff_det_m, eff_det_err = mean_azim(eff_det, dimage=eff_det * 0)
vazim_m, vazim_err = mean_azim(vazim, dimage=vazim * 0)
vr_m, vr_err = mean_azim(vr, dimage=vr * 0)
cs_m, cs_err = mean_azim(cs, dimage=dcs)
cs_g_m, cs_g_std, cs_g_err = mean_azim(cs_g, dimage=dcs_g, std=True)
cs_g_imag_m, cs_g_imag_err = mean_azim(cs_g_imag, dimage=dcs_g_imag)
r_axis = mean_azim(R_grid)

fig, ax = plt.subplots(1, 4, figsize=(6, 1.2))
r_values = [6, 9, 18, 27]
m_max = 10 - 4
m_min = -10 - 4
M_plot = np.arange(m_min, m_max + 1, 1)
hw_min = energies[-1] - hw_pump
hw_max = energies[0] - hw_pump
for i, r in enumerate(r_values):
    idx_r = idx(r_axis, r)
    # Evaluate the Bogo model on the full M array for the given r.
    bogo_1d = Bogo(
        (M_plot + C) / r,
        gn=gn_m[idx_r],
        alpha_sq=alpha_sq,
        m_LP=m_LP,
        E_pump=hw_pump,
        Delta=detuning,
        vy=vazim_m[idx_r],
        vx=vr_m[idx_r],
        kx=0,
    )

    bogo_1d_m = -Bogo(
        (-M_plot + C) / r,
        gn=gn_m[idx_r],
        alpha_sq=alpha_sq,
        m_LP=m_LP,
        E_pump=hw_pump,
        Delta=detuning,
        vy=vazim_m[idx_r],
        vx=vr_m[idx_r],
        kx=0,
    )

    ax[i].plot(
        M_plot, bogo_1d - hw_pump, label=r"$\hbar\omeg a_B^+ (\mathrm{meV})$", color="k"
    )
    ax[i].plot(
        M_plot,
        bogo_1d_m + hw_pump,
        label=r"$\hbar\omega_B^- (\mathrm{meV})$",
        color="tab:red",
    )
    ax[i].hlines(0, m_min, m_max, color="k", linestyle="-", linewidth=0.3)
    ax[i].vlines(0, hw_min, hw_max, color="k", linestyle="-", linewidth=0.3)

    ax[i].fill_between(
        M_plot,
        bogo_1d - hw_pump,
        hw_max,
        where=(bogo_1d - hw_pump) < hw_max,
        interpolate=True,
        color="k",
        alpha=0.3,
    )

    # Fill from curve to bottom (only where curve > hw_min)
    ax[i].fill_between(
        M_plot,
        hw_min,
        bogo_1d_m + hw_pump,
        where=(bogo_1d_m + hw_pump) > hw_min,
        interpolate=True,
        color="tab:red",
        alpha=0.3,
    )
    ax[i].set_ylim(hw_min, hw_max)
    ax[i].set_xlim(m_min, m_max)
    ax[i].set_xticks([4, -4, -12])  # Set xticks at 0, -5, and -10
    aspect_ratio = (m_max - m_min) / (hw_max - hw_min)
    ax[i].set_aspect(aspect_ratio)
    if i == 0:
        ax[i].set_yticks([-0.2, 0, 0.2])
        ax[i].set_yticklabels([-0.2, 0, 0.2])
        ax[i].set_ylabel(r"$\hbar\omega (\mathrm{meV})$")
    else:
        ax[i].set_yticklabels([])
    ax[i].set_xlabel("$m$")
dpi = 300
plt.subplots_adjust(left=0.2, bottom=0.4, right=0.98, top=1, wspace=0)
plt.savefig("wkb_w_m_%s_serie%s.pdf" % (date, serie), dpi=300)
# %% (MainFig2(i)-(k)) Plot probe at specific energies
hw_list = [hw_0, hw_1, hw_2]

m = -6
r = 25
dr = 50
p = 0.0

amplitudes = np.zeros((3, 1024, 1024))

p = 0
for i, hw in enumerate(hw_list):
    _, ampl = retrieve_probe_field(
        folder,
        date,
        serie,
        m,
        p,
        r,
        dr,
        hw,
        roi_far_field=75,
        plot=False,
        save=False,
        show=False,
    )
    amplitudes[i] = gaussian_filter(ampl, 1.5 / pixel_cam)

max_amplitude = np.max(amplitudes)

fig, ax = plt.subplots(1, 3, figsize=(5, 1.5))
for i, hw in enumerate(hw_list):
    extent = [-512 * pixel_cam, 512 * pixel_cam, -512 * pixel_cam, 512 * pixel_cam]
    if i < len(amplitudes):  # Ensure amplitudes[i] exists
        im = ax[i].imshow(amplitudes[i] / max_amplitude, "gray", extent=extent)
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax[i].set_xlabel(r"$x (\mu \mathrm{m})$")
        ax[i].set_ylabel(r"$y (\mu \mathrm{m})$")
        if i > 0:  # Remove tick labels for the top plots
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")

        else:
            ax[i].set_ylabel("$y (\mu \mathrm{m})$")
            ax[i].set_xlabel("$x (\mu \mathrm{m})$")

plt.subplots_adjust(left=0.13, bottom=0.2, right=0.90, wspace=0.85, hspace=0.1)
plt.savefig("probe_field_amplitudes_combined_serie%s.pdf" % (serie), dpi=300)
plt.close("all")

# %% Look where the interferogram are saturated density pixels will be replaced by max value

r = 25
dr = 50
p = 0
Nx = 1024

Transmission_mask = np.zeros((len(M), len(energies), Nx // 2))
for m in [-2, -1, 0, 1, 2]:
    for hw in energies[energies < (hw_pump - 0.02)]:
        hdf5_filename = os.path.join(
            folder,
            "Interferograms",
            f"{date}_serie{serie}",
            f"interferograms_m={m}_p={p}_r={r}_dr={dr}.h5",
        )

        # Open the HDF5 file
        with h5py.File(hdf5_filename, "r") as hdf:
            # Retrieve the dataset and load the specific interferogram
            interferograms = hdf["interferograms"]  # Reference to the dataset
            im = interferograms[idx(energies, hw)]  # Load only the specific slice

        im = np.roll(im, center, axis=(0, 1))

        mask = mean_azim(im >= 255) > 0
        try:
            r_sat = np.max(r_axis[r_axis < 20][mask[r_axis < 20]])
        except:
            r_sat = -1e9
        mask = r_axis < r_sat
        Transmission_mask[idx(M, m), idx(energies, hw), :] = (
            r_axis < r_sat
        )  # ((r_axis < 20) * mean_azim(im >= 255))>0
# save transmission mask
np.save(
    folder
    + "/Transmission"
    + "/transmission_saturation_mask_%s_serie%s_r=%s_dr=%s_p=%s.npy"
    % (date, serie, str(r), str(dr), str(p)),
    Transmission_mask,
)

# load transmission mask
Transmission_mask = np.load(
    folder
    + "/Transmission"
    + "/transmission_saturation_mask_%s_serie%s_r=%s_dr=%s_p=%s.npy"
    % (date, serie, str(r), str(dr), str(p))
)
# %% (MainFig3(c)(d)) Plot effective cavity density maps
m = 0
r = 25
dr = 50
p = 0
hw_1 = -0.0015 + hw_pump  # higher frequency cavity mode
hw_2 = -0.068 + hw_pump  # lower frequency cavity mode

hdf5_filename = os.path.join(
    folder,
    "Interferograms",
    f"{date}_serie{serie}",
    f"interferograms_m={m}_p={p}_r={r}_dr={dr}.h5",
)
with h5py.File(hdf5_filename, "r") as hdf:
    interferograms = hdf["interferograms"]  # Reference to the dataset
    im1 = interferograms[idx(energies, hw_1)]  # Load only the specific slice

field_probe1 = im_osc_fast(
    im1[:, :], center=(3 * im1.shape[-2] // 4, im1.shape[-1] // 4)
)
field_probe1 = np.roll(field_probe1, center, axis=(0, 1))
amplitude_probe1 = np.abs(field_probe1)

mask1 = mean_azim(im1 >= 255) > 0
r_sat1 = np.nanmax(r_axis[r_axis < 20][mask1[r_axis < 20]])
amplitude_probe1[R_grid < r_sat1] = np.max(amplitude_probe1)
phase_probe1 = np.angle(field_probe1)
amplitude_probe1 = gaussian_filter(amplitude_probe1, 1.5 / pixel_cam)
phase_probe1 = gaussian_filter(phase_probe1, 1.5 / pixel_cam)

hdf5_filename = os.path.join(
    folder,
    "Interferograms",
    f"{date}_serie{serie}",
    f"interferograms_m={m}_p={p}_r={r}_dr={dr}.h5",
)
with h5py.File(hdf5_filename, "r") as hdf:
    # Retrieve the dataset and load the specific interferogram
    interferograms = hdf["interferograms"]  # Reference to the dataset
    im2 = interferograms[idx(energies, hw_2)]  # Load only the specific slice

field_probe2 = im_osc_fast(
    im2[:, :], center=(3 * im2.shape[-2] // 4, im2.shape[-1] // 4)
)
field_probe2 = np.roll(field_probe2, center, axis=(0, 1))
amplitude_probe2 = np.abs(field_probe2)

mask2 = mean_azim(im2 >= 255) > 0
r_sat2 = np.nanmax(r_axis[r_axis < 20][mask2[r_axis < 20]])
amplitude_probe2[R_grid < r_sat2] = np.max(amplitude_probe2)
# Set saturated pixels to max amplitude
phase_probe2 = np.angle(field_probe2)
amplitude_probe2 = gaussian_filter(amplitude_probe2, 1.5 / pixel_cam)
phase_probe2 = gaussian_filter(phase_probe2, 1.5 / pixel_cam)

extent = [-512 * pixel_cam, 512 * pixel_cam, -512 * pixel_cam, 512 * pixel_cam]

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
# normalized intensities
I1 = (amplitude_probe1 / np.nanmax(amplitude_probe1)) ** 2
I2 = (amplitude_probe2 / np.nanmax(amplitude_probe2)) ** 2

# use same color scaling for both panels
vmin, vmax = 0.0, max(np.nanmax(I1), np.nanmax(I2))

im0 = axs[0].imshow(
    I1, cmap="gray", extent=extent, vmin=vmin, vmax=vmax, origin="upper"
)
axs[0].set_title(f"{hw_1 - hw_pump:.4f} meV")
axs[0].set_xlabel(r"$x (\mu \mathrm{m})$")
axs[0].set_yticks([])

im1 = axs[1].imshow(
    I2, cmap="gray", extent=extent, vmin=vmin, vmax=vmax, origin="upper"
)
axs[1].set_title(f"{hw_2 - hw_pump:.4f} meV")
axs[1].set_xlabel(r"$x (\mu \mathrm{m})$")
axs[1].set_yticks([])

# single colorbar for both axes
cbar = fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.9)
cbar.set_label("Normalized intensity")

# save combined figure

plt.savefig(
    "probe_field_m=%s_r=%s_hw_combined_%s_serie%s.pdf" % (m, r, date, serie),
    dpi=300,
    bbox_inches="tight",
)
plt.close("all")

# %% Compute filtered transmissions W_M(R)

M_list = np.array([-12, -10, -8, -5, -4, -3, 4, 5, 6]) + C

dr = 50
r = 25
p = 0
Nx, Ny = 1024, 1024

R_polar = polarTransform.convertToPolarImage(
    R_grid,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=Nx,
)[0].T
THETA_polar = polarTransform.convertToPolarImage(
    THETA,
    center=[Nx // 2, Ny // 2],
    finalRadius=Nx // 2,
    radiusSize=Nx // 2,
    angleSize=Nx,
)[0].T
nb_shot_for_plot = 80
filtered_transmission = np.zeros((nb_shot_for_plot, r_axis.shape[0]))
for m in M_list:
    if m == M[-1]:
        print("No raw data for m=", m)
        continue
    for n in tqdm(np.arange(0, n_shot, n_shot // nb_shot_for_plot)):
        hdf5_filename = os.path.join(
            folder,
            "Interferograms",
            f"{date}_serie{serie}",
            f"interferograms_m={m}_p={p}_r={r}_dr={dr}.h5",
        )

        # Open the HDF5 file
        with h5py.File(hdf5_filename, "r") as hdf:
            # Retrieve the dataset and load the specific interferogram
            interferograms = hdf["interferograms"]
            im = interferograms[n]

        field = im_osc_fast(im[:, :], center=(3 * im.shape[-2] // 4, im.shape[-1] // 4))

        field = gaussian_filter(field, 1.5 / pixel_cam)
        field = np.roll(field, center, axis=(0, 1))

        den = np.abs(field) ** 2
        den_azim = mean_azim(den)

        filtered_transmission[n, :] = den_azim[:]

        np.save(
            folder
            + "/Transmission"
            + "/filtered_transmission_%s_serie%s_p=%s_m=%s_fromraw.npy"
            % (date, serie, str(p), str(m)),
            filtered_transmission,
        )


dr = 50
r = 25
p = 0
nb_shot_for_plot = 80

norm = "log"

m = 0
filtered_transmission = np.load(
    folder
    + "/Transmission"
    + "/filtered_transmission_%s_serie%s_p=%s_m=%s_fromraw.npy"
    % (date, serie, str(p), str(m))
)

filtered_transmissions = np.zeros((len(M),) + filtered_transmission.shape)
filtered_transmissions_raw = np.zeros((len(M),) + filtered_transmission.shape)

M_list = np.array([-12, -10, -8, -5, -4, -3, 4, 5, 6]) + C

# Load all filtered transmissions for each m
for i, m in enumerate(M_list):
    try:
        filtered_transmissions_raw[i] = np.load(
            folder
            + "/Transmission"
            + f"/filtered_transmission_{date}_serie{serie}_p={p}_m={m}_fromraw.npy"
        )
        filtered_transmissions[i] = np.load(
            folder
            + "/Transmission"
            + f"/filtered_transmission_{date}_serie{serie}_p={p}_m={m}_fromraw.npy"
        )
        filtered_transmissions[i][Transmission_mask[idx(M, m)] == 1] = np.max(
            filtered_transmissions_raw[i]
        )  # Set the problematic transmission to NaN
        filtered_transmissions[i] = gaussian_filter(
            filtered_transmissions[i], (0, 1.5 / pixel_cam)
        )
    except:
        print("Loading processed data for m=", m)
        filtered_transmissions_raw[i] = np.load(
            folder
            + "/Transmission"
            + f"/filtered_transmission_{date}_serie{serie}_p={p}_m={m}.npy"
        )
        filtered_transmissions[i] = np.load(
            folder
            + "/Transmission"
            + f"/filtered_transmission_{date}_serie{serie}_p={p}_m={m}.npy"
        )
        filtered_transmissions[i][Transmission_mask[idx(M, m)] == 1] = np.max(
            filtered_transmissions_raw[i]
        )  # Set the problematic transmission to NaN
        filtered_transmissions[i] = gaussian_filter(
            filtered_transmissions[i], (0, 1.5 / pixel_cam)
        )
# %% (MainFig3(e)-(m)) Plot filtered transmissions W_R(M) with WKB model
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
import os
import imageio

norm = "linear"
cmap = "inferno"
vmax = 170

e_min = -0.15
e_max = 0.2
r_min = 6
rmax = 50
extent = [0, r_axis[-1], energies[-1] - hw_pump, energies[0] - hw_pump]
aspect_ratio = rmax / (e_max - e_min)
fig_num = np.sqrt(len(M_list)).astype(int)
fig, axs = plt.subplots(fig_num, fig_num, figsize=(4, 3))
images = []
lines = []


alpha = 0.95
for i, ax in enumerate(axs.flat):
    if i < len(M_list):
        m = M_list[i]
        bogo_wkb_2d = Bogo_2d(
            kx=m / R_grid,
            ky=0,
            gn=alpha * gn,
            alpha_sq=alpha_sq,
            m_LP=m_LP,
            E_pump=hw_pump,
            detuning=detuning,
            vx=vazim,
            vy=vr,
        )
        bogo_wkb_2d_plus = Bogo_2d(
            kx=m / R_grid,
            ky=0,
            gn=alpha * gn + dgn,
            alpha_sq=alpha_sq,
            m_LP=m_LP,
            E_pump=hw_pump,
            detuning=detuning,
            vx=vazim,
            vy=vr,
        )
        bogo_wkb_2d_minus = Bogo_2d(
            kx=m / R_grid,
            ky=0,
            gn=alpha * gn - dgn,
            alpha_sq=alpha_sq,
            m_LP=m_LP,
            E_pump=hw_pump,
            detuning=detuning,
            vx=vazim,
            vy=vr,
        )
        dbogo_wkb_2d = np.abs(bogo_wkb_2d_plus - bogo_wkb_2d_minus) / 2
        bogo_wkb_m, bogo_wkb_std, bogo_wkb_err = mean_azim(
            bogo_wkb_2d, std=True, dimage=dbogo_wkb_2d
        )
        bogo_wkb_m = bogo_wkb_m - hw_pump

        ax.set_xlim(0, rmax)
        ax.set_ylim(e_min, e_max)
        im = ax.imshow(
            filtered_transmissions[i],
            norm=norm,
            cmap=cmap,
            aspect=aspect_ratio,
            extent=extent,
            interpolation="nearest",
            vmax=vmax,
        )
        (line,) = ax.plot(
            r_axis[r_axis > r_min],
            bogo_wkb_m[r_axis > r_min],
            label=r"$\min\left(\hbar\omega_{\mathrm{Bog}}(m+C=%d)\right)$" % m,
            color="w",
        )
        ax.fill_between(
            r_axis[r_axis > r_min],
            bogo_wkb_m[r_axis > r_min] - bogo_wkb_err[r_axis > r_min],
            bogo_wkb_m[r_axis > r_min] + bogo_wkb_err[r_axis > r_min],
            alpha=0.2,
            color="w",
        )

        ax.text(
            0.02,
            0.01,
            r"$m=%d$" % (m - C),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize="small",
            color="white",
        )
        if (
            i == len(axs.flat) - fig_num
        ):  # Add ticks only for the lowest left corner plot
            ax.set_xlabel(r"$r (\mu m)$")
            ax.set_ylabel(r"$\hbar\omega(\mathrm{meV})$")
        else:  # Remove ticks for all other plots
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        images.append(im)
        lines.append(line)
        ax.grid(color="white", linestyle="-", linewidth=0.5)
    else:
        ax.axis("off")

plt.subplots_adjust(hspace=0.05, wspace=0.05, bottom=0.15)
cbar = plt.colorbar(im, ax=axs, pad=0.02, aspect=30)
custom_ticks = [1e-9 * vmax, vmax]
cbar.set_ticks(custom_ticks)
cbar.set_ticklabels([r"$0$", r"$1$"])

plt.savefig(
    "filtered_transmission_%s_serie%s_p=%s.pdf" % (date, serie, str(p)), dpi=300
)
plt.close("all")
# %% (MainFig3(a)(b)) Cavity mode plots
m = 0
azim_avgs = []
Nx, Ny = field_probe1.shape
for hw in [hw_1, hw_2]:
    hdf5_filename = os.path.join(
        folder,
        "Interferograms",
        f"{date}_serie{serie}",
        f"interferograms_m={m}_p={p}_r={r}_dr={dr}.h5",
    )

    # Open the HDF5 file
    with h5py.File(hdf5_filename, "r") as hdf:
        # Retrieve the dataset and load the specific interferogram
        interferograms = hdf["interferograms"]  # Reference to the dataset
        im = interferograms[idx(energies, hw)]  # Load only the specific slice

        field_probe = im_osc_fast(
            im[:, :], center=(3 * im.shape[-2] // 4, im.shape[-1] // 4)
        )
        field_probe = np.roll(field_probe, center, axis=(0, 1))
        # field_probe = gaussian_filter(field_probe, 1.5/pixel_cam)
        amplitude_probe = np.abs(field_probe)
        amplitude_probe = gaussian_filter(amplitude_probe, 1.5 / pixel_cam)
        mask = mean_azim(im >= 255) > 0
        if hw < -0.04 + hw_pump:
            r_sat = np.nanmax(r_axis[r_axis < 20][mask[r_axis < 20]])
            mask = r_axis < r_sat
        # amplitude_probe[R_grid<r_sat] = np.max(amplitude_probe)
        # Set saturated pixels to max amplitude
        phase_probe = np.angle(field_probe)
        # phase_probe = gaussian_filter(phase_probe, 1.5/pixel_cam)
        azim_avg = mean_azim(amplitude_probe) ** 2
        if hw < -0.04 + hw_pump:
            azim_avg[mask] = np.nan
        # azim_avg = gaussian_filter(azim_avg, sigma=1.5/pixel_cam)
        azim_avgs.append(azim_avg)

data_2d = filtered_transmissions[idx(M_list, 0)]
int_sum = np.sum(data_2d[:, r_axis < 20], axis=1)

extent = [0, r_axis[-1], energies[-1] - hw_pump, energies[0] - hw_pump]

e_min = -0.15
e_max = 0.2
r_min = 0
rmax = 50

aspect_ratio = rmax / (e_max - e_min)

e_max_zoom = 0.05
e_min_zoom = -0.15
r_min_zoom = 0
r_max_zoom = 20

data_zoom = data_2d[
    idx(energies, e_max_zoom + hw_pump) : idx(energies + hw_pump, e_min_zoom),
    idx(r_axis, r_min_zoom) : idx(r_axis, r_max_zoom),
]
zoom_aspect_ratio = (
    aspect_ratio
    * ((idx(r_axis, r_max_zoom) - idx(r_axis, r_min_zoom)))
    / ((idx(energies, e_min_zoom + hw_pump) - idx(energies, e_max_zoom + hw_pump)))
)

fig, ax = plt.subplots(2, 2, figsize=(3, 3))
plt.tight_layout(pad=2)

im0 = ax[0, 0].imshow(
    data_2d, cmap="inferno", extent=extent, interpolation="nearest", aspect=aspect_ratio
)
ax[0, 0].set_xlabel(r"$r (\mu \mathrm{m})$")
ax[0, 0].set_ylabel(r"$\hbar\omega_{\mathrm{probe}} (\mathrm{meV})$")
ax[0, 0].set_xlim(0, rmax)
ax[0, 0].set_ylim(e_min, e_max)
ax[0, 0].add_patch(
    plt.Rectangle(
        (r_min_zoom, e_max_zoom),
        r_max_zoom - r_min_zoom,
        e_min_zoom - e_max_zoom,
        fill=False,
        edgecolor="white",
        linewidth=1.5,
    )
)

im1 = ax[0, 1].imshow(
    data_2d, cmap="inferno", extent=extent, interpolation="nearest", aspect=aspect_ratio
)
ax[0, 1].set_ylim(e_min_zoom, e_max_zoom)
ax[0, 1].set_xlim(r_min_zoom, r_max_zoom)
ax[0, 1].hlines(hw_1 - hw_pump, r_min_zoom, r_max_zoom, "r", label="Pump energy")
ax[0, 1].hlines(hw_2 - hw_pump, r_min_zoom, r_max_zoom, "b", label="Probe energy")

ax[1, 0].plot(r_axis, azim_avgs[0] / np.nanmax(azim_avgs[0]))
ax[1, 0].plot(r_axis, azim_avgs[1] / np.nanmax(azim_avgs[1]))
ax[1, 0].set_xlim(0, r_max_zoom)
ax[1, 0].set_ylim(0, 1)
ax[1, 0].set_aspect(8)

ax[1, 1].plot(energies - hw_pump, int_sum / np.max(int_sum))
ax[1, 1].set_xlim(e_min_zoom, e_max_zoom)
ax[1, 1].set_aspect(0.1)
plt.savefig("cavity_modes_%s_serie%s_p=%s.pdf" % (date, serie, str(p)), dpi=300)
# %% (SMFig5) OAI method

date_illu = "20250810"
serie_illu = 0
dr_illu = 60
m_illu = -6
r_illu = 30
p_illu = 0.0
hw_illu = energies[40]
hdf5_filename = os.path.join(
    folder,
    "Interferograms",
    f"{date_illu}_serie{serie_illu}",
    f"interferograms_m={m_illu}_p={p_illu}_r={r_illu}_dr={dr_illu}.h5",
)

# Open the HDF5 file
with h5py.File(hdf5_filename, "r") as hdf:
    # Retrieve the dataset and load the specific interferogram
    interferograms = hdf["interferograms"]  # Reference to the dataset
    im = interferograms[idx(energies, hw_illu)]  # Load only the specific slice

field_probe = im_osc_fast(im[:, :], center=(3 * im.shape[-2] // 4, im.shape[-1] // 4))
field_probe = np.roll(field_probe, center, axis=(0, 1))
field_probe = gaussian_filter(field_probe, 1.5 / pixel_cam)
amplitude_probe = np.abs(field_probe)
phase_probe = np.angle(field_probe)

extent = [-512 * pixel_cam, 512 * pixel_cam, -512 * pixel_cam, 512 * pixel_cam]

fig, ax = plt.subplots(1, 3, figsize=(5, 2))

# Plot interferogram with inset
im_main = ax[0].imshow(im / np.max(im), cmap="gray", extent=extent)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im_main, cax=cax0)
ax[0].set_title("Interferogram")
ax[0].set_xlabel(r"$x (\mu \mathrm{m})$")
ax[0].set_ylabel(r"$y (\mu \mathrm{m})$")

# Define the ROI to zoom in on (e.g., around the center)
y1, y2 = -22, -18  # µm
x1, x2 = -2, 2  # µm

# Create inset axes
axins = inset_axes(ax[0], width="45%", height="45%", loc="upper right", borderpad=2)
im_inset = axins.imshow(im / np.max(im), cmap="gray", extent=extent)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([])
axins.set_yticks([])

# Draw a box and lines linking inset to main image
mark_inset(ax[0], axins, loc1=2, loc2=4, fc="none", ec="red")

# Plot phase
im1 = ax[1].imshow(
    phase_probe, "twilight_shifted", interpolation="None", extent=extent, label="Phase"
)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im1, cax=cax1, ticks=[-np.pi + 0.001, 0, np.pi - 0.001])
cbar.ax.set_yticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax[1].set_xlabel(r"$x (\mu \mathrm{m})$")
ax[1].set_ylabel(r"$y (\mu \mathrm{m})$")
ax[1].set_title("Phase")

# Plot amplitude
im2 = ax[2].imshow(
    amplitude_probe / np.max(amplitude_probe), "gray", extent=extent, label="Amplitude"
)
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)
ax[2].set_xlabel(r"$x (\mu \mathrm{m})$")
ax[2].set_ylabel(r"$y (\mu \mathrm{m})$")
ax[2].set_title("Amplitude")

plt.subplots_adjust(left=0.13, right=0.90, wspace=1, hspace=0.1)
plt.savefig("probe_field_combined.pdf", dpi=300)
plt.close("all")

im_fft = np.fft.fft2(im)
im_fft = np.fft.fftshift(im_fft)

roi_far_field = 512
ky_roi = (
    2 * np.pi * np.fft.rfftfreq(im.shape[0], pixel_cam) * roi_far_field / (im.shape[0])
)
kx_roi = (
    2 * np.pi * np.fft.rfftfreq(im.shape[1], pixel_cam) * roi_far_field / (im.shape[1])
)
kx = 2 * np.pi * np.fft.rfftfreq(im.shape[1], pixel_cam)
ky = 2 * np.pi * np.fft.rfftfreq(im.shape[0], pixel_cam)
extent_k = [-kx[-1], kx[-1], -ky[-1], ky[-1]]
extent_k_roi = [-kx_roi[-1], kx_roi[-1], -ky_roi[-1], ky_roi[-1]]
im_fft_roi = im_fft[
    im_fft.shape[0] // 4
    - roi_far_field // 2 : im_fft.shape[0] // 4
    + roi_far_field // 2,
    3 * im_fft.shape[1] // 4
    - roi_far_field // 2 : 3 * im_fft.shape[1] // 4
    + roi_far_field // 2,
]

fig, ax = plt.subplots(1, 3, figsize=(6, 2))
vmax = 0.3 * 1e15
im0 = ax[0].imshow(im / np.max(im), "gray", extent=extent)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im0, cax=cax0)

im1 = ax[1].imshow(np.abs(im_fft) ** 2, "gray", extent=extent_k, vmax=vmax)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

im2 = ax[2].imshow(np.abs(im_fft_roi) ** 2, "gray", extent=extent_k_roi, vmax=vmax)
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

plt.subplots_adjust(left=0.13, right=0.90, wspace=1, hspace=0.1)
plt.savefig("probe_field_fft.pdf", dpi=300)
plt.close("all")
# %%
