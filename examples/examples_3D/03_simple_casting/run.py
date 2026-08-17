import os

# Use GPU 0. Remove this line to let JAX select the device.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import matplotlib.pyplot as plt

from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids_postprocess import load_data, create_2D_animation, create_2D_figure


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

input_manager = InputManager(
    "casting.json",
    "numerical_setup.json"
)

initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)


# ---------------------------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------------------------

path = sim_manager.output_writer.save_path_domain

quantities = [
    "real_density",
    "real_velocity",
    "real_pressure",
    "density",
    "velocity",
    "pressure",
    "levelset",
    "volume_fraction"
]

jxf_data = load_data(path, quantities)

cell_centers = jxf_data.cell_centers
times = jxf_data.times
data = jxf_data.data


# ---------------------------------------------------------------------------
# 3D CASE VISUALIZATION
#
# Show an XY slice through the center of the mold.
# ---------------------------------------------------------------------------

os.makedirs("images", exist_ok=True)

plot_dict = {
    "density": data["real_density"],
    "velocityY": data["real_velocity"][:, 1],
    "pressure": data["real_pressure"],
    "volume_fraction": data["volume_fraction"]
}

create_2D_animation(
    data_dict=plot_dict,
    cell_centers=cell_centers,
    times=times,
    nrows_ncols=(2, 2),
    plane="xy",
    plane_value=0.0,
    interval=100
)

create_2D_figure(
    data_dict=plot_dict,
    cell_centers=cell_centers,
    plane="xy",
    plane_value=0.0,
    nrows_ncols=(2, 2),
    dpi=200,
    save_fig="images/casting_final.png"
)


# ---------------------------------------------------------------------------
# CASTING METRICS
# ---------------------------------------------------------------------------

X, Y, Z = np.meshgrid(
    cell_centers[0],
    cell_centers[1],
    cell_centers[2],
    indexing="ij"
)

dx, dy, dz = np.asarray(jxf_data.cell_sizes)
cell_volume = dx * dy * dz


# In this case the negative material is molten metal.
#
# JAX-Fluids' volume_fraction field follows the level-set representation.
# The following expression extracts the metal fraction from the saved data.
metal_fraction = 1.0 - data["volume_fraction"]

metal_density = data["density"][:, 1]
metal_velocity_y = data["velocity"][:, 1, 1]

metal_mass_density = metal_fraction * metal_density
metal_mass = metal_mass_density * cell_volume

total_metal_mass = np.sum(
    metal_mass,
    axis=(-3, -2, -1)
)

metal_com_y = (
    np.sum(
        metal_mass * Y,
        axis=(-3, -2, -1)
    )
    / total_metal_mass
)

metal_vertical_velocity = (
    np.sum(
        metal_mass * metal_velocity_y,
        axis=(-3, -2, -1)
    )
    / total_metal_mass
)


# ---------------------------------------------------------------------------
# METRICS PLOT
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(
    1,
    3,
    figsize=(10, 3)
)

ax[0].plot(times, total_metal_mass)
ax[0].set_title("Molten-metal mass")
ax[0].set_xlabel("time")

ax[1].plot(times, metal_com_y)
ax[1].set_title("Metal center of mass")
ax[1].set_xlabel("time")
ax[1].set_ylabel("y")

ax[2].plot(times, metal_vertical_velocity)
ax[2].set_title("Mean vertical velocity")
ax[2].set_xlabel("time")
ax[2].set_ylabel("v_y")

plt.tight_layout()
plt.savefig(
    "images/casting_metrics.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()
plt.close()
