import time
import json
import numpy as np
from dask.distributed import Client
from ljmd_parallel_dask_1 import lj_force_vectorized , kin_energy_vectorized, verlet_vectorized

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    # Read the input file using json
    with open('input_2916.json') as f:
        data = json.load(f)
    print(f"Loaded {data['n_particles']} particles with {data['n_steps']} timesteps.")

    # Read initial positions using numpy, from the json file
    with open(f"data/{data['initial_state']}") as initial_state:
        x_i, y_i, z_i = np.genfromtxt(initial_state, unpack=True, max_rows=data['n_particles'])

    # Initialize the arrays to store the positions, velocities, and forces
    x_s = np.zeros((data['n_steps'], data['n_particles']))
    y_s = np.zeros((data['n_steps'], data['n_particles']))
    z_s = np.zeros((data['n_steps'], data['n_particles']))
    vx_s = np.zeros((data['n_steps'], data['n_particles']))
    vy_s = np.zeros((data['n_steps'], data['n_particles']))
    vz_s = np.zeros((data['n_steps'], data['n_particles']))
    fx_s = np.zeros((data['n_steps'], data['n_particles']))
    fy_s = np.zeros((data['n_steps'], data['n_particles']))
    fz_s = np.zeros((data['n_steps'], data['n_particles']))

    # Initialize the arrays to store the energies
    kinetic_energy = np.zeros(data['n_steps'])
    potential_energy = np.zeros(data['n_steps'])
    temperature = np.zeros(data['n_steps'])

    # Add the initial positions to the arrays
    x_s[0] = x_i
    y_s[0] = y_i
    z_s[0] = z_i

    # Define client
    client = Client(n_workers=data['n_workers'], processes=True)

    # Calculate the initial forces
    fx_s[0], fy_s[0], fz_s[0], potential_energy[0] = lj_force_vectorized(x_s[0], y_s[0], z_s[0])

    client.close()

    # Calculate the initial kinetic energy and temperature
    kinetic_energy[0], temperature[0] = kin_energy_vectorized(vx_s[0], vy_s[0], vz_s[0])

    # Print a title for the output
    print(f"Starting simulation...")
    print(f"Step T(K)   KE  PE  TE")
    # Open both the files of the positions and the energies
    with (open(data["trajectory_file"], 'w') as pos, open(data["energy_file"], 'w') as en):
        # Start the timer
        start = time.time()
        for i in range(1, data["n_steps"]):
            # Propagate the positions, velocities and forces
            x_s[i], y_s[i], z_s[i], vx_s[i], vy_s[i], vz_s[i], fx_s[i], fy_s[i], fz_s[i], potential_energy[
                i] = verlet_vectorized(x_s[i - 1], y_s[i - 1], z_s[i - 1], vx_s[i - 1], vy_s[i - 1], vz_s[i - 1],
                                       fx_s[i - 1], fy_s[i - 1], fz_s[i - 1], data['delta_t'])
            # Calculate the kinetic energy and temperature
            kinetic_energy[i], temperature[i] = kin_energy_vectorized(vx_s[i], vy_s[i], vz_s[i])
            # Write the positions and energies to the files
            pos.write(f"Step: {i}, Total energy: {kinetic_energy[i] + potential_energy[i]}\n")
            np.savetxt(pos, np.c_[x_s[i], y_s[i], z_s[i]])
            np.savetxt(en, np.c_[temperature[i], kinetic_energy[i], potential_energy[i],
                                 kinetic_energy[i] + potential_energy[i]])
            if i % data["output_freq"] == 0:
                print(f"{i} {temperature[i]:.3f} {kinetic_energy[i]:.3f} {potential_energy[i]:.3f} "
                      f"{kinetic_energy[i] + potential_energy[i]:.3f}")
        # Stop the timer
        end = time.time()
    print(f"Simulation finished in {end - start:.3f} seconds.")


if __name__ == "__main__":
    main()
