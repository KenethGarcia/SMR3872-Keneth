"""
A script to run the Lennard-Jones Molecular Dynamics simulation using for loops in a sequential way.
Created on September 2023 by Keneth Garcia at the 3rd SMR3872-Colombia School.
"""
# Importing libraries
import json
import time
import numpy as np


def lj_force_looping(x, y, z):
    fx, fy, fz, u = [0] * len(x), [0] * len(x), [0] * len(x), 0

    def pbc(r, box_size_by_2):
        """ Minimum image convention
        In this approach, the box have a length 2 * box_size_by_2, and if it is outside the box divided by 2, it is moved to the other side of the box, to avoid wall effects. See: https://www.ucl.ac.uk/~ucfbasc/Theory/pbc-mi.html

        It is required to have a cutoff radius lower than box_size_by_2.

        :param r: Position of the particle, can be x, y or z.
        :param box_size_by_2: Half of the box size.
        :return: The position of the particle, with the minimum image convention.
        """
        while r > box_size_by_2:
            r -= 2 * box_size_by_2
        while r < -box_size_by_2:
            r += 2 * box_size_by_2
        return r

    c12 = 4 * data['epsilon'] * (data['sigma'] ** 12)
    c6 = 4 * data['epsilon'] * (data['sigma'] ** 6)
    r_cut2 = data['r_cut'] * data['r_cut']
    for i in range(0, data['n_particles']):
        for j in range(i + 1, data['n_particles']):
            if i != j:
                dx = pbc(x[i] - x[j], data['box_size'] * 0.5)
                dy = pbc(y[i] - y[j], data['box_size'] * 0.5)
                dz = pbc(z[i] - z[j], data['box_size'] * 0.5)
                r2 = dx * dx + dy * dy + dz * dz

                if r2 < r_cut2:
                    r2_inv = 1.0 / r2
                    r6_inv = r2_inv * r2_inv * r2_inv
                    f_aux = (12.0 * c12 * r6_inv - 6.0 * c6) * r6_inv * r2_inv
                    u += (c12 * r6_inv - c6) * r6_inv
                    fx[i] += f_aux * dx
                    fy[i] += f_aux * dy
                    fz[i] += f_aux * dz
                    # Take into account 3rd Newton's law
                    fx[j] -= f_aux * dx
                    fy[j] -= f_aux * dy
                    fz[j] -= f_aux * dz
    return fx, fy, fz, u


def kin_energy_loop(vx, vy, vz):
    mvsq2e = 2390.05736153349  # m*v^2 in kcal/mol
    kboltz = 0.0019872067  # boltzmann constant in kcal/mol/K
    k = 0.0
    for i in range(0, data['n_particles']):
        k += 0.5 * mvsq2e * data['mass'] * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i])
    temp = 2.0 * k / (3.0 * data['n_particles'] - 3.0) / kboltz
    return k, temp


def verlet_loop(x, y, z, vx, vy, vz, fx, fy, fz, dt):
    mvsq2e = 2390.05736153349  # m*v^2 in kcal/mol
    aux = 0.5 / (data['mass'] * mvsq2e)
    # First, propagate the velocities by half and the positions
    for i in range(0, data['n_particles']):
        vx[i] += aux * fx[i] * dt
        vy[i] += aux * fy[i] * dt
        vz[i] += aux * fz[i] * dt
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt
        z[i] += vz[i] * dt
    # Then, calculate the forces
    fx, fy, fz, u = lj_force_looping(x, y, z)
    # Finally, propagate the velocities by half step
    for i in range(0, data['n_particles']):
        vx[i] += aux * fx[i] * dt
        vy[i] += aux * fy[i] * dt
        vz[i] += aux * fz[i] * dt
    return x, y, z, vx, vy, vz, fx, fy, fz, u


if __name__ == "__main__":
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

    # Calculate the initial forces
    fx_s[0], fy_s[0], fz_s[0], potential_energy[0] = lj_force_looping(x_s[0], y_s[0], z_s[0])

    # Calculate the initial kinetic energy and temperature
    kinetic_energy[0], temperature[0] = kin_energy_loop(vx_s[0], vy_s[0], vz_s[0])

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
                i] = verlet_loop(x_s[i - 1], y_s[i - 1], z_s[i - 1], vx_s[i - 1], vy_s[i - 1], vz_s[i - 1], fx_s[i - 1],
                                 fy_s[i - 1], fz_s[i - 1], data['delta_t'])
            # Calculate the kinetic energy and temperature
            kinetic_energy[i], temperature[i] = kin_energy_loop(vx_s[i], vy_s[i], vz_s[i])
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
