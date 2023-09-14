import json
import numpy as np
import dask.array as da

with open('input_2916.json') as f:
    data = json.load(f)


def lj_force_vectorized(x, y, z):
    # # Calculate the distance between the particles
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dz = z[:, np.newaxis] - z[np.newaxis, :]
    # Apply the minimum image convention
    dx = dx - data['box_size'] * np.round(dx / data['box_size'])
    dy = dy - data['box_size'] * np.round(dy / data['box_size'])
    dz = dz - data['box_size'] * np.round(dz / data['box_size'])
    # Calculate the distance between the particles
    r2 = dx*dx + dy*dy + dz*dz
    # Set the conditions for the force
    rcut2 = data['r_cut'] * data['r_cut']
    conditions = np.logical_and(0.001 < r2, r2 < rcut2)
    # Calculate the force
    c12 = 4 * data['epsilon'] * (data['sigma']**12)
    c6 = 4 * data['epsilon'] * (data['sigma']**6)
    r2_inv = 1.0 / r2
    r6_inv = r2_inv * r2_inv * r2_inv
    f_aux = (12.0 * c12 * r6_inv - 6.0 * c6) * r6_inv * r2_inv
    # Apply the conditions
    f_aux[~conditions] = 0
    # Calculate the potential energy
    u = (c12 * r6_inv - c6) * r6_inv
    u[~conditions] = 0
    # Return the forces and the potential energy
    return np.sum(f_aux*dx, axis=1), np.sum(f_aux*dy, axis=1), np.sum(f_aux*dz, axis=1), 0.5 * np.sum(u)


def kin_energy_vectorized(vx, vy, vz):
    mvsq2e = 2390.05736153349  # m*v^2 in kcal/mol
    kboltz = 0.0019872067  # boltzmann constant in kcal/mol/K
    k = 0.5 * mvsq2e * data['mass'] * (vx * vx + vy * vy + vz * vz)
    temp = 2.0 * np.sum(k) / (3.0 * data['n_particles'] - 3.0) / kboltz
    return np.sum(k), temp


def verlet_vectorized(x, y, z, vx, vy, vz, fx, fy, fz, dt):
    x = da.from_array(x, chunks=data['n_particles'] // data['n_workers'])
    y = da.from_array(y, chunks=data['n_particles'] // data['n_workers'])
    z = da.from_array(z, chunks=data['n_particles'] // data['n_workers'])
    vx = da.from_array(vx, chunks=data['n_particles'] // data['n_workers'])
    vy = da.from_array(vy, chunks=data['n_particles'] // data['n_workers'])
    vz = da.from_array(vz, chunks=data['n_particles'] // data['n_workers'])
    fx = da.from_array(fx, chunks=data['n_particles'] // data['n_workers'])
    fy = da.from_array(fy, chunks=data['n_particles'] // data['n_workers'])
    fz = da.from_array(fz, chunks=data['n_particles'] // data['n_workers'])

    mvsq2e = 2390.05736153349  # m*v^2 in kcal/mol
    aux = 0.5 / (data['mass'] * mvsq2e)
    # First, propagate the velocities by half and the positions
    vx += aux * fx * dt
    vy += aux * fy * dt
    vz += aux * fz * dt
    x += vx * dt
    y += vy * dt
    z += vz * dt
    # Then, calculate the forces
    # fx, fy, fz, u = lj_force_vectorized(x, y, z)
    fx, fy, fz, u = lj_force_looping(x, y, z)
    # Finally, propagate the velocities by half step
    vx += aux * fx * dt
    vy += aux * fy * dt
    vz += aux * fz * dt
    return x, y, z, vx, vy, vz, fx, fy, fz, u
