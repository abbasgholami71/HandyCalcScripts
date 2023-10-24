# This script is designed to correct the broken molecules in the trajectory for shear simulations
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt


# This function calculates the maximum distance between a set of points in different dimensions
def calculate_max_distance(points):
    # Convert the points to a NumPy array for easy manipulation
    points_array = np.array(points)

    # Calculate the maximum distance for each dimension
    max_distance_x = np.max(points_array[:, 0]) - np.min(points_array[:, 0])
    max_distance_y = np.max(points_array[:, 1]) - np.min(points_array[:, 1])
    max_distance_z = np.max(points_array[:, 2]) - np.min(points_array[:, 2])

    return max_distance_x, max_distance_y, max_distance_z


# This function detects if the molecule is whole or broken (with the broken dimension) in the box with pbc
def is_whole(molecule, box):
    coordinates = molecule.atoms.positions

    dx, dy, dz = calculate_max_distance(coordinates)
    # finding the dimension that the molecule is broken
    pbc_break_direction = [int(dx > box[0] / 2), int(dy > box[1] / 2), int(dz > box[2] / 2)]
    # pbc_com_loc = [pbc_break_direction[0]*, pbc_break_direction[1]*, pbc_break_direction[2]*]
    result = all(element == 0 for element in pbc_break_direction)

    return result, pbc_break_direction


# This function return all indices of an array that are equal to a certain value
def find_all_indices(arr, target):
    return [i for i, elem in enumerate(arr) if elem == target]


# This function detect if the COM of the broken molecule is close to the upper or the lower edge of the box
def com_location(molecule, mid_point, index):
    coordinates = molecule.atoms.positions

    lower_weight = 0.0
    upper_weight = 0.0

    for i in range(len(coordinates)):
        if coordinates[i, index] < mid_point:
            # we have to normalize each atom's position by considering its distance from box edge
            lower_weight += (molecule.atoms.masses[i] * (coordinates[i, index]-0.0))

        if coordinates[i, index] > mid_point:
            # we have to normalize each atom's position by considering its distance from box edge
            upper_weight += (molecule.atoms.masses[i] * (2*mid_point-coordinates[i, index]))

    if lower_weight > upper_weight:
        return "lower"
    if lower_weight < upper_weight:
        return "upper"


# This function is supposed to shift the position (in x direction) of broken molecules (in z direction) in pbc
def image_box_shift(molecule, positions, box, shear, time):
    mid_point = box[2] / 2.0

    if com_location(molecule, mid_point, 2) == "lower":
        shift = (shear * time * box[2]) % box[0]
        for i in range(len(positions)):
            if positions[i, 2] > mid_point:
                if positions[i, 0] < shift:
                    positions[i, 0] = positions[i, 0] + box[0] - shift
                if positions[i, 0] > shift:
                    positions[i, 0] = positions[i, 0] - shift

    if com_location(molecule, mid_point, 2) == "upper":
        shift = box[0] - ((shear * time * box[2]) % box[0])
        for i in range(len(positions)):
            if positions[i, 2] < mid_point:
                if positions[i, 0] < shift:
                    positions[i, 0] = positions[i, 0] + box[0] - shift
                if positions[i, 0] > shift:
                    positions[i, 0] = positions[i, 0] - shift

    return positions


# This function calculates the COM for broken molecules
def com_calculator(molecule, box, pbc_cut, time, shear):
    indices = find_all_indices(pbc_cut, 1)
    coordinates = molecule.atoms.positions

    # loop over all broken dimensions
    for index in indices:

        mid_point = box[index] / 2.0

        # move the broken atoms from upper edge of the box to the lower edge
        if com_location(molecule, mid_point, index) == "lower":

            for i in range(len(coordinates)):
                if coordinates[i, index] > mid_point:
                    coordinates[i, index] -= box[index]

        # move the broken atoms from lowe edge of the box to the upper edge
        if com_location(molecule, mid_point, index) == "upper":

            for i in range(len(coordinates)):
                if coordinates[i, index] < mid_point:
                    coordinates[i, index] += box[index]

        # adjust positions in x direction when a molecule is broken in z direction according to the shift of image boxes
        if index == 2:
            coordinates = image_box_shift(molecule, coordinates, box, shear, time)

    com_molecule = np.average(coordinates, axis=0, weights=molecule.atoms.masses)
    return com_molecule


path = "E:\\Postdoc\\ionic_liquid\\simulations\\shear\\AA\\2x3x3_256iop_20ns\\set-ups\\"
shear_rates = [0]  # shear rate in x direction
trajectory = path + '{:.0e}'.format(shear_rates[0]).replace('e-0', 'e-') + "\\centered.xtc"
configuration = path + '{:.0e}'.format(shear_rates[0]).replace('e-0', 'e-') + "\\start.gro"
shear = shear_rates[0]

u = mda.Universe(configuration, trajectory)  # universe
box = u.dimensions[0:3]
com = np.zeros([u.trajectory.n_frames, len(u.residues), 3])

# Iterate over frames
for ts in u.trajectory:
    broken_molecules = []

    # iterate over all residues/molecules
    for res in range(1, max(u.atoms.resids) + 1):
        molecule = u.select_atoms('resid ' + str(res))

        # finding the COM for residue of those that are not broken due to pbc
        if is_whole(molecule, box)[0]:
            com[ts.frame, res - 1, :] = np.average(molecule.atoms.positions, axis=0, weights=molecule.atoms.masses)

        # finding the COM for residue of those that are broken due to pbc
        if not is_whole(molecule, box)[0]:
            pbc_cut = is_whole(molecule, box)[1]
            com[ts.frame, res - 1, :] = com_calculator(molecule, box, pbc_cut, ts.time, shear)

    print(ts.frame)
'''
    plt.plot(com[ts.frame, :, 0], com[ts.frame, :, 1], marker='o', linestyle='')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.axvline(x=box[0], color='r', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=box[1], color='r', linestyle='--')

    plt.show()
'''

# Write the numpy array to a text file
np.savetxt(path + '{:.0e}'.format(shear_rates[0]).replace('e-0', 'e-') + "\\com.txt", com.reshape(-1, com.shape[-1]), fmt='%.4f', delimiter='\t')