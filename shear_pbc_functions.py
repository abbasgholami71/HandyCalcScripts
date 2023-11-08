import numpy as np
import itertools
from scipy.spatial import distance
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt

# this function collects the required information for all particles and then removes the displacement of the com of box
def box_displacement_remover(particles, data, masses_cation, masses_anion, box):

    cation_data = []
    anion_data = []

    for i in range(len(particles)):
        if data[i, 1] == 1:
            cation_data.append([particles[i, 0], # x-coordinate
                                   particles[i, 1], # y-coordinate
                                   particles[i, 2], # z-coordinate
                                   masses_cation[data[i, 2]], # mass
                                   data[i, 0] # molecule id
                                   ])

        elif data[i, 1] == -1:
            anion_data.append([particles[i, 0], # x-coordinate
                                   particles[i, 1], # y-coordinate
                                   particles[i, 2], # z-coordinate
                                   masses_anion[data[i, 2]], # mass
                                   data[i, 0] # molecule id
                                   ])

    # creating an array for all atoms
    particles_data = np.array(cation_data + anion_data)

    # calculating the com at the current frame
    com = np.average(particles_data[:, 0:3], axis=0, weights=particles_data[:, 3])

    # calculating the displacement of the com from the center of the box
    displacement = com - 0.5*box

    # adjusting particles' coordinate by removing the com displacement
    cation_data = np.array(cation_data)
    anion_data = np.array(anion_data)
    cation_data[:, 0:3] -= displacement
    anion_data[:, 0:3] -= displacement

    return cation_data, anion_data


# this function separates the whole and broken molecules in the group of cations or anions
def whole_separator(group, box):
    whole = []
    broken = []

    # Get all unique residue IDs
    residue_ids = np.unique(group[:, -1])

    for resid in residue_ids:
        molecule = group[group[:, -1]==resid]
        #print(molecule.atoms.masses)

        if is_whole(molecule[:, 0:3], box):
            whole.extend(np.column_stack([molecule[:, :]]))
        else:
            broken.extend(np.column_stack([molecule[:, :]]))

    return np.array(whole), np.array(broken)


# this function puts the original box on the lattice that means it adds 26 copies of the original box around it in 3D
def lattice(particles, box, molecule_size, shift):
    n_particles = len(particles)

    # defining the big box that we are going to fill with following order for data
    # [ x-coordinate, y-coordinate, z-coordinate, mass, residue_id/molecule_id ]
    big_box = np.zeros([27, n_particles, len(particles[0])])
    # the format of combinations is as below:
    # [-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
    # [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1], [0, -1, 0], [0, -1, 1],
    # [0, 0, -1], [0, 0, 0], [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1], [1 -1 -1],
    # [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]
    combinations = np.array(list(itertools.product([-1, 0, 1], repeat=3)))

    # iterating over all possible combinations for image boxes
    for i in range(len(combinations)):

        # iterating ove all atoms in the original box
        for j in range(n_particles):

            big_box[i, j, 0] = particles[j, 0] + combinations[i, 0] * box[0]  # x-coordinate
            big_box[i, j, 1] = particles[j, 1] + combinations[i, 1] * box[1]  # y-coordinate
            big_box[i, j, 2] = particles[j, 2] + combinations[i, 2] * box[2]  # z-coordinate
            big_box[i, j, 3] = particles[j, 3]  # mass
            big_box[i, j, 4] = particles[j, 4]  # molecule_id/residue_id

            # due to shear on top and bottom of the box, they need to be shifted at each frame
            if shift <= 0.5 * box[0]:
                big_box[i, j, 0] += shift * combinations[i, 2]
            else:
                big_box[i, j, 0] -= (box[0] - shift) * combinations[i, 2]

    # filter the broken molecules and just consider broken molecules inside a shell around the original box
    # bufffer shell around the original box: it should at least be larger than the size of a molecule
    result = filter_lattice(big_box, box, buffer=[molecule_size, molecule_size, molecule_size])

    return result


# This function detects if the molecule is whole or broken (with the broken dimension) in the box with pbc
def is_whole(coordinates, box):
    # Calculate the maximum distances in each direction
    dx = max(coordinates[:, 0]) - min(coordinates[:, 0])
    dy = max(coordinates[:, 1]) - min(coordinates[:, 1])
    dz = max(coordinates[:, 2]) - min(coordinates[:, 2])

    # Find the dimension that the molecule is broken
    pbc_break_direction = [int(dx > 0.5 * box[0]), int(dy > 0.5 * box[1]), int(dz > 0.5 * box[2])]

    # Identify whether a molecule is whole or broken
    result = all(element == 0 for element in pbc_break_direction)

    return result


# this function filters the lattice boxes by considering atoms within a shell of width "buffer" around the original box
def filter_lattice(lattice, box, buffer):
    # Create boolean masks for each dimension
    condition_x = (-buffer[0] < lattice[:, :, 0]) & (lattice[:, :, 0] < (box[0] + buffer[0]))
    condition_y = (-buffer[1] < lattice[:, :, 1]) & (lattice[:, :, 1] < (box[1] + buffer[1]))
    condition_z = (-buffer[2] < lattice[:, :, 2]) & (lattice[:, :, 2] < (box[2] + buffer[2]))

    # Combine the conditions to filter the lattice
    mask = condition_x & condition_y & condition_z

    # Use the mask to filter the lattice
    result = lattice[mask]

    return result


# this function gets the broken molecule on the lattice and finds those that are those of interest
# firstly, it filters to just those that are around the original box within a buffer and then cluster molecules and
# find those that their com is located inside the original box
def pbc_fixer(filtered_molecule, broken_original, box, molecule_size, molecule_atom_number):
    result = []

    # finding the indices of all broken molecules
    #indices = []
    #[indices.append(int(x)) for x in broken_original[:, -1] if int(x) not in indices]
    # Create a set of unique broken molecule indices
    indices = set(broken_original[:, -1].astype(int))

    # iterating over all broken molecules
    for i in indices:
        
        # separating atoms with same residue IDs
        atoms = filtered_molecule[filtered_molecule[:, 4] == i]
        
        # clustering atoms to identify separate molecules based on their interparticle distances
        molecules = cluster_atoms(atoms, molecule_atom_number, d=molecule_size)
        
        # find the target molecule among canidiates: the one that its com is inside the box
        final_molecule = target_molecule(molecules, atoms, box)
        
        for atom in final_molecule:
            result.append(atom)

    return np.array(result)


# this function cluster atoms according to their inter-particle distances and treat them as a molecule
def cluster_atoms(atoms, molecule_atom_number, d):

    # Compute pairwise distances between atoms
    pairwise_distances = distance.cdist(atoms[:, :3], atoms[:, :3])

    # Create a mask for atoms that are within distance 'd'
    within_distance = pairwise_distances <= d
    
    # Find connected components in the graph
    _, labels = connected_components(within_distance, directed=False)
    
    # Count the number of atoms in each cluster
    cluster_sizes = np.bincount(labels)
    
    # Find clusters with the desired number of atoms
    filtered_clusters = np.where(cluster_sizes == molecule_atom_number)[0]

    # Extract the atom indices for the filtered clusters
    cluster_indices = [np.where(labels == cluster)[0] for cluster in filtered_clusters]


    return cluster_indices


# this function picks the final target molecule by looking into their COM and checking if it is inside the original box
def target_molecule(molecule_list, all_atoms, box):
    
    for molecule in molecule_list:

        com = np.average(all_atoms[molecule, 0:3], axis=0, weights=all_atoms[molecule, 3])
        condition_x = 0 < com[0] < box[0]
        condition_y = 0 < com[1] < box[1]
        condition_z = 0 < com[2] < box[2]
        
        if condition_x and condition_y and condition_z:

            return all_atoms[molecule, :]

    return None


# this function calculates the atom number and the maximum size of cations and anions
def size_finder(all_atoms):

    # finding the indices of all broken molecules
    indices = set(all_atoms[:, -1].astype(int))
    #indices = []
    #[indices.append(int(x)) for x in all_atoms[:, -1] if int(x) not in indices]

    max_distance = np.zeros(len(indices))

    # getting atoms info in each molecule
    for index, element in enumerate(indices):
        atoms = all_atoms[all_atoms[:, 4] == element, 0:3]

        atom_number = len(atoms)

        # Calculate pairwise distances between all points
        distances = distance.pdist(atoms, 'euclidean')

        # Find the maximum distance in the distance matrix
        max_distance[index] = np.max(distance.squareform(distances))

    return atom_number, max(max_distance)


# this function calculates the COM of molecules
def com_finder(atoms):

    indices = np.unique(atoms[:, -1].astype(int))


    com = np.zeros([len(indices), 3])

    for i, value in enumerate(indices):

        selected_atoms = atoms[atoms[:, -1] == value]

        com[i] = np.average(selected_atoms[:, 0:3], axis=0, weights=selected_atoms[:, 3])

    return com


