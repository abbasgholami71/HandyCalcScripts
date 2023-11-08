# This script is designed to correct the broken molecules in the trajectory for shear simulations
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import h5py
import shear_pbc_functions as functions



path = "/ptmp/gholamia/slurm/Epp/09_2023/26/AA_4/1e-2/out/"
trajectory = path + "result.h5"
shear = 1e-2

with h5py.File(trajectory, 'r') as h5:
    # particles position [frame number, particle ID, [x, y, z]]
    position = np.array(h5["particles/position"][:, :, :])
    # particles index [frame number, particle ID, [molecule number, molecule name, atom ID in molecule]]
    # Unfortunately, I have done a mistake in my simulation script and the write_h5md function hasn't worked correctly
    # I have specified the number of pairs to be 256 which is not correct and here it is 630
    # thus, as the data are written in order at each step, I will define the index matrix below
    #index = np.array(h5["particles/index"][:, :, :])
    box = np.array(h5["box/size"][:]) # [Lx, Ly, Lz]
    h5.close()

pairs_number = 630
time_step = 1.5
n_frame = len(position)
cation_n = 25
anion_n = 7

# defining the feature matrix as the simulation failed to write the correct data in h5["particles/index"][:, :, :]
index = np.zeros([pairs_number*(cation_n+anion_n), 3])

index[:pairs_number*cation_n, 1] = 1
index[pairs_number*cation_n:, 1] = -1

for j in range(pairs_number*cation_n):
    index[j, 2] = j%cation_n + 1
    index[j, 0] = j//cation_n + 1
for j in range(pairs_number*cation_n, len(index)):
    index[j, 2] = j%anion_n + 1
    index[j, 0] = (j-pairs_number*cation_n)//anion_n + 1


# atoms' masses read from the topology file; E++ writes particles data in order for cations (1-25) and anions (1-7)
masses_cation = {1: 14.0067, 2: 14.0067, 3: 1.0080, 4: 1.0080, 5: 1.0080, 6: 1.0080, 7: 1.0080, 8: 1.0080, 9: 1.0080,
                 10: 1.0080, 11: 1.0080, 12: 1.0080, 13: 1.0080, 14: 1.0080, 15: 1.0080, 16: 1.0080, 17: 1.0080,
                 18: 12.0110, 19: 12.0110, 20: 12.0110, 21: 12.0110, 22: 12.0110, 23: 12.0110, 24: 12.0110, 25: 12.0110}
masses_anion = {1: 30.9738, 2: 18.9984, 3: 18.9984, 4: 18.9984, 5: 18.9984, 6: 18.9984, 7: 18.9984}

# matrix for cation and anion to store their xyz coordinates, masses, and molecule index
cation = np.zeros([n_frame, pairs_number*cation_n, 5])
anion = np.zeros([n_frame, pairs_number*anion_n, 5])
cation_com = np.zeros([n_frame, pairs_number, 3])
anion_com = np.zeros([n_frame, pairs_number, 3])



# Iterate over frames
for frame in range(n_frame):
    
    # collecting trajectory information and removing the displacement of com of the box
    # defining cation and anion at the current frame of the trajectory
    cations, anions = functions.box_displacement_remover(position[frame], index, masses_cation, masses_anion, box)
       
    # finding all whole and broken cations and anions and recording them in separate arrays
    cation_whole, cation_broken = functions.whole_separator(cations, box)
    anion_whole, anion_broken = functions.whole_separator(anions, box)

    # finding the maximum size and number of atoms of cations and anions
    cation_atom_number, cation_size = functions.size_finder(cation_whole)
    anion_atom_number, anion_size = functions.size_finder(anion_whole)

    # putting broken molecules on lattice
    cation_lattice = functions.lattice(cation_broken, box, 1.1*cation_size, shift=(shear*time_step*frame*box[2]) % box[0])
    anion_lattice = functions.lattice(anion_broken, box, 1.1*anion_size, shift=(shear*time_step*frame*box[2]) % box[0])
    #print(anion_broken)

    # picking the whole molecules which their com is located inside the original box
    cation_fixed = functions.pbc_fixer(cation_lattice, cation_broken, box, 1.1*cation_size, cation_atom_number)
    anion_fixed = functions.pbc_fixer(anion_lattice, anion_broken, box, 1.1*anion_size, anion_atom_number)

    # adding fixed boundary molecules to the whole molecules
    cation[frame, :, :] = np.vstack([cation_whole, cation_fixed])
    anion[frame, :, :] = np.vstack([anion_whole, anion_fixed])

    # calculating COM
    cation_com[frame] = functions.com_finder(cation[frame])
    anion_com[frame] = functions.com_finder(anion[frame])

    print(frame)


# Write the numpy array to a text file
np.savetxt(path + "cation_com.txt", cation_com.reshape(-1, cation_com.shape[-1]), fmt='%.4f', delimiter='\t')
np.savetxt(path + "anion_com.txt", anion_com.reshape(-1, anion_com.shape[-1]), fmt='%.4f', delimiter='\t')
np.savetxt(path + "cation_trajectory.txt", cation.reshape(-1, cation.shape[-1]), fmt='%.4f', delimiter='\t')
np.savetxt(path + "anion_trajectory.txt", anion.reshape(-1, anion.shape[-1]), fmt='%.4f', delimiter='\t')

'''
    plt.plot(functions.com_finder(cation_fixed)[:,0], functions.com_finder(cation_fixed)[:,2], marker="o", linestyle="", color="blue", markersize=1)
    plt.plot(functions.com_finder(anion_fixed)[:,0], functions.com_finder(anion_fixed)[:,2], marker="o", linestyle="", color="red", markersize=1)
    plt.axvline(x=0, ymin=0, ymax=box[2], color='g', linestyle='--')
    plt.axvline(x=box[0], ymin=0, ymax=box[2], color='g', linestyle='--')
    plt.axhline(y=0, xmin=0, xmax=box[0], color='g', linestyle='--')
    plt.axhline(y=box[2],xmin=0, xmax=box[0], color='g', linestyle='--')


    plt.show()
'''
