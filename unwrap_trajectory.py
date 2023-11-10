# This script is designed to unwrap the already recorded trajectory for the com of anions ad cations
import numpy as np
import unwrap_trajectory_functions as functions

path = "/ptmp/gholamia/slurm/Epp/09_2023/26/AA_4/1e-2/out/"
file_cation = path + "cation_com.txt"
file_anion = path + "anion_com.txt"
shear = 1e-2
frame_number = 13336
pairs_number = 630
dt = 1.5
box = np.array([7.26001, 5.64668, 5.64668])


# read the com array
com_cation = np.loadtxt(file_cation, dtype=float, delimiter='\t').reshape((frame_number, pairs_number, 3))
com_anion = np.loadtxt(file_anion, dtype=float, delimiter='\t').reshape((frame_number, pairs_number, 3))

# generate the unwrapped trajectory
cation_com_unwrapped = functions.unwrap_trajectory(com_cation, box, dt, shear)
anion_com_unwrapped = functions.unwrap_trajectory(com_anion, box, dt, shear)

# write out the unwrapped trajectory
np.savetxt(path + "cation_com_unwrapped.txt", cation_com_unwrapped.reshape(-1, cation_com_unwrapped.shape[-1]), fmt='%.4f', delimiter='\t')
np.savetxt(path + "anion_com_unwrapped.txt", anion_com_unwrapped.reshape(-1, anion_com_unwrapped.shape[-1]), fmt='%.4f', delimiter='\t')
