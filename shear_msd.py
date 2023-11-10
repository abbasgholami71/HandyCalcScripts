import numpy as np
import matplotlib.pyplot as plt

path = "/ptmp/gholamia/slurm/Epp/09_2023/26/"
simulations = ["AA_1", "AA_2", "AA_3", "AA_4", "AA_5", "AA_6", "AA_8", "AA_9"]
shear = ["0", "1e-2", "3e-3", "5e-4", "6e-3"]
frame_number = 13336
dt = 1.5
skipped_frame = int(1000/dt) # skip the first nanosecond
pairs_number = 630
box = [7.26001, 5.64668, 5.64668]
msd_type = ["cation", "anion"]

time_sequence = np.arange(0, frame_number-skipped_frame, 1)*dt/1000 #ns


# iterate over all simulations
for sim in simulations:

    # iterate over all shear rates
    for shear_rate in shear:

        # iterate over all MSD types
        for type in msd_type:

            # reading data for the COM
            com = np.genfromtxt(path + sim + "/" + shear_rate + "/out/" + type + "_com_unwrapped.txt", dtype=float, delimiter='\t').reshape((frame_number, pairs_number, 3))
            
            msd = np.zeros(frame_number - skipped_frame)

            for d_frame in range(1, frame_number - skipped_frame):
                print(d_frame)
                # calculating displacement and mean squared displacements
                # note that MSD for any time interval is calculated in such a way the that specific time window
                # is moveing in the whole time span. Thus, the MSD is more accurate for smaller time windows than larger time windows
                displacement = com[skipped_frame + d_frame:, :, :] - com[skipped_frame:-d_frame, :, :]
                msd[d_frame] = np.mean(np.square(displacement[:, :, 1])+np.square(displacement[:, :, 2]))

          
            # writing down the MSD 
            np.savetxt(path + "MSD/yz/" + type + "_" + shear_rate + "_" + sim  + ".txt", np.vstack((time_sequence, msd)).reshape(-1, np.vstack((time_sequence, msd)).shape[-1]), fmt='%.4f', delimiter='\t')

