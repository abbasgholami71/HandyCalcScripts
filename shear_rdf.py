import numpy as np
import matplotlib.pyplot as plt


def calculate_distance(coordinates_ref, coordinates_sel, box, shear, time):
    delta_r = np.abs(coordinates_ref - coordinates_sel)

    # applying pbc in x direction
    delta_r[0] = delta_r[0] if delta_r[0] <= 0.5 * box[0] else box[0] - delta_r[0]
    # applying pbc in y direction
    delta_r[1] = delta_r[1] if delta_r[1] <= 0.5 * box[1] else box[1] - delta_r[1]
    # applying pbc in z direction
    # delta_r[2] = delta_r[2] if delta_r[2] <= 0.5 * box[2] else box[2] - delta_r[2]

    # shearing effect and applying pbc in z direction
    if delta_r[2] > 0.5 * box[2]:

        delta_r[2] = box[2] - delta_r[2]

        shift = (shear * time * box[2]) % box[0] #box[0] - ((shear * time * box[2]) % box[0])

        dr_x = coordinates_ref[0] - coordinates_sel[0]

        if coordinates_ref[2] > box[2]/2:
            delta_r[0] = np.abs(dr_x - shift) % box[0]
        if coordinates_ref[2] < box[2] / 2:
            delta_r[0] = np.abs(dr_x + shift) % box[0]

    # Calculate the distance between residues
    distance = np.linalg.norm(delta_r)

    return distance

def calculate_rdf(ref, sel, bin_width, max_distance, box, dt, type, shear):
    n_frames, n_particles, _ = ref.shape
    n_frames = 50
    rdf = np.zeros(int(max_distance / bin_width))

    for frame in range(n_frames):
        print(frame)
        time = (frame+1)*dt
        for ref_particle in range(len(ref[frame, :, 0])):
            for sel_particle in range(len(sel[frame, :, 0])):

                if (type == 'AMIM-PF6' or ref_particle != sel_particle):

                    distance = calculate_distance(ref[frame, ref_particle], sel[frame, sel_particle], box, shear, time)

                    if distance < max_distance:
                        bin_index = int(distance / bin_width)
                        rdf[bin_index] += 1  # Count each occurrence once


    # Normalize RDF
    shell_volume = 4/3 * np.pi * (np.arange(1, len(rdf) + 1) ** 3 - np.arange(0, len(rdf)) ** 3) * bin_width ** 3


    rdf = rdf/(2*n_particles*n_frames*shell_volume)

    return rdf



n_frames = 6667
n_residues = 512
dt = 3.0

path = "E:\\Postdoc\\ionic_liquid\\simulations\\shear\\AA\\2x3x3_256iop_20ns\\set-ups\\"
shear_rates = [1e-2]  # shear rate in x direction
file = path + '{:.0e}'.format(shear_rates[0]).replace('e-0', 'e-') + "\\com.txt"
shear = shear_rates[0]

box = [4.97147, 4.34942, 4.34942]

bin_width = 0.01
max_distance = 2.17
rdf_type = ['AMIM-AMIM', 'AMIM-PF6','PF6-PF6']

#read the com array
com = np.loadtxt(file, dtype=float, delimiter='\t').reshape((n_frames, n_residues, 3))/10.0

# iterate over different RDF types
for type in rdf_type:

    # set the reference and select groups of residues
    if type=='AMIM-AMIM':
        ref_group = com[:, :256, :]
        sel_group = com[:, :256, :]
    if type == 'AMIM-PF6':
        ref_group = com[:, :256, :]
        sel_group = com[:, 256:, :]
    if type=='PF6-PF6':
        ref_group = com[:, 256:, :]
        sel_group = com[:, 256:, :]


    RDF = calculate_rdf(ref_group, sel_group, bin_width, max_distance, box, dt, type, shear)

    # Plot RDF
    distances = np.arange(0, max_distance, bin_width)
    plt.plot(distances, RDF)
    plt.xlabel('Distance')
    plt.ylabel('Radial Distribution Function (RDF)')
    plt.title('Radial Distribution Function: '+type)
    plt.show()