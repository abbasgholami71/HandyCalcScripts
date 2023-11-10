import numpy as np


# this function detects if the particle has crossed the edge of the box during a time step
def cross_check(position_new, position_old, half_box):
    # displacement of a residue in the original wrapped trajectory
    displacement = np.abs(position_new - position_old)

    # check if the residue crosses the box edge
    cross = (displacement > half_box).astype(int)

    return cross


# this function unwrap the positions when the particle has crossed any of the edges of the box
def normal_unwrpper(position_unwrapped_last,position_new, position_old, box):
    half_box = 0.5*box
    # this parameter is -1 if the residue left the box from bottom and +1 if it left from up
    sgn = np.sign(half_box - position_new)
    # displacement
    displacement = position_new + sgn * box - position_old
    # adding the correct displacement of the residue to its old position to get the new location
    position_unwrapped = position_unwrapped_last + displacement

    return position_unwrapped


# this function updates the position of the particle when it doesn't cross any edge of the box
def no_cross_unwrpper(position_unwrapped_last, position_new, position_old):
    # displacement
    displacement = position_new - position_old
    # adding the correct displacement of the residue to its old position to get the new location
    position_unwrapped = position_unwrapped_last + displacement

    return position_unwrapped


# this function adjust the particles position in x-direction when they cross the box in z-direction
def shear_x_corrector(x_new, x_old, z_new, box, shift):

    half_box = 0.5*box[2]

    # adjusting the location in x direction according to the shift of the neighbour box due to shear
    if z_new < half_box:
        if x_old > shift:
            displacement = shift
        elif x_old < shift:
            displacement = shift - box[0]
    else:
        if x_old > (box[0] - shift):
            displacement = box[0] - shift
        elif x_old < (box[0] - shift):
            displacement = -shift

    x_shifted = x_new + displacement
    return x_shifted


# this function unwrap the trajectory
def unwrap_trajectory(traj, box, dt, shear):

    n_frames, n_residues, _ = traj.shape

    traj_unwrapped = np.zeros_like(traj)

    traj_unwrapped[0, :, :] = traj[0, :, :]

    # iterate over all frames
    for frame in range(1, n_frames):

        print(frame)

        shift = (shear * (frame - 1) * dt * box[2]) % box[0]  # box shift due to the shear at the last step

        # iterate over all residues
        for i in range(n_residues):

            # check if the residue crosses the box edge
            cross_initial = cross_check(position_new=traj[frame, i, :],
                                        position_old=traj[frame-1, i, :],
                                        half_box=0.5*box)

            # check if it crosses the box in z-direction in the case of shear
            if cross_initial[2] and shear != 0:
                # correct the position of particle in x-direction according to the shear
                traj[frame, i, 0] = shear_x_corrector(x_new=traj[frame, i, 0],
                                                      x_old=traj[frame-1, i, 0],
                                                      z_new=traj[frame, i, 2],
                                                      box=box,
                                                      shift=shift)

            # check if the residue crosses the box edge
            cross = cross_check(position_new=traj[frame, i, :],
                                position_old=traj[frame-1, i, :],
                                half_box=0.5*box)

            for index, condition in enumerate(cross):

                # if not crossed
                if not condition:
                    # adding the correct displacement of the residue to its old position to get the new location
                    traj_unwrapped[frame, i, index] = \
                        no_cross_unwrpper(position_unwrapped_last=traj_unwrapped[frame-1, i, index],
                                          position_new=traj[frame, i, index],
                                          position_old=traj[frame-1, i, index])

                # if crossed
                if condition:
                    traj_unwrapped[frame, i, index] = \
                        normal_unwrpper(position_unwrapped_last=traj_unwrapped[frame-1, i, index],
                                        position_new=traj[frame, i, index],
                                        position_old=traj[frame-1, i, index],
                                        box=box[index])

    return traj_unwrapped
