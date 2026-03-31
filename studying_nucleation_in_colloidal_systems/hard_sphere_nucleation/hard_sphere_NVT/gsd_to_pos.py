'''import gsd.hoomd
import numpy as np

# Define input and output file names
input_filename = "npt_hpmc_output_traj.gsd"  # Input trajectory file
output_filename = "npt_hpmc_restart.pos"  # Output .pos file

# Open the GSD file and the output file
with gsd.hoomd.open(input_filename, mode='r') as traj, open(output_filename, "w") as fout:
    # Check if the file contains any frames
    if len(traj) == 0:
        print("Error: The GSD file contains no frames.")
        exit(1)

    # Access the last frame
    frame = traj[-1]

    # Extract box dimensions from the configuration
    # HOOMD box: [Lx, Ly, Lz, xy, xz, yz]
    box = frame.configuration.box
    Lx, Ly, Lz = box[0], box[1], box[2]
    xy, xz, yz = box[3], box[4], box[5]

    # Extract particle data from the frame
    positions = frame.particles.position  # Shape: (N, 3)
    diameters = frame.particles.diameter  # Array of diameters
    type_ids = frame.particles.typeid     # Array of integer type IDs
    type_names = frame.particles.types    # List of type names (e.g., ["A"])

    # Use the first particle's diameter as the sphere diameter
    diameter = diameters[0]

    # Write the box matrix line in the specified format
    # Format: "boxMatrix Lx xy xz 0 Ly yz 0 0 Lz"
    fout.write(f"boxMatrix {Lx:.11f} {xy:.11f} {xz:.11f} 0 {Ly:.11f} {yz:.11f} 0 0 {Lz:.11f}\n")

    # Write the definition line
    # Format: def A "sphere <diameter> 97d9fc"
    fout.write(f'def A "sphere {diameter:.14f} 97d9fc"\n')

    # Write a line for each particle
    for pos, tid in zip(positions, type_ids):
        p_type = type_names[tid]
        fout.write(f"{p_type} {pos[0]} {pos[1]} {pos[2]}\n")

    # End with eof
    fout.write("eof\n")

print(f"Output written to {output_filename}")'''





















import gsd.hoomd
import numpy as np

# Define input and output file names
input_filename = "hard_sphere_4096_nvt_hpmc_pf0p58_final.gsd"  # Input trajectory file
output_filename = "hard_sphere_4096_nvt_hpmc_pf0p58_final.pos"  # Output .pos file

# Open the GSD file for reading and the output file for writing
with gsd.hoomd.open(input_filename, mode='r') as traj, open(output_filename, "w") as fout:
    # Initialize a flag to handle the first frame
    first_frame = True
    
    # Loop over every frame in the trajectory
    for frame in traj:
        # If this is not the first frame, write 'eof' before starting the new frame's data
        if not first_frame:
            fout.write("eof\n")
        else:
            first_frame = False
        
        # Extract box dimensions from the configuration
        # HOOMD stores the box as a 6-element list: [Lx, Ly, Lz, xy, xz, yz]
        box = frame.configuration.box
        Lx, Ly, Lz = box[0], box[1], box[2]
        xy, xz, yz = box[3], box[4], box[5]
        
        # Write the box matrix line in the specified format
        # Format: "boxMatrix Lx xy xz 0 Ly yz 0 0 Lz"
        fout.write(f"boxMatrix {Lx:.11f} {xy:.11f} {xz:.11f} 0 {Ly:.11f} {yz:.11f} 0 0 {Lz:.11f}\n")
        
        # Extract particle data from the frame
        positions = frame.particles.position  # Shape: (N, 3)
        diameters = frame.particles.diameter   # Array of diameters
        type_ids = frame.particles.typeid      # Array of integer type IDs
        type_names = frame.particles.types     # List of type names (e.g., ["A"])
        
        # Use the first particle's diameter as the sphere diameter
        # (Assuming all particles have the same diameter)
        diameter = diameters[0]
        
        # Write the definition line for the particle type
        # Format: def A "sphere <diameter> 97d9fc"
        fout.write(f'def A "sphere {diameter:.14f} 97d9fc"\n')
        
        # Write a line for each particle
        for pos, tid in zip(positions, type_ids):
            p_type = type_names[tid]  # Get the type name from type ID
            fout.write(f"{p_type} {pos[0]} {pos[1]} {pos[2]}\n")

# Print a confirmation message
print(f"Output written to {output_filename}")
