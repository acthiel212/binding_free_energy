def restart_simulation(simulation, checkpoint_filename):
    # Initialize checkpoint parameters
    loadCheckpoint(simulation, checkpoint_filename)

def loadCheckpoint(simulation, filename):

    simulation.loadCheckpoint(filename)
    print(f"States loaded from {filename}")

# Function to generate checkpoint filename
def get_checkpoint_filename(prefix):
    return f"{prefix}.chk"