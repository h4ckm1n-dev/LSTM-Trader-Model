import subprocess
import os

# Get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))

# List of scripts to run
scripts_to_run = ["inspect_csv.py", "preprocess.py", "inspect_data.py", "train.py"]

# Run each script
for script in scripts_to_run:
    script_path = os.path.join(current_directory, script)
    subprocess.run(["python3", script_path])
