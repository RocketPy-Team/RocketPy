import csv
import matplotlib.pyplot as plt
from docs.notebooks.monte_carlo_analysis.mc_example import test_run
import subprocess

def get_current_branch():
    # Run the git branch command
    result = subprocess.run(['git', 'branch'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Error running git command: {result.stderr}")

    # Parse the output to find the current branch
    branches = result.stdout.split('\n')
    for branch in branches:
        if branch.startswith('*'):
            return branch.split(' ')[1]

    raise Exception("Current branch not found")

def test_wrapper():
    case = str(get_current_branch()).replace("/", "_")
    n_workers = [2, 4, 6, 8, 10]
    n_sim = 1000
    append_mode = False
    
    print(f"Running test for {case}")
    
    time_array_light = []
    time_array_heavy = []
    for n in n_workers:
        print("Number of Workers:", n)
        time_array_light.append(test_run(n, n_sim, append_mode, True))
        time_array_heavy.append(test_run(n, n_sim, append_mode, False))
    
    # Export the results to a CSV file
    with open(f"results_{case}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_workers", "time"])
        for i in range(len(n_workers)):
            writer.writerow([n_workers[i], time_array_light[i], time_array_heavy[i]])
    
    # Plot the results
    plt.plot(n_workers, time_array_light, marker="o", label="Light")
    plt.plot(n_workers, time_array_heavy, marker="o", label="Heavy")
    plt.xlabel("Number of Workers")
    plt.ylabel("Time (s)")
    plt.title(f"Performance Analysis ({case})")
    plt.grid()
    plt.legend()
    plt.savefig(f"results_{case}.png")
    plt.show()
    
if __name__ == "__main__":
    test_wrapper()