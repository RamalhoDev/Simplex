import pulp
import sys


# Function to convert MPS to LP
def convert_mps_to_lp(mps_file_path, lp_file_path):
    # Read the MPS file
    _, prob = pulp.LpProblem.fromMPS(mps_file_path)

    # Write the problem to an LP file
    prob.writeLP(lp_file_path)
    print(f"Converted {mps_file_path} to {lp_file_path}")


# Example usage
mps_file_path = sys.argv[1]
lp_file_path = sys.argv[2]
convert_mps_to_lp(mps_file_path, lp_file_path)
