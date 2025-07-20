import matplotlib

# Ensure the backend is set to 'TkAgg' for interactive plotting
if matplotlib.get_backend() != 'TkAgg':
	matplotlib.use('TkAgg')

import hopfield_model
from hopfield_model.typing import Ordering

from exercise.scripts.initialization import general_init
from exercise.scripts.part_a import run_part_a
from exercise.scripts.part_b import run_part_b
from exercise.scripts.part_c import run_part_c, NOISE_LIST
from exercise.scripts.part_d import run_part_d


# ───────────────────────────────────────────────────────────────────────
# For the Vector, Matrix type aliases, look at hopfield_model/typing.py
# ───────────────────────────────────────────────────────────────────────


def main():
	# General initialization: Load patterns and create the Hopfield network
	patterns, network, letter = general_init()
	
	# Part A: Train on first letter of my name and check if it is a fixed point
	print("Starting Part A: Training the network on a single letter (the first letter of my name):")
	run_part_a(network, patterns, letter, Ordering.SEQUENTIAL)
	print()
	
	# Part B: Perturb the pattern and run convergence
	print("Starting Part B: Perturbing the target state and observing convergence:")
	run_part_b(network, patterns, letter, noise_level=0.2, max_cycles=100, order=Ordering.SEQUENTIAL,
			   with_saving=hopfield_model.SAVING_MODE,
			   with_interactive=True,
			   shape=(10, 9),
			   file_base_name=f"convergence_to_letter_{letter}",
			   directory_path_lst=["exercise", "plots", "network_convergence_plots", f"letter_{letter}"],
			   extension="png",
			   dpi=300,
			   is_initial_training=False)
	print()
	
	# Part C: Calculate probability of convergence for different noise levels
	noise_level_list = NOISE_LIST  # Noise levels to test in Part C
	print("Starting Part C: Calculating probability of convergence for different noise levels:")
	run_part_c(network, patterns, letter,
			   noise_level_list,
			   max_cycles=100, order=Ordering.SEQUENTIAL, trials_per_noise=2500,
			   with_saving=hopfield_model.SAVING_MODE,
			   with_interactive=True,
			   filename=f"probability_of_convergence_letter_{letter}",
			   directory_path_lst=["	exercise", "plots", "probability_vs_noise_plots", f"letter_{letter}"],
			   extension="png",
			   dpi=300,
			   is_initial_training=False)
	print()
	
	# Part D: Incrementally train the network on letter patterns and check stability percentages
	print("Starting Part D: Incrementally training the network on letter patterns and checking stability:")
	run_part_d(network, patterns, order=Ordering.SEQUENTIAL,
			   with_saving=hopfield_model.SAVING_MODE,
			   with_interactive=True,
			   filename=f"stability_capacity_tradeoff",
			   directory_path_lst=["exercise", "plots", "stability_vs_capacity_plots"],
			   extension="png",
			   dpi=300)
	print()
	
	print("Assignment completed successfully!")


if __name__ == "__main__":
	main()
