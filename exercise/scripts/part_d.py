import matplotlib

# Ensure the backend is set to 'TkAgg' for interactive plotting
if matplotlib.get_backend() != 'TkAgg':
	matplotlib.use('TkAgg')

from typing import Dict, Optional, List
import hopfield_model
from hopfield_model.network import HopfieldNetwork
from hopfield_model.typing import Vector, Ordering
from hopfield_model.visualization.plotting import create_stability_percentage_figure, save_figure
from matplotlib import pyplot as plt
from exercise.scripts.initialization import general_init


def get_sorted_patterns(patterns: Dict[str, Vector]) -> List[Vector]:
	"""
	Returns a list of patterns sorted by their letter (alphabetically).
	
	Parameters:
		patterns (Dict[str, Vector]): Dictionary mapping letters to their patterns.
		
		Returns:
			List[Vector]: List of patterns sorted by their corresponding letters.
	"""
	sorted_items = sorted(patterns.items(), key=lambda item: item[0])  # Sort by letter
	return [pattern for _, pattern in sorted_items]


def run_part_d(network: HopfieldNetwork, patterns: Dict[str, Vector],
			   order: Ordering, with_saving: bool = False,
			   with_interactive: bool = True, filename: Optional[str] = None,
			   directory_path_lst: Optional[List[str]] = None,
			   extension: Optional[str] = None, dpi: Optional[int] = None) -> None:
	"""
	Run Part D: Incrementally train the network on letter patterns and check stability percentages.

	Parameters:
		network (HopfieldNetwork): Instance of the Hopfield network (already trained).
		patterns (Dict[str, Vector]): Dictionary of letter patterns.
		order (Ordering): Order of neuron updates (sequential or random).
		with_saving (bool): If True, enables saving plots to disk.
		with_interactive (bool): If True, enables interactive plotting.
		filename (Optional[str]): Base name for saving files, if applicable.
		directory_path_lst (Optional[List[str]]): Relative or absolute path to the directory where files will be saved (given as a list of strings), if applicable.
		extension (Optional[str]): File extension for saving plots, if applicable.
		dpi (Optional[int]): DPI for saving plots, if applicable.
	"""
	assert isinstance(network, HopfieldNetwork), "Network must be an instance of HopfieldNetwork."
	assert with_saving or with_interactive, "At least one of with_saving or with_interactive must be True."
	
	sorted_patterns = get_sorted_patterns(patterns)
	num_letters = len(sorted_patterns)
	stability_percentages = []
	curr_patterns: List[Vector] = []
	
	for t in range(1, num_letters + 1):
		new_pattern = sorted_patterns[t - 1]
		is_initial = (t == 1)
		
		# Train incrementally on the new pattern only
		network.train([new_pattern], is_initial)
		
		# Accumulate patterns for fixed point checking
		curr_patterns.append(new_pattern)
		
		# Count how many stored patterns remain fixed points
		stability_count = sum(network.check_if_fixed_point(p, order=order) for p in curr_patterns)
		stability_percentage = 100 * stability_count / len(curr_patterns)
		stability_percentages.append(stability_percentage)
		
		print(
				f"Stored {t} patterns → Stability: {stability_percentage:.2f}% ({stability_count}/{len(curr_patterns)})")
	
	# Plot results
	num_stored = list(range(1, num_letters + 1))
	fig = create_stability_percentage_figure(num_stored, stability_percentages)
	if with_saving:
		save_figure(fig, filename, directory_path_lst, extension, dpi)
	if with_interactive:
		plt.show()
	plt.close(fig)  # Close the figure to free up resources


if __name__ == "__main__":
	# Run only part D
	
	# General initialization: Load patterns and create the Hopfield network
	patterns_dict, new_network, _ = general_init()
	
	# Part D: Incrementally train the network on letter patterns and check stability percentages
	print("Starting Part D: Incrementally training the network on letter patterns and checking stability:")
	run_part_d(new_network, patterns_dict, order=Ordering.SEQUENTIAL,
			   with_saving=hopfield_model.SAVING_MODE,
			   with_interactive=True,
			   filename="stability_percentage_vs_num_stored_patterns",
			   directory_path_lst=["exercise", "plots", "stability_percentage_plots"],
			   extension="png",
			   dpi=300)
	print("Part D completed successfully.\n")
