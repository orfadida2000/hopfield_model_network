import math
from typing import Tuple, Optional, List, Dict
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

import os
from hopfield_model.utilities import (get_unique_plot_filename, assert_valid_pattern, is_valid_fs_name,
									  is_valid_plt_extension)
from hopfield_model.typing import Vector
from hopfield_model.visualization.typing import (NetworkPlotType, MPLFig, Axes, AxesImage, BlackColor,
												 WhiteColor, PlottingVector)


# ───────────────────────────────────────────────────────────────────────
# For the Vector, Matrix type aliases, look at hopfield_model/typing.py
# ───────────────────────────────────────────────────────────────────────


def create_generic_figure(fig_title: str, fig_size: Tuple[float, float],
						  x: PlottingVector, y: PlottingVector,
						  x_label: str, y_label: str,
						  x_lim: Tuple[float, float], y_lim: Tuple[float, float],
						  x_ticks: Optional[Tuple[float, float, float]] = None,
						  y_ticks: Optional[Tuple[float, float, float]] = None,
						  tight_layout_args: Optional[Dict] = None,
						  window_title: Optional[str] = None) -> MPLFig:
	"""
	Creates a generic Matplotlib figure with specified parameters and returns it.
	
	Parameters:
		fig_title (str): Title of the plot.
		fig_size (Tuple[float, float]): Size of the figure in inches, as a tuple (width, height).
		x (PlottingVector): X-axis data.
		y (PlottingVector): Y-axis data.
		x_label (str): Label for the x-axis.
		y_label (str): Label for the y-axis.
		x_lim (Tuple[float, float]): Limits for the x-axis, as a tuple of (min, max).
		y_lim (Tuple[float, float]): Limits for the y-axis, as a tuple of (min, max).
		x_ticks (Optional[Tuple[float, float, float]]): Optional ticks for the x-axis, as a tuple of (start, end, step).
		y_ticks (Optional[Tuple[float, float, float]]): Optional ticks for the y-axis, as a tuple of (start, end, step).
		tight_layout_args (Optional[Dict]): Optional dictionary of arguments for `fig.tight_layout()`.
		window_title (Optional[str]): Optional title for the figure window.
		
	Returns:
		MPLFig: Matplotlib figure object containing the plot.
	
	"""
	assert isinstance(fig_title, str) and len(fig_title) > 0, "fig_title must be a non-empty string."
	assert isinstance(fig_size, Tuple) and len(fig_size) == 2 and all(
			isinstance(x, (int, float)) and x > 0 for x in
			fig_size), "fig_size must be a tuple of two positive numbers."
	assert isinstance(tight_layout_args,
					  (type(None), Dict)), "tight_layout_args must be a dictionary or None."
	assert isinstance(window_title, type(None)) or (isinstance(window_title, str) and len(
			window_title) > 0), "window_title must be a non-empty string or None."
	
	if tight_layout_args is None:
		tight_layout_args = {}
	
	fig, ax = plt.subplots(figsize=fig_size)
	
	ax.plot(x, y, marker='o', linestyle='-', color='tab:blue')
	ax.set_xlabel(x_label, fontsize=12)
	ax.set_ylabel(y_label, fontsize=12)
	ax.set_xlim(*x_lim)
	ax.set_ylim(*y_lim)
	if x_ticks is not None:
		ax.set_xticks(np.arange(*x_ticks))
	if y_ticks is not None:
		ax.set_yticks(np.arange(*y_ticks))
	
	ax.grid(True, linestyle='--', alpha=0.7)
	
	fig.suptitle(fig_title, fontsize=16, y=0.98)
	fig.tight_layout(**tight_layout_args)
	
	if window_title is not None:
		fig.canvas.manager.set_window_title(window_title)
	return fig


def create_stability_percentage_figure(num_stored: List[int], stability_percentages: List[float]) -> MPLFig:
	"""
	Creates a Matplotlib figure showing the percentage of stable letters as a function of the number of stored letters, and returns it.
	
	Parameters:
		num_stored (List[int]): List of integers representing the number of stored letters.
		stability_percentages (List[float]): List of percentages of stable letters corresponding to the number of stored letters.
		
	Returns:
		MPLFig: Matplotlib figure object containing the plot.
	
	"""
	assert isinstance(num_stored, list) and len(num_stored) > 0 and all(isinstance(x, int) and x > 0 for x in
																		num_stored), "num_stored must be a non-empty list of positive integers."
	assert isinstance(stability_percentages, list) and len(stability_percentages) > 0 and all(
			isinstance(x, float) and 0 <= x <= 100 for x in
			stability_percentages), "stability_percentages must be a non-empty list of floats between 0 and 100."
	
	fig = create_generic_figure(
			fig_title="Percentage of Stable Letters vs. Number of Stored Letters",
			fig_size=(10, 6),
			x=num_stored,
			y=stability_percentages,
			x_label="Number of Stored Letters (Capacity)",
			y_label="Stability Percentage (%)",
			x_lim=(-1, max(num_stored) + 1),
			y_lim=(-5, 105),
			x_ticks=(0, max(num_stored) + 1, 1),
			y_ticks=(0, 101, 10),
			tight_layout_args={'rect': (0, 0, 1, 0.96)},
			window_title="Stability Capacity Tradeoff Plot"
			)
	
	return fig


def create_probabilities_trajectory_figure(noise_levels: NDArray[np.float64],
										   probabilities: NDArray[np.float64], target_letter: str) -> MPLFig:
	"""
	Creates a Matplotlib figure showing the probability of convergence as a function of noise levels, and returns it.

	Parameters:
		noise_levels (NDArray[np.float64]): Sorted 1D array of noise levels as percentages (0 to 100).
		probabilities (NDArray[np.float64]): Corresponding 1D array of probabilities of convergence (0 to 1).
		target_letter (str): The letter for which the probabilities of convergence are calculated on.
	
	Returns:
		MPLFig: Matplotlib figure object containing the plot.
	"""
	assert isinstance(noise_levels,
					  np.ndarray) and noise_levels.ndim == 1 and noise_levels.dtype == np.float64 and len(
			noise_levels) > 0, "Noise levels must be a 1D numpy array of floats."
	assert np.all((noise_levels >= 0) & (noise_levels <= 100)), "Noise levels must be between 0 and 100."
	assert isinstance(probabilities,
					  np.ndarray) and probabilities.ndim == 1 and probabilities.dtype == np.float64 and len(
			probabilities) > 0, "Probabilities must be a 1D numpy array of floats."
	assert np.all((probabilities >= 0) & (probabilities <= 1)), "Probabilities must be between 0 and 1."
	assert len(noise_levels) == len(
			probabilities), "Noise levels and probabilities must have the same length."
	
	fig = create_generic_figure(
			fig_title=f"Probability of Convergence vs. Noise Level of '{target_letter}'",
			fig_size=(10, 6),
			x=noise_levels,
			y=probabilities,
			x_label="Noise Level (%)",
			y_label="Probability of Convergence",
			x_lim=(-5, 105),
			y_lim=(-0.05, 1.05),
			x_ticks=(0, 101, 10),
			y_ticks=(0, 1.1, 0.1),
			tight_layout_args={'rect': (0, 0, 1, 0.96)},
			window_title="Probability of Convergence Plot"
			)
	
	return fig


def create_letter_figure(state: Vector, shape: Tuple[int, int],
						 title: Optional[str] = None) -> Tuple[MPLFig, Axes, AxesImage]:
	"""
	Creates a Matplotlib figure displaying a 1D vector as a grayscale image and returns the figure, axes, and axes image objects.

	Parameters:
		state (Vector): Vector of values in {-1, +1}.
		shape (tuple[int, int]): Shape to reshape the vector into.
		title (Optional[str]): Optional plot title.
	
	Returns:
		Tuple[MPLFig, Axes, AxesImage]: Matplotlib figure, axes, and axes image objects.
	"""
	assert (isinstance(shape, Tuple) and len(shape) == 2 and
			isinstance(shape[0], int) and shape[0] > 0 and
			isinstance(shape[1], int) and shape[1] > 0), \
		"Shape must be a tuple of two positive integers."
	assert_valid_pattern(state, shape[0] * shape[1])
	assert isinstance(title, (str, type(None))), "Title must be a string or None."
	
	# Convert -1 to white (the value 1), +1 to black (the value 0).
	image = np.where(state == -1, WhiteColor, BlackColor).reshape(shape)
	fig, ax = plt.subplots(figsize=(6, 6))
	im = ax.imshow(image, cmap='gray', vmin=0, vmax=1, aspect='equal')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.tick_params(left=False, bottom=False)  # Hide ticks
	
	if title:
		fig.suptitle(title, fontsize=16, y=0.98)
	fig.subplots_adjust(top=0.85)  # Adjust top margin for title
	fig.canvas.manager.set_window_title(
			"Hopfield Network Dynamics Plot")  # Set the window title for the figure
	return fig, ax, im


def make_network_plot_title(t: int, plot_type: NetworkPlotType, steps_per_cycle: int) -> str:
	"""
	Generates a title for the network state plot based on the time step.
	
	Parameters:
		t (int): Time step, must be a non-negative integer.
		plot_type (NetworkPlotType): Type of the network plot, used to determine the title format.
		steps_per_cycle (int): Number of steps per cycle, used to determine the title format.
	
	Returns:
		str: Title for the plot.
	"""
	
	assert isinstance(t, int) and t >= 0, "Time step t must be a non-negative integer."
	assert isinstance(steps_per_cycle,
					  int) and steps_per_cycle > 0, "steps_per_cycle must be a positive integer."
	assert isinstance(plot_type, NetworkPlotType), "plot_type must be an instance of NetworkPlotType."
	
	vec_str = rf'$\vec{{S}}({t})$'
	if plot_type == NetworkPlotType.INITIAL:
		title = f"Initial state: {vec_str}"
	else:  # Means plot_type is NetworkPlotType.INTERMEDIATE or NetworkPlotType.FINAL
		if t == 1:
			title = f"State after 1 Update (cycle=1): {vec_str}"
		else:
			title = f"State after {t} Updates (cycle={math.ceil(t / steps_per_cycle)}): {vec_str}"
	
	return title


def figure_saving_assertion(filename: str, directory_path_lst: List[str], extension: str, dpi: int):
	assert is_valid_fs_name(
			filename), "File base name and directory name must be valid file system names."
	assert isinstance(directory_path_lst, list) and all(
			is_valid_fs_name(directory) for directory in
			directory_path_lst), "directory_path_lst must be a list of valid file system names."
	assert is_valid_plt_extension(
			extension), "Extension must be a valid matplotlib file extension (e.g., 'png', 'pdf')."
	assert isinstance(dpi, int) and dpi > 0, "DPI must be a positive integer."


def save_figure(fig: MPLFig, filename: str, directory_path_lst: List[str], extension: str, dpi: int):
	assert isinstance(fig, MPLFig), "fig must be a Matplotlib Figure instance."
	figure_saving_assertion(filename, directory_path_lst, extension, dpi)
	if len(directory_path_lst) > 0:
		directory_path = os.path.join(*directory_path_lst)
		os.makedirs(directory_path, exist_ok=True)  # Ensure the plots directory exists
		file_path = os.path.join(directory_path, filename)
	else:
		file_path = filename
	# Save the figure with a unique filename
	final_path = get_unique_plot_filename(file_path, extension)
	fig.savefig(final_path, dpi=dpi)
