import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Optional, List
from hopfield_model.utilities import assert_valid_pattern
import hopfield_model  # for DEBUG_MODE
from hopfield_model.typing import Vector
from hopfield_model.visualization.plotting import (create_letter_figure, make_network_plot_title, save_figure,
												   figure_saving_assertion)
from hopfield_model.visualization.typing import (NetworkPlotType, MPLFig, Axes, AxesImage, BlackColor,
												 WhiteColor)


# ───────────────────────────────────────────────────────────────────────
# For the Vector, Matrix type aliases, look at hopfield_model/typing.py
# ───────────────────────────────────────────────────────────────────────

class PlotController:
	"""
	Controller for plotting network states during asynchronous updates based on the plotting logic.
	Handles plotting conditions, step tracking, and visualization of network states.
	
	Attributes:
		with_visualization (bool): Whether to enable visualization, i.e. interactive plotting and saving.
		with_saving (bool): Whether to save plots to disk.
		with_interactive (bool): Whether to plot interactively.
		shape (Optional[Tuple[int, int]]): Shape of the network state for visualization, if visualization is enabled.
		file_base_name (Optional[str]): Base name for saving files, if saving is enabled.
		directory_path_lst (Optional[List[str]]): Relative or absolute path to the directory where files will be saved (given as a list of strings), if saving is enabled.
		extension (Optional[str]): File extension for saving plots, if saving is enabled.
		dpi (Optional[int]): DPI for saving plots, if saving is enabled.
		curr_step (int): Current step number in the simulation which is about to be performed.
		last_visualized_step (int): Last step at which a plot was generated or saved, reflecting the network state after it was updated.
		fig (Optional[MPLFig]): Matplotlib figure object for the general plot, if visualization is enabled.
		ax (Optional[Axes]): Matplotlib axes object for the general plot, if visualization is enabled.
		im (Optional[AxesImage]): Matplotlib axes image object for the network state visualization, if visualization is enabled.
	"""
	
	def __init__(self, with_visualization: bool, with_saving: bool, with_interactive: bool,
				 shape: Optional[Tuple[int, int]], file_base_name: Optional[str],
				 directory_path_lst: Optional[List[str]], extension: Optional[str],
				 dpi: Optional[int]):
		"""
		Initializes the Plot Controller with plotting settings and network shape.
		
		Parameters:
			with_visualization (bool): Whether to enable visualization, i.e. interactive plotting and saving.
			with_saving (bool): Whether to save plots to disk.
			with_interactive (bool): Whether to plot interactively.
			shape (Optional[Tuple[int, int]]): Shape of the network state for visualization, if applicable.
			file_base_name (Optional[str]): Base name for saving files, if applicable.
			directory_path_lst (Optional[List[str]]): Relative or absolute path to the directory where files will be saved (given as a list of strings), if applicable.
			extension (Optional[str]): File extension for saving plots, if applicable.
			dpi (Optional[int]): DPI for saving plots, if applicable.
		"""
		self._init_assertion(with_visualization, with_saving, with_interactive, shape, file_base_name,
							 directory_path_lst, extension, dpi)
		
		self.with_visualization: bool = with_visualization
		self.with_saving: bool = with_saving
		self.with_interactive: bool = with_interactive
		self.shape: Tuple[int, int] = shape
		self.file_base_name: Optional[str] = file_base_name  # Base name for saving files, if applicable.
		self.directory_path_lst: Optional[
			List[str]] = directory_path_lst  # Directory path for saving files (as a list), if applicable.
		self.extension: Optional[str] = extension  # File extension for saving plots, if applicable.
		self.dpi: Optional[int] = dpi  # DPI for saving plots, if applicable.
		self.curr_step: int = 0  # Initial step is 0, before the first update.
		self.last_visualized_step: int = -1  # No plots have been made yet.
		self.fig = None
		self.ax = None
		self.im = None
		
		if self.with_visualization:
			plt.close("all")  # Close all existing figures to start fresh.
			dummy_state = np.ones(self.shape[0] * self.shape[1],
								  dtype=int)  # Dummy state for initial plotting
			fig, ax, im = create_letter_figure(dummy_state, self.shape)
			self.fig: MPLFig = fig  # Store the figure for later updates
			self.ax: Axes = ax  # Store the axis for later updates
			self.im: AxesImage = im  # Store the image for later updates
	
	@staticmethod
	def _init_assertion(with_visualization: bool, with_saving: bool, with_interactive: bool,
						shape: Optional[Tuple[int, int]],
						file_base_name: Optional[str], directory_path_lst: Optional[List[str]],
						extension: Optional[str],
						dpi: Optional[int]) -> None:
		"""
		Initial assertions to validate the parameters for the __init__ method of PlotController.
		
		Parameters:
			with_visualization (bool): Whether to enable visualization, i.e. interactive plotting and saving.
			with_saving (bool): Whether to save plots to disk.
			with_interactive (bool): Whether to plot interactively.
			shape (Optional[Tuple[int, int]]): Shape of the network state for visualization, if applicable.
			file_base_name (Optional[str]): Base name for saving files, if applicable.
			directory_path_lst (Optional[List[str]]): Relative or absolute path to the directory where files will be saved (given as a list of strings), if applicable.
			extension (Optional[str]): File extension for saving plots, if applicable.
			dpi (Optional[int]): DPI for saving plots, if applicable.
		"""
		
		assert isinstance(with_visualization, bool), "with_visualization must be a boolean."
		assert isinstance(with_saving, bool), "with_saving must be a boolean."
		assert isinstance(with_interactive, bool), "with_interactive must be a boolean."
		if with_visualization:
			assert (isinstance(shape, Tuple) and len(shape) == 2 and
					isinstance(shape[0], int) and shape[0] > 0 and
					isinstance(shape[1], int) and shape[1] > 0), \
				"Shape must be a tuple of two positive integers."
			assert shape[0] > 0 and shape[1] > 0, "Shape dimensions must be positive integers."
			assert with_saving or with_interactive, "If visualization is enabled, at least one of saving or interactive must also be enabled."
		if with_saving:
			figure_saving_assertion(file_base_name, directory_path_lst, extension, dpi)
	
	def _visualize_state(self, state: Vector, plot_type: NetworkPlotType) -> None:
		"""
		Unified plotting flow for initial, intermediate, and final plotting.

		- Updates `im` with new state.
		- Updates title.
		- Optionally saves figure.
		- Optionally updates interactive window.

		Parameters:
			state (Vector): Current network state.
			plot_type (NetworkPlotType): Type of the plot, e.g., 'initial', 'intermediate', 'final'.
		"""
		# --- Update figure content ---
		assert_valid_pattern(state, self.shape[0] * self.shape[1])
		assert isinstance(plot_type, NetworkPlotType), "plot_type must be an instance of NetworkPlotType."
		
		# Create image by converting -1 to white (the value 1), +1 to black (the value 0) and reshaping it to the specified shape.
		image = np.where(state == -1, WhiteColor, BlackColor).reshape(self.shape)
		self.im.set_data(image)  # Update the image data in the AxesImage object
		# Generate title based on step and plot type
		title = make_network_plot_title(self.curr_step, plot_type, self.shape[0] * self.shape[1])
		self.fig.suptitle(title, fontsize=16, y=0.98)  # Update the figure title
		
		# --- Saving ---
		if self.with_saving:
			filename = f"{self.file_base_name}_step_{self.curr_step}"
			save_figure(self.fig, filename, self.directory_path_lst, self.extension, self.dpi)
		
		# --- Interactive GUI ---
		if self.with_interactive:
			if plot_type == NetworkPlotType.INITIAL:
				plt.show(block=False)  # Creates window + renders initially
				plt.pause(0.3)  # Ensure GUI responsiveness
			elif plot_type == NetworkPlotType.INTERMEDIATE:
				self.fig.canvas.draw_idle()  # Async refresh
				plt.pause(0.3)  # Ensure GUI responsiveness
			elif plot_type == NetworkPlotType.FINAL:
				print("Close the interactive plot window to end the simulation.")
				plt.show(block=True)  # Blocking final view, no need for draw_idle/pause
		
		self.last_visualized_step = self.curr_step  # Update last output step to current step
	
	def save_and_plot_initial_if_needed(self, state: Vector):
		if self.with_visualization:
			self._visualize_state(state, NetworkPlotType.INITIAL)
		self.curr_step += 1
	
	def save_and_plot_intermediate_if_needed(self, curr_state: Vector, prev_state: Vector):
		
		should_visualize = self.with_visualization and (
				hopfield_model.DEBUG_MODE or not np.array_equal(curr_state, prev_state))
		if should_visualize:
			self._visualize_state(curr_state, NetworkPlotType.INTERMEDIATE)
		self.curr_step += 1
	
	def save_and_plot_final_if_needed(self, state: Vector):
		# Final step was already performed, so curr_step reflect the next step number to be performed (in this case, there isn't any).
		final_step = self.curr_step - 1
		self.curr_step = final_step  # Update curr_step to the final step number that was performed.
		
		if not self.with_visualization:
			return
		
		if self.last_visualized_step != final_step:  # Network State at the end of the simulation was not visualized yet.
			self._visualize_state(state, NetworkPlotType.FINAL)
		else:
			# Network State at the end of the simulation was visualized already, so no need to save it again (if with_saving is enabled),
			# but a final blocking interactive plot is still needed (if with_interactive is enabled).
			self.with_saving = False  # No need to save again, so disable saving.
			self._visualize_state(state, NetworkPlotType.FINAL)
	
	def close_figure_if_needed(self) -> None:
		"""
		Close the figure if it exists and reset the figure, axes, and image attributes to None for safe cleanup.
	 	"""
		if self.fig is not None:  # Only close if the figure was created, meaning visualization was enabled.
			plt.close(self.fig)
		self.fig = None
		self.ax = None
		self.im = None
