import numpy as np
from typing import List, Tuple, Optional
from hopfield_model.typing import Vector, Matrix, Ordering
from hopfield_model.utilities import assert_valid_pattern, is_valid_pattern
from hopfield_model.visualization.plot_controller import PlotController


# ───────────────────────────────────────────────────────────────────────
# For the Vector, Matrix type aliases, look at hopfield_model/typing.py
# ───────────────────────────────────────────────────────────────────────

class HopfieldNetwork:
	"""
	Hopfield Network with asynchronous updates and internal state.

	Attributes:
		n (int): Number of neurons.
		W (Matrix): Weight matrix (n x n).
		s (Vector): Current state vector (n,).
	"""
	
	def __init__(self, n: int) -> None:
		"""
		Initializes the network structure (but not weights or state).

		Parameters:
			n (int): Number of neurons.
		"""
		assert n > 0, "Number of neurons must be positive."
		self.n: int = n
		self.W: Matrix = np.zeros((n, n))
		self.s: Vector = np.zeros(n, dtype=int)
	
	def train(self, patterns: List[Vector], is_initial: bool = True) -> None:
		"""
		Learns patterns using Hebbian learning rule.

		Parameters:
			patterns (List[Vector]): List of 1D vectors (length n, values in {-1, +1}).
			is_initial (bool): Is this the initial training? If True, clears existing weights before training, default is True.
		"""
		assert isinstance(patterns, list) and len(patterns) > 0, "Patterns must be a non-empty list."
		assert isinstance(is_initial, bool), "is_initial must be a boolean."
		if is_initial:
			W = np.zeros_like(self.W)
		else:
			W = np.copy(self.W)
		for p in patterns:
			self.assert_network_valid_pattern(p)
			W += 1 / self.n * np.outer(p, p)
		np.fill_diagonal(W, 0)
		self.W = W
	
	def is_network_valid_pattern(self, pattern: Vector) -> Tuple[bool, str]:
		"""
		Checks if the provided pattern is valid for the network, and returns a tuple with validity status and an error message if invalid.

		Parameters:
			pattern (Vector): 1D vector of shape (n, ) with values in {-1, +1}.
		
		Returns:
			Tuple[bool, str]: Tuple containing a boolean indicating validity and an error message if invalid.
		"""
		return is_valid_pattern(pattern, self.n)
	
	def assert_network_valid_pattern(self, pattern: Vector) -> None:
		"""
		Asserts that the provided pattern is valid for the network.

		Parameters:
			pattern (Vector): 1D vector of shape (n, ) with values in {-1, +1}.
		"""
		assert_valid_pattern(pattern, self.n)
	
	def get_state(self) -> Vector:
		"""
		Returns a copy of the current state vector.

		Returns:
			Vector: Current state vector  of shape (n, ).
		"""
		return self.s.copy()
	
	def set_state(self, state: Vector) -> None:
		"""
		Sets the initial network state.

		Parameters:
			state (Vector): 1D vector of shape (n, ) with values in {-1, +1}.
		"""
		self.assert_network_valid_pattern(state)
		self.s = state.copy()
	
	def get_weights(self) -> Matrix:
		"""
		Returns a copy of the weight matrix.

		Returns:
			Matrix: Current Weight matrix of shape (n, n).
		"""
		return self.W.copy()
	
	def clear(self) -> None:
		"""
		Clears the weight matrix and resets the state vector.
		"""
		self.W.fill(0)
		self.s.fill(0)
	
	def _update_single_neuron(self, i: int) -> Vector:
		"""
		Updates a single neuron (index i) asynchronously, and returns the updated state.

		Parameters:
			i (int): Neuron index.

		Returns:
			Vector: Updated state vector after the neuron update.
		"""
		h_i = np.dot(self.W[i], self.s)
		self.s[i] = 1 if h_i >= 0 else -1
		return self.get_state()
	
	def run_cycle(self, controller: PlotController, order: Ordering = Ordering.SEQUENTIAL) -> None:
		"""
		Runs one asynchronous update cycle over all neurons.

		Parameters:
			controller (PlotController): Controller for network states visualization.
			order (Ordering): Order of neuron updates (sequential or random), default is sequential.
		"""
		
		indices = range(self.n)
		if order == Ordering.RANDOM:
			indices = list(indices)
			rng = np.random.default_rng()
			rng.shuffle(indices)
		
		for i in indices:
			prev_state = self.get_state()
			curr_state = self._update_single_neuron(i)
			controller.save_and_plot_intermediate_if_needed(curr_state, prev_state)
	
	def run_until_convergence(self, max_cycles: int = 100, order: Ordering = Ordering.SEQUENTIAL,
							  with_visualization: bool = False, with_saving: bool = False,
							  with_interactive: bool = False,
							  shape: Optional[Tuple[int, int]] = None,
							  file_base_name: Optional[str] = None,
							  directory_path_lst: Optional[List[str]] = None,
							  extension: Optional[str] = None,
							  dpi: Optional[int] = None) -> Tuple[bool, Vector]:
		"""
		Runs update cycles until convergence or max_cycles.

		Parameters:
			max_cycles (int): Max number of cycles to run, default is 100.
			order (Ordering): Order of neuron updates (sequential or random), default is sequential.
			with_visualization (bool): If True, enables visualization, default is False.
			with_saving (bool): If True, enables saving plots to disk, default is False.
			with_interactive (bool): If True, enables interactive plotting, default is False.
			shape (Optional[Tuple[int, int]]): Shape of the network state for visualization if applicable, default is None.
			file_base_name (Optional[str]): Base name for saving files if applicable, default is None.
			directory_path_lst (Optional[str]): Relative or absolute path to the directory where files will be saved (given as a list of strings) if applicable, default is None.
			extension (Optional[str]): File extension for saving plots if applicable, default is None.
			dpi (Optional[int]): DPI for saving plots if applicable, default is None.
			

		Returns:
			Tuple[bool, Vector]: Convergence status and final state vector.
		"""
		assert isinstance(max_cycles, int) and max_cycles > 0, "max_cycles must be a positive integer."
		controller = PlotController(with_visualization=with_visualization,
									with_saving=with_saving,
									with_interactive=with_interactive,
									shape=shape,
									file_base_name=file_base_name,
									directory_path_lst=directory_path_lst,
									extension=extension,
									dpi=dpi)
		converged = False
		controller.save_and_plot_initial_if_needed(self.get_state())
		
		for _ in range(max_cycles):
			prev = self.get_state()
			self.run_cycle(controller, order)
			if np.array_equal(self.s, prev):
				converged = True
				break
		
		controller.save_and_plot_final_if_needed(self.get_state())
		controller.close_figure_if_needed()
		return converged, self.get_state()
	
	def check_if_fixed_point(self, state: Vector, order: Ordering = Ordering.SEQUENTIAL) -> bool:
		"""
		Checks if a given state is a fixed point of the network.

		Parameters:
			state (Vector): 1D vector of shape (n, ) with values in {-1, +1}.
			order (Ordering): Order of neuron updates (sequential or random), default is sequential.

		Returns:
			bool: True if state is a fixed point, False otherwise.
		"""
		prev_state = self.s  # Save current state, must be a valid pattern
		self.set_state(state)
		# Because max_cycles=1, so if converged = True, then the state is a fixed point (no updates needed, the network converged to the same state).
		converged, _ = self.run_until_convergence(max_cycles=1,
												  order=order)  # Run one cycle to check if fixed point, no visualization needed (it defaults to False).
		self.s = prev_state  # Restore original state
		return converged
