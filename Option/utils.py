import numpy as np

from State.buffer import HierarchicalReplayBuffer, VectorHierarchicalReplayBufferManager


def get_new_indices(last_index, buffer):
	if isinstance(buffer, HierarchicalReplayBuffer):
		new_last = buffer._index
		if new_last > last_index:
			new_indices = np.arange(last_index, new_last)
		elif new_last < last_index:
			new_indices = np.concatenate([np.arange(last_index, buffer.maxsize),
										  np.arange(buffer.maxsize, new_last)])
		else:
			new_indices = np.array([])
		return new_last, new_indices
	else:
		if last_index is None:
			last_index = buffer._offset.copy()

		# get new indices
		new_indices = []
		for old_last, new_last, offset_l, offset_r in zip(last_index, buffer.last_index,
														  buffer._offset, buffer._extend_offset[1:]):
			if new_last > old_last:
				new_indices.append(np.arange(old_last, new_last))
			elif new_last < old_last:
				new_indices.append(np.arange(old_last, offset_r))
				new_indices.append(np.arange(offset_l, new_last))

		if len(new_indices) == 0:
			return last_index, np.array([])

		new_indices = np.concatenate(new_indices)
		last_index = buffer.last_index.copy()
		return last_index, new_indices
