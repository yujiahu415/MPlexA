class MultiplexImageError(RuntimeError):
	'''Base exception for multiplex image access errors.'''



class UnsupportedImageError(MultiplexImageError):
	'''Raised when an image format or dimensional layout is unsupported.'''



class LazyReadUnavailableError(MultiplexImageError):
	'''Raised when a lazy backend required for a large image is unavailable.'''



class InvalidRegionError(MultiplexImageError):
	'''Raised when a requested image region or channel is invalid.'''



class TilingError(MultiplexImageError):
	'''Raised when a tiling plan or tile operation is invalid.'''



class CheckpointError(MultiplexImageError):
	'''Raised for invalid tile-checkpoint operations.'''



class CheckpointMismatchError(CheckpointError):
	'''Raised when a checkpoint does not match the selected grid or image.'''