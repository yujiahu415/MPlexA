class MultiplexImageError(RuntimeError):



class UnsupportedImageError(MultiplexImageError):



class LazyReadUnavailableError(MultiplexImageError):



class InvalidRegionError(MultiplexImageError):



class TilingError(MultiplexImageError):



class CheckpointError(MultiplexImageError):



class CheckpointMismatchError(CheckpointError):
