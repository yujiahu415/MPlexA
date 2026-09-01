from __future__ import annotations
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any



@dataclass(frozen=True)
class ImageLevelMetadata:
	level:int
	shape:tuple[int,...]
	axes:str


	def to_dict(self)->dict[str,Any]:
		return asdict(self)



@dataclass(frozen=True)
class MultiplexImageMetadata:
	path:str
	format:str
	series:int
	axes:str
	shape:tuple[int,...]
	dtype:str
	channel_names:tuple[str,...]
	levels:tuple[ImageLevelMetadata,...]
	pixel_size_x:float|None=None
	pixel_size_y:float|None=None
	pixel_size_unit:str|None=None


	@property
	def channel_count(self)->int:
		return len(self.channel_names)


	@property
	def width(self)->int:
		return int(self.shape[self.axes.index('X')])


	@property
	def height(self)->int:
		return int(self.shape[self.axes.index('Y')])


	@property
	def name(self)->str:
		return Path(self.path).name


	def to_dict(self)->dict[str,Any]:
		result=asdict(self)
		result['levels']=[level.to_dict()for level in self.levels]
		result['channel_count']=self.channel_count
		result['width']=self.width
		result['height']=self.height
		return result


	def summary(self,max_channels:int=12)->str:
		shown=list(self.channel_names[:max_channels])
		if self.channel_count>max_channels:
			shown.append(f'... ({self.channel_count-max_channels} more)')
		pixel_size='unknown'
		if self.pixel_size_x is not None and self.pixel_size_y is not None:
			unit= f' {self.pixel_size_unit}' if self.pixel_size_unit else''
			pixel_size= f'{self.pixel_size_x:g} x {self.pixel_size_y:g}{unit}'
		level_shapes=', '.join(
			f'L{level.level}: {level.shape}' for level in self.levels
		)
		return(
			f'File: {self.path}\n'
			f'Format: {self.format}\n'
			f'Series: {self.series}\n'
			f'Axes: {self.axes}\n'
			f'Shape: {self.shape}\n'
			f'Data type: {self.dtype}\n'
			f'Channels: {self.channel_count}\n'
			f'Channel names: {shown}\n'
			f'Pixel size: {pixel_size}\n'
			f'Resolution levels: {level_shapes}'
		)
