from __future__ import annotations
from bisect import bisect_right
from dataclasses import asdict,dataclass
import hashlib
import json
from typing import Any,Iterator,Mapping,Sequence
import numpy as np
from.exceptions import InvalidRegionError,TilingError
TILING_SCHEMA_VERSION=1



@dataclass(frozen=True,slots=True)
class Bounds:
	x:int
	y:int
	width:int
	height:int


	def __post_init__(self)->None:
		values=(self.x,self.y,self.width,self.height)
		if any(not isinstance(value,(int,np.integer))for value in values):
			raise TypeError('Bounds values must be integers.')
		if self.width<0 or self.height<0:
			raise TilingError('Bounds width and height cannot be negative.')


	@property
	def x1(self)->int:
		return int(self.x+self.width)


	@property
	def y1(self)->int:
		return int(self.y+self.height)


	@property
	def area(self)->int:
		return int(self.width*self.height)


	@property
	def slices_yx(self)->tuple[slice,slice]:
		return slice(self.y,self.y1),slice(self.x,self.x1)


	def contains_point(self,x:float,y:float)->bool:
		return self.x<=x<self.x1 and self.y<=y<self.y1


	def contains_bounds(self,other:'Bounds')->bool:
		return(
			self.x<=other.x
			and self.y<=other.y
			and self.x1>=other.x1
			and self.y1>=other.y1
		)


	def intersection(self,other:'Bounds')->'Bounds | None':
		x0=max(self.x,other.x)
		y0=max(self.y,other.y)
		x1=min(self.x1,other.x1)
		y1=min(self.y1,other.y1)
		if x0>=x1 or y0>=y1:
			return None
		return Bounds(x0,y0,x1-x0,y1-y0)


	def translated(self,dx:int,dy:int)->'Bounds':
		return Bounds(self.x+int(dx),self.y+int(dy),self.width,self.height)


	def to_dict(self)->dict[str,int]:
		return{key:int(value)for key,value in asdict(self).items()}



@dataclass(frozen=True,slots=True)
class Padding:
	left:int=0
	top:int=0
	right:int=0
	bottom:int=0


	@property
	def required(self)->bool:
		return any((self.left,self.top,self.right,self.bottom))


	def to_dict(self)->dict[str,int]:
		return{key:int(value)for key,value in asdict(self).items()}



@dataclass(frozen=True,slots=True)
class Tile:
	tile_id:str
	index:int
	row:int
	column:int
	level:int
	read_bounds:Bounds
	core_bounds:Bounds
	target_width:int
	target_height:int


	@property
	def core_local_bounds(self)->Bounds:
		return self.core_bounds.translated(-self.read_bounds.x,-self.read_bounds.y)


	@property
	def padding(self)->Padding:
		return Padding(
			right=max(0,int(self.target_width-self.read_bounds.width)),
			bottom=max(0,int(self.target_height-self.read_bounds.height)),
		)


	def owns_global_point(self,x:float,y:float)->bool:
		return self.core_bounds.contains_point(x,y)


	def global_to_local(
		self,x:float,y:float,*,require_inside_read:bool=False
	)->tuple[float,float]:
		if require_inside_read and not self.read_bounds.contains_point(x,y):
			raise InvalidRegionError(
				f'Global point ({x}, {y}) is outside tile {self.tile_id}.'
			)
		return x-self.read_bounds.x,y-self.read_bounds.y


	def local_to_global(
		self,x:float,y:float,*,require_inside_read:bool=False
	)->tuple[float,float]:
		if require_inside_read and not(
			0<=x<self.read_bounds.width and 0<=y<self.read_bounds.height
		):
			raise InvalidRegionError(
				f'Local point ({x}, {y}) is outside tile {self.tile_id}.'
			)
		return x+self.read_bounds.x,y+self.read_bounds.y


	def global_bounds_to_local(
		self,bounds:Bounds,*,require_inside_read:bool=False
	)->Bounds:
		if require_inside_read and not self.read_bounds.contains_bounds(bounds):
			raise InvalidRegionError(
				f'Global bounds {bounds} are not contained by tile {self.tile_id}.'
			)
		return bounds.translated(-self.read_bounds.x,-self.read_bounds.y)


	def local_bounds_to_global(
		self,bounds:Bounds,*,require_inside_read:bool=False
	)->Bounds:
		local_read=Bounds(0,0,self.read_bounds.width,self.read_bounds.height)
		if require_inside_read and not local_read.contains_bounds(bounds):
			raise InvalidRegionError(
				f'Local bounds {bounds} are not contained by tile {self.tile_id}.'
			)
		return bounds.translated(self.read_bounds.x,self.read_bounds.y)


	def pad_array(self,array:np.ndarray,constant_value:int|float=0)->np.ndarray:
		data=np.asarray(array)
		if data.ndim<2:
			raise TilingError('A tile array must have at least two dimensions.')
		expected=(self.read_bounds.height,self.read_bounds.width)
		if tuple(data.shape[-2:])!=expected:
			raise TilingError(
				f'Tile {self.tile_id} expects spatial shape {expected}; got '
				f'{tuple(data.shape[-2:])}.'
			)
		padding=self.padding
		if not padding.required:
			return data
		pad_width=[(0,0)]*data.ndim
		pad_width[-2]=(padding.top,padding.bottom)
		pad_width[-1]=(padding.left,padding.right)
		return np.pad(data,pad_width,mode='constant',constant_values=constant_value)


	def extract_core(self,array:np.ndarray)->np.ndarray:
		data=np.asarray(array)
		if data.ndim<2:
			raise TilingError('A tile array must have at least two dimensions.')
		expected=(self.read_bounds.height,self.read_bounds.width)
		if tuple(data.shape[-2:])!=expected:
			raise TilingError(
				f'Tile {self.tile_id} expects spatial shape {expected}; got '
				f'{tuple(data.shape[-2:])}.'
			)
		local=self.core_local_bounds
		return data[...,local.y:local.y1,local.x:local.x1]


	def read_from(
		self,
		image:Any,
		*,
		channels:int|str|Sequence[int|str]|None=None,
		position:Mapping[str,int]|None=None,
		pad:bool=False,
		constant_value:int|float=0,
	)->np.ndarray:
		data=image.read_region(
			x=self.read_bounds.x,
			y=self.read_bounds.y,
			width=self.read_bounds.width,
			height=self.read_bounds.height,
			channels=channels,
			level=self.level,
			position=position,
		)
		return self.pad_array(data,constant_value)if pad else data


	def to_dict(self)->dict[str,Any]:
		return{
			'tile_id':self.tile_id,
			'index':self.index,
			'row':self.row,
			'column':self.column,
			'level':self.level,
			'read_bounds':self.read_bounds.to_dict(),
			'core_bounds':self.core_bounds.to_dict(),
			'core_local_bounds':self.core_local_bounds.to_dict(),
			'padding':self.padding.to_dict(),
			'target_width':self.target_width,
			'target_height':self.target_height,
		}



class TileGrid:


	def __init__(
		self,
		image_width:int,
		image_height:int,
		*,
		tile_width:int=2048,
		tile_height:int|None=None,
		overlap:int|tuple[int,int]|None=None,
		overlap_ratio:float|tuple[float,float]|None=None,
		level:int=0,
	)->None:
		self.image_width=self._positive_int('image_width',image_width)
		self.image_height=self._positive_int('image_height',image_height)
		self.tile_width=self._positive_int('tile_width',tile_width)
		self.tile_height=self._positive_int(
			'tile_height',tile_height if tile_height is not None else tile_width
		)
		self.level=self._nonnegative_int('level',level)
		if overlap is not None and overlap_ratio is not None:
			raise TilingError('Specify either overlap pixels or overlap ratios, not both.')
		if overlap_ratio is not None:
			if isinstance(overlap_ratio,tuple):
				if len(overlap_ratio)!=2:
					raise TilingError(
						'Overlap-ratio tuple must contain (ratio_x, ratio_y).'
					)
				ratio_x,ratio_y=overlap_ratio
			else:
				ratio_x=ratio_y=overlap_ratio
			self.overlap_ratio_x=self._ratio('overlap_ratio_x',ratio_x)
			self.overlap_ratio_y=self._ratio('overlap_ratio_y',ratio_y)
			self.overlap_x=self._ratio_to_pixels(
				self.tile_width,self.overlap_ratio_x
			)
			self.overlap_y=self._ratio_to_pixels(
				self.tile_height,self.overlap_ratio_y
			)
		else:
			if overlap is None:
				overlap=128
			if isinstance(overlap,tuple):
				if len(overlap)!=2:
					raise TilingError(
						'Overlap tuple must contain (overlap_x, overlap_y).'
					)
				overlap_x,overlap_y=overlap
			else:
				overlap_x=overlap_y=overlap
			self.overlap_x=self._nonnegative_int('overlap_x',overlap_x)
			self.overlap_y=self._nonnegative_int('overlap_y',overlap_y)
			self.overlap_ratio_x=self.overlap_x/self.tile_width
			self.overlap_ratio_y=self.overlap_y/self.tile_height
		if self.overlap_x>=self.tile_width:
			raise TilingError('Horizontal overlap must be smaller than tile width.')
		if self.overlap_y>=self.tile_height:
			raise TilingError('Vertical overlap must be smaller than tile height.')
		self._x_starts=self._axis_starts(
			self.image_width,self.tile_width,self.overlap_x
		)
		self._y_starts=self._axis_starts(
			self.image_height,self.tile_height,self.overlap_y
		)
		self._x_core_boundaries=self._ownership_boundaries(
			self.image_width,self.tile_width,self._x_starts
		)
		self._y_core_boundaries=self._ownership_boundaries(
			self.image_height,self.tile_height,self._y_starts
		)
		self._tiles=self._make_tiles()
		self._by_id={tile.tile_id:tile for tile in self._tiles}


	@staticmethod
	def _positive_int(name:str,value:Any)->int:
		if not isinstance(value,(int,np.integer))or int(value)<=0:
			raise TilingError(f'{name} must be a positive integer.')
		return int(value)


	@staticmethod
	def _nonnegative_int(name:str,value:Any)->int:
		if not isinstance(value,(int,np.integer))or int(value)<0:
			raise TilingError(f'{name} must be a non-negative integer.')
		return int(value)


	@staticmethod
	def _ratio(name:str,value:Any)->float:
		if isinstance(value,(bool,np.bool_))or not isinstance(
			value,(int,float,np.integer,np.floating)
		):
			raise TilingError(f'{name} must be a numeric ratio in [0, 1).')
		ratio=float(value)
		if not np.isfinite(ratio)or ratio<0 or ratio>=1:
			raise TilingError(f'{name} must be in the range [0, 1).')
		return ratio


	@staticmethod
	def _ratio_to_pixels(tile_size:int,ratio:float)->int:
		pixels=int(np.floor(tile_size*ratio+0.5))
		return min(max(pixels,0),tile_size-1)


	@staticmethod
	def _axis_starts(length:int,tile_size:int,overlap:int)->tuple[int,...]:
		if length<=tile_size:
			return(0,)
		stride=tile_size-overlap
		last_start=length-tile_size
		starts=list(range(0,last_start+1,stride))
		if starts[-1]!=last_start:
			starts.append(last_start)
		return tuple(starts)


	@staticmethod
	def _ownership_boundaries(
		length:int,tile_size:int,starts:Sequence[int]
	)->tuple[int,...]:
		boundaries=[0]
		for left_start,right_start in zip(starts[:-1],starts[1:]):
			left_end=min(length,left_start+tile_size)
			boundaries.append((left_end+right_start)//2)
		boundaries.append(length)
		return tuple(boundaries)


	@property
	def rows(self)->int:
		return len(self._y_starts)


	@property
	def columns(self)->int:
		return len(self._x_starts)


	@property
	def stride_x(self)->int:
		return self.tile_width-self.overlap_x


	@property
	def stride_y(self)->int:
		return self.tile_height-self.overlap_y


	def _signature_payload(self)->dict[str,Any]:
		return{
			'schema_version':TILING_SCHEMA_VERSION,
			'image_width':self.image_width,
			'image_height':self.image_height,
			'tile_width':self.tile_width,
			'tile_height':self.tile_height,
			'overlap_x':self.overlap_x,
			'overlap_y':self.overlap_y,
			'level':self.level,
			'rows':self.rows,
			'columns':self.columns,
			'tile_count':len(self),
		}


	@property
	def signature(self)->str:
		payload=json.dumps(
			self._signature_payload(),sort_keys=True,separators=(',',':')
		)
		return hashlib.sha256(payload.encode('utf-8')).hexdigest()


	def _make_tiles(self)->tuple[Tile,...]:
		tiles:list[Tile]=[]
		index=0
		for row,y in enumerate(self._y_starts):
			read_height=min(self.tile_height,self.image_height-y)
			core_y0=self._y_core_boundaries[row]
			core_y1=self._y_core_boundaries[row+1]
			for column,x in enumerate(self._x_starts):
				read_width=min(self.tile_width,self.image_width-x)
				core_x0=self._x_core_boundaries[column]
				core_x1=self._x_core_boundaries[column+1]
				tile_id=(
					f'L{self.level:03d}_R{row:06d}_C{column:06d}_'
					f'Y{y:09d}_X{x:09d}'
				)
				tile=Tile(
					tile_id=tile_id,
					index=index,
					row=row,
					column=column,
					level=self.level,
					read_bounds=Bounds(x,y,read_width,read_height),
					core_bounds=Bounds(
						core_x0,core_y0,core_x1-core_x0,core_y1-core_y0
					),
					target_width=self.tile_width,
					target_height=self.tile_height,
				)
				if not tile.read_bounds.contains_bounds(tile.core_bounds):
					raise TilingError(
						f'Internal tiling error: core of {tile.tile_id} is outside its read bounds.'
					)
				tiles.append(tile)
				index+=1
		return tuple(tiles)


	def __len__(self)->int:
		return len(self._tiles)


	def __iter__(self)->Iterator[Tile]:
		return iter(self._tiles)


	def __getitem__(self,index:int|slice)->Tile|tuple[Tile,...]:
		return self._tiles[index]


	def get(self,tile_id:str)->Tile:
		try:
			return self._by_id[tile_id]
		except KeyError as error:
			raise KeyError(f'Unknown tile ID: {tile_id}')from error


	def tile_at_global_point(self,x:float,y:float)->Tile:
		if not(0<=x<self.image_width and 0<=y<self.image_height):
			raise InvalidRegionError(
				f'Point ({x}, {y}) is outside the image '
				f'({self.image_width} x {self.image_height}).'
			)
		column=bisect_right(self._x_core_boundaries,x)-1
		row=bisect_right(self._y_core_boundaries,y)-1
		column=min(max(column,0),self.columns-1)
		row=min(max(row,0),self.rows-1)
		tile=self._tiles[row*self.columns+column]
		if not tile.owns_global_point(x,y):
			raise TilingError(f'No tile owns image point ({x}, {y}).')
		return tile


	def to_dict(self)->dict[str,Any]:
		return{
			**self._signature_payload(),
			'overlap_ratio_x':self.overlap_ratio_x,
			'overlap_ratio_y':self.overlap_ratio_y,
		}


	@classmethod
	def from_dict(cls,data:Mapping[str,Any])->'TileGrid':
		common={
			'tile_width':int(data['tile_width']),
			'tile_height':int(data['tile_height']),
			'level':int(data.get('level',0)),
		}
		if'overlap_ratio_x'in data and'overlap_ratio_y'in data:
			common['overlap_ratio']=(
				float(data['overlap_ratio_x']),
				float(data['overlap_ratio_y']),
			)
		else:
			common['overlap']=(
				int(data['overlap_x']),int(data['overlap_y'])
			)
		return cls(
			int(data['image_width']),
			int(data['image_height']),
			**common,
		)


	def summary(self)->str:
		read_pixels=sum(tile.read_bounds.area for tile in self._tiles)
		unique_pixels=self.image_width*self.image_height
		overhead=read_pixels/unique_pixels if unique_pixels else 1.0
		first=self._tiles[0]
		last=self._tiles[-1]
		return(
			f'Resolution level: {self.level}\n'
			f'Image dimensions: {self.image_width} x {self.image_height} px\n'
			f'Read tile size: {self.tile_width} x {self.tile_height} px\n'
			f'Requested overlap ratio: {self.overlap_ratio_x:.4f} x '
			f'{self.overlap_ratio_y:.4f}\n'
			f'Resolved overlap: {self.overlap_x} x {self.overlap_y} px\n'
			f'Stride: {self.stride_x} x {self.stride_y} px\n'
			f'Grid: {self.rows} rows x {self.columns} columns\n'
			f'Total tiles: {len(self):,}\n'
			f'Read-pixel overhead: {overhead:.3f}x\n'
			f'First tile: {first.tile_id}, read={first.read_bounds}, core={first.core_bounds}\n'
			f'Last tile: {last.tile_id}, read={last.read_bounds}, core={last.core_bounds}\n'
			f'Grid signature: {self.signature}'
		)
