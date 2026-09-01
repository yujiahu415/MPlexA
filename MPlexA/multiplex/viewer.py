from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any,Iterable,Sequence
import cv2
import numpy as np
from skimage.segmentation import find_boundaries
from.exceptions import MultiplexImageError
from.image_source import open_multiplex_image
from.reconciliation import ChunkedInstanceLabelStore
from.segmentation import load_tile_predictions
from.tiling import TileGrid



class ViewerError(MultiplexImageError):
DEFAULT_CHANNEL_COLORS:tuple[tuple[int,int,int],...]=(
	(0,0,255),
	(0,255,0),
	(255,0,0),
	(0,255,255),
	(255,0,255),
	(255,255,0),
	(255,255,255),
	(255,128,0),
)



@dataclass(frozen=True,slots=True)
class ChannelDisplaySettings:
	channel:int|str
	color:tuple[int,int,int]=(255,255,255)
	minimum:float=0.0
	maximum:float=65535.0
	gamma:float=1.0
	opacity:float=1.0
	visible:bool=True


	def __post_init__(self)->None:
		if len(self.color)!=3 or any(int(value)<0 or int(value)>255 for value in self.color):
			raise ViewerError('Channel color must be an RGB tuple in the range 0..255.')
		if float(self.maximum)<=float(self.minimum):
			raise ViewerError('Channel display maximum must exceed minimum.')
		if float(self.gamma)<=0:
			raise ViewerError('Channel gamma must be positive.')
		if not 0<=float(self.opacity)<=1:
			raise ViewerError('Channel opacity must be between 0 and 1.')



@dataclass(frozen=True,slots=True)
class Viewport:
	x:float
	y:float
	width:float
	height:float
	screen_width:int
	screen_height:int



@dataclass(frozen=True,slots=True)
class RenderResult:
	rgb:np.ndarray
	level:int
	source_bounds:tuple[int,int,int,int]



class MultiplexCompositeRenderer:


	def __init__(self,image_path:str|Path,*,series:int=0,position:dict[str,int]|None=None)->None:
		self.image_path=Path(image_path).expanduser().resolve()
		self.series=int(series)
		self.position=dict(position or{})
		with open_multiplex_image(self.image_path,series=self.series)as image:
			self.metadata=image.metadata
		base=self.metadata.levels[0]
		self.base_width=int(base.shape[base.axes.index('X')])
		self.base_height=int(base.shape[base.axes.index('Y')])


	def _level_dimensions(self,level:int)->tuple[int,int]:
		item=self.metadata.levels[level]
		return int(item.shape[item.axes.index('X')]),int(item.shape[item.axes.index('Y')])


	def choose_level(self,viewport:Viewport)->int:
		desired_x=max(1e-12,float(viewport.screen_width)/max(1.0,float(viewport.width)))
		desired_y=max(1e-12,float(viewport.screen_height)/max(1.0,float(viewport.height)))
		desired=min(desired_x,desired_y)# screen pixels per base pixel
		best_level=0
		best_score=float('inf')
		for level in range(len(self.metadata.levels)):
			width,height=self._level_dimensions(level)
			scale=min(width/self.base_width,height/self.base_height)
			score=abs(np.log(max(1e-12,desired/max(scale,1e-12))))
			if score<best_score:
				best_score=float(score)
				best_level=level
		return best_level


	@staticmethod
	def _normalize(image:np.ndarray,setting:ChannelDisplaySettings)->np.ndarray:
		values=np.asarray(image,dtype=np.float32)
		values=(values-float(setting.minimum))/(float(setting.maximum)-float(setting.minimum))
		np.clip(values,0.0,1.0,out=values)
		if float(setting.gamma)!=1.0:
			values=np.power(values,1.0/float(setting.gamma),dtype=np.float32)
		return values*float(setting.opacity)


	def render(self,viewport:Viewport,settings:Sequence[ChannelDisplaySettings])->RenderResult:
		active=[setting for setting in settings if setting.visible and float(setting.opacity)>0]
		if not active:
			return RenderResult(
				np.zeros((int(viewport.screen_height),int(viewport.screen_width),3),dtype=np.uint8),
				0,
				(int(viewport.x),int(viewport.y),int(viewport.width),int(viewport.height)),
			)
		level=self.choose_level(viewport)
		level_width,level_height=self._level_dimensions(level)
		scale_x=level_width/self.base_width
		scale_y=level_height/self.base_height
		x0=max(0,min(level_width-1,int(np.floor(float(viewport.x)*scale_x))))
		y0=max(0,min(level_height-1,int(np.floor(float(viewport.y)*scale_y))))
		x1=max(x0+1,min(level_width,int(np.ceil((float(viewport.x)+float(viewport.width))*scale_x))))
		y1=max(y0+1,min(level_height,int(np.ceil((float(viewport.y)+float(viewport.height))*scale_y))))
		channels=[setting.channel for setting in active]
		with open_multiplex_image(self.image_path,series=self.series)as image:
			data=image.read_region(
				channels=channels,
				x=x0,
				y=y0,
				width=x1-x0,
				height=y1-y0,
				level=level,
				position=self.position,
			)
		composite=np.zeros((data.shape[1],data.shape[2],3),dtype=np.float32)
		for index,setting in enumerate(active):
			normalized=self._normalize(data[index],setting)
			color=np.asarray(setting.color,dtype=np.float32)/255.0
			composite+=normalized[...,None]*color[None,None,:]
		np.clip(composite,0.0,1.0,out=composite)
		rgb=(composite*255.0+0.5).astype(np.uint8)
		rgb=cv2.resize(
			rgb,
			(max(1,int(viewport.screen_width)),max(1,int(viewport.screen_height))),
			interpolation=cv2.INTER_LINEAR,
		)
		return RenderResult(rgb=rgb,level=level,source_bounds=(x0,y0,x1-x0,y1-y0))


	def auto_contrast(
		self,
		viewport:Viewport,
		channel:int|str,
		*,
		low_percentile:float=1.0,
		high_percentile:float=99.8,
	)->tuple[float,float]:
		level=self.choose_level(viewport)
		level_width,level_height=self._level_dimensions(level)
		scale_x=level_width/self.base_width
		scale_y=level_height/self.base_height
		x0=max(0,min(level_width-1,int(np.floor(float(viewport.x)*scale_x))))
		y0=max(0,min(level_height-1,int(np.floor(float(viewport.y)*scale_y))))
		x1=max(x0+1,min(level_width,int(np.ceil((float(viewport.x)+float(viewport.width))*scale_x))))
		y1=max(y0+1,min(level_height,int(np.ceil((float(viewport.y)+float(viewport.height))*scale_y))))
		with open_multiplex_image(self.image_path,series=self.series)as image:
			plane=image.read_region(
				channels=[channel],x=x0,y=y0,width=x1-x0,height=y1-y0,
				level=level,position=self.position,
			)[0]
		low,high=np.percentile(plane[np.isfinite(plane)],[float(low_percentile),float(high_percentile)])
		if not np.isfinite(low)or not np.isfinite(high)or high<=low:
			low=float(np.nanmin(plane))
			high=float(np.nanmax(plane))
			if high<=low:
				high=low+1.0
		return float(low),float(high)



@dataclass(frozen=True,slots=True)
class SegmentationOverlayResult:
	fill:np.ndarray
	boundaries:np.ndarray
	prediction_count:int
	tile_count:int
	truncated:bool=False
	hidden_for_zoom:bool=False



class SegmentationOverlayIndex:


	def __init__(self,segmentation_directory:str|Path,*,base_width:int,base_height:int)->None:
		self.directory=Path(segmentation_directory).expanduser().resolve()
		self.config_path=self.directory/'segmentation_config.json'
		self.tiles_directory=self.directory/'tiles'
		if not self.config_path.is_file():
			raise ViewerError(f'Segmentation configuration not found: {self.config_path}')
		if not self.tiles_directory.is_dir():
			raise ViewerError(f'Segmentation tile directory not found: {self.tiles_directory}')
		try:
			config=json.loads(self.config_path.read_text(encoding='utf-8'))
			self.grid=TileGrid.from_dict(config['grid'])
		except(OSError,ValueError,KeyError,TypeError,json.JSONDecodeError)as error:
			raise ViewerError(f'Unable to read Module 2 segmentation configuration: {error}')from error
		self.base_width=int(base_width)
		self.base_height=int(base_height)
		if self.base_width<=0 or self.base_height<=0:
			raise ViewerError('Base image dimensions must be positive.')
		self.scale_x=self.base_width/float(self.grid.image_width)
		self.scale_y=self.base_height/float(self.grid.image_height)
		self.method=str(config.get('segmentation_method','detectron2'))
		self.channel_name=str(config.get('channel_name',''))


	def _segmentation_bounds(self,viewport:Viewport)->tuple[float,float,float,float]:
		x0=float(viewport.x)/self.scale_x
		y0=float(viewport.y)/self.scale_y
		x1=float(viewport.x+viewport.width)/self.scale_x
		y1=float(viewport.y+viewport.height)/self.scale_y
		return x0,y0,x1,y1


	@staticmethod
	def _intersects(bounds:Any,x0:float,y0:float,x1:float,y1:float)->bool:
		return not(float(bounds.x1)<=x0 or float(bounds.x)>=x1 or float(bounds.y1)<=y0 or float(bounds.y)>=y1)


	def render(
		self,
		viewport:Viewport,
		*,
		owned_only:bool=True,
		max_predictions:int=50_000,
		max_segmentation_pixels:int=25_000_000,
	)->SegmentationOverlayResult:
		sx0,sy0,sx1,sy1=self._segmentation_bounds(viewport)
		source_width=max(1.0,sx1-sx0)
		source_height=max(1.0,sy1-sy0)
		screen_w=max(1,int(viewport.screen_width))
		screen_h=max(1,int(viewport.screen_height))
		empty=np.zeros((screen_h,screen_w),dtype=bool)
		if source_width*source_height>int(max_segmentation_pixels):
			return SegmentationOverlayResult(empty,empty.copy(),0,0,hidden_for_zoom=True)
		fill=np.zeros((screen_h,screen_w),dtype=np.uint8)
		boundaries=np.zeros((screen_h,screen_w),dtype=np.uint8)
		prediction_count=0
		tile_count=0
		truncated=False
		for tile in self.grid:
			if not self._intersects(tile.read_bounds,sx0,sy0,sx1,sy1):
				continue
			archive_path=self.tiles_directory/ f'{tile.tile_id}.npz'
			if not archive_path.is_file():
				continue
			archive=load_tile_predictions(archive_path)
			tile_count+=1
			for index in range(archive.count):
				if owned_only and not bool(archive.owned_by_core[index]):
					continue
				gx0,gy0,gx1,gy1=(float(value)for value in archive.global_boxes[index])
				if gx1<=sx0 or gx0>=sx1 or gy1<=sy0 or gy0>=sy1:
					continue
				if prediction_count>=int(max_predictions):
					truncated=True
					break
				mask=archive.decode_cropped_mask(index)
				ix0=max(gx0,sx0);iy0=max(gy0,sy0)
				ix1=min(gx1,sx1);iy1=min(gy1,sy1)
				mx0=max(0,int(np.floor(ix0-gx0)));my0=max(0,int(np.floor(iy0-gy0)))
				mx1=min(mask.shape[1],int(np.ceil(ix1-gx0)));my1=min(mask.shape[0],int(np.ceil(iy1-gy0)))
				if mx1<=mx0 or my1<=my0:
					continue
				clipped=mask[my0:my1,mx0:mx1].astype(np.uint8)
				bx0=gx0+mx0;by0=gy0+my0
				bx1=gx0+mx1;by1=gy0+my1
				dx0=max(0,min(screen_w,int(np.floor((bx0-sx0)/source_width*screen_w))))
				dy0=max(0,min(screen_h,int(np.floor((by0-sy0)/source_height*screen_h))))
				dx1=max(dx0+1,min(screen_w,int(np.ceil((bx1-sx0)/source_width*screen_w))))
				dy1=max(dy0+1,min(screen_h,int(np.ceil((by1-sy0)/source_height*screen_h))))
				if dx0>=screen_w or dy0>=screen_h or dx1<=0 or dy1<=0:
					continue
				resized=cv2.resize(clipped,(dx1-dx0,dy1-dy0),interpolation=cv2.INTER_NEAREST)
				fill[dy0:dy1,dx0:dx1]|=resized
				padded=np.pad(clipped.astype(bool),1,mode='constant',constant_values=False)
				source_boundary=find_boundaries(padded,mode='inner')[1:-1,1:-1].astype(np.uint8)
				object_boundary=cv2.resize(source_boundary,(dx1-dx0,dy1-dy0),interpolation=cv2.INTER_NEAREST)
				boundaries[dy0:dy1,dx0:dx1]|=object_boundary
				prediction_count+=1
			if truncated:
				break
		fill_bool=fill.astype(bool)
		return SegmentationOverlayResult(fill_bool,boundaries.astype(bool),prediction_count,tile_count,truncated=truncated)



class ClusterOverlayIndex:


	def __init__(self,clustering_directory:str|Path)->None:
		self.database=Path(clustering_directory).expanduser().resolve()/'clustering.sqlite'
		if not self.database.is_file():
			raise ViewerError(f'Clustering database not found: {self.database}')


	def cells_in_region(self,x:float,y:float,width:float,height:float,limit:int=200_000)->list[dict[str,Any]]:
		x1=float(x)+float(width)
		y1=float(y)+float(height)
		with sqlite3.connect(self.database)as conn:
			conn.row_factory=sqlite3.Row
			rows=conn.execute(
				'''SELECT c.global_cell_id,c.centroid_x,c.centroid_y,c.cluster_id,c.cluster_name,
                          c.pca1,c.pca2,c.embedding_x,c.embedding_y,c.features_json
                   FROM cell_rtree r JOIN cells c ON c.global_cell_id=r.global_cell_id
                   WHERE r.max_x>=? AND r.min_x<=? AND r.max_y>=? AND r.min_y<=?
                   LIMIT ?''',
				(float(x),x1,float(y),y1,int(limit)),
			).fetchall()
		return[dict(row)for row in rows]


	def nearest_cell(self,x:float,y:float,radius:float=30.0)->dict[str,Any]|None:
		candidates=self.cells_in_region(x-radius,y-radius,radius*2,radius*2,limit=10000)
		if not candidates:
			return None
		best=min(
			candidates,
			key=lambda row:(float(row['centroid_x'])-x)**2+(float(row['centroid_y'])-y)**2,
		)
		distance=((float(best['centroid_x'])-x)**2+(float(best['centroid_y'])-y)**2)**0.5
		if distance>radius:
			return None
		result=dict(best)
		try:
			result['features']=json.loads(str(result.pop('features_json')))
		except Exception:
			result['features']={}
		return result


def label_boundaries_for_viewport(
	label_store_path:str|Path,
	viewport:Viewport,
	*,
	max_source_pixels:int=20_000_000,
)->np.ndarray|None:
	width=max(1,int(np.ceil(viewport.width)))
	height=max(1,int(np.ceil(viewport.height)))
	if width*height>int(max_source_pixels):
		return None
	store=ChunkedInstanceLabelStore.open(label_store_path)
	x0=max(0,min(store.metadata.width-1,int(np.floor(viewport.x))))
	y0=max(0,min(store.metadata.height-1,int(np.floor(viewport.y))))
	x1=min(store.metadata.width,max(x0+1,int(np.ceil(viewport.x+viewport.width))))
	y1=min(store.metadata.height,max(y0+1,int(np.ceil(viewport.y+viewport.height))))
	labels=store.read_region(x=x0,y=y0,width=x1-x0,height=y1-y0)
	boundaries=find_boundaries(labels,mode='inner')
	resized=cv2.resize(
		boundaries.astype(np.uint8),
		(max(1,int(viewport.screen_width)),max(1,int(viewport.screen_height))),
		interpolation=cv2.INTER_NEAREST,
	)
	return resized.astype(bool)
__all__=[
	'ChannelDisplaySettings',
	'ClusterOverlayIndex',
	'DEFAULT_CHANNEL_COLORS',
	'MultiplexCompositeRenderer',
	'RenderResult',
	'SegmentationOverlayIndex',
	'SegmentationOverlayResult',
	'ViewerError',
	'Viewport',
	'label_boundaries_for_viewport',
]
