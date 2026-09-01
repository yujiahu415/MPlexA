from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable,Iterable,Sequence
import csv
import json
import math
import re
import threading
import numpy as np
import tifffile
from.image_source import open_multiplex_image



class ROIExtractionError(RuntimeError):



class ROIExtractionCancelled(ROIExtractionError):



@dataclass(frozen=True,slots=True)
class ROIChannel:
	index:int
	name:str
	color:tuple[int,int,int]


	def __post_init__(self)->None:
		if int(self.index)<0:
			raise ROIExtractionError('Channel index must be non-negative.')
		if len(self.color)!=3 or any(int(v)<0 or int(v)>255 for v in self.color):
			raise ROIExtractionError('Channel color must be an RGB tuple in the range 0..255.')



@dataclass(frozen=True,slots=True)
class SquareROI:
	row:int
	column:int
	x:int
	y:int
	size:int
	valid_width:int
	valid_height:int


	@property
	def pad_right(self)->int:
		return self.size-self.valid_width


	@property
	def pad_bottom(self)->int:
		return self.size-self.valid_height


	@property
	def roi_id(self)->str:
		return f'R{self.row:05d}_C{self.column:05d}_Y{self.y:09d}_X{self.x:09d}'



@dataclass(frozen=True,slots=True)
class ROIExtractionConfig:
	roi_size:int=1024
	overlap_ratio:float=0.10
	padding:str='black'
	series:int=0


	def __post_init__(self)->None:
		if int(self.roi_size)<=0:
			raise ROIExtractionError('ROI size must be a positive integer.')
		if not 0.0<=float(self.overlap_ratio)<1.0:
			raise ROIExtractionError('Overlap ratio must be in the range 0.0 <= ratio < 1.0.')
		if str(self.padding).lower()not in{'black','white'}:
			raise ROIExtractionError('Padding must be either \'black\' or \'white\'.')
		if int(self.series)<0:
			raise ROIExtractionError('Series index must be non-negative.')
		if self.overlap_pixels>=int(self.roi_size):
			raise ROIExtractionError('Overlap leaves no positive ROI stride.')


	@property
	def overlap_pixels(self)->int:
		return int(math.floor(int(self.roi_size)*float(self.overlap_ratio)+0.5))


	@property
	def stride(self)->int:
		return int(self.roi_size)-self.overlap_pixels


	@property
	def pad_value(self)->int:
		return 0 if str(self.padding).lower()=='black'else 255


	def to_dict(self)->dict[str,object]:
		return{
			'roi_size':int(self.roi_size),
			'overlap_ratio':float(self.overlap_ratio),
			'overlap_pixels':self.overlap_pixels,
			'stride':self.stride,
			'padding':str(self.padding).lower(),
			'series':int(self.series),
			'normalization':'independent per ROI per source channel; min-max to 0..255',
			'compositing':'additive RGB, clipped to 0..255',
		}



@dataclass(frozen=True,slots=True)
class ROIExtractionProgress:
	completed:int
	total:int
	source_index:int
	source_count:int
	source_name:str
	roi_id:str|None=None
	message:str=''


	@property
	def fraction(self)->float:
		return 1.0 if self.total<=0 else min(1.0,max(0.0,self.completed/self.total))



@dataclass(frozen=True,slots=True)
class ROIExtractionSummary:
	output_directory:str
	source_count:int
	roi_count:int
	cancelled:bool


	def summary(self)->str:
		status='cancelled'if self.cancelled else'completed'
		return(
			f'ROI extraction {status}.\n'
			f'Source images: {self.source_count}\n'
			f'Exported ROIs: {self.roi_count}\n'
			f'Output: {self.output_directory}'
		)


def _axis_starts(length:int,size:int,stride:int)->tuple[int,...]:
	length=int(length)
	size=int(size)
	stride=int(stride)
	if length<=0:
		raise ROIExtractionError('Image dimensions must be positive.')
	starts:list[int]=[0]
	while starts[-1]+size<length:
		starts.append(starts[-1]+stride)
	return tuple(starts)


def plan_square_rois(width:int,height:int,config:ROIExtractionConfig)->tuple[SquareROI,...]:
	xs=_axis_starts(int(width),int(config.roi_size),int(config.stride))
	ys=_axis_starts(int(height),int(config.roi_size),int(config.stride))
	rois:list[SquareROI]=[]
	for row,y in enumerate(ys):
		valid_height=min(int(config.roi_size),int(height)-y)
		for column,x in enumerate(xs):
			valid_width=min(int(config.roi_size),int(width)-x)
			rois.append(
				SquareROI(
					row=row,
					column=column,
					x=x,
					y=y,
					size=int(config.roi_size),
					valid_width=max(0,valid_width),
					valid_height=max(0,valid_height),
				)
			)
	return tuple(rois)


def rescale_channel_to_uint8(channel:np.ndarray)->np.ndarray:
	array=np.asarray(channel)
	if array.ndim!=2:
		raise ROIExtractionError(f'Expected a 2-D channel ROI, got shape {array.shape}.')
	values=array.astype(np.float32,copy=False)
	finite=np.isfinite(values)
	if not np.any(finite):
		return np.zeros(array.shape,dtype=np.uint8)
	minimum=float(np.min(values[finite]))
	maximum=float(np.max(values[finite]))
	if maximum<=minimum:
		return np.zeros(array.shape,dtype=np.uint8)
	scaled=(values-minimum)*(255.0/(maximum-minimum))
	scaled=np.where(finite,scaled,0.0)
	return np.clip(np.rint(scaled),0,255).astype(np.uint8)


def compose_rgb_roi(
	channel_data:np.ndarray,
	colors:Sequence[tuple[int,int,int]],
	*,
	roi_size:int,
	padding:str,
)->np.ndarray:
	data=np.asarray(channel_data)
	if data.ndim!=3:
		raise ROIExtractionError(f'Expected channel x y x x data, got shape {data.shape}.')
	if data.shape[0]!=len(colors):
		raise ROIExtractionError('Number of channel colors does not match number of channels.')
	if data.shape[1]>roi_size or data.shape[2]>roi_size:
		raise ROIExtractionError('Read ROI is larger than the requested square ROI size.')
	valid_height,valid_width=int(data.shape[1]),int(data.shape[2])
	composite=np.zeros((valid_height,valid_width,3),dtype=np.float32)
	for channel_index,color in enumerate(colors):
		normalized=rescale_channel_to_uint8(data[channel_index]).astype(np.float32)
		rgb=np.asarray(color,dtype=np.float32)/255.0
		composite+=normalized[...,None]*rgb[None,None,:]
	composite8=np.clip(np.rint(composite),0,255).astype(np.uint8)
	pad_value=0 if str(padding).lower()=='black'else 255
	output=np.full((int(roi_size),int(roi_size),3),pad_value,dtype=np.uint8)
	output[:valid_height,:valid_width]=composite8
	return output


def _safe_stem(path:str|Path)->str:
	stem=Path(path).name
	for suffix in('.ome.zarr','.ome.tiff','.ome.tif','.qptiff','.tiff','.tif','.zarr'):
		if stem.lower().endswith(suffix):
			stem=stem[:-len(suffix)]
			break
	cleaned=re.sub('[^A-Za-z0-9_.-]+','_',stem).strip('_.')
	return cleaned or'image'



class SquareROIExtractor:


	def __init__(
		self,
		source_paths:Sequence[str|Path],
		output_directory:str|Path,
		channels:Sequence[ROIChannel],
		config:ROIExtractionConfig,
	)->None:
		if not source_paths:
			raise ROIExtractionError('Select at least one source image.')
		if not channels:
			raise ROIExtractionError('Select at least one source channel.')
		self.source_paths=tuple(Path(path).expanduser()for path in source_paths)
		self.output_directory=Path(output_directory).expanduser()
		self.channels=tuple(channels)
		self.config=config


	def _validate_source_channels(self,source)->None:
		for channel in self.channels:
			if channel.index>=len(source.channel_names):
				raise ROIExtractionError(
					f'Channel {channel.index} ({channel.name}) is unavailable in {source.metadata.path}.'
				)
			actual=str(source.channel_names[channel.index])
			if actual!=channel.name:
				raise ROIExtractionError(
					'Selected source images do not have the same channel layout. '
					f'Expected channel {channel.index} to be {channel.name!r}, found {actual!r} '
					f'in {source.metadata.path}.'
				)


	def count_rois(self)->int:
		total=0
		for path in self.source_paths:
			with open_multiplex_image(path,series=self.config.series)as source:
				self._validate_source_channels(source)
				total+=len(plan_square_rois(source.metadata.width,source.metadata.height,self.config))
		return total


	def run(
		self,
		*,
		progress_callback:Callable[[ROIExtractionProgress],None]|None=None,
		cancel_event:threading.Event|None=None,
	)->ROIExtractionSummary:
		self.output_directory.mkdir(parents=True,exist_ok=True)
		total=self.count_rois()
		completed=0
		cancelled=False
		manifest_path=self.output_directory/'roi_manifest.csv'
		config_path=self.output_directory/'roi_extraction_config.json'
		config_payload=self.config.to_dict()
		config_payload['channels']=[
			{'index':ch.index,'name':ch.name,'color':list(ch.color)}for ch in self.channels
		]
		config_payload['sources']=[str(path.resolve())for path in self.source_paths]
		config_path.write_text(json.dumps(config_payload,indent=2),encoding='utf-8')
		fieldnames=[
			'filename','source_index','source_path','source_name','series',
			'row','column','x','y','roi_size','valid_width','valid_height',
			'pad_right','pad_bottom','overlap_ratio','overlap_pixels','stride',
			'padding','channel_indices','channel_names','channel_colors_rgb',
		]
		with manifest_path.open('w',newline='',encoding='utf-8')as manifest_file:
			writer=csv.DictWriter(manifest_file,fieldnames=fieldnames)
			writer.writeheader()
			for source_index,path in enumerate(self.source_paths):
				if cancel_event is not None and cancel_event.is_set():
					cancelled=True
					break
				with open_multiplex_image(path,series=self.config.series)as source:
					self._validate_source_channels(source)
					rois=plan_square_rois(source.metadata.width,source.metadata.height,self.config)
					safe_stem=_safe_stem(path)
					for roi in rois:
						if cancel_event is not None and cancel_event.is_set():
							cancelled=True
							break
						data=source.read_region(
							x=roi.x,
							y=roi.y,
							width=roi.valid_width,
							height=roi.valid_height,
							channels=[channel.index for channel in self.channels],
							level=0,
						)
						rgb=compose_rgb_roi(
							data,
							[channel.color for channel in self.channels],
							roi_size=self.config.roi_size,
							padding=self.config.padding,
						)
						filename= f'S{source_index:03d}_{safe_stem}__{roi.roi_id}.tif'
						output_path=self.output_directory/filename
						tifffile.imwrite(
							output_path,
							rgb,
							photometric='rgb',
							metadata={
								'axes':'YXS',
								'MPlexA_roi':True,
								'source':str(path),
								'x':roi.x,
								'y':roi.y,
								'roi_size':self.config.roi_size,
								'channels':[channel.name for channel in self.channels],
								'colors':[list(channel.color)for channel in self.channels],
							},
						)
						writer.writerow(
							{
								'filename':filename,
								'source_index':source_index,
								'source_path':str(path),
								'source_name':path.name,
								'series':self.config.series,
								'row':roi.row,
								'column':roi.column,
								'x':roi.x,
								'y':roi.y,
								'roi_size':roi.size,
								'valid_width':roi.valid_width,
								'valid_height':roi.valid_height,
								'pad_right':roi.pad_right,
								'pad_bottom':roi.pad_bottom,
								'overlap_ratio':self.config.overlap_ratio,
								'overlap_pixels':self.config.overlap_pixels,
								'stride':self.config.stride,
								'padding':self.config.padding,
								'channel_indices':json.dumps([ch.index for ch in self.channels]),
								'channel_names':json.dumps([ch.name for ch in self.channels]),
								'channel_colors_rgb':json.dumps([list(ch.color)for ch in self.channels]),
							}
						)
						completed+=1
						if progress_callback is not None:
							progress_callback(
								ROIExtractionProgress(
									completed=completed,
									total=total,
									source_index=source_index,
									source_count=len(self.source_paths),
									source_name=path.name,
									roi_id=roi.roi_id,
									message=f'Saved {filename}',
								)
							)
					if cancelled:
						break
		return ROIExtractionSummary(
			output_directory=str(self.output_directory),
			source_count=len(self.source_paths),
			roi_count=completed,
			cancelled=cancelled,
		)
__all__=[
	'ROIChannel',
	'ROIExtractionCancelled',
	'ROIExtractionConfig',
	'ROIExtractionError',
	'ROIExtractionProgress',
	'ROIExtractionSummary',
	'SquareROI',
	'SquareROIExtractor',
	'compose_rgb_roi',
	'plan_square_rois',
	'rescale_channel_to_uint8',
]
