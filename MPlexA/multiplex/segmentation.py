from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any,Callable,Mapping,Protocol,Sequence
import numpy as np
from scipy import ndimage as ndi
from skimage import feature,filters,measure,morphology
from skimage.segmentation import watershed
from.checkpoints import CheckpointProgress,TileCheckpointStore
from.exceptions import MultiplexImageError,TilingError
from.image_source import MultiplexImageSource,open_multiplex_image
from.tiling import Tile,TileGrid
SEGMENTATION_SCHEMA_VERSION=1


def _utc_now()->str:
	return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _canonical_json(value:Any)->str:
	return json.dumps(value,sort_keys=True,separators=(',',':'),default=str)


def _path_identity(path:Path)->dict[str,Any]:
	stat=path.stat()
	identity:dict[str,Any]={
		'path':str(path),
		'is_directory':path.is_dir(),
		'size':int(stat.st_size),
		'mtime_ns':int(stat.st_mtime_ns),
	}
	if path.is_dir():
		metadata_files=[]
		for name in('.zattrs','.zgroup','.zmetadata','zarr.json'):
			candidate=path/name
			if candidate.exists():
				item_stat=candidate.stat()
				metadata_files.append(
					{'name':name,'size':int(item_stat.st_size),'mtime_ns':int(item_stat.st_mtime_ns)}
				)
		identity['metadata_files']=metadata_files
	return identity



class SegmentationError(MultiplexImageError):



class SegmentationCancelled(SegmentationError):



@dataclass(frozen=True,slots=True)
class DetectorMetadata:
	path:str
	cell_names:tuple[str,...]
	cell_mapping:dict[int,str]
	inferencing_framesize:int
	black_background:bool
	fingerprint:str


	def to_dict(self)->dict[str,Any]:
		return{
			'path':self.path,
			'cell_names':list(self.cell_names),
			'cell_mapping':{str(key):value for key,value in self.cell_mapping.items()},
			'inferencing_framesize':self.inferencing_framesize,
			'black_background':self.black_background,
			'fingerprint':self.fingerprint,
		}



@dataclass(frozen=True,slots=True)
class IntensityNormalization:
	low_value:float
	high_value:float
	low_percentile:float=1.0
	high_percentile:float=99.8
	sample_count:int=0
	sampled_pixels:int=0
	method:str='sampled_percentiles'


	def __post_init__(self)->None:
		values=(self.low_value,self.high_value,self.low_percentile,self.high_percentile)
		if not all(np.isfinite(value)for value in values):
			raise SegmentationError('Normalization values must be finite.')
		if self.high_value<=self.low_value:
			raise SegmentationError('Normalization high value must exceed low value.')
		if not 0<=self.low_percentile<self.high_percentile<=100:
			raise SegmentationError('Normalization percentiles must satisfy 0 <= low < high <= 100.')


	def apply(self,image:np.ndarray)->np.ndarray:
		data=np.asarray(image,dtype=np.float32)
		scaled=(data-float(self.low_value))*(255.0/(float(self.high_value)-float(self.low_value)))
		scaled=np.nan_to_num(scaled,nan=0.0,posinf=255.0,neginf=0.0)
		return np.clip(scaled,0.0,255.0).astype(np.uint8)


	def to_dict(self)->dict[str,Any]:
		return asdict(self)



@dataclass(frozen=True,slots=True)
class SegmentationConfig:
	channel:int|str='DAPI'
	score_threshold:float=0.5
	batch_size:int=1
	low_percentile:float=1.0
	high_percentile:float=99.8
	normalization_sample_size:int=512
	normalization_samples:int=16
	retry_failed:bool=False
	position:Mapping[str,int]|None=None


	def __post_init__(self)->None:
		if not 0<=float(self.score_threshold)<=1:
			raise SegmentationError('Detection score threshold must be between 0 and 1.')
		if int(self.batch_size)<=0:
			raise SegmentationError('Batch size must be positive.')
		if not 0<=float(self.low_percentile)<float(self.high_percentile)<=100:
			raise SegmentationError('Normalization percentiles must satisfy 0 <= low < high <= 100.')
		if int(self.normalization_sample_size)<=0 or int(self.normalization_samples)<=0:
			raise SegmentationError('Normalization sample size and sample count must be positive.')


	def to_dict(self)->dict[str,Any]:
		result=asdict(self)
		if self.position is not None:
			result['position']=dict(self.position)
		return result



@dataclass(frozen=True,slots=True)
class InferenceResult:
	masks:np.ndarray
	class_ids:np.ndarray
	scores:np.ndarray


	def __post_init__(self)->None:
		masks=np.asarray(self.masks)
		class_ids=np.asarray(self.class_ids)
		scores=np.asarray(self.scores)
		if masks.ndim!=3:
			raise SegmentationError('Inference masks must have shape N x Y x X.')
		if class_ids.ndim!=1 or scores.ndim!=1:
			raise SegmentationError('Inference classes and scores must be one-dimensional.')
		if len(masks)!=len(class_ids)or len(masks)!=len(scores):
			raise SegmentationError('Inference masks, classes, and scores have inconsistent lengths.')



@dataclass(frozen=True,slots=True)
class LabelInferenceResult:
	labels:np.ndarray
	object_labels:np.ndarray
	class_ids:np.ndarray
	scores:np.ndarray


	def __post_init__(self)->None:
		labels=np.asarray(self.labels)
		object_labels=np.asarray(self.object_labels)
		class_ids=np.asarray(self.class_ids)
		scores=np.asarray(self.scores)
		if labels.ndim!=2:
			raise SegmentationError('Label inference image must be two-dimensional.')
		if object_labels.ndim!=1 or class_ids.ndim!=1 or scores.ndim!=1:
			raise SegmentationError('Label inference metadata must be one-dimensional.')
		if len(object_labels)!=len(class_ids)or len(object_labels)!=len(scores):
			raise SegmentationError('Label inference labels, classes, and scores have inconsistent lengths.')



class InferenceAdapter(Protocol):
	metadata:DetectorMetadata


	def infer_batch(self,images:Sequence[np.ndarray])->list[InferenceResult]:



@dataclass(frozen=True,slots=True)
class TilePredictionSummary:
	tile_id:str
	prediction_count:int
	owned_prediction_count:int
	output_path:str


	def to_dict(self)->dict[str,Any]:
		return asdict(self)



@dataclass(frozen=True,slots=True)
class SegmentationRunSummary:
	output_directory:str
	checkpoint_path:str
	channel_index:int
	channel_name:str
	normalization:IntensityNormalization
	progress:CheckpointProgress
	predictions:int
	owned_predictions:int
	cancelled:bool
	started_at:str
	finished_at:str


	def summary(self)->str:
		return(
			f'Channel: {self.channel_name} ({self.channel_index})\n'
			f'Predictions saved: {self.predictions:,}\n'
			f'Core-owned predictions: {self.owned_predictions:,}\n'
			f'Cancelled: {self.cancelled}\n'
			f'{self.progress.summary()}\n'
			f'Output: {self.output_directory}'
		)



@dataclass(frozen=True,slots=True)
class TilePredictionArchive:
	path:Path
	tile_id:str
	read_bounds:np.ndarray
	core_bounds:np.ndarray
	scores:np.ndarray
	class_ids:np.ndarray
	class_names:np.ndarray
	local_boxes:np.ndarray
	global_boxes:np.ndarray
	local_centroids:np.ndarray
	global_centroids:np.ndarray
	areas:np.ndarray
	owned_by_core:np.ndarray
	touches_read_edge:np.ndarray
	mask_heights:np.ndarray
	mask_widths:np.ndarray
	mask_offsets:np.ndarray
	mask_data:np.ndarray


	@property
	def count(self)->int:
		return int(len(self.scores))


	def decode_cropped_mask(self,index:int)->np.ndarray:
		if index<0:
			index+=self.count
		if index<0 or index>=self.count:
			raise IndexError(index)
		start=int(self.mask_offsets[index])
		stop=int(self.mask_offsets[index+1])
		height=int(self.mask_heights[index])
		width=int(self.mask_widths[index])
		bits=np.unpackbits(self.mask_data[start:stop],bitorder='little')
		return bits[:height*width].reshape(height,width).astype(bool)


	def decode_full_mask(self,index:int)->np.ndarray:
		height=int(self.read_bounds[3])
		width=int(self.read_bounds[2])
		output=np.zeros((height,width),dtype=bool)
		x0,y0,x1,y1=(int(value)for value in self.local_boxes[index])
		output[y0:y1,x0:x1]=self.decode_cropped_mask(index)
		return output
DETECTOR_REQUIRED_FILENAMES=(
	'model_parameters.txt',
	'config.yaml',
	'model_final.pth',
)


def is_mplexa_detector_directory(path:str|Path)->bool:
	'''Return ``True`` when *path* contains a complete MPlexA detector bundle.'''
	candidate=Path(path).expanduser()
	return candidate.is_dir()and all(
		(candidate/filename).is_file()for filename in DETECTOR_REQUIRED_FILENAMES
	)


def discover_mplexa_detectors(detectors_directory:str|Path)->tuple[Path,...]:
	root=Path(detectors_directory).expanduser()
	if not root.is_dir():
		return()
	detectors=[
		item.resolve()
		for item in root.iterdir()
		if item.is_dir()
		and not item.name.startswith('.')
		and item.name!='__pycache__'
		and is_mplexa_detector_directory(item)
	]
	return tuple(sorted(detectors,key=lambda item:item.name.casefold()))


def read_detector_metadata(path_to_detector:str|Path)->DetectorMetadata:
	path=Path(path_to_detector).expanduser().resolve()
	parameters_path=path/DETECTOR_REQUIRED_FILENAMES[0]
	config_path=path/DETECTOR_REQUIRED_FILENAMES[1]
	model_path=path/DETECTOR_REQUIRED_FILENAMES[2]
	missing=[str(item.name)for item in(parameters_path,config_path,model_path)if not item.is_file()]
	if missing:
		raise SegmentationError(
			'Detector folder is missing required file(s): {}.'.format(', '.join(missing))
		)
	try:
		data=json.loads(parameters_path.read_text(encoding='utf-8'))
		frame_size=int(data['inferencing_framesize'])
		raw_mapping=data['cell_mapping']
	except(OSError,ValueError,KeyError,TypeError,json.JSONDecodeError)as error:
		raise SegmentationError(f'Unable to read detector metadata: {error}')from error
	mapping={int(key):str(value)for key,value in raw_mapping.items()}
	names=tuple(str(value)for value in data.get('cell_names',mapping.values()))
	if frame_size<=0:
		raise SegmentationError('Detector inference frame size must be positive.')
	background_flag=int(data.get('black_background',0))
	fingerprint_payload:list[dict[str,Any]]=[]
	for item in(parameters_path,config_path,model_path):
		stat=item.stat()
		fingerprint_payload.append(
			{
				'name':item.name,
				'size':stat.st_size,
				'mtime_ns':stat.st_mtime_ns,
			}
		)
	fingerprint=hashlib.sha256(_canonical_json(fingerprint_payload).encode('utf-8')).hexdigest()
	return DetectorMetadata(
		path=str(path),
		cell_names=names,
		cell_mapping=mapping,
		inferencing_framesize=frame_size,
		black_background=background_flag==0,
		fingerprint=fingerprint,
	)


def build_detector_tile_grid(
	image_width:int,
	image_height:int,
	detector:DetectorMetadata,
	*,
	overlap_ratio:float|tuple[float,float]=0.10,
	level:int=0,
)->TileGrid:
	frame=int(detector.inferencing_framesize)
	if frame<=0:
		raise SegmentationError('Detector inference frame size must be positive.')
	return TileGrid(
		image_width,
		image_height,
		tile_width=frame,
		tile_height=frame,
		overlap_ratio=overlap_ratio,
		level=level,
	)


def build_threshold_tile_grid(
	image_width:int,
	image_height:int,
	*,
	tile_size:int=2048,
	overlap_ratio:float|tuple[float,float]=0.10,
	level:int=0,
)->TileGrid:
	tile_size=int(tile_size)
	if tile_size<=0:
		raise SegmentationError('Threshold-segmentation tile size must be positive.')
	return TileGrid(
		image_width,
		image_height,
		tile_width=tile_size,
		tile_height=tile_size,
		overlap_ratio=overlap_ratio,
		level=level,
	)


def choose_pyramid_level_for_pixel_size(metadata:Any,requested_pixel_size:float)->int:
	requested=float(requested_pixel_size)
	if requested<=0:
		return 0
	base_x=getattr(metadata,'pixel_size_x',None)
	base_y=getattr(metadata,'pixel_size_y',None)
	if base_x is None or base_y is None:
		raise SegmentationError(
			'Requested-pixel-size mode requires physical pixel-size metadata. '
			'Use the Pyramid level control directly for this image.'
		)
	levels=tuple(getattr(metadata,'levels',()))
	axes=str(getattr(metadata,'axes',''))
	shape=tuple(getattr(metadata,'shape',()))
	if not levels or'X'not in axes or'Y'not in axes:
		raise SegmentationError('Image pyramid metadata are unavailable.')
	base_width=float(shape[axes.index('X')])
	base_height=float(shape[axes.index('Y')])
	candidates:list[tuple[float,int]]=[]
	for level in levels:
		level_axes=str(level.axes)
		width=float(level.shape[level_axes.index('X')])
		height=float(level.shape[level_axes.index('Y')])
		px_x=float(base_x)*(base_width/width)
		px_y=float(base_y)*(base_height/height)
		effective=math.sqrt(px_x*px_y)
		distance=abs(math.log(max(effective,1e-12)/requested))
		candidates.append((distance,int(level.level)))
	return min(candidates)[1]



@dataclass(frozen=True,slots=True)
class ThresholdSegmentationConfig:
	channel:int|str='DAPI'
	threshold_value:float=25.0
	foreground:str='bright'
	background_radius:int=15
	background_by_reconstruction:bool=True
	median_radius:int=0
	gaussian_sigma:float=3.0
	min_area:int=10
	max_area:int=1000
	split_touching:bool=True
	watershed_min_distance:int=3
	refine_boundaries:bool=True
	retain_core_owned_only:bool=True
	tile_size:int=2048
	batch_size:int=1
	cpu_workers:int=0
	fast_archives:bool=True
	low_percentile:float=1.0
	high_percentile:float=99.8
	normalization_sample_size:int=512
	normalization_samples:int=16
	retry_failed:bool=False
	fill_holes:bool=True
	position:Mapping[str,int]|None=None


	def __post_init__(self)->None:
		if not 0<=float(self.threshold_value)<=255:
			raise SegmentationError('Threshold value must be between 0 and 255.')
		if str(self.foreground).lower()not in{'bright','dark'}:
			raise SegmentationError('Threshold foreground must be \'bright\' or \'dark\'.')
		if int(self.background_radius)<0:
			raise SegmentationError('Background radius cannot be negative.')
		if int(self.median_radius)<0:
			raise SegmentationError('Median radius cannot be negative.')
		if float(self.gaussian_sigma)<0:
			raise SegmentationError('Gaussian sigma cannot be negative.')
		if int(self.min_area)<=0:
			raise SegmentationError('Minimum object area must be positive.')
		if int(self.max_area)<int(self.min_area):
			raise SegmentationError('Maximum object area must be >= minimum object area.')
		if int(self.watershed_min_distance)<=0:
			raise SegmentationError('Watershed minimum distance must be positive.')
		if int(self.tile_size)<=0:
			raise SegmentationError('Threshold-segmentation tile size must be positive.')
		if int(self.batch_size)<=0:
			raise SegmentationError('Batch size must be positive.')
		if int(self.cpu_workers)<0:
			raise SegmentationError('CPU workers cannot be negative; use 0 for Auto.')
		if not 0<=float(self.low_percentile)<float(self.high_percentile)<=100:
			raise SegmentationError('Normalization percentiles must satisfy 0 <= low < high <= 100.')
		if int(self.normalization_sample_size)<=0 or int(self.normalization_samples)<=0:
			raise SegmentationError('Normalization sample size and sample count must be positive.')


	def to_dict(self)->dict[str,Any]:
		result=asdict(self)
		if self.position is not None:
			result['position']=dict(self.position)
		return result


	def resolved_cpu_workers(self)->int:
		requested=int(self.cpu_workers)
		if requested>0:
			return requested
		available=max(1,int(os.cpu_count()or 1))
		return max(1,min(4,available))


def _disk_footprint(radius:int)->np.ndarray:
	radius=int(radius)
	if radius<=0:
		return np.ones((1,1),dtype=bool)
	return morphology.disk(radius).astype(bool,copy=False)


def _estimate_adaptive_background(
	image:np.ndarray,
	*,
	radius:int,
	by_reconstruction:bool,
)->np.ndarray:
	data=np.asarray(image,dtype=np.float32)
	radius=int(radius)
	if radius<=0:
		return np.zeros_like(data,dtype=np.float32)
	footprint=_disk_footprint(radius)
	eroded=morphology.erosion(data,footprint=footprint)
	if by_reconstruction:
		return morphology.reconstruction(eroded,data,method='dilation',footprint=footprint).astype(
			np.float32,copy=False
		)
	return morphology.dilation(eroded,footprint=footprint).astype(np.float32,copy=False)


def _shape_watershed(binary:np.ndarray,min_distance:int)->np.ndarray:
	binary=np.asarray(binary,dtype=bool)
	if not np.any(binary):
		return np.zeros(binary.shape,dtype=np.int32)
	distance=ndi.distance_transform_edt(binary)
	coordinates=feature.peak_local_max(
		distance,
		labels=binary.astype(np.uint8),
		min_distance=max(1,int(min_distance)),
		exclude_border=False,
	)
	markers=np.zeros(binary.shape,dtype=np.int32)
	if coordinates.size:
		markers[tuple(coordinates.T)]=np.arange(1,len(coordinates)+1,dtype=np.int32)
	else:
		markers=measure.label(binary,connectivity=1).astype(np.int32,copy=False)
	return watershed(-distance,markers,mask=binary).astype(np.int32,copy=False)


def _label_counts_and_means(labels:np.ndarray,intensity:np.ndarray)->tuple[np.ndarray,np.ndarray]:
	label_array=np.asarray(labels,dtype=np.int32)
	intensity_array=np.asarray(intensity,dtype=np.float32)
	if label_array.shape!=intensity_array.shape:
		raise SegmentationError('Label and intensity images must have the same shape.')
	flat_labels=label_array.reshape(-1)
	max_label=int(flat_labels.max(initial=0))
	if max_label<=0:
		return np.zeros(1,dtype=np.int64),np.zeros(1,dtype=np.float64)
	counts=np.bincount(flat_labels,minlength=max_label+1).astype(np.int64,copy=False)
	sums=np.bincount(
		flat_labels,
		weights=intensity_array.reshape(-1).astype(np.float64,copy=False),
		minlength=max_label+1,
	)
	means=np.divide(
		sums,
		counts,
		out=np.zeros_like(sums,dtype=np.float64),
		where=counts>0,
	)
	return counts,means


def _empty_label_inference(shape:tuple[int,int])->LabelInferenceResult:
	return LabelInferenceResult(
		labels=np.zeros(shape,dtype=np.int32),
		object_labels=np.empty(0,dtype=np.int32),
		class_ids=np.empty(0,dtype=np.int32),
		scores=np.empty(0,dtype=np.float32),
	)


def segment_intensity_threshold_labels(
	image_uint8:np.ndarray,
	config:ThresholdSegmentationConfig,
)->LabelInferenceResult:
	image=np.asarray(image_uint8)
	if image.ndim!=2:
		raise SegmentationError('Threshold segmentation expects a 2-D single-channel tile.')
	if image.dtype!=np.uint8:
		image=np.clip(image,0,255).astype(np.uint8)
	bright=str(config.foreground).lower()=='bright'
	working=image.astype(np.float32,copy=False)if bright else(255.0-image.astype(np.float32,copy=False))
	median_radius=int(config.median_radius)
	if median_radius>0:
		working=filters.median(working,footprint=_disk_footprint(median_radius)).astype(
			np.float32,copy=False
		)
	background_radius=int(config.background_radius)
	if background_radius>0:
		background=_estimate_adaptive_background(
			working,
			radius=background_radius,
			by_reconstruction=bool(config.background_by_reconstruction),
		)
		measured=np.maximum(working-background,0.0).astype(np.float32,copy=False)
	else:
		measured=working.astype(np.float32,copy=False)
	sigma=float(config.gaussian_sigma)
	smoothed=ndi.gaussian_filter(measured,sigma=sigma,mode='nearest')if sigma>0 else measured
	log_response=-ndi.laplace(smoothed,mode='nearest')
	positive_log=log_response>0
	if not np.any(positive_log):
		return _empty_label_inference(image.shape)
	regional_maxima=morphology.local_maxima(log_response)&positive_log
	markers=measure.label(regional_maxima,connectivity=1).astype(np.int32,copy=False)
	if int(markers.max())==0:
		return _empty_label_inference(image.shape)
	initial=watershed(-log_response,markers,mask=positive_log).astype(np.int32,copy=False)
	_,initial_means=_label_counts_and_means(initial,measured)
	threshold=float(config.threshold_value)
	accepted_labels=initial_means>threshold
	accepted_labels[0]=False
	accepted=accepted_labels[initial]
	if not np.any(accepted):
		return _empty_label_inference(image.shape)
	merged=ndi.maximum_filter(accepted.astype(np.uint8),size=3,mode='nearest')>0
	merged&=positive_log
	if config.fill_holes and np.any(merged):
		merged=ndi.binary_fill_holes(merged)
	if config.split_touching:
		labels=_shape_watershed(merged,int(config.watershed_min_distance))
	else:
		labels=measure.label(merged,connectivity=1).astype(np.int32,copy=False)
	if bool(config.refine_boundaries)and sigma>1.5 and np.any(labels):
		fine_smoothed=ndi.gaussian_filter(working,sigma=1.0,mode='nearest')
		fine_positive=-ndi.laplace(fine_smoothed,mode='nearest')>0
		current=labels>0
		eroded=ndi.minimum_filter(current.astype(np.uint8),size=3,mode='nearest')>0
		refined=eroded|(current&fine_positive)
		labels=(
			_shape_watershed(refined,int(config.watershed_min_distance))
			if config.split_touching
			else measure.label(refined,connectivity=1).astype(np.int32,copy=False)
		)
	counts,means=_label_counts_and_means(labels,measured)
	if len(counts)<=1:
		return _empty_label_inference(image.shape)
	label_ids=np.arange(len(counts),dtype=np.int32)
	keep=(
		(label_ids>0)
		&(counts>=int(config.min_area))
		&(counts<=int(config.max_area))
		&(means>threshold)
	)
	object_labels=label_ids[keep]
	scores=np.clip(means[keep]/255.0,0.0,1.0).astype(np.float32,copy=False)
	return LabelInferenceResult(
		labels=labels,
		object_labels=object_labels.astype(np.int32,copy=False),
		class_ids=np.zeros(len(object_labels),dtype=np.int32),
		scores=scores,
	)


def segment_intensity_threshold(
	image_uint8:np.ndarray,
	config:ThresholdSegmentationConfig,
)->InferenceResult:
	result=segment_intensity_threshold_labels(image_uint8,config)
	if len(result.object_labels)==0:
		return InferenceResult(
			masks=np.empty((0,result.labels.shape[0],result.labels.shape[1]),dtype=bool),
			class_ids=result.class_ids,
			scores=result.scores,
		)
	masks=np.stack([result.labels==int(label)for label in result.object_labels]).astype(bool,copy=False)
	return InferenceResult(masks=masks,class_ids=result.class_ids,scores=result.scores)



class MPlexADetectorAdapter:


	def __init__(self,path_to_detector:str|Path)->None:
		self.metadata=read_detector_metadata(path_to_detector)
		# Import lazily so metadata/tests do not require a working Detectron2 binary.
		from MPlexA.detector import Detector
		self._detector=Detector()
		self._detector.load(self.metadata.path,list(self.metadata.cell_names))


	def infer_batch(self,images:Sequence[np.ndarray])->list[InferenceResult]:
		if not images:
			return[]
		import torch
		inputs:list[dict[str,Any]]=[]
		for image in images:
			array=np.asarray(image)
			if array.ndim!=3 or array.shape[2]!=3:
				raise SegmentationError('Detector input must have shape H x W x 3.')
			inputs.append(
				{
					'image':torch.as_tensor(array.astype('float32').transpose(2,0,1)),
					'height':int(array.shape[0]),
					'width':int(array.shape[1]),
				}
			)
		raw_outputs=self._detector.inference(inputs)
		results:list[InferenceResult]=[]
		for output in raw_outputs:
			instances=output['instances'].to('cpu')
			if hasattr(instances,'pred_masks'):
				masks=instances.pred_masks.numpy().astype(bool,copy=False)
			else:
				image_size=tuple(int(value)for value in instances.image_size)
				masks=np.empty((0,*image_size),dtype=bool)
			class_ids=instances.pred_classes.numpy().astype(np.int32,copy=False)
			scores=instances.scores.numpy().astype(np.float32,copy=False)
			results.append(InferenceResult(masks,class_ids,scores))
		return results


def _sample_starts(length:int,sample_size:int,count:int)->tuple[int,...]:
	size=min(int(sample_size),int(length))
	if count<=1 or length<=size:
		return(max(0,(length-size)//2),)
	starts=np.linspace(0,length-size,count,dtype=np.int64)
	return tuple(dict.fromkeys(int(value)for value in starts))


def estimate_intensity_normalization(
	image:MultiplexImageSource,
	*,
	channel:int|str,
	level:int=0,
	low_percentile:float=1.0,
	high_percentile:float=99.8,
	sample_size:int=512,
	max_samples:int=16,
	max_pixels:int=1_000_000,
	position:Mapping[str,int]|None=None,
)->IntensityNormalization:
	if not 0<=low_percentile<high_percentile<=100:
		raise SegmentationError('Normalization percentiles must satisfy 0 <= low < high <= 100.')
	if sample_size<=0 or max_samples<=0 or max_pixels<=0:
		raise SegmentationError('Normalization sampling parameters must be positive.')
	if level<0 or level>=image.level_count:
		raise SegmentationError(f'Resolution level {level} is unavailable.')
	level_meta=image.metadata.levels[level]
	width=int(level_meta.shape[level_meta.axes.index('X')])
	height=int(level_meta.shape[level_meta.axes.index('Y')])
	axis_count=max(1,int(math.ceil(math.sqrt(max_samples))))
	x_starts=_sample_starts(width,sample_size,axis_count)
	y_starts=_sample_starts(height,sample_size,axis_count)
	locations=[(x,y)for y in y_starts for x in x_starts][:max_samples]
	collected:list[np.ndarray]=[]
	remaining=int(max_pixels)
	for sample_index,(x,y)in enumerate(locations):
		if remaining<=0:
			break
		tile=image.read_region(
			x=x,
			y=y,
			width=min(sample_size,width-x),
			height=min(sample_size,height-y),
			channels=channel,
			level=level,
			position=position,
		)[0]
		values=np.asarray(tile).reshape(-1)
		values=values[np.isfinite(values)]
		if values.size==0:
			continue
		samples_left=max(1,len(locations)-sample_index)
		allowance=max(1,remaining//samples_left)
		if values.size>allowance:
			step=int(math.ceil(values.size/allowance))
			values=values[::step][:allowance]
		values=values.astype(np.float64,copy=False)
		collected.append(values)
		remaining-=int(values.size)
	if not collected:
		raise SegmentationError('No finite pixels were available for intensity normalization.')
	pixels=np.concatenate(collected)
	low,high=np.percentile(pixels,[low_percentile,high_percentile])
	if not np.isfinite(low)or not np.isfinite(high):
		raise SegmentationError('Unable to estimate finite normalization limits.')
	if high<=low:
		data_min=float(np.min(pixels))
		data_max=float(np.max(pixels))
		if data_max>data_min:
			low,high=data_min,data_max
		else:
			high=float(low)+1.0
	return IntensityNormalization(
		low_value=float(low),
		high_value=float(high),
		low_percentile=float(low_percentile),
		high_percentile=float(high_percentile),
		sample_count=len(collected),
		sampled_pixels=int(pixels.size),
	)


def _prediction_arrays(
	tile:Tile,
	result:InferenceResult,
	*,
	class_mapping:Mapping[int,str],
	score_threshold:float,
	core_owned_only:bool=False,
)->dict[str,np.ndarray]:
	read_height=tile.read_bounds.height
	read_width=tile.read_bounds.width
	local_boxes:list[tuple[int,int,int,int]]=[]
	global_boxes:list[tuple[int,int,int,int]]=[]
	local_centroids:list[tuple[float,float]]=[]
	global_centroids:list[tuple[float,float]]=[]
	areas:list[int]=[]
	scores:list[float]=[]
	class_ids:list[int]=[]
	class_names:list[str]=[]
	owned:list[bool]=[]
	touches_edge:list[bool]=[]
	mask_heights:list[int]=[]
	mask_widths:list[int]=[]
	packed_masks:list[np.ndarray]=[]
	for mask,class_id,score in zip(result.masks,result.class_ids,result.scores):
		if float(score)<float(score_threshold):
			continue
		clipped=np.asarray(mask,dtype=bool)[:read_height,:read_width]
		ys,xs=np.nonzero(clipped)
		if xs.size==0:
			continue
		x0=int(xs.min())
		y0=int(ys.min())
		x1=int(xs.max())+1
		y1=int(ys.max())+1
		cropped=clipped[y0:y1,x0:x1]
		area=int(cropped.sum())
		if area<=0:
			continue
		cx=float(xs.mean())
		cy=float(ys.mean())
		gx,gy=tile.local_to_global(cx,cy)
		is_owned=tile.owns_global_point(gx,gy)
		if core_owned_only and not is_owned:
			continue
		local_boxes.append((x0,y0,x1,y1))
		global_boxes.append(
			(
				x0+tile.read_bounds.x,
				y0+tile.read_bounds.y,
				x1+tile.read_bounds.x,
				y1+tile.read_bounds.y,
			)
		)
		local_centroids.append((cx,cy))
		global_centroids.append((gx,gy))
		areas.append(area)
		scores.append(float(score))
		class_value=int(class_id)
		class_ids.append(class_value)
		class_names.append(str(class_mapping.get(class_value,class_value)))
		owned.append(is_owned)
		touches_edge.append(x0==0 or y0==0 or x1==read_width or y1==read_height)
		mask_heights.append(int(cropped.shape[0]))
		mask_widths.append(int(cropped.shape[1]))
		packed_masks.append(np.packbits(cropped.reshape(-1),bitorder='little'))
	offsets=[0]
	for packed in packed_masks:
		offsets.append(offsets[-1]+len(packed))
	mask_data=np.concatenate(packed_masks).astype(np.uint8,copy=False)if packed_masks else np.empty(0,dtype=np.uint8)
	count=len(scores)
	return{
		'scores':np.asarray(scores,dtype=np.float32),
		'class_ids':np.asarray(class_ids,dtype=np.int32),
		'class_names':np.asarray(class_names,dtype='U128'),
		'local_boxes':np.asarray(local_boxes,dtype=np.int32).reshape(count,4),
		'global_boxes':np.asarray(global_boxes,dtype=np.int64).reshape(count,4),
		'local_centroids':np.asarray(local_centroids,dtype=np.float32).reshape(count,2),
		'global_centroids':np.asarray(global_centroids,dtype=np.float64).reshape(count,2),
		'areas':np.asarray(areas,dtype=np.int64),
		'owned_by_core':np.asarray(owned,dtype=bool),
		'touches_read_edge':np.asarray(touches_edge,dtype=bool),
		'mask_heights':np.asarray(mask_heights,dtype=np.int32),
		'mask_widths':np.asarray(mask_widths,dtype=np.int32),
		'mask_offsets':np.asarray(offsets,dtype=np.int64),
		'mask_data':mask_data,
	}


def _label_prediction_arrays(
	tile:Tile,
	result:LabelInferenceResult,
	*,
	class_mapping:Mapping[int,str],
	core_owned_only:bool=False,
)->dict[str,np.ndarray]:
	labels=np.asarray(result.labels,dtype=np.int32)
	read_height=min(int(tile.read_bounds.height),int(labels.shape[0]))
	read_width=min(int(tile.read_bounds.width),int(labels.shape[1]))
	labels=labels[:read_height,:read_width]
	object_slices=ndi.find_objects(labels)
	local_boxes:list[tuple[int,int,int,int]]=[]
	global_boxes:list[tuple[int,int,int,int]]=[]
	local_centroids:list[tuple[float,float]]=[]
	global_centroids:list[tuple[float,float]]=[]
	areas:list[int]=[]
	scores:list[float]=[]
	class_ids:list[int]=[]
	class_names:list[str]=[]
	owned:list[bool]=[]
	touches_edge:list[bool]=[]
	mask_heights:list[int]=[]
	mask_widths:list[int]=[]
	packed_masks:list[np.ndarray]=[]
	for object_label,class_id,score in zip(result.object_labels,result.class_ids,result.scores):
		label_value=int(object_label)
		slice_index=label_value-1
		if slice_index<0 or slice_index>=len(object_slices):
			continue
		object_slice=object_slices[slice_index]
		if object_slice is None:
			continue
		y_slice,x_slice=object_slice
		y0,y1=int(y_slice.start),min(int(y_slice.stop),read_height)
		x0,x1=int(x_slice.start),min(int(x_slice.stop),read_width)
		if y0>=y1 or x0>=x1:
			continue
		cropped=labels[y0:y1,x0:x1]==label_value
		ys,xs=np.nonzero(cropped)
		if xs.size==0:
			continue
		area=int(xs.size)
		cx=float(x0+xs.mean())
		cy=float(y0+ys.mean())
		gx,gy=tile.local_to_global(cx,cy)
		is_owned=tile.owns_global_point(gx,gy)
		if core_owned_only and not is_owned:
			continue
		local_boxes.append((x0,y0,x1,y1))
		global_boxes.append(
			(
				x0+tile.read_bounds.x,
				y0+tile.read_bounds.y,
				x1+tile.read_bounds.x,
				y1+tile.read_bounds.y,
			)
		)
		local_centroids.append((cx,cy))
		global_centroids.append((gx,gy))
		areas.append(area)
		scores.append(float(score))
		class_value=int(class_id)
		class_ids.append(class_value)
		class_names.append(str(class_mapping.get(class_value,class_value)))
		owned.append(is_owned)
		touches_edge.append(x0==0 or y0==0 or x1==read_width or y1==read_height)
		mask_heights.append(int(cropped.shape[0]))
		mask_widths.append(int(cropped.shape[1]))
		packed_masks.append(np.packbits(cropped.reshape(-1),bitorder='little'))
	offsets=[0]
	for packed in packed_masks:
		offsets.append(offsets[-1]+len(packed))
	mask_data=(
		np.concatenate(packed_masks).astype(np.uint8,copy=False)
		if packed_masks
		else np.empty(0,dtype=np.uint8)
	)
	count=len(scores)
	return{
		'scores':np.asarray(scores,dtype=np.float32),
		'class_ids':np.asarray(class_ids,dtype=np.int32),
		'class_names':np.asarray(class_names,dtype='U128'),
		'local_boxes':np.asarray(local_boxes,dtype=np.int32).reshape(count,4),
		'global_boxes':np.asarray(global_boxes,dtype=np.int64).reshape(count,4),
		'local_centroids':np.asarray(local_centroids,dtype=np.float32).reshape(count,2),
		'global_centroids':np.asarray(global_centroids,dtype=np.float64).reshape(count,2),
		'areas':np.asarray(areas,dtype=np.int64),
		'owned_by_core':np.asarray(owned,dtype=bool),
		'touches_read_edge':np.asarray(touches_edge,dtype=bool),
		'mask_heights':np.asarray(mask_heights,dtype=np.int32),
		'mask_widths':np.asarray(mask_widths,dtype=np.int32),
		'mask_offsets':np.asarray(offsets,dtype=np.int64),
		'mask_data':mask_data,
	}


def _save_prediction_archive(
	destination:Path,
	tile:Tile,
	arrays:Mapping[str,np.ndarray],
	*,
	compressed:bool,
)->None:
	handle=tempfile.NamedTemporaryFile(
		mode='wb',suffix='.npz',prefix=destination.stem+'.',dir=destination.parent,delete=False
	)
	temporary=Path(handle.name)
	handle.close()
	writer=np.savez_compressed if compressed else np.savez
	try:
		writer(
			temporary,
			schema_version=np.asarray([SEGMENTATION_SCHEMA_VERSION],dtype=np.int16),
			tile_id=np.asarray(tile.tile_id),
			read_bounds=np.asarray(
				[tile.read_bounds.x,tile.read_bounds.y,tile.read_bounds.width,tile.read_bounds.height],
				dtype=np.int64,
			),
			core_bounds=np.asarray(
				[tile.core_bounds.x,tile.core_bounds.y,tile.core_bounds.width,tile.core_bounds.height],
				dtype=np.int64,
			),
			**arrays,
		)
		os.replace(temporary,destination)
	except Exception:
		temporary.unlink(missing_ok=True)
		raise


def save_tile_predictions(
	path:str|Path,
	tile:Tile,
	result:InferenceResult,
	*,
	class_mapping:Mapping[int,str],
	score_threshold:float,
	core_owned_only:bool=False,
	compressed:bool=True,
)->TilePredictionSummary:
	destination=Path(path).expanduser().resolve()
	destination.parent.mkdir(parents=True,exist_ok=True)
	arrays=_prediction_arrays(
		tile,
		result,
		class_mapping=class_mapping,
		score_threshold=score_threshold,
		core_owned_only=core_owned_only,
	)
	_save_prediction_archive(destination,tile,arrays,compressed=bool(compressed))
	return TilePredictionSummary(
		tile_id=tile.tile_id,
		prediction_count=int(len(arrays['scores'])),
		owned_prediction_count=int(np.count_nonzero(arrays['owned_by_core'])),
		output_path=str(destination),
	)


def save_label_tile_predictions(
	path:str|Path,
	tile:Tile,
	result:LabelInferenceResult,
	*,
	class_mapping:Mapping[int,str],
	core_owned_only:bool=False,
	compressed:bool=False,
)->TilePredictionSummary:
	destination=Path(path).expanduser().resolve()
	destination.parent.mkdir(parents=True,exist_ok=True)
	arrays=_label_prediction_arrays(
		tile,
		result,
		class_mapping=class_mapping,
		core_owned_only=core_owned_only,
	)
	_save_prediction_archive(destination,tile,arrays,compressed=bool(compressed))
	return TilePredictionSummary(
		tile_id=tile.tile_id,
		prediction_count=int(len(arrays['scores'])),
		owned_prediction_count=int(np.count_nonzero(arrays['owned_by_core'])),
		output_path=str(destination),
	)


def load_tile_predictions(path:str|Path)->TilePredictionArchive:
	source=Path(path).expanduser().resolve()
	try:
		with np.load(source,allow_pickle=False)as data:
			version=int(data['schema_version'][0])
			if version!=SEGMENTATION_SCHEMA_VERSION:
				raise SegmentationError(
					f'Prediction archive schema {version} is incompatible with {SEGMENTATION_SCHEMA_VERSION}.'
				)
			return TilePredictionArchive(
				path=source,
				tile_id=str(data['tile_id'].item()),
				read_bounds=data['read_bounds'].copy(),
				core_bounds=data['core_bounds'].copy(),
				scores=data['scores'].copy(),
				class_ids=data['class_ids'].copy(),
				class_names=data['class_names'].copy(),
				local_boxes=data['local_boxes'].copy(),
				global_boxes=data['global_boxes'].copy(),
				local_centroids=data['local_centroids'].copy(),
				global_centroids=data['global_centroids'].copy(),
				areas=data['areas'].copy(),
				owned_by_core=data['owned_by_core'].copy(),
				touches_read_edge=data['touches_read_edge'].copy(),
				mask_heights=data['mask_heights'].copy(),
				mask_widths=data['mask_widths'].copy(),
				mask_offsets=data['mask_offsets'].copy(),
				mask_data=data['mask_data'].copy(),
			)
	except(OSError,KeyError,ValueError)as error:
		raise SegmentationError(f'Unable to load tile prediction archive {source}: {error}')from error



class TiledDapiSegmenter:


	def __init__(
		self,
		path_to_detector:str|Path|None=None,
		*,
		adapter:InferenceAdapter|None=None,
	)->None:
		if adapter is None and path_to_detector is None:
			raise SegmentationError('A detector folder or inference adapter is required.')
		self.adapter=adapter if adapter is not None else MPlexADetectorAdapter(path_to_detector)# type: ignore[arg-type]
		self.detector_metadata=self.adapter.metadata


	def _validate_grid(self,image:MultiplexImageSource,grid:TileGrid)->None:
		if grid.level<0 or grid.level>=image.level_count:
			raise TilingError(f'Grid level {grid.level} is unavailable in the selected image.')
		level=image.metadata.levels[grid.level]
		width=int(level.shape[level.axes.index('X')])
		height=int(level.shape[level.axes.index('Y')])
		if(grid.image_width,grid.image_height)!=(width,height):
			raise TilingError(
				'Tile grid dimensions do not match the selected image level: '
				f'grid={grid.image_width}x{grid.image_height}, image={width}x{height}.'
			)
		frame=self.detector_metadata.inferencing_framesize
		if grid.tile_width!=frame or grid.tile_height!=frame:
			raise TilingError(
				f'This detector was trained for {frame} x {frame} px inference tiles. '
				'Set both read tile dimensions to the detector inference frame size.'
			)


	def run(
		self,
		*,
		image_path:str|Path,
		series:int,
		grid:TileGrid,
		output_directory:str|Path,
		config:SegmentationConfig,
		normalization:IntensityNormalization|None=None,
		checkpoint_path:str|Path|None=None,
		cancel_event:threading.Event|None=None,
		on_progress:Callable[[CheckpointProgress,TilePredictionSummary|None],None]|None=None,
		on_log:Callable[[str],None]|None=None,
	)->SegmentationRunSummary:
		started_at=_utc_now()
		cancel_event=cancel_event or threading.Event()
		output=Path(output_directory).expanduser().resolve()
		tiles_directory=output/'tiles'
		tiles_directory.mkdir(parents=True,exist_ok=True)
		checkpoint=Path(checkpoint_path).expanduser().resolve()if checkpoint_path else output/'segmentation.sqlite'
		image_path_resolved=Path(image_path).expanduser().resolve()
		total_predictions=0
		total_owned=0
		cancelled=False


		def log(message:str)->None:
			if on_log is not None:
				on_log(message)
		with open_multiplex_image(image_path_resolved,series=series)as image:
			self._validate_grid(image,grid)
			channel_index=image.channel_index(config.channel)
			channel_name=image.channel_names[channel_index]
			if normalization is None:
				log('Estimating image-wide DAPI intensity normalization...')
				normalization=estimate_intensity_normalization(
					image,
					channel=channel_index,
					level=grid.level,
					low_percentile=config.low_percentile,
					high_percentile=config.high_percentile,
					sample_size=config.normalization_sample_size,
					max_samples=config.normalization_samples,
					position=config.position,
				)
			context={
				'segmentation_schema_version':SEGMENTATION_SCHEMA_VERSION,
				'image':_path_identity(image_path_resolved),
				'series':int(series),
				'level':int(grid.level),
				'channel_index':int(channel_index),
				'channel_name':channel_name,
				'detector_fingerprint':self.detector_metadata.fingerprint,
				'detector_mapping':{str(key):value for key,value in self.detector_metadata.cell_mapping.items()},
				'score_threshold':float(config.score_threshold),
				'position':dict(config.position or{}),
				'normalization':normalization.to_dict(),
			}
			run_configuration={
				'created_at':started_at,
				'grid':grid.to_dict(),
				'detector':self.detector_metadata.to_dict(),
				'execution_config':config.to_dict(),
				**context,
			}
			config_path=output/'segmentation_config.json'
			padding_value=0 if self.detector_metadata.black_background else 255
			with TileCheckpointStore(
				checkpoint,
				grid,
				job_name='MPlexA tiled DAPI segmentation',
				context=context,
				reset_interrupted=True,
			)as store:
				config_path.write_text(json.dumps(run_configuration,indent=2),encoding='utf-8')
				if config.retry_failed:
					reset_count=store.reset_failed()
					if reset_count:
						log(f'Reset {reset_count} failed tile(s) for one retry pass.')
				initial_progress=store.progress()
				if on_progress is not None:
					on_progress(initial_progress,None)
				while True:
					if cancel_event.is_set():
						cancelled=True
						log('Cancellation requested; stopping before the next batch.')
						break
					batch_tiles:list[Tile]=[]
					for _ in range(int(config.batch_size)):
						tile=store.claim_next(include_failed=False)
						if tile is None:
							break
						batch_tiles.append(tile)
					if not batch_tiles:
						break
					images:list[np.ndarray]=[]
					try:
						for tile in batch_tiles:
							raw=tile.read_from(
								image,
								channels=channel_index,
								position=config.position,
								pad=False,
							)[0]
							normalized=normalization.apply(raw)
							if tile.padding.required:
								normalized=tile.pad_array(normalized,constant_value=padding_value)
							three_channel=np.repeat(normalized[:,:,None],3,axis=2)
							images.append(np.ascontiguousarray(three_channel))
						results=self.adapter.infer_batch(images)
						if len(results)!=len(batch_tiles):
							raise SegmentationError(
								f'Detector returned {len(results)} outputs for {len(batch_tiles)} inputs.'
							)
					except Exception as error:
						for tile in batch_tiles:
							store.mark_failed(tile.tile_id,error)
						progress=store.progress()
						if on_progress is not None:
							on_progress(progress,None)
						log(f'Batch failed: {error}')
						continue
					for tile,result in zip(batch_tiles,results):
						if cancel_event.is_set():
							store.mark_pending(tile.tile_id)
							cancelled=True
							continue
						try:
							tile_output=tiles_directory/ f'{tile.tile_id}.npz'
							summary=save_tile_predictions(
								tile_output,
								tile,
								result,
								class_mapping=self.detector_metadata.cell_mapping,
								score_threshold=config.score_threshold,
							)
							store.mark_completed(tile.tile_id,summary.to_dict())
							total_predictions+=summary.prediction_count
							total_owned+=summary.owned_prediction_count
							progress=store.progress()
							if on_progress is not None:
								on_progress(progress,summary)
						except Exception as error:
							store.mark_failed(tile.tile_id,error)
							log(f'Tile {tile.tile_id} failed: {error}')
							progress=store.progress()
							if on_progress is not None:
								on_progress(progress,None)
					if cancelled:
						break
				final_progress=store.progress()
				# Include predictions from completed tiles on resumed runs.
				total_predictions=0
				total_owned=0
				for tile in store.iter_tiles(('completed',)):
					status=store.status(tile.tile_id)
					output_data=status.get('output')or{}
					total_predictions+=int(output_data.get('prediction_count',0))
					total_owned+=int(output_data.get('owned_prediction_count',0))
		finished_at=_utc_now()
		summary=SegmentationRunSummary(
			output_directory=str(output),
			checkpoint_path=str(checkpoint),
			channel_index=channel_index,
			channel_name=channel_name,
			normalization=normalization,
			progress=final_progress,
			predictions=total_predictions,
			owned_predictions=total_owned,
			cancelled=cancelled,
			started_at=started_at,
			finished_at=finished_at,
		)
		(output/'segmentation_summary.json').write_text(
			json.dumps(
				{
					'output_directory':summary.output_directory,
					'checkpoint_path':summary.checkpoint_path,
					'channel_index':summary.channel_index,
					'channel_name':summary.channel_name,
					'normalization':summary.normalization.to_dict(),
					'progress':asdict(summary.progress),
					'predictions':summary.predictions,
					'owned_predictions':summary.owned_predictions,
					'cancelled':summary.cancelled,
					'started_at':summary.started_at,
					'finished_at':summary.finished_at,
				},
				indent=2,
			),
			encoding='utf-8',
		)
		return summary



class TiledThresholdSegmenter:
	class_mapping:Mapping[int,str]={0:'Nucleus'}


	def _validate_grid(
		self,
		image:MultiplexImageSource,
		grid:TileGrid,
		config:ThresholdSegmentationConfig,
	)->None:
		if grid.level<0 or grid.level>=image.level_count:
			raise TilingError(f'Grid level {grid.level} is unavailable in the selected image.')
		level=image.metadata.levels[grid.level]
		width=int(level.shape[level.axes.index('X')])
		height=int(level.shape[level.axes.index('Y')])
		if(grid.image_width,grid.image_height)!=(width,height):
			raise TilingError(
				'Tile grid dimensions do not match the selected image level: '
				f'grid={grid.image_width}x{grid.image_height}, image={width}x{height}.'
			)
		tile_size=int(config.tile_size)
		if grid.tile_width!=tile_size or grid.tile_height!=tile_size:
			raise TilingError(
				f'Intensity-threshold segmentation is configured for square {tile_size} x {tile_size} px '
				'processing tiles.'
			)


	def run(
		self,
		*,
		image_path:str|Path,
		series:int,
		grid:TileGrid,
		output_directory:str|Path,
		config:ThresholdSegmentationConfig,
		normalization:IntensityNormalization|None=None,
		checkpoint_path:str|Path|None=None,
		cancel_event:threading.Event|None=None,
		on_progress:Callable[[CheckpointProgress,TilePredictionSummary|None],None]|None=None,
		on_log:Callable[[str],None]|None=None,
	)->SegmentationRunSummary:
		started_at=_utc_now()
		cancel_event=cancel_event or threading.Event()
		output=Path(output_directory).expanduser().resolve()
		tiles_directory=output/'tiles'
		tiles_directory.mkdir(parents=True,exist_ok=True)
		checkpoint=(
			Path(checkpoint_path).expanduser().resolve()
			if checkpoint_path
			else output/'segmentation.sqlite'
		)
		image_path_resolved=Path(image_path).expanduser().resolve()
		cancelled=False


		def log(message:str)->None:
			if on_log is not None:
				on_log(message)
		with open_multiplex_image(image_path_resolved,series=series)as image:
			self._validate_grid(image,grid,config)
			channel_index=image.channel_index(config.channel)
			channel_name=image.channel_names[channel_index]
			if normalization is None:
				log('Estimating image-wide intensity normalization for threshold segmentation...')
				normalization=estimate_intensity_normalization(
					image,
					channel=channel_index,
					level=grid.level,
					low_percentile=config.low_percentile,
					high_percentile=config.high_percentile,
					sample_size=config.normalization_sample_size,
					max_samples=config.normalization_samples,
					position=config.position,
				)
			result_settings={
				'threshold_algorithm':'adaptive_watershed_v3',
				'threshold_value':float(config.threshold_value),
				'foreground':str(config.foreground).lower(),
				'background_radius':int(config.background_radius),
				'background_by_reconstruction':bool(config.background_by_reconstruction),
				'median_radius':int(config.median_radius),
				'gaussian_sigma':float(config.gaussian_sigma),
				'min_area':int(config.min_area),
				'max_area':int(config.max_area),
				'split_touching':bool(config.split_touching),
				'watershed_min_distance':int(config.watershed_min_distance),
				'refine_boundaries':bool(config.refine_boundaries),
				'retain_core_owned_only':bool(config.retain_core_owned_only),
				'fill_holes':bool(config.fill_holes),
				'tile_size':int(config.tile_size),
			}
			context={
				'segmentation_schema_version':SEGMENTATION_SCHEMA_VERSION,
				'segmentation_method':'intensity_threshold',
				'threshold_algorithm':'adaptive_watershed_v3',
				'image':_path_identity(image_path_resolved),
				'series':int(series),
				'level':int(grid.level),
				'channel_index':int(channel_index),
				'channel_name':channel_name,
				'position':dict(config.position or{}),
				'normalization':normalization.to_dict(),
				'threshold_settings':result_settings,
			}
			run_configuration={
				'created_at':started_at,
				'segmentation_method':'intensity_threshold',
				'grid':grid.to_dict(),
				'threshold_segmentation':config.to_dict(),
				**context,
			}
			config_path=output/'segmentation_config.json'
			padding_value=0 if str(config.foreground).lower()=='bright'else 255
			if len(grid)>1:
				if grid.overlap_x<=0 or grid.overlap_y<=0:
					raise SegmentationError(
						'Adaptive watershed segmentation requires overlapping tiles. '
						'Use a positive X and Y overlap ratio to prevent tile-boundary artifacts.'
					)
				context_margin_x=grid.overlap_x/2.0
				context_margin_y=grid.overlap_y/2.0
				filter_margin=max(
					float(config.background_radius),
					float(config.median_radius),
					3.0*float(config.gaussian_sigma),
				)
				if min(context_margin_x,context_margin_y)<filter_margin:
					log(
						'Warning: half of the tile overlap is smaller than the filtering context '
						f'({filter_margin:.1f} px). Increase overlap to further isolate tile borders.'
					)
				equivalent_max_diameter=2.0*math.sqrt(float(config.max_area)/math.pi)
				if min(grid.overlap_x,grid.overlap_y)<equivalent_max_diameter:
					log(
						'Warning: tile overlap is smaller than the equivalent diameter implied by '
						f'Max area ({equivalent_max_diameter:.1f} px). Increase overlap beyond '
						'the largest expected object so border detections can be resolved safely.'
					)
			log(
				'Using MPlexA adaptive watershed detection with core-only tile ownership; '
				'detections outside each tile\'s central ownership region are discarded before Module 3.'
			)
			worker_count=int(config.resolved_cpu_workers())
			log(
				f'High-throughput threshold engine: {worker_count} CPU worker(s), vectorized label statistics, '
				+('fast bit-packed NPZ archives.'if config.fast_archives else'compressed bit-packed NPZ archives.')
			)
			with TileCheckpointStore(
				checkpoint,
				grid,
				job_name='MPlexA tiled intensity-threshold segmentation',
				context=context,
				reset_interrupted=True,
			)as store:
				config_path.write_text(json.dumps(run_configuration,indent=2),encoding='utf-8')
				if config.retry_failed:
					reset_count=store.reset_failed()
					if reset_count:
						log(f'Reset {reset_count} failed tile(s) for one retry pass.')
				initial_progress=store.progress()
				if on_progress is not None:
					on_progress(initial_progress,None)


				def process_tile(tile:Tile,normalized_tile:np.ndarray)->TilePredictionSummary:
					inference=segment_intensity_threshold_labels(normalized_tile,config)
					tile_output=tiles_directory/ f'{tile.tile_id}.npz'
					return save_label_tile_predictions(
						tile_output,
						tile,
						inference,
						class_mapping=self.class_mapping,
						core_owned_only=bool(config.retain_core_owned_only),
						compressed=not bool(config.fast_archives),
					)
				with ThreadPoolExecutor(max_workers=worker_count,thread_name_prefix='MPlexA-threshold')as executor:
					while True:
						if cancel_event.is_set():
							cancelled=True
							log('Cancellation requested; stopping before the next threshold worker wave.')
							break
						claimed:list[Tile]=[]
						for _ in range(worker_count):
							tile=store.claim_next(include_failed=False)
							if tile is None:
								break
							claimed.append(tile)
						if not claimed:
							break
						futures:dict[Any,Tile]={}
						for index,tile in enumerate(claimed):
							if cancel_event.is_set():
								store.mark_pending(tile.tile_id)
								for remaining in claimed[index+1:]:
									store.mark_pending(remaining.tile_id)
								cancelled=True
								break
							try:
								raw=tile.read_from(
									image,
									channels=channel_index,
									position=config.position,
									pad=False,
								)[0]
								normalized_tile=normalization.apply(raw)
								if tile.padding.required:
									normalized_tile=tile.pad_array(
										normalized_tile,
										constant_value=padding_value,
									)
								futures[executor.submit(process_tile,tile,normalized_tile)]=tile
							except Exception as error:
								store.mark_failed(tile.tile_id,error)
								log(f'Tile {tile.tile_id} failed during image reading: {error}')
								progress=store.progress()
								if on_progress is not None:
									on_progress(progress,None)
						for future in as_completed(futures):
							tile=futures[future]
							try:
								summary=future.result()
								store.mark_completed(tile.tile_id,summary.to_dict())
								progress=store.progress()
								if on_progress is not None:
									on_progress(progress,summary)
							except Exception as error:
								store.mark_failed(tile.tile_id,error)
								log(f'Tile {tile.tile_id} failed during threshold processing: {error}')
								progress=store.progress()
								if on_progress is not None:
									on_progress(progress,None)
						if cancelled:
							break
				final_progress=store.progress()
				total_predictions=0
				total_owned=0
				for tile in store.iter_tiles(('completed',)):
					status=store.status(tile.tile_id)
					output_data=status.get('output')or{}
					total_predictions+=int(output_data.get('prediction_count',0))
					total_owned+=int(output_data.get('owned_prediction_count',0))
		finished_at=_utc_now()
		summary=SegmentationRunSummary(
			output_directory=str(output),
			checkpoint_path=str(checkpoint),
			channel_index=channel_index,
			channel_name=channel_name,
			normalization=normalization,
			progress=final_progress,
			predictions=total_predictions,
			owned_predictions=total_owned,
			cancelled=cancelled,
			started_at=started_at,
			finished_at=finished_at,
		)
		(output/'segmentation_summary.json').write_text(
			json.dumps(
				{
					'segmentation_method':'intensity_threshold',
					'output_directory':summary.output_directory,
					'checkpoint_path':summary.checkpoint_path,
					'channel_index':summary.channel_index,
					'channel_name':summary.channel_name,
					'normalization':summary.normalization.to_dict(),
					'progress':asdict(summary.progress),
					'predictions':summary.predictions,
					'owned_predictions':summary.owned_predictions,
					'cancelled':summary.cancelled,
					'started_at':summary.started_at,
					'finished_at':summary.finished_at,
				},
				indent=2,
			),
			encoding='utf-8',
		)
		return summary
