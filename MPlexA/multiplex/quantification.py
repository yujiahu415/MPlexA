from __future__ import annotations
from dataclasses import asdict,dataclass,field
from datetime import datetime,timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from typing import Any,Callable,Iterable,Mapping,Sequence
import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from skimage.segmentation import expand_labels,watershed
from.checkpoints import TileCheckpointStore
from.exceptions import CheckpointMismatchError,MultiplexImageError
from.image_source import open_multiplex_image
from.reconciliation import ChunkedInstanceLabelStore,resolve_global_label_store,ReconciliationError
from.tiling import Bounds,TileGrid
REGION_SCHEMA_VERSION=1
CELL_REGION_LABEL_STORE_NAME='regions.mplexa-labels'
LEGACY_CELL_REGION_LABEL_STORE_NAMES=('regions.cellan-labels',)


def resolve_cell_region_label_store(directory:str|Path)->Path:
	base=Path(directory).expanduser().resolve()
	preferred=base/CELL_REGION_LABEL_STORE_NAME
	if preferred.is_dir():
		return preferred
	for name in LEGACY_CELL_REGION_LABEL_STORE_NAMES:
		candidate=base/name
		if candidate.is_dir():
			return candidate
	return preferred
QUANTIFICATION_SCHEMA_VERSION=1
EXCEL_MAX_ROWS=1_048_576
EXCEL_MAX_COLUMNS=16_384


def _utc_now()->str:
	return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _canonical_json(value:Any)->str:
	return json.dumps(value,sort_keys=True,separators=(',',':'),default=str)


def _path_identity(path:str|Path)->dict[str,Any]:
	item=Path(path).expanduser().resolve()
	try:
		stat=item.stat()
	except OSError as error:
		raise QuantificationError(f'Unable to inspect {item}: {error}')from error
	return{
		'path':str(item),
		'size':int(stat.st_size)if item.is_file()else None,
		'mtime_ns':int(stat.st_mtime_ns),
	}


def _json_atomic(path:Path,payload:Mapping[str,Any])->None:
	path.parent.mkdir(parents=True,exist_ok=True)
	handle=tempfile.NamedTemporaryFile(
		mode='w',suffix='.json',prefix=path.stem+'.',dir=path.parent,
		delete=False,encoding='utf-8'
	)
	temporary=Path(handle.name)
	try:
		json.dump(payload,handle,indent=2)
		handle.flush()
		os.fsync(handle.fileno())
		handle.close()
		os.replace(temporary,path)
	except Exception:
		handle.close()
		temporary.unlink(missing_ok=True)
		raise


def _file_sha256(path:Path,block_size:int=1024*1024)->str:
	digest=hashlib.sha256()
	with path.open('rb')as handle:
		while True:
			block=handle.read(block_size)
			if not block:
				break
			digest.update(block)
	return digest.hexdigest()


def _clipped_halo(bounds:Bounds,halo:int,width:int,height:int)->Bounds:
	x0=max(0,bounds.x-int(halo))
	y0=max(0,bounds.y-int(halo))
	x1=min(int(width),bounds.x1+int(halo))
	y1=min(int(height),bounds.y1+int(halo))
	return Bounds(x0,y0,x1-x0,y1-y0)


def _crop_to_core(array:np.ndarray,read_bounds:Bounds,core_bounds:Bounds)->np.ndarray:
	y0=core_bounds.y-read_bounds.y
	x0=core_bounds.x-read_bounds.x
	return np.asarray(array)[y0:y0+core_bounds.height,x0:x0+core_bounds.width]



class QuantificationError(MultiplexImageError):



class QuantificationCancelled(QuantificationError):



@dataclass(frozen=True,slots=True)
class CellRegionConfig:
	mode:str='nuclear'
	expansion_distance:int=12
	membrane_channel:int|str|None=None
	membrane_sigma:float=1.0
	chunk_size:int|None=None
	retry_failed_chunks:bool=False
	position:Mapping[str,int]|None=None


	def __post_init__(self)->None:
		if self.mode not in{'nuclear','fixed','voronoi','watershed'}:
			raise QuantificationError(
				'Cell-region mode must be nuclear, fixed, voronoi, or watershed.'
			)
		if int(self.expansion_distance)<0:
			raise QuantificationError('Expansion distance cannot be negative.')
		if self.mode!='nuclear'and int(self.expansion_distance)==0:
			raise QuantificationError('Expanded cell-region modes require a positive distance.')
		if self.mode=='watershed'and self.membrane_channel is None:
			raise QuantificationError('Membrane-guided watershed requires a membrane channel.')
		if float(self.membrane_sigma)<0:
			raise QuantificationError('Membrane smoothing sigma cannot be negative.')
		if self.chunk_size is not None and int(self.chunk_size)<=0:
			raise QuantificationError('Region chunk size must be positive.')


	def to_dict(self)->dict[str,Any]:
		data=asdict(self)
		data['position']=dict(self.position or{})
		return data



@dataclass(frozen=True,slots=True)
class CellRegionProgress:
	completed:int
	total:int
	failed:int
	message:str=''


	@property
	def fraction(self)->float:
		return 0.0 if self.total<=0 else min(1.0,max(0.0,self.completed/self.total))



@dataclass(frozen=True,slots=True)
class CellRegionRunSummary:
	output_directory:str
	label_store_path:str
	mode:str
	completed_chunks:int
	failed_chunks:int
	nonzero_pixels:int
	cancelled:bool
	started_at:str
	finished_at:str


	def summary(self)->str:
		return(
			f'Cell-region mode: {self.mode}\n'
			f'Completed chunks: {self.completed_chunks:,}\n'
			f'Failed chunks: {self.failed_chunks:,}\n'
			f'Cell-region pixels: {self.nonzero_pixels:,}\n'
			f'Cancelled: {self.cancelled}\n'
			f'Output: {self.output_directory}'
		)



@dataclass(frozen=True,slots=True)
class QuantificationConfig:
	channels:tuple[int|str,...]|None=None
	channel_batch_size:int=8
	positive_threshold:float=0.0
	positive_thresholds:Mapping[int|str,float]|None=None
	cytoplasmic_ring_width:int=3
	membrane_ring_width:int=2
	export_csv:bool=True
	export_excel:bool=False
	retry_failed_units:bool=False
	position:Mapping[str,int]|None=None


	def __post_init__(self)->None:
		if int(self.channel_batch_size)<=0:
			raise QuantificationError('Channel batch size must be positive.')
		if int(self.cytoplasmic_ring_width)<0:
			raise QuantificationError('Cytoplasmic-ring width cannot be negative.')
		if int(self.membrane_ring_width)<0:
			raise QuantificationError('Membrane-ring width cannot be negative.')
		if not self.export_csv and not self.export_excel:
			raise QuantificationError('Select CSV, Excel, or both output formats.')


	def to_dict(self)->dict[str,Any]:
		return{
			'channels':list(self.channels)if self.channels is not None else None,
			'channel_batch_size':int(self.channel_batch_size),
			'positive_threshold':float(self.positive_threshold),
			'positive_thresholds':{
				str(key):float(value)for key,value in(self.positive_thresholds or{}).items()
			},
			'cytoplasmic_ring_width':int(self.cytoplasmic_ring_width),
			'membrane_ring_width':int(self.membrane_ring_width),
			'export_csv':bool(self.export_csv),
			'export_excel':bool(self.export_excel),
			'retry_failed_units':bool(self.retry_failed_units),
			'position':dict(self.position or{}),
		}



@dataclass(frozen=True,slots=True)
class QuantificationProgress:
	stage:str
	completed:int
	total:int
	failed:int=0
	message:str=''


	@property
	def fraction(self)->float:
		return 0.0 if self.total<=0 else min(1.0,max(0.0,self.completed/self.total))



@dataclass(frozen=True,slots=True)
class QuantificationRunSummary:
	output_directory:str
	database_path:str
	csv_path:str|None
	excel_path:str|None
	cell_count:int
	channel_count:int
	completed_units:int
	failed_units:int
	cancelled:bool
	started_at:str
	finished_at:str


	def summary(self)->str:
		lines=[
			f'Cells: {self.cell_count:,}',
			f'Channels: {self.channel_count:,}',
			f'Completed chunk/channel units: {self.completed_units:,}',
			f'Failed units: {self.failed_units:,}',
			f'Cancelled: {self.cancelled}',
		]
		if self.csv_path:
			lines.append(f'CSV: {self.csv_path}')
		if self.excel_path:
			lines.append(f'Excel: {self.excel_path}')
		lines.append(f'Output: {self.output_directory}')
		return'\n'.join(lines)



class CellRegionGenerator:


	def __init__(self,module3_directory:str|Path)->None:
		self.module3_directory=Path(module3_directory).expanduser().resolve()
		self.nuclear_store_path=resolve_global_label_store(self.module3_directory)
		self.reconciliation_db=self.module3_directory/'reconciliation.sqlite'
		if not self.nuclear_store_path.is_dir():
			raise QuantificationError(
				f'Missing Module 3 nuclear labels: {self.nuclear_store_path}'
			)
		if not self.reconciliation_db.is_file():
			raise QuantificationError(
				f'Missing Module 3 global-cell database: {self.reconciliation_db}'
			)
		self.nuclear_store=ChunkedInstanceLabelStore.open(self.nuclear_store_path)


	def _context(
		self,
		config:CellRegionConfig,
		image_path:str|Path|None,
		series:int,
		membrane_channel_index:int|None,
	)->dict[str,Any]:
		metadata_path=self.nuclear_store_path/ChunkedInstanceLabelStore.METADATA_FILENAME
		context:dict[str,Any]={
			'region_schema_version':REGION_SCHEMA_VERSION,
			'nuclear_labels':{
				'path':str(self.nuclear_store_path),
				'metadata_sha256':_file_sha256(metadata_path),
			},
			'reconciliation_database':_path_identity(self.reconciliation_db),
			'mode':config.mode,
			'expansion_distance':int(config.expansion_distance),
			'membrane_sigma':float(config.membrane_sigma),
			'membrane_channel_index':membrane_channel_index,
			'position':dict(config.position or{}),
			'series':int(series),
		}
		if image_path is not None:
			context['image']=_path_identity(image_path)
		return context


	@staticmethod
	def _emit(
		callback:Callable[[CellRegionProgress],None]|None,
		completed:int,
		total:int,
		failed:int,
		message:str='',
	)->None:
		if callback is not None:
			callback(CellRegionProgress(completed,total,failed,message))


	@staticmethod
	def _voronoi_labels(
		nuclear:np.ndarray,
		read_bounds:Bounds,
		centroids:Sequence[sqlite3.Row],
		max_distance:int,
	)->np.ndarray:
		markers=np.zeros_like(nuclear)
		height,width=markers.shape
		for row in centroids:
			cell_id=int(row['global_cell_id'])
			x=int(round(float(row['centroid_x'])))-read_bounds.x
			y=int(round(float(row['centroid_y'])))-read_bounds.y
			if 0<=x<width and 0<=y<height:
				if markers[y,x]==0:
					markers[y,x]=cell_id
				elif markers[y,x]!=cell_id:
					yy,xx=np.nonzero(nuclear==cell_id)
					if yy.size:
						markers[int(yy[0]),int(xx[0])]=cell_id
		nuclear_ids=np.unique(nuclear[nuclear>0])
		marker_ids=set(int(value)for value in np.unique(markers[markers>0]))
		missing=[int(value)for value in nuclear_ids if int(value)not in marker_ids]
		for cell_id in missing:
			yy,xx=np.nonzero(nuclear==int(cell_id))
			if yy.size:
				markers[int(yy[yy.size//2]),int(xx[xx.size//2])]=int(cell_id)
		if not np.any(markers):
			return nuclear.copy()
		distance,indices=ndi.distance_transform_edt(markers==0,return_indices=True)
		nearest=markers[tuple(indices)]
		result=np.where(distance<=int(max_distance),nearest,0).astype(nuclear.dtype,copy=False)
		result[nuclear>0]=nuclear[nuclear>0]
		return result


	@staticmethod
	def _watershed_labels(
		nuclear:np.ndarray,
		membrane:np.ndarray,
		max_distance:int,
		sigma:float,
	)->np.ndarray:
		if not np.any(nuclear):
			return nuclear.copy()
		allowed=expand_labels(nuclear,distance=int(max_distance))>0
		image=np.asarray(membrane,dtype=np.float32)
		finite=np.isfinite(image)
		if np.any(finite):
			low,high=np.percentile(image[finite],[1.0,99.0])
			if high>low:
				image=np.clip((image-low)/(high-low),0,1)
			else:
				image=np.zeros_like(image)
		else:
			image=np.zeros_like(image)
		if sigma>0:
			image=ndi.gaussian_filter(image,float(sigma))
		elevation=filters.sobel(image)
		result=watershed(elevation,markers=nuclear,mask=allowed)
		return np.asarray(result,dtype=nuclear.dtype)


	def run(
		self,
		*,
		output_directory:str|Path|None=None,
		config:CellRegionConfig|None=None,
		image_path:str|Path|None=None,
		series:int=0,
		cancel_event:threading.Event|None=None,
		on_progress:Callable[[CellRegionProgress],None]|None=None,
		on_log:Callable[[str],None]|None=None,
	)->CellRegionRunSummary:
		config=config or CellRegionConfig()
		cancel_event=cancel_event or threading.Event()
		started_at=_utc_now()
		destination=(
			Path(output_directory).expanduser().resolve()
			if output_directory is not None
			else self.module3_directory/'cell_regions'
		)
		destination.mkdir(parents=True,exist_ok=True)
		output_store_path=destination/CELL_REGION_LABEL_STORE_NAME
		chunk_size=int(config.chunk_size or self.nuclear_store.metadata.chunk_width)
		output_store=ChunkedInstanceLabelStore.create(
			output_store_path,
			width=self.nuclear_store.metadata.width,
			height=self.nuclear_store.metadata.height,
			chunk_size=chunk_size,
			dtype=self.nuclear_store.dtype,
			level=self.nuclear_store.metadata.level,
			global_cell_count=self.nuclear_store.metadata.global_cell_count,
		)
		membrane_index:int|None=None
		image=None
		if config.mode=='watershed':
			if image_path is None:
				raise QuantificationError('Membrane-guided watershed requires the source multiplex image.')
			image=open_multiplex_image(image_path,series=series)
			membrane_index=image.channel_index(config.membrane_channel)# type: ignore[arg-type]
			level_meta=image.metadata.levels[self.nuclear_store.metadata.level]
			level_width=int(level_meta.shape[level_meta.axes.index('X')])
			level_height=int(level_meta.shape[level_meta.axes.index('Y')])
			if level_width!=self.nuclear_store.metadata.width or level_height!=self.nuclear_store.metadata.height:
				image.close()
				raise QuantificationError(
					'The source image dimensions at the selected level do not match Module 3 labels.'
				)
		context=self._context(config,image_path,series,membrane_index)
		config_path=destination/'region_config.json'
		grid=output_store.grid()
		checkpoint_path=destination/'region_chunks.sqlite'
		cancelled=False
		total_nonzero=0
		centroid_connection:sqlite3.Connection|None=None
		if config.mode=='voronoi':
			centroid_connection=sqlite3.connect(self.reconciliation_db)
			centroid_connection.row_factory=sqlite3.Row
		try:
			with TileCheckpointStore(
				checkpoint_path,
				grid,
				job_name='MPlexA cell-region generation',
				context=context,
				reset_interrupted=True,
			)as checkpoint:
				_json_atomic(config_path,{
					'created_at':started_at,
					'module3_directory':str(self.module3_directory),
					'nuclear_label_store':str(self.nuclear_store_path),
					'cell_label_store':str(output_store_path),
					'image_path':str(Path(image_path).expanduser().resolve())if image_path is not None else None,
					'series':int(series),
					'level':int(self.nuclear_store.metadata.level),
					'membrane_channel_index':membrane_index,
					'config':config.to_dict(),
					'context':context,
				})
				if config.retry_failed_chunks:
					checkpoint.reset_failed()
				for tile in checkpoint.iter_tiles(('completed',)):
					status=checkpoint.status(tile.tile_id)
					total_nonzero+=int((status.get('output')or{}).get('nonzero_pixels',0))
				initial=checkpoint.progress()
				self._emit(on_progress,initial.completed,initial.total,initial.failed)
				while True:
					if cancel_event.is_set():
						cancelled=True
						break
					tile=checkpoint.claim_next()
					if tile is None:
						break
					core=output_store.chunk_bounds(tile.row,tile.column)
					halo=0 if config.mode=='nuclear'else int(config.expansion_distance)
					if config.mode=='watershed':
						halo+=int(math.ceil(3*float(config.membrane_sigma)))+2
					read=_clipped_halo(
						core,halo,self.nuclear_store.metadata.width,self.nuclear_store.metadata.height
					)
					try:
						nuclear=self.nuclear_store.read_region(
							x=read.x,y=read.y,width=read.width,height=read.height
						)
						if config.mode=='nuclear':
							generated=nuclear
						elif config.mode=='fixed':
							generated=expand_labels(
								nuclear,distance=int(config.expansion_distance)
							).astype(nuclear.dtype,copy=False)
						elif config.mode=='voronoi':
							assert centroid_connection is not None
							rows=centroid_connection.execute(
								'SELECT c.global_cell_id, c.centroid_x, c.centroid_y '
								'FROM global_cells_rtree r JOIN global_cells c USING(global_cell_id) '
								'WHERE r.min_x<? AND r.max_x>? AND r.min_y<? AND r.max_y>?',
								(read.x1,read.x,read.y1,read.y),
							).fetchall()
							generated=self._voronoi_labels(
								nuclear,read,rows,int(config.expansion_distance)
							)
						else:
							assert image is not None and membrane_index is not None
							membrane=image.read_region(
								x=read.x,y=read.y,width=read.width,height=read.height,
								channels=membrane_index,level=self.nuclear_store.metadata.level,
								position=config.position,
							)[0]
							generated=self._watershed_labels(
								nuclear,membrane,int(config.expansion_distance),
								float(config.membrane_sigma),
							)
						core_array=_crop_to_core(generated,read,core)
						output_store.write_chunk(tile.row,tile.column,core_array)
						nonzero=int(np.count_nonzero(core_array))
						checkpoint.mark_completed(tile.tile_id,{
							'chunk_path':str(output_store.chunk_path(tile.row,tile.column)),
							'nonzero_pixels':nonzero,
						})
						total_nonzero+=nonzero
					except Exception as error:
						checkpoint.mark_failed(tile.tile_id,error)
					progress=checkpoint.progress()
					self._emit(
						on_progress,progress.completed,progress.total,progress.failed,
						f'{config.mode} regions',
					)
				final=checkpoint.progress()
				if final.failed and not cancelled:
					raise QuantificationError(
						f'Cell-region generation has {final.failed:,} failed chunk(s). '
						'Enable retry failed chunks and resume.'
					)
		finally:
			if image is not None:
				image.close()
			if centroid_connection is not None:
				centroid_connection.close()
		summary=CellRegionRunSummary(
			output_directory=str(destination),
			label_store_path=str(output_store_path),
			mode=config.mode,
			completed_chunks=final.completed,
			failed_chunks=final.failed,
			nonzero_pixels=total_nonzero,
			cancelled=cancelled,
			started_at=started_at,
			finished_at=_utc_now(),
		)
		_json_atomic(destination/'region_summary.json',asdict(summary))
		return summary


def _inner_label_boundary(labels:np.ndarray)->np.ndarray:
	data=np.asarray(labels)
	padded=np.pad(data,1,mode='constant',constant_values=0)
	center=padded[1:-1,1:-1]
	boundary=np.zeros_like(center,dtype=bool)
	for neighbor in(
		padded[:-2,1:-1],padded[2:,1:-1],
		padded[1:-1,:-2],padded[1:-1,2:],
	):
		boundary|=(center>0)&(neighbor!=center)
	return boundary


def build_compartment_labels(
	nuclear_labels:np.ndarray,
	cell_labels:np.ndarray,
	*,
	cytoplasmic_ring_width:int=3,
	membrane_ring_width:int=2,
)->dict[str,np.ndarray]:
	nuclei=np.asarray(nuclear_labels)
	cells=np.asarray(cell_labels)
	if nuclei.shape!=cells.shape:
		raise QuantificationError('Nuclear and cell label arrays must have the same shape.')
	if nuclei.dtype.kind not in'ui'or cells.dtype.kind not in'ui':
		raise QuantificationError('Compartment inputs must be integer instance-label arrays.')
	nuclear=np.where((nuclei>0)&(cells==nuclei),nuclei,0).astype(cells.dtype,copy=False)
	cytoplasm=np.where((cells>0)&(nuclear==0),cells,0).astype(cells.dtype,copy=False)
	cytoplasmic_ring=np.zeros_like(cells)
	if int(cytoplasmic_ring_width)>0 and np.any(nuclear):
		distance,indices=ndi.distance_transform_edt(nuclear==0,return_indices=True)
		nearest_nucleus=nuclear[tuple(indices)]
		ring=(
			(cells>0)&(nuclear==0)&(nearest_nucleus==cells)
			&(distance<=int(cytoplasmic_ring_width))
		)
		cytoplasmic_ring[ring]=cells[ring]
	membrane_ring=np.zeros_like(cells)
	if int(membrane_ring_width)>0 and np.any(cells):
		boundary=_inner_label_boundary(cells)
		distance=ndi.distance_transform_edt(~boundary)
		ring=(cells>0)&(distance<=int(membrane_ring_width))
		membrane_ring[ring]=cells[ring]
	return{
		'cell':cells,
		'nuclear':nuclear,
		'cytoplasm':cytoplasm,
		'cytoplasmic_ring':cytoplasmic_ring,
		'membrane_ring':membrane_ring,
	}


def _aggregate_values(
	labels:np.ndarray,
	values:np.ndarray,
	threshold:float,
)->dict[int,tuple[int,float,float,float,float,int]]:
	label_flat=np.asarray(labels).ravel()
	value_flat=np.asarray(values,dtype=np.float64).ravel()
	selected=label_flat>0
	if not np.any(selected):
		return{}
	ids,inverse=np.unique(label_flat[selected],return_inverse=True)
	data=value_flat[selected]
	count=np.bincount(inverse)
	total=np.bincount(inverse,weights=data)
	total_sq=np.bincount(inverse,weights=data*data)
	positive=np.bincount(inverse,weights=(data>float(threshold)).astype(np.float64))
	minimum=np.full(ids.size,np.inf)
	maximum=np.full(ids.size,-np.inf)
	np.minimum.at(minimum,inverse,data)
	np.maximum.at(maximum,inverse,data)
	return{
		int(cell_id):(
			int(count[index]),float(total[index]),float(total_sq[index]),
			float(minimum[index]),float(maximum[index]),int(positive[index]),
		)
		for index,cell_id in enumerate(ids)
	}


def _aggregate_sum_count(labels:np.ndarray,values:np.ndarray)->dict[int,tuple[int,float]]:
	label_flat=np.asarray(labels).ravel()
	value_flat=np.asarray(values,dtype=np.float64).ravel()
	selected=label_flat>0
	if not np.any(selected):
		return{}
	ids,inverse=np.unique(label_flat[selected],return_inverse=True)
	count=np.bincount(inverse)
	total=np.bincount(inverse,weights=value_flat[selected])
	return{
		int(cell_id):(int(count[index]),float(total[index]))
		for index,cell_id in enumerate(ids)
	}


def _safe_channel_labels(names:Sequence[str],indices:Sequence[int])->list[str]:
	seen:dict[str,int]={}
	result:list[str]=[]
	for index,name in zip(indices,names):
		clean=' '.join(str(name).replace('\n',' ').replace('\r',' ').split())or f'Channel {index}'
		occurrence=seen.get(clean.casefold(),0)+1
		seen[clean.casefold()]=occurrence
		if occurrence>1:
			clean= f'{clean} [{index}]'
		result.append(clean)
	return result



class QuantificationDatabase:


	def __init__(self,path:str|Path,context:Mapping[str,Any])->None:
		self.path=Path(path).expanduser().resolve()
		self.path.parent.mkdir(parents=True,exist_ok=True)
		self.connection=sqlite3.connect(self.path,timeout=120.0)
		self.connection.row_factory=sqlite3.Row
		self.connection.execute('PRAGMA journal_mode=WAL')
		self.connection.execute('PRAGMA synchronous=NORMAL')
		self._create_schema()
		self._validate_context(context)


	def _create_schema(self)->None:
		with self.connection:
			self.connection.execute(
				'CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS channels (
                    channel_order INTEGER PRIMARY KEY,
                    channel_index INTEGER NOT NULL,
                    channel_name TEXT NOT NULL,
                    output_name TEXT NOT NULL,
                    positive_threshold REAL NOT NULL
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS units (
                    unit_id TEXT PRIMARY KEY,
                    row_index INTEGER NOT NULL,
                    column_index INTEGER NOT NULL,
                    channel_start INTEGER NOT NULL,
                    channel_end INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )
                '''
			)
			self.connection.execute(
				'CREATE INDEX IF NOT EXISTS idx_units_status ON units(status, row_index, column_index, channel_start)'
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS measurements (
                    global_cell_id INTEGER NOT NULL,
                    channel_order INTEGER NOT NULL,
                    cell_count INTEGER NOT NULL,
                    cell_sum REAL NOT NULL,
                    cell_sumsq REAL NOT NULL,
                    cell_min REAL NOT NULL,
                    cell_max REAL NOT NULL,
                    positive_count INTEGER NOT NULL,
                    nuclear_count INTEGER NOT NULL,
                    nuclear_sum REAL NOT NULL,
                    cytoplasm_count INTEGER NOT NULL,
                    cytoplasm_sum REAL NOT NULL,
                    cytoplasmic_ring_count INTEGER NOT NULL,
                    cytoplasmic_ring_sum REAL NOT NULL,
                    membrane_ring_count INTEGER NOT NULL,
                    membrane_ring_sum REAL NOT NULL,
                    PRIMARY KEY(global_cell_id, channel_order)
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS cell_areas (
                    global_cell_id INTEGER PRIMARY KEY,
                    cell_area INTEGER NOT NULL,
                    nuclear_area INTEGER NOT NULL,
                    cytoplasm_area INTEGER NOT NULL,
                    cytoplasmic_ring_area INTEGER NOT NULL,
                    membrane_ring_area INTEGER NOT NULL
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS channel_backgrounds (
                    channel_order INTEGER PRIMARY KEY,
                    background_count INTEGER NOT NULL,
                    background_sum REAL NOT NULL
                )
                '''
			)


	def _validate_context(self,context:Mapping[str,Any])->None:
		expected={
			'schema_version':str(QUANTIFICATION_SCHEMA_VERSION),
			'context':_canonical_json(context),
		}
		existing={
			str(row['key']):str(row['value'])
			for row in self.connection.execute('SELECT key, value FROM metadata')
		}
		if existing:
			if existing.get('schema_version')!=expected['schema_version']:
				raise CheckpointMismatchError('Quantification database schema is incompatible.')
			if existing.get('context')!=expected['context']:
				raise CheckpointMismatchError(
					'The existing quantification output uses different image, labels, channels, thresholds, or ring settings.'
				)
			return
		with self.connection:
			self.connection.executemany(
				'INSERT INTO metadata(key, value) VALUES (?, ?)',
				[
					('schema_version',expected['schema_version']),
					('context',expected['context']),
					('created_at',_utc_now()),
				],
			)


	def initialize_channels(
		self,
		indices:Sequence[int],
		names:Sequence[str],
		output_names:Sequence[str],
		thresholds:Sequence[float],
	)->None:
		with self.connection:
			for order,(index,name,output,threshold)in enumerate(
				zip(indices,names,output_names,thresholds)
			):
				self.connection.execute(
					'INSERT OR IGNORE INTO channels(channel_order, channel_index, channel_name, output_name, positive_threshold) '
					'VALUES (?, ?, ?, ?, ?)',
					(order,int(index),str(name),str(output),float(threshold)),
				)


	def initialize_units(self,grid:TileGrid,channel_count:int,batch_size:int)->None:
		rows=[]
		now=_utc_now()
		for tile in grid:
			for start in range(0,int(channel_count),int(batch_size)):
				end=min(int(channel_count),start+int(batch_size))
				unit_id= f'{tile.tile_id}_C{start:05d}_{end:05d}'
				rows.append((unit_id,tile.row,tile.column,start,end,now))
		with self.connection:
			self.connection.executemany(
				'INSERT OR IGNORE INTO units(unit_id, row_index, column_index, channel_start, channel_end, updated_at) '
				'VALUES (?, ?, ?, ?, ?, ?)',rows,
			)


	def reset_interrupted(self)->None:
		with self.connection:
			self.connection.execute(
				'UPDATE units SET status=\'pending\', error=NULL, updated_at=? WHERE status=\'running\'',
				(_utc_now(),),
			)


	def reset_failed(self)->int:
		with self.connection:
			cursor=self.connection.execute(
				'UPDATE units SET status=\'pending\', error=NULL, updated_at=? WHERE status=\'failed\'',
				(_utc_now(),),
			)
		return int(cursor.rowcount)


	def claim_next(self)->sqlite3.Row|None:
		with self.connection:
			row=self.connection.execute(
				'SELECT * FROM units WHERE status=\'pending\' '
				'ORDER BY row_index, column_index, channel_start LIMIT 1'
			).fetchone()
			if row is None:
				return None
			updated=self.connection.execute(
				'UPDATE units SET status=\'running\', attempts=attempts+1, updated_at=? '
				'WHERE unit_id=? AND status=\'pending\'',
				(_utc_now(),row['unit_id']),
			)
			if updated.rowcount!=1:
				return None
			return self.connection.execute(
				'SELECT * FROM units WHERE unit_id=?',(row['unit_id'],)
			).fetchone()


	def mark_failed(self,unit_id:str,error:Exception|str)->None:
		with self.connection:
			self.connection.execute(
				'UPDATE units SET status=\'failed\', error=?, updated_at=? WHERE unit_id=?',
				(str(error),_utc_now(),unit_id),
			)


	def progress(self)->tuple[int,int,int,int]:
		counts={
			str(row[0]):int(row[1])
			for row in self.connection.execute('SELECT status, COUNT(*) FROM units GROUP BY status')
		}
		total=sum(counts.values())
		return counts.get('completed',0),total,counts.get('failed',0),counts.get('pending',0)


	@staticmethod
	def _upsert_measurement(
		connection:sqlite3.Connection,
		cell_id:int,
		channel_order:int,
		whole:tuple[int,float,float,float,float,int],
		nuclear:tuple[int,float],
		cytoplasm:tuple[int,float],
		cytoplasmic_ring:tuple[int,float],
		membrane_ring:tuple[int,float],
	)->None:
		connection.execute(
			'''
            INSERT INTO measurements(
                global_cell_id, channel_order, cell_count, cell_sum, cell_sumsq,
                cell_min, cell_max, positive_count, nuclear_count, nuclear_sum,
                cytoplasm_count, cytoplasm_sum, cytoplasmic_ring_count,
                cytoplasmic_ring_sum, membrane_ring_count, membrane_ring_sum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(global_cell_id, channel_order) DO UPDATE SET
                cell_count=cell_count+excluded.cell_count,
                cell_sum=cell_sum+excluded.cell_sum,
                cell_sumsq=cell_sumsq+excluded.cell_sumsq,
                cell_min=MIN(cell_min, excluded.cell_min),
                cell_max=MAX(cell_max, excluded.cell_max),
                positive_count=positive_count+excluded.positive_count,
                nuclear_count=nuclear_count+excluded.nuclear_count,
                nuclear_sum=nuclear_sum+excluded.nuclear_sum,
                cytoplasm_count=cytoplasm_count+excluded.cytoplasm_count,
                cytoplasm_sum=cytoplasm_sum+excluded.cytoplasm_sum,
                cytoplasmic_ring_count=cytoplasmic_ring_count+excluded.cytoplasmic_ring_count,
                cytoplasmic_ring_sum=cytoplasmic_ring_sum+excluded.cytoplasmic_ring_sum,
                membrane_ring_count=membrane_ring_count+excluded.membrane_ring_count,
                membrane_ring_sum=membrane_ring_sum+excluded.membrane_ring_sum
            ''',
			(
				int(cell_id),int(channel_order),*whole,
				int(nuclear[0]),float(nuclear[1]),
				int(cytoplasm[0]),float(cytoplasm[1]),
				int(cytoplasmic_ring[0]),float(cytoplasmic_ring[1]),
				int(membrane_ring[0]),float(membrane_ring[1]),
			),
		)


	def commit_unit(
		self,
		unit_id:str,
		channel_orders:Sequence[int],
		thresholds:Sequence[float],
		values:np.ndarray,
		compartments:Mapping[str,np.ndarray],
		*,
		update_areas:bool,
	)->None:
		cells=compartments['cell']
		with self.connection:
			if update_areas:
				area_maps={
					name:_aggregate_sum_count(labels,np.ones(labels.shape,dtype=np.float32))
					for name,labels in compartments.items()
				}
				all_ids=set().union(*(mapping.keys()for mapping in area_maps.values()))
				for cell_id in all_ids:
					counts=[int(area_maps[name].get(cell_id,(0,0))[0])for name in(
						'cell','nuclear','cytoplasm','cytoplasmic_ring','membrane_ring'
					)]
					self.connection.execute(
						'''
                        INSERT INTO cell_areas(global_cell_id, cell_area, nuclear_area, cytoplasm_area,
                            cytoplasmic_ring_area, membrane_ring_area)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(global_cell_id) DO UPDATE SET
                            cell_area=cell_area+excluded.cell_area,
                            nuclear_area=nuclear_area+excluded.nuclear_area,
                            cytoplasm_area=cytoplasm_area+excluded.cytoplasm_area,
                            cytoplasmic_ring_area=cytoplasmic_ring_area+excluded.cytoplasmic_ring_area,
                            membrane_ring_area=membrane_ring_area+excluded.membrane_ring_area
                        ''',
						(int(cell_id),*counts),
					)
			background_mask=cells==0
			for local_index,channel_order in enumerate(channel_orders):
				channel_values=np.asarray(values[local_index])
				threshold=float(thresholds[local_index])
				whole=_aggregate_values(cells,channel_values,threshold)
				nuclear=_aggregate_sum_count(compartments['nuclear'],channel_values)
				cytoplasm=_aggregate_sum_count(compartments['cytoplasm'],channel_values)
				cytoplasmic_ring=_aggregate_sum_count(compartments['cytoplasmic_ring'],channel_values)
				membrane_ring=_aggregate_sum_count(compartments['membrane_ring'],channel_values)
				all_ids=set(whole)
				for cell_id in all_ids:
					self._upsert_measurement(
						self.connection,cell_id,int(channel_order),whole[cell_id],
						nuclear.get(cell_id,(0,0.0)),
						cytoplasm.get(cell_id,(0,0.0)),
						cytoplasmic_ring.get(cell_id,(0,0.0)),
						membrane_ring.get(cell_id,(0,0.0)),
					)
				background_values=channel_values[background_mask]
				background_count=int(background_values.size)
				background_sum=float(np.sum(background_values,dtype=np.float64))
				self.connection.execute(
					'''
                    INSERT INTO channel_backgrounds(channel_order, background_count, background_sum)
                    VALUES (?, ?, ?)
                    ON CONFLICT(channel_order) DO UPDATE SET
                        background_count=background_count+excluded.background_count,
                        background_sum=background_sum+excluded.background_sum
                    ''',
					(int(channel_order),background_count,background_sum),
				)
			self.connection.execute(
				'UPDATE units SET status=\'completed\', error=NULL, updated_at=? WHERE unit_id=?',
				(_utc_now(),unit_id),
			)


	def close(self)->None:
		self.connection.close()


	def __enter__(self)->'QuantificationDatabase':
		return self


	def __exit__(self,exc_type:Any,exc_value:Any,traceback:Any)->None:
		self.close()



class MarkerQuantifier:
	BASE_FIELDS=[
		'global_cell_id','class_id','class_name','score','centroid_x','centroid_y',
		'x0','y0','x1','y1','source_count','touches_image_edge',
		'cell_area','nuclear_area','cytoplasm_area','cytoplasmic_ring_area','membrane_ring_area',
	]
	CHANNEL_METRICS=[
		'mean','sum','maximum','minimum','standard_deviation',
		'positive_pixel_fraction','background_corrected_mean','nuclear_mean',
		'cytoplasmic_mean','cytoplasmic_ring_mean','membrane_ring_mean',
	]


	def __init__(
		self,
		image_path:str|Path,
		region_directory:str|Path,
		*,
		series:int=0,
	)->None:
		self.image_path=Path(image_path).expanduser().resolve()
		self.region_directory=Path(region_directory).expanduser().resolve()
		self.series=int(series)
		config_path=self.region_directory/'region_config.json'
		if not config_path.is_file():
			raise QuantificationError(f'Missing Module 4 region configuration: {config_path}')
		try:
			self.region_config=json.loads(config_path.read_text(encoding='utf-8'))
		except(OSError,ValueError,json.JSONDecodeError)as error:
			raise QuantificationError(f'Unable to read {config_path}: {error}')from error
		self.cell_store_path=Path(self.region_config['cell_label_store']).expanduser().resolve()
		self.nuclear_store_path=Path(self.region_config['nuclear_label_store']).expanduser().resolve()
		module3_value=self.region_config.get('module3_directory')or self.region_config.get('module4_directory')
		if module3_value is None:
			raise QuantificationError('Region configuration does not identify the Module 3 output directory.')
		self.module3_directory=Path(module3_value).expanduser().resolve()
		self.reconciliation_db=self.module3_directory/'reconciliation.sqlite'
		self.cell_store=ChunkedInstanceLabelStore.open(self.cell_store_path)
		self.nuclear_store=ChunkedInstanceLabelStore.open(self.nuclear_store_path)
		if self.cell_store.metadata.width!=self.nuclear_store.metadata.width or self.cell_store.metadata.height!=self.nuclear_store.metadata.height:
			raise QuantificationError('Cell and nuclear label stores have different dimensions.')
		if not self.reconciliation_db.is_file():
			raise QuantificationError(f'Missing Module 3 database: {self.reconciliation_db}')


	@staticmethod
	def _thresholds(
		image:Any,
		indices:Sequence[int],
		config:QuantificationConfig,
	)->list[float]:
		overrides=config.positive_thresholds or{}
		result=[]
		for index in indices:
			name=image.channel_names[index]
			if index in overrides:
				result.append(float(overrides[index]))
			elif str(index)in overrides:
				result.append(float(overrides[str(index)]))
			elif name in overrides:
				result.append(float(overrides[name]))
			else:
				result.append(float(config.positive_threshold))
		return result


	def _context(
		self,
		config:QuantificationConfig,
		channel_indices:Sequence[int],
		thresholds:Sequence[float],
	)->dict[str,Any]:
		return{
			'quantification_schema_version':QUANTIFICATION_SCHEMA_VERSION,
			'image':_path_identity(self.image_path),
			'series':self.series,
			'level':int(self.cell_store.metadata.level),
			'cell_labels_metadata':_file_sha256(
				self.cell_store_path/ChunkedInstanceLabelStore.METADATA_FILENAME
			),
			'nuclear_labels_metadata':_file_sha256(
				self.nuclear_store_path/ChunkedInstanceLabelStore.METADATA_FILENAME
			),
			'region_context':self.region_config.get('context',{}),
			'channels':list(map(int,channel_indices)),
			'thresholds':list(map(float,thresholds)),
			'cytoplasmic_ring_width':int(config.cytoplasmic_ring_width),
			'membrane_ring_width':int(config.membrane_ring_width),
			'position':dict(config.position or{}),
			'channel_batch_size':int(config.channel_batch_size),
		}


	@staticmethod
	def _emit(
		callback:Callable[[QuantificationProgress],None]|None,
		stage:str,
		completed:int,
		total:int,
		failed:int=0,
		message:str='',
	)->None:
		if callback is not None:
			callback(QuantificationProgress(stage,completed,total,failed,message))


	@staticmethod
	def _measurement_values(row:sqlite3.Row|None,background:float)->list[Any]:
		if row is None or int(row['cell_count'])<=0:
			return['']*len(MarkerQuantifier.CHANNEL_METRICS)
		count=int(row['cell_count'])
		total=float(row['cell_sum'])
		mean=total/count
		variance=max(0.0,float(row['cell_sumsq'])/count-mean*mean)
		std=math.sqrt(variance)


		def compartment(prefix:str)->float|str:
			amount=int(row[f'{prefix}_count'])
			return float(row[f'{prefix}_sum'])/amount if amount else''
		return[
			mean,
			total,
			float(row['cell_max']),
			float(row['cell_min']),
			std,
			int(row['positive_count'])/count,
			mean-background if math.isfinite(background)else'',
			compartment('nuclear'),
			compartment('cytoplasm'),
			compartment('cytoplasmic_ring'),
			compartment('membrane_ring'),
		]


	def _export_rows(
		self,
		quant_db:sqlite3.Connection,
	)->tuple[list[str],Iterable[list[Any]],int]:
		channels=quant_db.execute(
			'SELECT * FROM channels ORDER BY channel_order'
		).fetchall()
		backgrounds={
			int(row['channel_order']):(
				float(row['background_sum'])/int(row['background_count'])
				if int(row['background_count'])else float('nan')
			)
			for row in quant_db.execute('SELECT * FROM channel_backgrounds')
		}
		headers=list(self.BASE_FIELDS)
		for channel in channels:
			prefix=str(channel['output_name'])
			headers.extend(f'{prefix}__{metric}' for metric in self.CHANNEL_METRICS)
		global_connection=sqlite3.connect(self.reconciliation_db)
		global_connection.row_factory=sqlite3.Row
		cell_count=int(global_connection.execute('SELECT COUNT(*) FROM global_cells').fetchone()[0])


		def iterator()->Iterable[list[Any]]:
			try:
				measurement_cursor=iter(quant_db.execute(
					'SELECT * FROM measurements ORDER BY global_cell_id, channel_order'
				))
				current=next(measurement_cursor,None)
				area_cursor=iter(quant_db.execute(
					'SELECT * FROM cell_areas ORDER BY global_cell_id'
				))
				current_area=next(area_cursor,None)
				for cell in global_connection.execute(
					'SELECT * FROM global_cells ORDER BY global_cell_id'
				):
					cell_id=int(cell['global_cell_id'])
					while current_area is not None and int(current_area['global_cell_id'])<cell_id:
						current_area=next(area_cursor,None)
					area=current_area if current_area is not None and int(current_area['global_cell_id'])==cell_id else None
					if area is not None:
						current_area=next(area_cursor,None)
					row_values:list[Any]=[
						cell_id,int(cell['class_id']),str(cell['class_name']),float(cell['score']),
						float(cell['centroid_x']),float(cell['centroid_y']),
						int(cell['x0']),int(cell['y0']),int(cell['x1']),int(cell['y1']),
						int(cell['source_count']),int(cell['touches_image_edge']),
					]
					if area is None:
						row_values.extend([0,0,0,0,0])
					else:
						row_values.extend([
							int(area['cell_area']),int(area['nuclear_area']),
							int(area['cytoplasm_area']),int(area['cytoplasmic_ring_area']),
							int(area['membrane_ring_area']),
						])
					rows_by_channel:dict[int,sqlite3.Row]={}
					while current is not None and int(current['global_cell_id'])<cell_id:
						current=next(measurement_cursor,None)
					while current is not None and int(current['global_cell_id'])==cell_id:
						rows_by_channel[int(current['channel_order'])]=current
						current=next(measurement_cursor,None)
					for channel in channels:
						order=int(channel['channel_order'])
						row_values.extend(self._measurement_values(
							rows_by_channel.get(order),backgrounds.get(order,float('nan'))
						))
					yield row_values
			finally:
				global_connection.close()
		return headers,iterator(),cell_count


	def _export_csv(self,path:Path,connection:sqlite3.Connection)->int:
		headers,rows,cell_count=self._export_rows(connection)
		temporary=path.with_suffix(path.suffix+'.tmp')
		with temporary.open('w',newline='',encoding='utf-8-sig')as handle:
			writer=csv.writer(handle)
			writer.writerow(headers)
			writer.writerows(rows)
		os.replace(temporary,path)
		return cell_count


	def _export_background_csv(self,path:Path,connection:sqlite3.Connection)->None:
		temporary=path.with_suffix(path.suffix+'.tmp')
		with temporary.open('w',newline='',encoding='utf-8-sig')as handle:
			writer=csv.writer(handle)
			writer.writerow([
				'channel_index','channel_name','positive_threshold',
				'background_mean','background_pixels',
			])
			for row in connection.execute(
				'''
                SELECT c.channel_index, c.channel_name, c.positive_threshold,
                    b.background_sum, b.background_count
                FROM channels c LEFT JOIN channel_backgrounds b USING(channel_order)
                ORDER BY c.channel_order
                '''
			):
				count=int(row['background_count']or 0)
				mean=float(row['background_sum']or 0)/count if count else''
				writer.writerow([
					int(row['channel_index']),str(row['channel_name']),
					float(row['positive_threshold']),mean,count,
				])
		os.replace(temporary,path)


	def _export_excel(self,path:Path,connection:sqlite3.Connection)->int:
		try:
			import xlsxwriter
		except ImportError as error:
			raise QuantificationError('Excel export requires xlsxwriter.')from error
		headers,rows,cell_count=self._export_rows(connection)
		if len(headers)>EXCEL_MAX_COLUMNS:
			raise QuantificationError(
				f'The wide result has {len(headers):,} columns, exceeding Excel\'s {EXCEL_MAX_COLUMNS:,}-column limit. '
				'Export CSV only or select fewer channels.'
			)
		temporary=path.with_name(path.stem+'.tmp'+path.suffix)
		workbook=xlsxwriter.Workbook(str(temporary),{'constant_memory':True})
		try:
			worksheet=None
			row_index=EXCEL_MAX_ROWS
			sheet_index=0
			for values in rows:
				if row_index>=EXCEL_MAX_ROWS:
					sheet_index+=1
					worksheet=workbook.add_worksheet(f'Cells_{sheet_index:03d}')
					for column,header in enumerate(headers):
						worksheet.write(0,column,header)
					worksheet.freeze_panes(1,0)
					row_index=1
				assert worksheet is not None
				for column,value in enumerate(values):
					if value==''or value is None:
						worksheet.write_blank(row_index,column,None)
					elif isinstance(value,(int,float,np.integer,np.floating)):
						worksheet.write_number(row_index,column,float(value))
					else:
						worksheet.write(row_index,column,value)
				row_index+=1
			backgrounds=workbook.add_worksheet('Channel_backgrounds')
			backgrounds.write_row(0,0,['channel_index','channel_name','positive_threshold','background_mean','background_pixels'])
			for row_number,row in enumerate(connection.execute(
				'''
                SELECT c.channel_index, c.channel_name, c.positive_threshold,
                    b.background_sum, b.background_count
                FROM channels c LEFT JOIN channel_backgrounds b USING(channel_order)
                ORDER BY c.channel_order
                '''
			),start=1):
				count=int(row['background_count']or 0)
				mean=float(row['background_sum']or 0)/count if count else None
				backgrounds.write_row(row_number,0,[
					int(row['channel_index']),str(row['channel_name']),
					float(row['positive_threshold']),mean,count,
				])
		finally:
			workbook.close()
		os.replace(temporary,path)
		return cell_count


	def run(
		self,
		*,
		output_directory:str|Path|None=None,
		config:QuantificationConfig|None=None,
		cancel_event:threading.Event|None=None,
		on_progress:Callable[[QuantificationProgress],None]|None=None,
		on_log:Callable[[str],None]|None=None,
	)->QuantificationRunSummary:
		config=config or QuantificationConfig()
		cancel_event=cancel_event or threading.Event()
		started_at=_utc_now()
		destination=(
			Path(output_directory).expanduser().resolve()
			if output_directory is not None
			else self.region_directory/'marker_quantification'
		)
		destination.mkdir(parents=True,exist_ok=True)
		database_path=destination/'quantification.sqlite'
		csv_path=destination/'cell_marker_measurements.csv'if config.export_csv else None
		excel_path=destination/'cell_marker_measurements.xlsx'if config.export_excel else None
		cancelled=False


		def log(message:str)->None:
			if on_log is not None:
				on_log(message)
		with open_multiplex_image(self.image_path,series=self.series)as image:
			level=int(self.cell_store.metadata.level)
			level_meta=image.metadata.levels[level]
			level_width=int(level_meta.shape[level_meta.axes.index('X')])
			level_height=int(level_meta.shape[level_meta.axes.index('Y')])
			if level_width!=self.cell_store.metadata.width or level_height!=self.cell_store.metadata.height:
				raise QuantificationError(
					'The source image dimensions at the selected level do not match the generated cell regions.'
				)
			if config.channels is None:
				channel_indices=list(range(image.metadata.channel_count))
			else:
				channel_indices=[image.channel_index(channel)for channel in config.channels]
			if not channel_indices:
				raise QuantificationError('Select at least one channel for marker quantification.')
			channel_names=[image.channel_names[index]for index in channel_indices]
			output_names=_safe_channel_labels(channel_names,channel_indices)
			thresholds=self._thresholds(image,channel_indices,config)
			context=self._context(config,channel_indices,thresholds)
			grid=self.cell_store.grid()
			halo=max(int(config.cytoplasmic_ring_width),int(config.membrane_ring_width),1)+1
			with QuantificationDatabase(database_path,context)as database:
				_json_atomic(destination/'quantification_config.json',{
					'created_at':started_at,
					'image_path':str(self.image_path),
					'series':self.series,
					'region_directory':str(self.region_directory),
					'channels':[
						{'channel_index':int(index),'channel_name':str(name),
							'output_name':str(output),'positive_threshold':float(threshold)}
						for index,name,output,threshold in zip(
							channel_indices,channel_names,output_names,thresholds
						)
					],
					'config':config.to_dict(),
					'context':context,
				})
				database.initialize_channels(channel_indices,channel_names,output_names,thresholds)
				database.initialize_units(grid,len(channel_indices),int(config.channel_batch_size))
				database.reset_interrupted()
				if config.retry_failed_units:
					reset=database.reset_failed()
					if reset:
						log(f'Reset {reset:,} failed quantification unit(s).')
				completed,total,failed,_=database.progress()
				self._emit(on_progress,'quantifying',completed,total,failed)
				while True:
					if cancel_event.is_set():
						cancelled=True
						break
					unit=database.claim_next()
					if unit is None:
						break
					unit_id=str(unit['unit_id'])
					row=int(unit['row_index'])
					column=int(unit['column_index'])
					start=int(unit['channel_start'])
					end=int(unit['channel_end'])
					core=self.cell_store.chunk_bounds(row,column)
					read=_clipped_halo(core,halo,self.cell_store.metadata.width,self.cell_store.metadata.height)
					try:
						cell_read=self.cell_store.read_region(
							x=read.x,y=read.y,width=read.width,height=read.height
						)
						nuclear_read=self.nuclear_store.read_region(
							x=read.x,y=read.y,width=read.width,height=read.height
						)
						compartment_read=build_compartment_labels(
							nuclear_read,cell_read,
							cytoplasmic_ring_width=int(config.cytoplasmic_ring_width),
							membrane_ring_width=int(config.membrane_ring_width),
						)
						compartments={
							name:_crop_to_core(labels,read,core)
							for name,labels in compartment_read.items()
						}
						batch_indices=channel_indices[start:end]
						batch_values=image.read_region(
							x=core.x,y=core.y,width=core.width,height=core.height,
							channels=batch_indices,level=level,position=config.position,
						)
						database.commit_unit(
							unit_id,
							list(range(start,end)),
							thresholds[start:end],
							batch_values,
							compartments,
							update_areas=(start==0),
						)
					except Exception as error:
						database.mark_failed(unit_id,error)
					completed,total,failed,_=database.progress()
					self._emit(
						on_progress,'quantifying',completed,total,failed,
						f'chunk ({row}, {column}), channels {start+1}-{end}',
					)
				completed,total,failed,_=database.progress()
				if failed and not cancelled:
					raise QuantificationError(
						f'Marker quantification has {failed:,} failed unit(s). '
						'Enable retry failed units and resume.'
					)
				with sqlite3.connect(self.reconciliation_db)as global_connection:
					cell_count=int(global_connection.execute(
						'SELECT COUNT(*) FROM global_cells'
					).fetchone()[0])
				if not cancelled and completed==total:
					self._export_background_csv(destination/'channel_backgrounds.csv',database.connection)
					if csv_path is not None:
						self._emit(on_progress,'exporting CSV',0,cell_count)
						cell_count=self._export_csv(csv_path,database.connection)
						self._emit(on_progress,'exporting CSV',cell_count,cell_count)
					if excel_path is not None:
						self._emit(on_progress,'exporting Excel',0,cell_count)
						cell_count=self._export_excel(excel_path,database.connection)
						self._emit(on_progress,'exporting Excel',cell_count,cell_count)
		summary=QuantificationRunSummary(
			output_directory=str(destination),
			database_path=str(database_path),
			csv_path=str(csv_path)if csv_path is not None and csv_path.exists()else None,
			excel_path=str(excel_path)if excel_path is not None and excel_path.exists()else None,
			cell_count=cell_count,
			channel_count=len(channel_indices),
			completed_units=completed,
			failed_units=failed,
			cancelled=cancelled,
			started_at=started_at,
			finished_at=_utc_now(),
		)
		_json_atomic(destination/'quantification_summary.json',asdict(summary))
		return summary
