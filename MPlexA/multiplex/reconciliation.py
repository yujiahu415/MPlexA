from __future__ import annotations
from collections import OrderedDict
from dataclasses import asdict,dataclass
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
import time
from typing import Any,Callable,Iterable,Iterator,Mapping,Sequence
import numpy as np
from.checkpoints import CheckpointProgress,TileCheckpointStore
from.exceptions import CheckpointMismatchError,MultiplexImageError,TilingError
from.segmentation import SegmentationError,TilePredictionArchive,load_tile_predictions
from.tiling import Bounds,Tile,TileGrid
RECONCILIATION_SCHEMA_VERSION=1
GLOBAL_LABEL_STORE_NAME='labels.mplexa-labels'
LEGACY_GLOBAL_LABEL_STORE_NAMES=('labels.cellan-labels',)


def resolve_global_label_store(directory:str|Path)->Path:
	base=Path(directory).expanduser().resolve()
	preferred=base/GLOBAL_LABEL_STORE_NAME
	if preferred.is_dir():
		return preferred
	for name in LEGACY_GLOBAL_LABEL_STORE_NAMES:
		candidate=base/name
		if candidate.is_dir():
			return candidate
	return preferred
LABEL_STORE_SCHEMA_VERSION=1


def _utc_now()->str:
	return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _canonical_json(value:Any)->str:
	return json.dumps(value,sort_keys=True,separators=(',',':'),default=str)


def _sha256_file(path:Path,block_size:int=1024*1024)->str:
	digest=hashlib.sha256()
	with path.open('rb')as handle:
		while True:
			block=handle.read(block_size)
			if not block:
				break
			digest.update(block)
	return digest.hexdigest()



class ReconciliationError(MultiplexImageError):



class ReconciliationCancelled(ReconciliationError):



@dataclass(frozen=True,slots=True)
class ReconciliationConfig:
	iou_threshold:float=0.30
	containment_threshold:float=0.65
	same_class_only:bool=True
	mask_strategy:str='best'
	chunk_size:int=1024
	retry_failed_chunks:bool=False
	archive_cache_size:int=12


	def __post_init__(self)->None:
		if not 0<=float(self.iou_threshold)<=1:
			raise ReconciliationError('Mask IoU threshold must be between 0 and 1.')
		if not 0<=float(self.containment_threshold)<=1:
			raise ReconciliationError('Mask containment threshold must be between 0 and 1.')
		if self.iou_threshold==0 and self.containment_threshold==0:
			raise ReconciliationError('At least one duplicate-match threshold must be greater than zero.')
		if self.mask_strategy not in{'best','union'}:
			raise ReconciliationError('Mask strategy must be \'best\' or \'union\'.')
		if int(self.chunk_size)<=0:
			raise ReconciliationError('Label chunk size must be positive.')
		if int(self.archive_cache_size)<=0:
			raise ReconciliationError('Archive cache size must be positive.')


	def to_dict(self)->dict[str,Any]:
		return asdict(self)



@dataclass(frozen=True,slots=True)
class ReconciliationProgress:
	stage:str
	current:int
	total:int
	message:str=''


	@property
	def fraction(self)->float:
		return 0.0 if self.total<=0 else min(1.0,max(0.0,self.current/self.total))



@dataclass(frozen=True,slots=True)
class ReconciliationRunSummary:
	segmentation_directory:str
	output_directory:str
	database_path:str
	label_store_path:str
	prediction_count:int
	duplicate_link_count:int
	global_cell_count:int
	label_chunks:int
	conflict_pixels:int
	cancelled:bool
	started_at:str
	finished_at:str


	def summary(self)->str:
		return(
			f'Input predictions: {self.prediction_count:,}\n'
			f'Duplicate links: {self.duplicate_link_count:,}\n'
			f'Global cells: {self.global_cell_count:,}\n'
			f'Label chunks: {self.label_chunks:,}\n'
			f'Conflicting pixels resolved: {self.conflict_pixels:,}\n'
			f'Cancelled: {self.cancelled}\n'
			f'Output: {self.output_directory}'
		)



@dataclass(frozen=True,slots=True)
class LabelStoreMetadata:
	width:int
	height:int
	chunk_width:int
	chunk_height:int
	dtype:str
	level:int=0
	fill_value:int=0
	schema_version:int=LABEL_STORE_SCHEMA_VERSION
	axes:str='YX'
	global_cell_count:int=0
	created_at:str=''


	def __post_init__(self)->None:
		if min(self.width,self.height,self.chunk_width,self.chunk_height)<=0:
			raise ReconciliationError('Label-store dimensions and chunks must be positive.')
		dtype=np.dtype(self.dtype)
		if dtype.kind!='u':
			raise ReconciliationError('Global instance labels must use an unsigned integer dtype.')


	def to_dict(self)->dict[str,Any]:
		return asdict(self)



class ChunkedInstanceLabelStore:
	METADATA_FILENAME='metadata.json'


	def __init__(self,path:str|Path,metadata:LabelStoreMetadata)->None:
		self.path=Path(path).expanduser().resolve()
		self.metadata=metadata
		self.chunks_directory=self.path/'chunks'


	@classmethod
	def create(
		cls,
		path:str|Path,
		*,
		width:int,
		height:int,
		chunk_size:int|tuple[int,int]=1024,
		dtype:str|np.dtype[Any]=np.uint32,
		level:int=0,
		global_cell_count:int=0,
		overwrite:bool=False,
	)->'ChunkedInstanceLabelStore':
		destination=Path(path).expanduser().resolve()
		if isinstance(chunk_size,Sequence)and not isinstance(chunk_size,(str,bytes)):
			values=tuple(int(value)for value in chunk_size)
			if len(values)!=2:
				raise ReconciliationError('Chunk size must be one integer or (width, height).')
			chunk_width,chunk_height=values
		else:
			chunk_width=chunk_height=int(chunk_size)
		metadata=LabelStoreMetadata(
			width=int(width),
			height=int(height),
			chunk_width=chunk_width,
			chunk_height=chunk_height,
			dtype=np.dtype(dtype).name,
			level=int(level),
			global_cell_count=int(global_cell_count),
			created_at=_utc_now(),
		)
		metadata_path=destination/cls.METADATA_FILENAME
		if metadata_path.exists()and not overwrite:
			existing=cls.open(destination)
			comparable_existing=existing.metadata.to_dict().copy()
			comparable_requested=metadata.to_dict().copy()
			comparable_existing.pop('created_at',None)
			comparable_requested.pop('created_at',None)
			if comparable_existing!=comparable_requested:
				raise CheckpointMismatchError(
					'The existing global label store uses different dimensions, chunk settings, or cell count.'
				)
			return existing
		destination.mkdir(parents=True,exist_ok=True)
		chunks=destination/'chunks'
		chunks.mkdir(parents=True,exist_ok=True)
		if overwrite:
			for item in chunks.glob('*.npy'):
				item.unlink()
		cls._write_json_atomic(metadata_path,metadata.to_dict())
		return cls(destination,metadata)


	@classmethod
	def open(cls,path:str|Path)->'ChunkedInstanceLabelStore':
		destination=Path(path).expanduser().resolve()
		metadata_path=destination/cls.METADATA_FILENAME
		try:
			data=json.loads(metadata_path.read_text(encoding='utf-8'))
			metadata=LabelStoreMetadata(**data)
		except(OSError,ValueError,TypeError,json.JSONDecodeError)as error:
			raise ReconciliationError(f'Unable to open MPlexA label store {destination}: {error}')from error
		return cls(destination,metadata)


	@staticmethod
	def _write_json_atomic(path:Path,data:Mapping[str,Any])->None:
		path.parent.mkdir(parents=True,exist_ok=True)
		handle=tempfile.NamedTemporaryFile(
			mode='w',suffix='.json',prefix=path.stem+'.',dir=path.parent,
			delete=False,encoding='utf-8'
		)
		temporary=Path(handle.name)
		try:
			json.dump(data,handle,indent=2)
			handle.flush()
			os.fsync(handle.fileno())
			handle.close()
			os.replace(temporary,path)
		except Exception:
			handle.close()
			temporary.unlink(missing_ok=True)
			raise


	@property
	def dtype(self)->np.dtype[Any]:
		return np.dtype(self.metadata.dtype)


	@property
	def columns(self)->int:
		return int(math.ceil(self.metadata.width/self.metadata.chunk_width))


	@property
	def rows(self)->int:
		return int(math.ceil(self.metadata.height/self.metadata.chunk_height))


	def chunk_bounds(self,row:int,column:int)->Bounds:
		if row<0 or row>=self.rows or column<0 or column>=self.columns:
			raise IndexError((row,column))
		x=column*self.metadata.chunk_width
		y=row*self.metadata.chunk_height
		return Bounds(
			x,
			y,
			min(self.metadata.chunk_width,self.metadata.width-x),
			min(self.metadata.chunk_height,self.metadata.height-y),
		)


	def chunk_path(self,row:int,column:int)->Path:
		bounds=self.chunk_bounds(row,column)
		return self.chunks_directory/ f'Y{bounds.y:09d}_X{bounds.x:09d}.npy'


	def write_chunk(self,row:int,column:int,array:np.ndarray)->Path:
		bounds=self.chunk_bounds(row,column)
		data=np.asarray(array,dtype=self.dtype)
		if data.shape!=(bounds.height,bounds.width):
			raise ReconciliationError(
				f'Chunk {(row,column)} expects {(bounds.height,bounds.width)}; got {data.shape}.'
			)
		destination=self.chunk_path(row,column)
		destination.parent.mkdir(parents=True,exist_ok=True)
		if not np.any(data):
			destination.unlink(missing_ok=True)
			return destination
		handle=tempfile.NamedTemporaryFile(
			mode='wb',suffix='.npy',prefix=destination.stem+'.',
			dir=destination.parent,delete=False
		)
		temporary=Path(handle.name)
		handle.close()
		try:
			np.save(temporary,data,allow_pickle=False)
			delays=(0.0,0.05,0.20,0.50)
			last_error:OSError|None=None
			for delay in delays:
				if delay:
					time.sleep(delay)
				try:
					os.replace(temporary,destination)
					last_error=None
					break
				except OSError as error:
					winerror=getattr(error,'winerror',None)
					if not isinstance(error,PermissionError)and winerror not in{5,32,33}:
						raise
					last_error=error
			if last_error is not None:
				raise last_error
		except Exception:
			temporary.unlink(missing_ok=True)
			raise
		return destination


	def read_chunk(self,row:int,column:int,*,mmap_mode:str|None=None)->np.ndarray:
		bounds=self.chunk_bounds(row,column)
		path=self.chunk_path(row,column)
		if not path.exists():
			return np.full((bounds.height,bounds.width),self.metadata.fill_value,dtype=self.dtype)
		try:
			data=np.load(path,mmap_mode=mmap_mode,allow_pickle=False)
		except(OSError,ValueError)as error:
			raise ReconciliationError(f'Unable to read label chunk {path}: {error}')from error
		if tuple(data.shape)!=(bounds.height,bounds.width)or data.dtype!=self.dtype:
			raise ReconciliationError(f'Label chunk {path} does not match its store metadata.')
		return data


	def read_region(self,*,x:int,y:int,width:int,height:int)->np.ndarray:
		request=Bounds(int(x),int(y),int(width),int(height))
		image_bounds=Bounds(0,0,self.metadata.width,self.metadata.height)
		if request.area<=0 or not image_bounds.contains_bounds(request):
			raise ReconciliationError('Requested label region is empty or outside the global label image.')
		output=np.full((request.height,request.width),self.metadata.fill_value,dtype=self.dtype)
		col0=request.x//self.metadata.chunk_width
		col1=(request.x1-1)//self.metadata.chunk_width
		row0=request.y//self.metadata.chunk_height
		row1=(request.y1-1)//self.metadata.chunk_height
		for row in range(row0,row1+1):
			for column in range(col0,col1+1):
				chunk_bounds=self.chunk_bounds(row,column)
				intersection=request.intersection(chunk_bounds)
				if intersection is None:
					continue
				chunk=self.read_chunk(row,column,mmap_mode='r')
				source_y0=intersection.y-chunk_bounds.y
				source_x0=intersection.x-chunk_bounds.x
				target_y0=intersection.y-request.y
				target_x0=intersection.x-request.x
				output[
					target_y0:target_y0+intersection.height,
					target_x0:target_x0+intersection.width,
				]=chunk[
					source_y0:source_y0+intersection.height,
					source_x0:source_x0+intersection.width,
				]
		return output


	def grid(self)->TileGrid:
		return TileGrid(
			self.metadata.width,
			self.metadata.height,
			tile_width=self.metadata.chunk_width,
			tile_height=self.metadata.chunk_height,
			overlap=0,
			level=self.metadata.level,
		)



class _ArchiveCache:


	def __init__(self,capacity:int)->None:
		self.capacity=int(capacity)
		self._items:OrderedDict[str,TilePredictionArchive]=OrderedDict()


	def get(self,path:str|Path)->TilePredictionArchive:
		key=str(Path(path).resolve())
		archive=self._items.pop(key,None)
		if archive is None:
			archive=load_tile_predictions(key)
		self._items[key]=archive
		while len(self._items)>self.capacity:
			self._items.popitem(last=False)
		return archive



class _UnionFind:


	def __init__(self,size:int)->None:
		self.parent=np.arange(size+1,dtype=np.int64)
		self.rank=np.zeros(size+1,dtype=np.uint8)


	def find(self,value:int)->int:
		parent=self.parent
		root=int(value)
		while int(parent[root])!=root:
			root=int(parent[root])
		current=int(value)
		while int(parent[current])!=current:
			next_value=int(parent[current])
			parent[current]=root
			current=next_value
		return root


	def union(self,first:int,second:int)->int:
		root_a=self.find(first)
		root_b=self.find(second)
		if root_a==root_b:
			return root_a
		rank=self.rank
		if rank[root_a]<rank[root_b]:
			root_a,root_b=root_b,root_a
		self.parent[root_b]=root_a
		if rank[root_a]==rank[root_b]:
			rank[root_a]+=1
		return root_a


	def compress_all(self)->np.ndarray:
		for index in range(1,len(self.parent)):
			self.parent[index]=self.find(index)
		return self.parent



class ReconciliationDatabase:


	def __init__(self,path:str|Path,*,context:Mapping[str,Any])->None:
		self.path=Path(path).expanduser().resolve()
		self.path.parent.mkdir(parents=True,exist_ok=True)
		self.connection=sqlite3.connect(self.path,timeout=60.0)
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
                CREATE TABLE IF NOT EXISTS archives (
                    tile_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    prediction_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tile_id TEXT NOT NULL,
                    archive_path TEXT NOT NULL,
                    local_index INTEGER NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    area INTEGER NOT NULL,
                    centroid_x REAL NOT NULL,
                    centroid_y REAL NOT NULL,
                    x0 INTEGER NOT NULL,
                    y0 INTEGER NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    owned_by_core INTEGER NOT NULL,
                    touches_read_edge INTEGER NOT NULL,
                    matched INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(tile_id, local_index)
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE VIRTUAL TABLE IF NOT EXISTS predictions_rtree USING rtree(
                    prediction_id, min_x, max_x, min_y, max_y
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS duplicate_links (
                    first_prediction_id INTEGER NOT NULL,
                    second_prediction_id INTEGER NOT NULL,
                    mask_iou REAL NOT NULL,
                    containment REAL NOT NULL,
                    intersection_pixels INTEGER NOT NULL,
                    PRIMARY KEY(first_prediction_id, second_prediction_id)
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS global_cells (
                    global_cell_id INTEGER PRIMARY KEY,
                    representative_prediction_id INTEGER NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    area INTEGER NOT NULL,
                    centroid_x REAL NOT NULL,
                    centroid_y REAL NOT NULL,
                    x0 INTEGER NOT NULL,
                    y0 INTEGER NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    source_count INTEGER NOT NULL,
                    owned_source_count INTEGER NOT NULL,
                    touches_image_edge INTEGER NOT NULL,
                    mask_strategy TEXT NOT NULL
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE VIRTUAL TABLE IF NOT EXISTS global_cells_rtree USING rtree(
                    global_cell_id, min_x, max_x, min_y, max_y
                )
                '''
			)
			self.connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS cell_sources (
                    global_cell_id INTEGER NOT NULL,
                    prediction_id INTEGER NOT NULL,
                    is_representative INTEGER NOT NULL,
                    PRIMARY KEY(global_cell_id, prediction_id)
                )
                '''
			)
			self.connection.execute(
				'CREATE INDEX IF NOT EXISTS idx_predictions_matched ON predictions(matched, prediction_id)'
			)
			self.connection.execute(
				'CREATE INDEX IF NOT EXISTS idx_predictions_tile ON predictions(tile_id, local_index)'
			)
			self.connection.execute(
				'CREATE INDEX IF NOT EXISTS idx_sources_cell ON cell_sources(global_cell_id)'
			)


	def _metadata(self)->dict[str,str]:
		return{
			str(row['key']):str(row['value'])
			for row in self.connection.execute('SELECT key, value FROM metadata')
		}


	def _validate_context(self,context:Mapping[str,Any])->None:
		expected={
			'schema_version':str(RECONCILIATION_SCHEMA_VERSION),
			'context':_canonical_json(context),
		}
		existing=self._metadata()
		if existing:
			if existing.get('schema_version')!=expected['schema_version']:
				raise CheckpointMismatchError('Reconciliation database schema is incompatible.')
			if existing.get('context')!=expected['context']:
				raise CheckpointMismatchError(
					'The existing reconciliation database belongs to different segmentation inputs or settings.'
				)
			return
		with self.connection:
			self.connection.executemany(
				'INSERT INTO metadata(key, value) VALUES (?, ?)',
				[('schema_version',expected['schema_version']),('context',expected['context']),
					('created_at',_utc_now()),('phase','indexing')],
			)


	def set_metadata(self,key:str,value:Any)->None:
		with self.connection:
			self.connection.execute(
				'INSERT INTO metadata(key, value) VALUES (?, ?) '
				'ON CONFLICT(key) DO UPDATE SET value=excluded.value',
				(str(key),_canonical_json(value)if not isinstance(value,str)else value),
			)


	def get_metadata(self,key:str,default:Any=None)->Any:
		row=self.connection.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
		if row is None:
			return default
		value=str(row[0])
		try:
			return json.loads(value)
		except json.JSONDecodeError:
			return value


	def close(self)->None:
		self.connection.close()


	def __enter__(self)->'ReconciliationDatabase':
		return self


	def __exit__(self,exc_type:Any,exc_value:Any,traceback:Any)->None:
		self.close()



@dataclass(frozen=True,slots=True)
class _PredictionReference:
	prediction_id:int
	tile_id:str
	archive_path:str
	local_index:int
	class_id:int
	class_name:str
	score:float
	area:int
	centroid_x:float
	centroid_y:float
	bounds:Bounds
	owned_by_core:bool
	touches_read_edge:bool


def _row_to_prediction(row:sqlite3.Row)->_PredictionReference:
	return _PredictionReference(
		prediction_id=int(row['prediction_id']),
		tile_id=str(row['tile_id']),
		archive_path=str(row['archive_path']),
		local_index=int(row['local_index']),
		class_id=int(row['class_id']),
		class_name=str(row['class_name']),
		score=float(row['score']),
		area=int(row['area']),
		centroid_x=float(row['centroid_x']),
		centroid_y=float(row['centroid_y']),
		bounds=Bounds(
			int(row['x0']),int(row['y0']),
			int(row['x1'])-int(row['x0']),
			int(row['y1'])-int(row['y0']),
		),
		owned_by_core=bool(row['owned_by_core']),
		touches_read_edge=bool(row['touches_read_edge']),
	)


def _prediction_mask(reference:_PredictionReference,cache:_ArchiveCache)->np.ndarray:
	archive=cache.get(reference.archive_path)
	return archive.decode_cropped_mask(reference.local_index)


def _mask_overlap(
	first:_PredictionReference,
	second:_PredictionReference,
	cache:_ArchiveCache,
)->tuple[int,float,float]:
	intersection_bounds=first.bounds.intersection(second.bounds)
	if intersection_bounds is None:
		return 0,0.0,0.0
	maximum=min(first.area,second.area,intersection_bounds.area)
	minimum_area=min(first.area,second.area)
	maximum_containment=maximum/minimum_area if minimum_area else 0.0
	maximum_iou=maximum/(first.area+second.area-maximum)if maximum else 0.0
	if maximum<=0:
		return 0,0.0,0.0
	first_mask=_prediction_mask(first,cache)
	second_mask=_prediction_mask(second,cache)
	f_y0=intersection_bounds.y-first.bounds.y
	f_x0=intersection_bounds.x-first.bounds.x
	s_y0=intersection_bounds.y-second.bounds.y
	s_x0=intersection_bounds.x-second.bounds.x
	first_piece=first_mask[
		f_y0:f_y0+intersection_bounds.height,
		f_x0:f_x0+intersection_bounds.width,
	]
	second_piece=second_mask[
		s_y0:s_y0+intersection_bounds.height,
		s_x0:s_x0+intersection_bounds.width,
	]
	intersection=int(np.count_nonzero(first_piece&second_piece))
	if intersection==0:
		return 0,0.0,0.0
	union=first.area+second.area-intersection
	iou=intersection/union if union else 0.0
	containment=intersection/minimum_area if minimum_area else 0.0
	return intersection,float(iou),float(containment)


def _representative_quality(reference:_PredictionReference)->tuple[int,int,float,int,int]:
	return(
		int(reference.owned_by_core),
		int(not reference.touches_read_edge),
		float(reference.score),
		int(reference.area),
		-int(reference.prediction_id),
	)


def _segmentation_inputs(segmentation_directory:Path)->tuple[dict[str,Any],TileGrid,list[Path]]:
	config_path=segmentation_directory/'segmentation_config.json'
	checkpoint_path=segmentation_directory/'segmentation.sqlite'
	tiles_directory=segmentation_directory/'tiles'
	if not config_path.is_file():
		raise ReconciliationError(f'Missing Module 2 configuration: {config_path}')
	if not checkpoint_path.is_file():
		raise ReconciliationError(f'Missing Module 2 checkpoint: {checkpoint_path}')
	if not tiles_directory.is_dir():
		raise ReconciliationError(f'Missing Module 2 tile predictions: {tiles_directory}')
	try:
		with sqlite3.connect(checkpoint_path)as checkpoint_connection:
			counts={
				str(status):int(count)
				for status,count in checkpoint_connection.execute(
					'SELECT status, COUNT(*) FROM tiles GROUP BY status'
				)
			}
	except sqlite3.Error as error:
		raise ReconciliationError(f'Unable to read the Module 2 segmentation checkpoint: {error}')from error
	incomplete=sum(counts.get(status,0)for status in('pending','running','failed'))
	if incomplete:
		raise ReconciliationError(
			'Module 2 segmentation is not complete: '+
			'pending={}, running={}, '.format(counts.get('pending',0),counts.get('running',0))+
			'failed={}. Finish or retry segmentation before assigning global IDs.'.format(counts.get('failed',0))
		)
	try:
		configuration=json.loads(config_path.read_text(encoding='utf-8'))
		grid=TileGrid.from_dict(configuration['grid'])
	except(OSError,ValueError,KeyError,TypeError,json.JSONDecodeError)as error:
		raise ReconciliationError(f'Unable to read Module 2 configuration: {error}')from error
	archives=sorted(tiles_directory.glob('*.npz'),key=lambda item:item.name)
	if not archives:
		raise ReconciliationError('No completed tile-prediction archives were found.')
	if len(archives)!=len(grid):
		raise ReconciliationError(
			f'Module 2 contains {len(archives):,} tile archive(s), but the saved grid requires '
			f'{len(grid):,}. Finish segmentation or restore the missing archives before reconciliation.'
		)
	return configuration,grid,archives


def _segmentation_fingerprint(
	segmentation_directory:Path,
	configuration:Mapping[str,Any],
	archives:Sequence[Path],
)->str:
	entries:list[dict[str,Any]]=[]
	for path in archives:
		stat=path.stat()
		entries.append({'name':path.name,'size':int(stat.st_size),'mtime_ns':int(stat.st_mtime_ns)})
	config_path=segmentation_directory/'segmentation_config.json'
	checkpoint_path=segmentation_directory/'segmentation.sqlite'
	try:
		with sqlite3.connect(checkpoint_path)as checkpoint_connection:
			checkpoint_status={
				str(status):int(count)
				for status,count in checkpoint_connection.execute(
					'SELECT status, COUNT(*) FROM tiles GROUP BY status'
				)
			}
	except sqlite3.Error as error:
		raise ReconciliationError(f'Unable to fingerprint the Module 2 checkpoint: {error}')from error
	payload={
		'configuration_sha256':_sha256_file(config_path),
		'grid':configuration.get('grid'),
		'checkpoint_status':checkpoint_status,
		'archives':entries,
	}
	return hashlib.sha256(_canonical_json(payload).encode('utf-8')).hexdigest()



class GlobalMaskReconciler:


	def __init__(self,segmentation_directory:str|Path)->None:
		self.segmentation_directory=Path(segmentation_directory).expanduser().resolve()
		self.configuration,self.segmentation_grid,self.archive_paths=_segmentation_inputs(
			self.segmentation_directory
		)


	def _emit(
		self,
		callback:Callable[[ReconciliationProgress],None]|None,
		stage:str,
		current:int,
		total:int,
		message:str='',
	)->None:
		if callback is not None:
			callback(ReconciliationProgress(stage,int(current),int(total),message))


	def _check_cancel(self,cancel_event:threading.Event)->None:
		if cancel_event.is_set():
			raise ReconciliationCancelled('Reconciliation cancelled by the user.')


	def _index_predictions(
		self,
		database:ReconciliationDatabase,
		*,
		cancel_event:threading.Event,
		on_progress:Callable[[ReconciliationProgress],None]|None,
	)->int:
		connection=database.connection
		indexed={
			str(row['tile_id']):(int(row['size']),int(row['mtime_ns']))
			for row in connection.execute('SELECT tile_id, size, mtime_ns FROM archives')
		}
		total=len(self.archive_paths)
		for index,path in enumerate(self.archive_paths,start=1):
			self._check_cancel(cancel_event)
			stat=path.stat()
			archive=load_tile_predictions(path)
			existing=indexed.get(archive.tile_id)
			identity=(int(stat.st_size),int(stat.st_mtime_ns))
			if existing is not None:
				if existing!=identity:
					raise CheckpointMismatchError(
						f'Tile archive {path.name} changed after reconciliation began. Use a new output folder.'
					)
				self._emit(on_progress,'indexing',index,total, f'Indexed {path.name}')
				continue
			rows=[]
			for local_index in range(archive.count):
				x0,y0,x1,y1=(int(value)for value in archive.global_boxes[local_index])
				cx,cy=(float(value)for value in archive.global_centroids[local_index])
				rows.append(
					(
						archive.tile_id,
						str(path.resolve()),
						int(local_index),
						int(archive.class_ids[local_index]),
						str(archive.class_names[local_index]),
						float(archive.scores[local_index]),
						int(archive.areas[local_index]),
						cx,
						cy,
						x0,
						y0,
						x1,
						y1,
						int(bool(archive.owned_by_core[local_index])),
						int(bool(archive.touches_read_edge[local_index])),
					)
				)
			with connection:
				before=connection.total_changes
				connection.executemany(
					'''
                    INSERT INTO predictions(
                        tile_id, archive_path, local_index, class_id, class_name,
                        score, area, centroid_x, centroid_y, x0, y0, x1, y1,
                        owned_by_core, touches_read_edge
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
					rows,
				)
				prediction_ids=[
					int(row[0])
					for row in connection.execute(
						'SELECT prediction_id FROM predictions WHERE tile_id=? ORDER BY local_index',
						(archive.tile_id,),
					)
				]
				if len(prediction_ids)!=archive.count:
					raise ReconciliationError(f'Prediction indexing failed for {archive.tile_id}.')
				rtree_rows=[]
				for prediction_id,box in zip(prediction_ids,archive.global_boxes):
					x0,y0,x1,y1=(int(value)for value in box)
					rtree_rows.append((prediction_id,x0,x1,y0,y1))
				connection.executemany(
					'INSERT INTO predictions_rtree(prediction_id, min_x, max_x, min_y, max_y) '
					'VALUES (?, ?, ?, ?, ?)',
					rtree_rows,
				)
				connection.execute(
					'INSERT INTO archives(tile_id, path, size, mtime_ns, prediction_count, indexed_at) '
					'VALUES (?, ?, ?, ?, ?, ?)',
					(archive.tile_id,str(path.resolve()),identity[0],identity[1],archive.count,_utc_now()),
				)
			self._emit(on_progress,'indexing',index,total, f'Indexed {path.name}')
		database.set_metadata('phase','matching')
		return int(connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0])


	def _match_duplicates(
		self,
		database:ReconciliationDatabase,
		config:ReconciliationConfig,
		*,
		cancel_event:threading.Event,
		on_progress:Callable[[ReconciliationProgress],None]|None,
	)->int:
		connection=database.connection
		total=int(connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0])
		completed=int(connection.execute('SELECT COUNT(*) FROM predictions WHERE matched=1').fetchone()[0])
		cache=_ArchiveCache(config.archive_cache_size)
		while True:
			self._check_cancel(cancel_event)
			row=connection.execute(
				'SELECT * FROM predictions WHERE matched=0 ORDER BY prediction_id LIMIT 1'
			).fetchone()
			if row is None:
				break
			current=_row_to_prediction(row)
			parameters:list[Any]=[
				current.prediction_id,
				current.tile_id,
				current.bounds.x1,
				current.bounds.x,
				current.bounds.y1,
				current.bounds.y,
			]
			class_sql=''
			if config.same_class_only:
				class_sql=' AND p.class_id=?'
				parameters.append(current.class_id)
			candidates=connection.execute(
				'''
                SELECT p.* FROM predictions_rtree r
                JOIN predictions p ON p.prediction_id=r.prediction_id
                WHERE p.prediction_id>? AND p.tile_id<>?
                  AND r.min_x<? AND r.max_x>?
                  AND r.min_y<? AND r.max_y>?
                '''+class_sql+' ORDER BY p.prediction_id',
				tuple(parameters),
			).fetchall()
			links:list[tuple[int,int,float,float,int]]=[]
			for candidate_row in candidates:
				candidate=_row_to_prediction(candidate_row)
				intersection_bounds=current.bounds.intersection(candidate.bounds)
				if intersection_bounds is None:
					continue
				maximum=min(current.area,candidate.area,intersection_bounds.area)
				minimum_area=min(current.area,candidate.area)
				max_containment=maximum/minimum_area if minimum_area else 0.0
				max_iou=maximum/(current.area+candidate.area-maximum)if maximum else 0.0
				if max_iou<config.iou_threshold and max_containment<config.containment_threshold:
					continue
				intersection,iou,containment=_mask_overlap(current,candidate,cache)
				if iou>=config.iou_threshold or containment>=config.containment_threshold:
					links.append(
						(
							current.prediction_id,
							candidate.prediction_id,
							iou,
							containment,
							intersection,
						)
					)
			with connection:
				if links:
					connection.executemany(
						'INSERT OR IGNORE INTO duplicate_links('
						'first_prediction_id, second_prediction_id, mask_iou, containment, intersection_pixels'
						') VALUES (?, ?, ?, ?, ?)',
						links,
					)
				connection.execute(
					'UPDATE predictions SET matched=1 WHERE prediction_id=?',
					(current.prediction_id,),
				)
			completed+=1
			if completed==total or completed%50==0:
				self._emit(
					on_progress,
					'matching',
					completed,
					total,
					f'Matched {completed:,}/{total:,} predictions',
				)
		database.set_metadata('phase','grouping')
		return int(connection.execute('SELECT COUNT(*) FROM duplicate_links').fetchone()[0])


	def _group_mask(
		self,
		sources:Sequence[_PredictionReference],
		strategy:str,
		cache:_ArchiveCache,
	)->tuple[Bounds,np.ndarray,_PredictionReference]:
		representative=max(sources,key=_representative_quality)
		if strategy=='best'or len(sources)==1:
			return representative.bounds,_prediction_mask(representative,cache),representative
		x0=min(source.bounds.x for source in sources)
		y0=min(source.bounds.y for source in sources)
		x1=max(source.bounds.x1 for source in sources)
		y1=max(source.bounds.y1 for source in sources)
		union=np.zeros((y1-y0,x1-x0),dtype=bool)
		for source in sources:
			mask=_prediction_mask(source,cache)
			sy=source.bounds.y-y0
			sx=source.bounds.x-x0
			union[sy:sy+source.bounds.height,sx:sx+source.bounds.width]|=mask
		ys,xs=np.nonzero(union)
		if not len(xs):
			raise ReconciliationError('A duplicate group produced an empty union mask.')
		shrink=Bounds(
			x0+int(xs.min()),
			y0+int(ys.min()),
			int(xs.max()-xs.min()+1),
			int(ys.max()-ys.min()+1),
		)
		cropped=union[
			shrink.y-y0:shrink.y1-y0,
			shrink.x-x0:shrink.x1-x0,
		]
		return shrink,cropped,representative


	def _build_global_cells(
		self,
		database:ReconciliationDatabase,
		config:ReconciliationConfig,
		*,
		cancel_event:threading.Event,
		on_progress:Callable[[ReconciliationProgress],None]|None,
	)->int:
		connection=database.connection
		existing=int(connection.execute('SELECT COUNT(*) FROM global_cells').fetchone()[0])
		phase=str(database.get_metadata('phase','grouping'))
		if existing and phase in{'rendering','completed'}:
			return existing
		if existing:
			with connection:
				connection.execute('DELETE FROM cell_sources')
				connection.execute('DELETE FROM global_cells_rtree')
				connection.execute('DELETE FROM global_cells')
		prediction_count=int(connection.execute('SELECT COUNT(*) FROM predictions').fetchone()[0])
		union_find=_UnionFind(prediction_count)
		for row in connection.execute(
			'SELECT first_prediction_id, second_prediction_id FROM duplicate_links ORDER BY first_prediction_id'
		):
			union_find.union(int(row[0]),int(row[1]))
		roots=union_find.compress_all()
		references:dict[int,_PredictionReference]={}
		group_members:dict[int,list[int]]={}
		for row in connection.execute('SELECT * FROM predictions ORDER BY prediction_id'):
			reference=_row_to_prediction(row)
			references[reference.prediction_id]=reference
			root=int(roots[reference.prediction_id])
			group_members.setdefault(root,[]).append(reference.prediction_id)
		ordered_groups:list[tuple[float,float,int,int,int]]=[]
		representative_by_root:dict[int,int]={}
		for root,member_ids in group_members.items():
			representative=max((references[item]for item in member_ids),key=_representative_quality)
			representative_by_root[root]=representative.prediction_id
			ordered_groups.append(
				(
					representative.centroid_y,
					representative.centroid_x,
					representative.class_id,
					min(member_ids),
					root,
				)
			)
		ordered_groups.sort()
		cache=_ArchiveCache(config.archive_cache_size)
		image_width=self.segmentation_grid.image_width
		image_height=self.segmentation_grid.image_height
		total_groups=len(ordered_groups)
		with connection:
			connection.execute('DELETE FROM cell_sources')
			connection.execute('DELETE FROM global_cells_rtree')
			connection.execute('DELETE FROM global_cells')
		for global_id,(_,_,_,_,root)in enumerate(ordered_groups,start=1):
			self._check_cancel(cancel_event)
			members=[references[item]for item in group_members[root]]
			bounds,mask,representative=self._group_mask(members,config.mask_strategy,cache)
			ys,xs=np.nonzero(mask)
			area=int(len(xs))
			if area<=0:
				continue
			centroid_x=bounds.x+float(xs.mean())
			centroid_y=bounds.y+float(ys.mean())
			touches_image_edge=int(
				bounds.x==0 or bounds.y==0 or bounds.x1==image_width or bounds.y1==image_height
			)
			owned_count=sum(int(item.owned_by_core)for item in members)
			with connection:
				connection.execute(
					'''
                    INSERT INTO global_cells(
                        global_cell_id, representative_prediction_id, class_id, class_name,
                        score, area, centroid_x, centroid_y, x0, y0, x1, y1,
                        source_count, owned_source_count, touches_image_edge, mask_strategy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
					(
						global_id,
						representative.prediction_id,
						representative.class_id,
						representative.class_name,
						representative.score,
						area,
						centroid_x,
						centroid_y,
						bounds.x,
						bounds.y,
						bounds.x1,
						bounds.y1,
						len(members),
						owned_count,
						touches_image_edge,
						config.mask_strategy,
					),
				)
				connection.execute(
					'INSERT INTO global_cells_rtree(global_cell_id, min_x, max_x, min_y, max_y) '
					'VALUES (?, ?, ?, ?, ?)',
					(global_id,bounds.x,bounds.x1,bounds.y,bounds.y1),
				)
				connection.executemany(
					'INSERT INTO cell_sources(global_cell_id, prediction_id, is_representative) '
					'VALUES (?, ?, ?)',
					[
						(global_id,item.prediction_id,int(item.prediction_id==representative.prediction_id))
						for item in members
					],
				)
			if global_id==total_groups or global_id%100==0:
				self._emit(
					on_progress,
					'grouping',
					global_id,
					total_groups,
					f'Assigned {global_id:,}/{total_groups:,} global cell IDs',
				)
		database.set_metadata('phase','rendering')
		return int(connection.execute('SELECT COUNT(*) FROM global_cells').fetchone()[0])


	def _sources_for_cell(
		self,
		connection:sqlite3.Connection,
		global_cell_id:int,
		strategy:str,
	)->list[_PredictionReference]:
		if strategy=='best':
			rows=connection.execute(
				'''
                SELECT p.* FROM cell_sources s
                JOIN predictions p ON p.prediction_id=s.prediction_id
                WHERE s.global_cell_id=? AND s.is_representative=1
                ''',
				(global_cell_id,),
			).fetchall()
		else:
			rows=connection.execute(
				'''
                SELECT p.* FROM cell_sources s
                JOIN predictions p ON p.prediction_id=s.prediction_id
                WHERE s.global_cell_id=? ORDER BY p.prediction_id
                ''',
				(global_cell_id,),
			).fetchall()
		return[_row_to_prediction(row)for row in rows]


	@staticmethod
	def _write_label_failure_report(checkpoint:TileCheckpointStore,path:Path)->None:
		rows=checkpoint.failed_rows()
		if not rows:
			path.unlink(missing_ok=True)
			return
		temporary=path.with_suffix(path.suffix+'.tmp')
		with temporary.open('w',newline='',encoding='utf-8')as handle:
			writer=csv.DictWriter(
				handle,
				fieldnames=['tile_id','tile_index','tile_row','tile_column','attempts','error'],
			)
			writer.writeheader()
			writer.writerows(rows)
		os.replace(temporary,path)


	def _render_labels(
		self,
		database:ReconciliationDatabase,
		label_store:ChunkedInstanceLabelStore,
		config:ReconciliationConfig,
		*,
		output_directory:Path,
		context:Mapping[str,Any],
		cancel_event:threading.Event,
		on_progress:Callable[[ReconciliationProgress],None]|None,
	)->tuple[int,int]:
		connection=database.connection
		grid=label_store.grid()
		checkpoint_path=output_directory/'label_chunks.sqlite'
		cache=_ArchiveCache(config.archive_cache_size)
		total_conflicts=0
		with TileCheckpointStore(
			checkpoint_path,
			grid,
			job_name='MPlexA global instance-label rendering',
			context=context,
			reset_interrupted=True,
		)as checkpoint:
			failure_report=output_directory/'label_chunk_failures.csv'
			if config.retry_failed_chunks:
				checkpoint.reset_failed()
			for tile in checkpoint.iter_tiles(('completed',)):
				status=checkpoint.status(tile.tile_id)
				output=status.get('output')or{}
				total_conflicts+=int(output.get('conflict_pixels',0))


			def process_pending_pass()->None:
				nonlocal total_conflicts
				while True:
					self._check_cancel(cancel_event)
					tile=checkpoint.claim_next()
					if tile is None:
						break
					bounds=label_store.chunk_bounds(tile.row,tile.column)
					try:
						rows=connection.execute(
							'''
                            SELECT c.* FROM global_cells_rtree r
                            JOIN global_cells c ON c.global_cell_id=r.global_cell_id
                            WHERE r.min_x<? AND r.max_x>? AND r.min_y<? AND r.max_y>?
                            ORDER BY c.score DESC, c.global_cell_id ASC
                            ''',
							(bounds.x1,bounds.x,bounds.y1,bounds.y),
						).fetchall()
						chunk=np.zeros((bounds.height,bounds.width),dtype=label_store.dtype)
						conflict_pixels=0
						painted_cells=0
						for cell_row in rows:
							global_id=int(cell_row['global_cell_id'])
							sources=self._sources_for_cell(connection,global_id,config.mask_strategy)
							cell_bounds,mask,_=self._group_mask(sources,config.mask_strategy,cache)
							intersection=bounds.intersection(cell_bounds)
							if intersection is None:
								continue
							source_y0=intersection.y-cell_bounds.y
							source_x0=intersection.x-cell_bounds.x
							target_y0=intersection.y-bounds.y
							target_x0=intersection.x-bounds.x
							piece=mask[
								source_y0:source_y0+intersection.height,
								source_x0:source_x0+intersection.width,
							]
							target=chunk[
								target_y0:target_y0+intersection.height,
								target_x0:target_x0+intersection.width,
							]
							conflicts=piece&(target!=0)
							conflict_pixels+=int(np.count_nonzero(conflicts))
							writable=piece&(target==0)
							if np.any(writable):
								target[writable]=global_id
								painted_cells+=1
						row=tile.row
						column=tile.column
						label_store.write_chunk(row,column,chunk)
						output={
							'chunk_path':str(label_store.chunk_path(row,column)),
							'nonzero_pixels':int(np.count_nonzero(chunk)),
							'painted_cells':painted_cells,
							'conflict_pixels':conflict_pixels,
						}
						checkpoint.mark_completed(tile.tile_id,output)
						total_conflicts+=conflict_pixels
					except Exception as error:
						checkpoint.mark_failed(tile.tile_id,error)
						self._emit(
							on_progress,
							'rendering',
							checkpoint.progress().completed,
							checkpoint.progress().total,
							f'Label chunk {tile.row},{tile.column} failed: {type(error).__name__}: {error}',
						)
					progress=checkpoint.progress()
					self._emit(
						on_progress,
						'rendering',
						progress.completed,
						progress.total,
						progress.summary(),
					)
			process_pending_pass()
			first_pass=checkpoint.progress()
			if config.retry_failed_chunks and first_pass.failed:
				checkpoint.reset_failed()
				self._emit(
					on_progress,
					'rendering',
					first_pass.completed,
					first_pass.total,
					f'Retrying {first_pass.failed:,} failed global-label chunk(s) once...',
				)
				process_pending_pass()
			final=checkpoint.progress()
			if final.failed:
				self._write_label_failure_report(checkpoint,failure_report)
				groups=checkpoint.failed_error_groups(limit=3)
				details='; '.join(
					f'{count} chunk(s): {message}' for message,count in groups
				)
				raise ReconciliationError(
					f'Global-label rendering has {final.failed:,} failed chunk(s). '
					f'Most common error(s): {details}. '
					f'Full failure report: {failure_report}'
				)
			failure_report.unlink(missing_ok=True)
		return final.completed,total_conflicts


	def _export_cells_csv(self,database:ReconciliationDatabase,path:Path)->None:
		fields=[
			'global_cell_id','class_id','class_name','score','area',
			'centroid_x','centroid_y','x0','y0','x1','y1',
			'source_count','owned_source_count','touches_image_edge',
			'representative_prediction_id','mask_strategy',
		]
		temporary=path.with_suffix(path.suffix+'.tmp')
		with temporary.open('w',newline='',encoding='utf-8')as handle:
			writer=csv.DictWriter(handle,fieldnames=fields)
			writer.writeheader()
			for row in database.connection.execute(
				'SELECT '+', '.join(fields)+' FROM global_cells ORDER BY global_cell_id'
			):
				writer.writerow({field:row[field]for field in fields})
		os.replace(temporary,path)


	def run(
		self,
		*,
		output_directory:str|Path|None=None,
		config:ReconciliationConfig|None=None,
		cancel_event:threading.Event|None=None,
		on_progress:Callable[[ReconciliationProgress],None]|None=None,
		on_log:Callable[[str],None]|None=None,
	)->ReconciliationRunSummary:
		config=config or ReconciliationConfig()
		cancel_event=cancel_event or threading.Event()
		started_at=_utc_now()
		destination=(
			Path(output_directory).expanduser().resolve()
			if output_directory is not None
			else self.segmentation_directory/'global_instances'
		)
		destination.mkdir(parents=True,exist_ok=True)
		database_path=destination/'reconciliation.sqlite'
		label_store_path=destination/GLOBAL_LABEL_STORE_NAME
		fingerprint=_segmentation_fingerprint(
			self.segmentation_directory,self.configuration,self.archive_paths
		)
		result_config={
			'iou_threshold':float(config.iou_threshold),
			'containment_threshold':float(config.containment_threshold),
			'same_class_only':bool(config.same_class_only),
			'mask_strategy':config.mask_strategy,
			'chunk_size':int(config.chunk_size),
		}
		context={
			'reconciliation_schema_version':RECONCILIATION_SCHEMA_VERSION,
			'segmentation_directory':str(self.segmentation_directory),
			'segmentation_fingerprint':fingerprint,
			'segmentation_grid_signature':self.segmentation_grid.signature,
			'result_config':result_config,
		}
		cancelled=False
		prediction_count=duplicate_links=global_cells=label_chunks=conflict_pixels=0


		def log(message:str)->None:
			if on_log is not None:
				on_log(message)
		try:
			with ReconciliationDatabase(database_path,context=context)as database:
				log('Indexing compact tile predictions...')
				prediction_count=self._index_predictions(
					database,cancel_event=cancel_event,on_progress=on_progress
				)
				log('Matching duplicate masks across overlapping tiles...')
				duplicate_links=self._match_duplicates(
					database,config,cancel_event=cancel_event,on_progress=on_progress
				)
				log('Assigning deterministic global cell IDs...')
				global_cells=self._build_global_cells(
					database,config,cancel_event=cancel_event,on_progress=on_progress
				)
				dtype=np.uint32 if global_cells<=np.iinfo(np.uint32).max else np.uint64
				label_store=ChunkedInstanceLabelStore.create(
					label_store_path,
					width=self.segmentation_grid.image_width,
					height=self.segmentation_grid.image_height,
					chunk_size=config.chunk_size,
					dtype=dtype,
					level=self.segmentation_grid.level,
					global_cell_count=global_cells,
				)
				log('Rendering chunked global instance labels...')
				label_context={
					'reconciliation_context':context,
					'global_cell_count':global_cells,
					'label_dtype':np.dtype(dtype).name,
					'chunk_size':config.chunk_size,
				}
				label_chunks,conflict_pixels=self._render_labels(
					database,
					label_store,
					config,
					output_directory=destination,
					context=label_context,
					cancel_event=cancel_event,
					on_progress=on_progress,
				)
				self._export_cells_csv(database,destination/'global_cells.csv')
				database.set_metadata('phase','completed')
		except ReconciliationCancelled:
			cancelled=True
			log('Module 3 cancellation requested; completed work is preserved for resume.')
		finished_at=_utc_now()
		summary=ReconciliationRunSummary(
			segmentation_directory=str(self.segmentation_directory),
			output_directory=str(destination),
			database_path=str(database_path),
			label_store_path=str(label_store_path),
			prediction_count=prediction_count,
			duplicate_link_count=duplicate_links,
			global_cell_count=global_cells,
			label_chunks=label_chunks,
			conflict_pixels=conflict_pixels,
			cancelled=cancelled,
			started_at=started_at,
			finished_at=finished_at,
		)
		ChunkedInstanceLabelStore._write_json_atomic(
			destination/'reconciliation_summary.json',asdict(summary)
		)
		return summary
