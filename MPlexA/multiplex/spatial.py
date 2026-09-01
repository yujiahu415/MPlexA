from __future__ import annotations
from dataclasses import asdict,dataclass
import csv
import json
import math
from pathlib import Path
import sqlite3
from typing import Any,Callable,Mapping
import numpy as np
from scipy.spatial import Delaunay,cKDTree
from.exceptions import MultiplexImageError
from.reconciliation import ChunkedInstanceLabelStore
SPATIAL_GRAPH_SCHEMA_VERSION=1



class SpatialGraphError(MultiplexImageError):



class SpatialGraphCancelled(SpatialGraphError):



@dataclass(frozen=True,slots=True)
class SpatialGraphConfig:
	method:str='radius'
	radius:float=30.0
	k_neighbors:int=6
	use_physical_units:bool=True
	max_delaunay_cells:int=300_000
	query_block_size:int=5000


	def __post_init__(self)->None:
		if self.method not in{'radius','knn','delaunay','contact'}:
			raise SpatialGraphError('Graph method must be radius, knn, delaunay, or contact.')
		if float(self.radius)<=0:
			raise SpatialGraphError('Radius must be positive.')
		if int(self.k_neighbors)<=0:
			raise SpatialGraphError('k-nearest-neighbor count must be positive.')
		if int(self.max_delaunay_cells)<=2:
			raise SpatialGraphError('Delaunay cell limit must exceed two.')
		if int(self.query_block_size)<=0:
			raise SpatialGraphError('Spatial query block size must be positive.')


	def to_dict(self)->dict[str,Any]:
		return asdict(self)



@dataclass(frozen=True,slots=True)
class SpatialGraphProgress:
	stage:str
	completed:int
	total:int
	message:str=''


	@property
	def fraction(self)->float:
		return 0.0 if self.total<=0 else min(1.0,max(0.0,self.completed/self.total))



@dataclass(frozen=True,slots=True)
class SpatialGraphRunSummary:
	output_directory:str
	edges_csv:str
	interaction_counts_csv:str
	interaction_enrichment_csv:str
	cell_count:int
	edge_count:int
	cluster_count:int
	method:str


	def summary(self)->str:
		return(
			f'Cells: {self.cell_count:,}\n'
			f'Edges: {self.edge_count:,}\n'
			f'Phenotypes/clusters: {self.cluster_count:,}\n'
			f'Graph: {self.method}\n'
			f'Edges: {self.edges_csv}\n'
			f'Interaction counts: {self.interaction_counts_csv}\n'
			f'Interaction enrichment: {self.interaction_enrichment_csv}\n'
			f'Output: {self.output_directory}'
		)



class SpatialGraphBuilder:


	def __init__(self,clustering_directory:str|Path)->None:
		self.clustering_directory=Path(clustering_directory).expanduser().resolve()
		self.clustering_db=self.clustering_directory/'clustering.sqlite'
		if not self.clustering_db.is_file():
			raise SpatialGraphError(f'Clustering database not found: {self.clustering_db}')


	@staticmethod
	def _emit(
		callback:Callable[[SpatialGraphProgress],None]|None,
		stage:str,
		completed:int,
		total:int,
		message:str,
	)->None:
		if callback is not None:
			callback(SpatialGraphProgress(stage,completed,total,message))


	def _load_cells(self)->tuple[np.ndarray,np.ndarray,np.ndarray,list[str]]:
		with sqlite3.connect(self.clustering_db)as conn:
			rows=conn.execute(
				'SELECT global_cell_id, centroid_x, centroid_y, cluster_id, cluster_name '
				'FROM cells ORDER BY global_cell_id'
			).fetchall()
		if not rows:
			raise SpatialGraphError('The clustering database does not contain any cells.')
		ids=np.asarray([row[0]for row in rows],dtype=np.int64)
		coords=np.asarray([[row[1],row[2]]for row in rows],dtype=np.float64)
		clusters=np.asarray([row[3]for row in rows],dtype=np.int32)
		names_by_id:dict[int,str]={}
		for row in rows:
			names_by_id[int(row[3])]=str(row[4])
		max_cluster=int(max(names_by_id))
		names=[names_by_id.get(index, f'Cluster {index}')for index in range(1,max_cluster+1)]
		return ids,coords,clusters,names


	@staticmethod
	def _create_edge_database(path:Path)->sqlite3.Connection:
		if path.exists():
			path.unlink()
		conn=sqlite3.connect(path)
		conn.execute('PRAGMA journal_mode=WAL')
		conn.execute('PRAGMA synchronous=NORMAL')
		conn.execute(
			'''CREATE TABLE edges(
                source_cell_id INTEGER NOT NULL,
                target_cell_id INTEGER NOT NULL,
                distance_px REAL,
                PRIMARY KEY(source_cell_id, target_cell_id)
            ) WITHOUT ROWID'''
		)
		return conn


	@staticmethod
	def _insert_edges(conn:sqlite3.Connection,rows:list[tuple[int,int,float|None]])->None:
		if not rows:
			return
		with conn:
			conn.executemany(
				'INSERT OR IGNORE INTO edges(source_cell_id,target_cell_id,distance_px) VALUES(?,?,?)',
				rows,
			)


	def _build_radius_or_knn(
		self,
		conn:sqlite3.Connection,
		ids:np.ndarray,
		coords:np.ndarray,
		config:SpatialGraphConfig,
		cancel_event:Any|None,
		progress_callback:Callable[[SpatialGraphProgress],None]|None,
	)->None:
		tree=cKDTree(coords)
		total=len(ids)
		block=int(config.query_block_size)
		for start in range(0,total,block):
			if cancel_event is not None and cancel_event.is_set():
				raise SpatialGraphCancelled('Spatial graph construction cancelled.')
			stop=min(total,start+block)
			rows:list[tuple[int,int,float|None]]=[]
			if config.method=='radius':
				neighbors=tree.query_ball_point(coords[start:stop],r=float(config.radius))
				for offset,linked in enumerate(neighbors):
					source_index=start+offset
					for target_index in linked:
						target_index=int(target_index)
						if target_index<=source_index:
							continue
						distance=float(np.linalg.norm(coords[source_index]-coords[target_index]))
						rows.append((int(ids[source_index]),int(ids[target_index]),distance))
			else:
				k=min(int(config.k_neighbors)+1,total)
				distances,neighbors=tree.query(coords[start:stop],k=k)
				if k==1:
					distances=distances[:,None]
					neighbors=neighbors[:,None]
				for offset in range(stop-start):
					source_index=start+offset
					for position in range(1,neighbors.shape[1]):
						target_index=int(neighbors[offset,position])
						if target_index==source_index:
							continue
						first,second=sorted((int(ids[source_index]),int(ids[target_index])))
						rows.append((first,second,float(distances[offset,position])))
			self._insert_edges(conn,rows)
			self._emit(progress_callback,'graph',stop,total,
						f'Building {config.method} graph: {stop:,}/{total:,} cells')


	def _build_delaunay(
		self,
		conn:sqlite3.Connection,
		ids:np.ndarray,
		coords:np.ndarray,
		config:SpatialGraphConfig,
		cancel_event:Any|None,
		progress_callback:Callable[[SpatialGraphProgress],None]|None,
	)->None:
		if len(ids)>int(config.max_delaunay_cells):
			raise SpatialGraphError(
				f'Delaunay graph is limited to {config.max_delaunay_cells:,} cells; '
				f'this dataset contains {len(ids):,}. Use radius or kNN instead.'
			)
		triangulation=Delaunay(coords)
		edge_set:set[tuple[int,int]]=set()
		simplices=triangulation.simplices
		for index,triangle in enumerate(simplices):
			if cancel_event is not None and cancel_event.is_set():
				raise SpatialGraphCancelled('Spatial graph construction cancelled.')
			a,b,c=[int(value)for value in triangle]
			edge_set.add(tuple(sorted((a,b))))
			edge_set.add(tuple(sorted((a,c))))
			edge_set.add(tuple(sorted((b,c))))
			if index%20000==0:
				self._emit(progress_callback,'graph',index,len(simplices),
							f'Building Delaunay graph: {index:,}/{len(simplices):,} triangles')
		rows=[]
		for a,b in sorted(edge_set):
			rows.append((int(ids[a]),int(ids[b]),float(np.linalg.norm(coords[a]-coords[b]))))
			if len(rows)>=100_000:
				self._insert_edges(conn,rows)
				rows=[]
		self._insert_edges(conn,rows)
		self._emit(progress_callback,'graph',len(simplices),len(simplices),'Delaunay graph complete')


	def _build_contact(
		self,
		conn:sqlite3.Connection,
		label_store_path:str|Path,
		coords_by_id:Mapping[int,tuple[float,float]],
		cancel_event:Any|None,
		progress_callback:Callable[[SpatialGraphProgress],None]|None,
	)->None:
		store=ChunkedInstanceLabelStore.open(label_store_path)
		total=store.rows*store.columns
		completed=0
		for row in range(store.rows):
			for column in range(store.columns):
				if cancel_event is not None and cancel_event.is_set():
					raise SpatialGraphCancelled('Spatial graph construction cancelled.')
				bounds=store.chunk_bounds(row,column)
				width=bounds.width+(1 if bounds.x1<store.metadata.width else 0)
				height=bounds.height+(1 if bounds.y1<store.metadata.height else 0)
				labels=store.read_region(x=bounds.x,y=bounds.y,width=width,height=height)
				pairs:set[tuple[int,int]]=set()
				horizontal=np.column_stack((labels[:,:-1].ravel(),labels[:,1:].ravel()))
				vertical=np.column_stack((labels[:-1,:].ravel(),labels[1:,:].ravel()))
				for pair_array in(horizontal,vertical):
					mask=(pair_array[:,0]>0)&(pair_array[:,1]>0)&(pair_array[:,0]!=pair_array[:,1])
					for first,second in pair_array[mask]:
						pairs.add(tuple(sorted((int(first),int(second)))))
				rows_to_insert=[]
				for first,second in pairs:
					first_xy=coords_by_id.get(first)
					second_xy=coords_by_id.get(second)
					distance=None
					if first_xy is not None and second_xy is not None:
						distance=float(np.linalg.norm(np.asarray(first_xy)-np.asarray(second_xy)))
					rows_to_insert.append((first,second,distance))
				self._insert_edges(conn,rows_to_insert)
				completed+=1
				self._emit(progress_callback,'graph',completed,total,
							f'Scanning contact labels: {completed:,}/{total:,} chunks')


	@staticmethod
	def _write_outputs(
		conn:sqlite3.Connection,
		output:Path,
		cluster_by_id:Mapping[int,int],
		cluster_names:list[str],
		pixel_size:float|None,
	)->tuple[Path,Path,Path,int]:
		edges_csv=output/'spatial_edges.csv'
		counts_csv=output/'interaction_counts.csv'
		enrichment_csv=output/'interaction_enrichment.csv'
		counts:dict[tuple[int,int],int]={}
		degree_sums=np.zeros(len(cluster_names)+1,dtype=np.int64)
		edge_count=0
		with edges_csv.open('w',newline='',encoding='utf-8')as handle:
			writer=csv.writer(handle)
			writer.writerow([
				'source_cell_id','target_cell_id','distance_px','distance_physical',
				'source_cluster_id','source_cluster_name','target_cluster_id','target_cluster_name',
			])
			cursor=conn.execute(
				'SELECT source_cell_id,target_cell_id,distance_px FROM edges ORDER BY source_cell_id,target_cell_id'
			)
			for source,target,distance in cursor:
				source_cluster=int(cluster_by_id.get(int(source),0))
				target_cluster=int(cluster_by_id.get(int(target),0))
				source_name=cluster_names[source_cluster-1]if source_cluster>0 else'Unassigned'
				target_name=cluster_names[target_cluster-1]if target_cluster>0 else'Unassigned'
				physical=None if distance is None or pixel_size is None else float(distance)*float(pixel_size)
				writer.writerow([
					int(source),int(target),''if distance is None else float(distance),
					''if physical is None else physical,
					source_cluster,source_name,target_cluster,target_name,
				])
				if source_cluster>0 and target_cluster>0:
					pair=tuple(sorted((source_cluster,target_cluster)))
					counts[pair]=counts.get(pair,0)+1
					degree_sums[source_cluster]+=1
					degree_sums[target_cluster]+=1
				edge_count+=1
		with counts_csv.open('w',newline='',encoding='utf-8')as handle:
			writer=csv.writer(handle)
			writer.writerow(['cluster_a_id','cluster_a','cluster_b_id','cluster_b','observed_edges'])
			for a in range(1,len(cluster_names)+1):
				for b in range(a,len(cluster_names)+1):
					writer.writerow([a,cluster_names[a-1],b,cluster_names[b-1],counts.get((a,b),0)])
		total_stubs=int(np.sum(degree_sums))
		with enrichment_csv.open('w',newline='',encoding='utf-8')as handle:
			writer=csv.writer(handle)
			writer.writerow([
				'cluster_a_id','cluster_a','cluster_b_id','cluster_b',
				'observed_edges','expected_edges','observed_expected_ratio',
			])
			denominator=max(1,total_stubs-1)
			for a in range(1,len(cluster_names)+1):
				for b in range(a,len(cluster_names)+1):
					observed=counts.get((a,b),0)
					if a==b:
						expected=float(degree_sums[a]*max(0,degree_sums[a]-1)/(2.0*denominator))
					else:
						expected=float(degree_sums[a]*degree_sums[b]/denominator)
					ratio=float(observed/expected)if expected>0 else math.nan
					writer.writerow([a,cluster_names[a-1],b,cluster_names[b-1],observed,expected,ratio])
		return edges_csv,counts_csv,enrichment_csv,edge_count


	def run(
		self,
		output_directory:str|Path,
		config:SpatialGraphConfig,
		*,
		label_store_path:str|Path|None=None,
		pixel_size:float|None=None,
		cancel_event:Any|None=None,
		progress_callback:Callable[[SpatialGraphProgress],None]|None=None,
	)->SpatialGraphRunSummary:
		output=Path(output_directory).expanduser().resolve()
		output.mkdir(parents=True,exist_ok=True)
		ids,coords,clusters,cluster_names=self._load_cells()
		graph_radius=float(config.radius)
		if config.use_physical_units:
			if pixel_size is None or float(pixel_size)<=0:
				raise SpatialGraphError(
					'Physical-unit graph distances require a positive image pixel size. '
					'Disable physical units or provide pixel size.'
				)
			graph_radius=float(config.radius)/float(pixel_size)
		effective=SpatialGraphConfig(
			method=config.method,
			radius=graph_radius,
			k_neighbors=config.k_neighbors,
			use_physical_units=False,
			max_delaunay_cells=config.max_delaunay_cells,
			query_block_size=config.query_block_size,
		)
		edge_db=output/'spatial_graph.sqlite'
		conn=self._create_edge_database(edge_db)
		try:
			if config.method in{'radius','knn'}:
				self._build_radius_or_knn(conn,ids,coords,effective,cancel_event,progress_callback)
			elif config.method=='delaunay':
				self._build_delaunay(conn,ids,coords,effective,cancel_event,progress_callback)
			else:
				if label_store_path is None:
					raise SpatialGraphError('Direct-contact graph requires a cell-region label store.')
				coords_by_id={int(cell_id):(float(x),float(y))for cell_id,(x,y)in zip(ids,coords)}
				self._build_contact(conn,label_store_path,coords_by_id,cancel_event,progress_callback)
			cluster_by_id={int(cell_id):int(cluster)for cell_id,cluster in zip(ids,clusters)}
			edges_csv,counts_csv,enrichment_csv,edge_count=self._write_outputs(
				conn,output,cluster_by_id,cluster_names,pixel_size
			)
			(output/'spatial_graph_config.json').write_text(
				json.dumps({
					'schema_version':SPATIAL_GRAPH_SCHEMA_VERSION,
					'config':config.to_dict(),
					'effective_radius_px':graph_radius,
					'pixel_size':pixel_size,
					'clustering_directory':str(self.clustering_directory),
					'label_store_path':None if label_store_path is None else str(Path(label_store_path).resolve()),
				},indent=2),
				encoding='utf-8',
			)
		finally:
			conn.close()
		return SpatialGraphRunSummary(
			output_directory=str(output),
			edges_csv=str(edges_csv),
			interaction_counts_csv=str(counts_csv),
			interaction_enrichment_csv=str(enrichment_csv),
			cell_count=int(len(ids)),
			edge_count=int(edge_count),
			cluster_count=len(cluster_names),
			method=config.method,
		)
__all__=[
	'SpatialGraphBuilder',
	'SpatialGraphCancelled',
	'SpatialGraphConfig',
	'SpatialGraphError',
	'SpatialGraphProgress',
	'SpatialGraphRunSummary',
]
