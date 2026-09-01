from __future__ import annotations
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import csv
import json
import math
from pathlib import Path
import sqlite3
from typing import Any,Callable,Iterable,Mapping,Sequence
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree
from.exceptions import MultiplexImageError
PHENOTYPING_SCHEMA_VERSION=1



class PhenotypingError(MultiplexImageError):



class PhenotypingCancelled(PhenotypingError):



@dataclass(frozen=True,slots=True)
class PhenotypingConfig:
	feature_columns:tuple[str,...]
	transform:str='arcsinh'
	arcsinh_cofactor:float=5.0
	winsor_low:float=0.5
	winsor_high:float=99.5
	n_pcs:int=20
	method:str='kmeans'
	n_clusters:int=12
	leiden_neighbors:int=15
	leiden_resolution:float=1.0
	embedding:str='pca'
	sample_size:int=100_000
	chunk_rows:int=50_000
	random_seed:int=13


	def __post_init__(self)->None:
		if not self.feature_columns:
			raise PhenotypingError('Select at least one marker feature for clustering.')
		if self.transform not in{'none','arcsinh','signed_log1p'}:
			raise PhenotypingError('Transform must be none, arcsinh, or signed_log1p.')
		if self.transform=='arcsinh'and float(self.arcsinh_cofactor)<=0:
			raise PhenotypingError('Arcsinh cofactor must be positive.')
		if not 0<=float(self.winsor_low)<float(self.winsor_high)<=100:
			raise PhenotypingError('Winsor percentiles must satisfy 0 <= low < high <= 100.')
		if int(self.n_pcs)<=0:
			raise PhenotypingError('Number of principal components must be positive.')
		if self.method not in{'kmeans','leiden'}:
			raise PhenotypingError('Clustering method must be kmeans or leiden.')
		if int(self.n_clusters)<2:
			raise PhenotypingError('K-means requires at least two clusters.')
		if int(self.leiden_neighbors)<2:
			raise PhenotypingError('Leiden neighbor count must be at least two.')
		if float(self.leiden_resolution)<=0:
			raise PhenotypingError('Leiden resolution must be positive.')
		if self.embedding not in{'pca','umap'}:
			raise PhenotypingError('Embedding must be pca or umap.')
		if int(self.sample_size)<100:
			raise PhenotypingError('Clustering sample size must be at least 100 cells.')
		if int(self.chunk_rows)<=0:
			raise PhenotypingError('CSV streaming chunk size must be positive.')


	def to_dict(self)->dict[str,Any]:
		return asdict(self)



@dataclass(frozen=True,slots=True)
class PhenotypingProgress:
	stage:str
	completed:int
	total:int
	message:str=''


	@property
	def fraction(self)->float:
		return 0.0 if self.total<=0 else min(1.0,max(0.0,self.completed/self.total))



@dataclass(frozen=True,slots=True)
class PhenotypingRunSummary:
	output_directory:str
	assignments_csv:str
	cluster_summary_csv:str
	sqlite_path:str
	cell_count:int
	cluster_count:int
	feature_count:int
	embedding:str
	cancelled:bool


	def summary(self)->str:
		return(
			f'Cells clustered: {self.cell_count:,}\n'
			f'Clusters: {self.cluster_count:,}\n'
			f'Marker features: {self.feature_count:,}\n'
			f'Embedding: {self.embedding}\n'
			f'Cancelled: {self.cancelled}\n'
			f'Assignments: {self.assignments_csv}\n'
			f'Output: {self.output_directory}'
		)


def discover_marker_features(csv_path:str|Path)->tuple[str,...]:
	path=Path(csv_path).expanduser().resolve()
	if not path.is_file():
		raise PhenotypingError(f'Marker table not found: {path}')
	columns=list(pd.read_csv(path,nrows=0).columns)
	ignored_prefixes={
		'global_cell_id','class_id','class_name','score','area',
		'cell_area','nuclear_area','cytoplasmic_area','centroid_x',
		'centroid_y','x0','y0','x1','y1','source_count',
		'owned_source_count','touches_image_edge','representative_prediction_id',
		'mask_strategy',
	}
	return tuple(
		column for column in columns
		if'__'in column and column not in ignored_prefixes
	)


def feature_columns_for_metric(columns:Sequence[str],metric:str)->tuple[str,...]:
	suffix='__'+str(metric)
	return tuple(column for column in columns if column.endswith(suffix))



class CellPhenotyper:
	ID_COLUMNS=('global_cell_id','centroid_x','centroid_y')


	def __init__(self,marker_csv:str|Path)->None:
		self.marker_csv=Path(marker_csv).expanduser().resolve()
		if not self.marker_csv.is_file():
			raise PhenotypingError(f'Marker table not found: {self.marker_csv}')
		self.columns=tuple(pd.read_csv(self.marker_csv,nrows=0).columns)
		missing=[name for name in self.ID_COLUMNS if name not in self.columns]
		if missing:
			raise PhenotypingError(
				'Marker table is missing required columns: '+', '.join(missing)
			)


	@staticmethod
	def _emit(
		callback:Callable[[PhenotypingProgress],None]|None,
		stage:str,
		completed:int,
		total:int,
		message:str='',
	)->None:
		if callback is not None:
			callback(PhenotypingProgress(stage,completed,total,message))


	def _row_count(self)->int:
		with self.marker_csv.open('rb')as handle:
			count=sum(block.count(b'\n')for block in iter(lambda:handle.read(8*1024*1024),b''))
		return max(0,count-1)


	@staticmethod
	def _transform(values:np.ndarray,config:PhenotypingConfig)->np.ndarray:
		values=np.asarray(values,dtype=np.float64)
		if config.transform=='arcsinh':
			return np.arcsinh(values/float(config.arcsinh_cofactor))
		if config.transform=='signed_log1p':
			return np.sign(values)*np.log1p(np.abs(values))
		return values


	def _priority_sample(
		self,
		config:PhenotypingConfig,
		cancel_event:Any|None,
		progress_callback:Callable[[PhenotypingProgress],None]|None,
		total_rows:int,
	)->tuple[np.ndarray,np.ndarray]:
		rng=np.random.default_rng(int(config.random_seed))
		keep_values:np.ndarray|None=None
		keep_keys:np.ndarray|None=None
		seen=0
		usecols=list(config.feature_columns)
		for chunk in pd.read_csv(
			self.marker_csv,
			usecols=usecols,
			chunksize=int(config.chunk_rows),
		):
			if cancel_event is not None and cancel_event.is_set():
				raise PhenotypingCancelled('Clustering cancelled while sampling cells.')
			values=chunk.to_numpy(dtype=np.float64,copy=True)
			keys=rng.random(values.shape[0])
			if keep_values is None:
				keep_values=values
				keep_keys=keys
			else:
				keep_values=np.vstack((keep_values,values))
				keep_keys=np.concatenate((keep_keys,keys))
			target=int(config.sample_size)
			if keep_values.shape[0]>max(target*2,target+int(config.chunk_rows)):
				indices=np.argpartition(keep_keys,target-1)[:target]
				keep_values=keep_values[indices]
				keep_keys=keep_keys[indices]
			seen+=values.shape[0]
			self._emit(progress_callback,'sampling',min(seen,total_rows),total_rows,
						f'Sampling marker profiles: {seen:,}/{total_rows:,}')
		if keep_values is None or keep_values.shape[0]<2:
			raise PhenotypingError('The marker table does not contain enough cells to cluster.')
		target=min(int(config.sample_size),keep_values.shape[0])
		if keep_values.shape[0]>target:
			indices=np.argpartition(keep_keys,target-1)[:target]
			keep_values=keep_values[indices]
			keep_keys=keep_keys[indices]
		return keep_values,keep_keys


	@staticmethod
	def _fit_preprocessing(
		sample:np.ndarray,
		config:PhenotypingConfig,
	)->tuple[np.ndarray,dict[str,np.ndarray]]:
		transformed=CellPhenotyper._transform(sample,config)
		finite=np.isfinite(transformed)
		medians=np.zeros(transformed.shape[1],dtype=np.float64)
		for index in range(transformed.shape[1]):
			valid=transformed[finite[:,index],index]
			medians[index]=float(np.median(valid))if valid.size else 0.0
			transformed[~finite[:,index],index]=medians[index]
		lows=np.percentile(transformed,float(config.winsor_low),axis=0)
		highs=np.percentile(transformed,float(config.winsor_high),axis=0)
		transformed=np.clip(transformed,lows,highs)
		means=np.mean(transformed,axis=0)
		stds=np.std(transformed,axis=0)
		stds[stds<1e-12]=1.0
		standardized=(transformed-means)/stds
		return standardized,{
			'medians':medians,
			'lows':lows,
			'highs':highs,
			'means':means,
			'stds':stds,
		}


	@staticmethod
	def _apply_preprocessing(
		values:np.ndarray,
		config:PhenotypingConfig,
		parameters:Mapping[str,np.ndarray],
	)->np.ndarray:
		transformed=CellPhenotyper._transform(values,config)
		medians=np.asarray(parameters['medians'])
		for index in range(transformed.shape[1]):
			bad=~np.isfinite(transformed[:,index])
			if np.any(bad):
				transformed[bad,index]=medians[index]
		transformed=np.clip(transformed,parameters['lows'],parameters['highs'])
		return(transformed-parameters['means'])/parameters['stds']


	@staticmethod
	def _fit_pca(sample:np.ndarray,n_pcs:int)->tuple[np.ndarray,np.ndarray]:
		center=np.mean(sample,axis=0)
		centered=sample-center
		_,_,vt=np.linalg.svd(centered,full_matrices=False)
		count=max(1,min(int(n_pcs),vt.shape[0],sample.shape[0]-1))
		components=vt[:count]
		return center,components


	@staticmethod
	def _nearest_centroid(points:np.ndarray,centroids:np.ndarray)->np.ndarray:
		distances=(
			np.sum(points*points,axis=1,keepdims=True)
			-2.0*points@centroids.T
			+np.sum(centroids*centroids,axis=1)[None,:]
		)
		return np.argmin(distances,axis=1).astype(np.int32)


	@staticmethod
	def _fit_clusters(
		sample_pca:np.ndarray,
		config:PhenotypingConfig,
	)->tuple[np.ndarray,np.ndarray]:
		if config.method=='kmeans':
			count=min(int(config.n_clusters),sample_pca.shape[0])
			centroids,labels=kmeans2(
				sample_pca,
				count,
				minit='++',
				iter=50,
				seed=int(config.random_seed),
			)
			return np.asarray(centroids,dtype=np.float64),np.asarray(labels,dtype=np.int32)
		try:
			import igraph as ig# type: ignore
			import leidenalg# type: ignore
		except Exception as error:
			raise PhenotypingError(
				'Leiden clustering requires optional packages \'python-igraph\' and '
				'\'leidenalg\'. Install them or select K-means.'
			)from error
		neighbors=min(int(config.leiden_neighbors),max(2,sample_pca.shape[0]-1))
		tree=cKDTree(sample_pca)
		distances,indices=tree.query(sample_pca,k=neighbors+1)
		edge_weights:dict[tuple[int,int],float]={}
		for source in range(sample_pca.shape[0]):
			for offset in range(1,indices.shape[1]):
				target=int(indices[source,offset])
				if target==source:
					continue
				edge=(min(source,target),max(source,target))
				weight=1.0/(1.0+float(distances[source,offset]))
				edge_weights[edge]=max(edge_weights.get(edge,0.0),weight)
		graph=ig.Graph(n=sample_pca.shape[0],edges=list(edge_weights),directed=False)
		graph.es['weight']=[edge_weights[edge]for edge in edge_weights]
		partition=leidenalg.find_partition(
			graph,
			leidenalg.RBConfigurationVertexPartition,
			weights='weight',
			resolution_parameter=float(config.leiden_resolution),
			seed=int(config.random_seed),
		)
		labels=np.asarray(partition.membership,dtype=np.int32)
		unique=np.unique(labels)
		centroids=np.vstack([np.mean(sample_pca[labels==label],axis=0)for label in unique])
		remap={int(label):index for index,label in enumerate(unique)}
		labels=np.asarray([remap[int(label)]for label in labels],dtype=np.int32)
		return centroids,labels


	@staticmethod
	def _fit_umap(sample_pca:np.ndarray,config:PhenotypingConfig)->Any|None:
		if config.embedding!='umap':
			return None
		try:
			import umap# type: ignore
		except Exception as error:
			raise PhenotypingError(
				'UMAP embedding requires the optional package \'umap-learn\'. '
				'Install it or select PCA embedding.'
			)from error
		model=umap.UMAP(
			n_components=2,
			n_neighbors=min(15,max(2,sample_pca.shape[0]-1)),
			min_dist=0.1,
			random_state=int(config.random_seed),
			transform_seed=int(config.random_seed),
		)
		model.fit(sample_pca)
		return model


	@staticmethod
	def _create_sqlite(path:Path,feature_columns:Sequence[str])->sqlite3.Connection:
		if path.exists():
			path.unlink()
		conn=sqlite3.connect(path)
		conn.execute('PRAGMA journal_mode=WAL')
		conn.execute('PRAGMA synchronous=NORMAL')
		conn.execute(
			'''CREATE TABLE cells (
                global_cell_id INTEGER PRIMARY KEY,
                centroid_x REAL NOT NULL,
                centroid_y REAL NOT NULL,
                cluster_id INTEGER NOT NULL,
                cluster_name TEXT NOT NULL,
                pca1 REAL,
                pca2 REAL,
                embedding_x REAL,
                embedding_y REAL,
                features_json TEXT NOT NULL
            )'''
		)
		conn.execute('CREATE VIRTUAL TABLE cell_rtree USING rtree(global_cell_id, min_x, max_x, min_y, max_y)')
		conn.execute('CREATE INDEX idx_cells_cluster ON cells(cluster_id)')
		conn.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
		conn.execute('INSERT INTO metadata VALUES(?,?)',('feature_columns',json.dumps(list(feature_columns))))
		return conn


	def run(
		self,
		output_directory:str|Path,
		config:PhenotypingConfig,
		*,
		cancel_event:Any|None=None,
		progress_callback:Callable[[PhenotypingProgress],None]|None=None,
	)->PhenotypingRunSummary:
		missing=[name for name in config.feature_columns if name not in self.columns]
		if missing:
			raise PhenotypingError('Unknown marker feature(s): '+', '.join(missing[:10]))
		output=Path(output_directory).expanduser().resolve()
		output.mkdir(parents=True,exist_ok=True)
		assignments_path=output/'cluster_assignments.csv'
		cluster_summary_path=output/'cluster_summary.csv'
		feature_summary_path=output/'feature_preprocessing.csv'
		sqlite_path=output/'clustering.sqlite'
		config_path=output/'clustering_config.json'
		total_rows=self._row_count()
		if total_rows<2:
			raise PhenotypingError('At least two cells are required for clustering.')
		sample_raw,_=self._priority_sample(config,cancel_event,progress_callback,total_rows)
		sample_std,preprocessing=self._fit_preprocessing(sample_raw,config)
		pca_center,pca_components=self._fit_pca(sample_std,int(config.n_pcs))
		sample_pca=(sample_std-pca_center)@pca_components.T
		centroids,_=self._fit_clusters(sample_pca,config)
		umap_model=self._fit_umap(sample_pca,config)
		payload=config.to_dict()
		payload.update({
			'schema_version':PHENOTYPING_SCHEMA_VERSION,
			'marker_csv':str(self.marker_csv),
			'cell_count':int(total_rows),
			'actual_pcs':int(pca_components.shape[0]),
			'actual_clusters':int(centroids.shape[0]),
			'created_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
		})
		config_path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
		with feature_summary_path.open('w',newline='',encoding='utf-8')as handle:
			writer=csv.writer(handle)
			writer.writerow(['feature','median','winsor_low','winsor_high','mean','std'])
			for index,name in enumerate(config.feature_columns):
				writer.writerow([
					name,
					float(preprocessing['medians'][index]),
					float(preprocessing['lows'][index]),
					float(preprocessing['highs'][index]),
					float(preprocessing['means'][index]),
					float(preprocessing['stds'][index]),
				])
		conn=self._create_sqlite(sqlite_path,config.feature_columns)
		usecols=list(self.ID_COLUMNS)+list(config.feature_columns)
		output_columns=[
			'global_cell_id','centroid_x','centroid_y','cluster_id','cluster_name',
			'pca1','pca2','embedding_x','embedding_y',
		]+list(config.feature_columns)
		completed=0
		cluster_counts=np.zeros(centroids.shape[0],dtype=np.int64)
		try:
			with assignments_path.open('w',newline='',encoding='utf-8')as handle:
				writer=csv.writer(handle)
				writer.writerow(output_columns)
				for chunk in pd.read_csv(
					self.marker_csv,
					usecols=usecols,
					chunksize=int(config.chunk_rows),
				):
					if cancel_event is not None and cancel_event.is_set():
						raise PhenotypingCancelled('Clustering cancelled while assigning cells.')
					ids=chunk['global_cell_id'].to_numpy(dtype=np.int64)
					xs=chunk['centroid_x'].to_numpy(dtype=np.float64)
					ys=chunk['centroid_y'].to_numpy(dtype=np.float64)
					raw=chunk[list(config.feature_columns)].to_numpy(dtype=np.float64,copy=True)
					standardized=self._apply_preprocessing(raw,config,preprocessing)
					pca=(standardized-pca_center)@pca_components.T
					labels=self._nearest_centroid(pca,centroids)
					embedding=(
						np.asarray(umap_model.transform(pca),dtype=np.float64)
						if umap_model is not None
						else pca[:,:min(2,pca.shape[1])]
					)
					if embedding.shape[1]==1:
						embedding=np.column_stack((embedding[:,0],np.zeros(embedding.shape[0])))
					pca2=pca[:,:min(2,pca.shape[1])]
					if pca2.shape[1]==1:
						pca2=np.column_stack((pca2[:,0],np.zeros(pca2.shape[0])))
					rows_sql=[]
					rows_rtree=[]
					for index in range(len(ids)):
						cluster_id=int(labels[index])+1
						cluster_name= f'Cluster {cluster_id}'
						features=[float(value)if np.isfinite(value)else None for value in raw[index]]
						writer.writerow([
							int(ids[index]),float(xs[index]),float(ys[index]),cluster_id,cluster_name,
							float(pca2[index,0]),float(pca2[index,1]),
							float(embedding[index,0]),float(embedding[index,1]),
							*features,
						])
						feature_json=json.dumps(dict(zip(config.feature_columns,features)),separators=(',',':'))
						rows_sql.append((
							int(ids[index]),float(xs[index]),float(ys[index]),cluster_id,cluster_name,
							float(pca2[index,0]),float(pca2[index,1]),
							float(embedding[index,0]),float(embedding[index,1]),feature_json,
						))
						rows_rtree.append((int(ids[index]),float(xs[index]),float(xs[index]),float(ys[index]),float(ys[index])))
						cluster_counts[cluster_id-1]+=1
					with conn:
						conn.executemany(
							'INSERT INTO cells VALUES(?,?,?,?,?,?,?,?,?,?)',
							rows_sql,
						)
						conn.executemany(
							'INSERT INTO cell_rtree VALUES(?,?,?,?,?)',
							rows_rtree,
						)
					completed+=len(ids)
					self._emit(progress_callback,'assignment',completed,total_rows,
								f'Assigned cells: {completed:,}/{total_rows:,}')
			with cluster_summary_path.open('w',newline='',encoding='utf-8')as handle:
				writer=csv.writer(handle)
				writer.writerow(['cluster_id','cluster_name','cell_count','fraction'])
				for index,count in enumerate(cluster_counts):
					writer.writerow([index+1, f'Cluster {index+1}',int(count),float(count/total_rows)])
			with conn:
				conn.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('config',json.dumps(payload)))
				conn.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',('cluster_count',str(int(centroids.shape[0]))))
		finally:
			conn.close()
		return PhenotypingRunSummary(
			output_directory=str(output),
			assignments_csv=str(assignments_path),
			cluster_summary_csv=str(cluster_summary_path),
			sqlite_path=str(sqlite_path),
			cell_count=int(total_rows),
			cluster_count=int(centroids.shape[0]),
			feature_count=len(config.feature_columns),
			embedding=config.embedding,
			cancelled=False,
		)


def rename_clusters(
	clustering_directory:str|Path,
	names:Mapping[int,str],
)->None:
	directory=Path(clustering_directory).expanduser().resolve()
	database=directory/'clustering.sqlite'
	if not database.is_file():
		raise PhenotypingError(f'Clustering database not found: {database}')
	cleaned:dict[int,str]={}
	with sqlite3.connect(database)as conn:
		for cluster_id,name in names.items():
			clean=str(name).strip()
			if clean:
				cleaned[int(cluster_id)]=clean
				conn.execute(
					'UPDATE cells SET cluster_name=? WHERE cluster_id=?',
					(clean,int(cluster_id)),
				)
	summary_path=directory/'cluster_summary.csv'
	if cleaned and summary_path.is_file():
		table=pd.read_csv(summary_path)
		if'cluster_id'in table.columns and'cluster_name'in table.columns:
			for cluster_id,clean in cleaned.items():
				table.loc[table['cluster_id']==int(cluster_id),'cluster_name']=clean
			table.to_csv(summary_path,index=False)
__all__=[
	'CellPhenotyper',
	'PhenotypingCancelled',
	'PhenotypingConfig',
	'PhenotypingError',
	'PhenotypingProgress',
	'PhenotypingRunSummary',
	'discover_marker_features',
	'feature_columns_for_metric',
	'rename_clusters',
]
