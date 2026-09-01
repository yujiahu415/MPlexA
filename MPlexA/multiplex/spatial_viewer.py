from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any
from.exceptions import MultiplexImageError
SPATIAL_VIEW_INDEX_SCHEMA_VERSION=1



class SpatialGraphViewError(MultiplexImageError):



@dataclass(frozen=True,slots=True)
class SpatialGraphViewData:
	nodes:tuple[tuple[int,float,float,int],...]
	edges:tuple[tuple[int,int,float,float,float,float,int,int],...]
	nodes_truncated:bool=False
	edges_truncated:bool=False



class SpatialGraphOverlayIndex:


	def __init__(
		self,
		spatial_directory:str|Path,
		clustering_directory:str|Path|None=None,
		*,
		rebuild:bool=False,
	)->None:
		self.spatial_directory=Path(spatial_directory).expanduser().resolve()
		self.graph_db=self.spatial_directory/'spatial_graph.sqlite'
		if not self.graph_db.is_file():
			raise SpatialGraphViewError(f'Spatial graph database not found: {self.graph_db}')
		if clustering_directory is None:
			config_path=self.spatial_directory/'spatial_graph_config.json'
			if not config_path.is_file():
				raise SpatialGraphViewError(
					'Cannot determine the clustering directory because spatial_graph_config.json is missing.'
				)
			payload=json.loads(config_path.read_text(encoding='utf-8'))
			clustering_directory=payload.get('clustering_directory')
		if not clustering_directory:
			raise SpatialGraphViewError('A clustering directory is required for graph visualization.')
		self.clustering_directory=Path(clustering_directory).expanduser().resolve()
		self.clustering_db=self.clustering_directory/'clustering.sqlite'
		if not self.clustering_db.is_file():
			raise SpatialGraphViewError(f'Clustering database not found: {self.clustering_db}')
		self.index_db=self.spatial_directory/'spatial_graph_viewer.sqlite'
		if rebuild or not self._is_current():
			self._build_index()


	@staticmethod
	def _stamp(path:Path)->dict[str,Any]:
		stat=path.stat()
		return{'path':str(path),'size':int(stat.st_size),'mtime_ns':int(stat.st_mtime_ns)}


	def _fingerprint(self)->dict[str,Any]:
		return{
			'schema_version':SPATIAL_VIEW_INDEX_SCHEMA_VERSION,
			'graph':self._stamp(self.graph_db),
			'clustering':self._stamp(self.clustering_db),
		}


	def _is_current(self)->bool:
		if not self.index_db.is_file():
			return False
		try:
			with sqlite3.connect(self.index_db)as conn:
				row=conn.execute('SELECT value FROM metadata WHERE key=\'fingerprint\'').fetchone()
			return row is not None and json.loads(row[0])==self._fingerprint()
		except Exception:
			return False


	def _build_index(self)->None:
		temp=self.index_db.with_suffix('.tmp.sqlite')
		if temp.exists():
			temp.unlink()
		conn=sqlite3.connect(temp)
		try:
			conn.execute('PRAGMA journal_mode=WAL')
			conn.execute('PRAGMA synchronous=NORMAL')
			conn.execute('PRAGMA temp_store=MEMORY')
			conn.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
			conn.execute(
				'CREATE TABLE nodes(global_cell_id INTEGER PRIMARY KEY, x REAL NOT NULL, y REAL NOT NULL, cluster_id INTEGER NOT NULL)'
			)
			conn.execute('CREATE VIRTUAL TABLE node_rtree USING rtree(global_cell_id,min_x,max_x,min_y,max_y)')
			conn.execute(
				'''CREATE TABLE edge_geometry(
                    edge_id INTEGER PRIMARY KEY,
                    source_cell_id INTEGER NOT NULL,
                    target_cell_id INTEGER NOT NULL,
                    sx REAL NOT NULL, sy REAL NOT NULL,
                    tx REAL NOT NULL, ty REAL NOT NULL,
                    source_cluster_id INTEGER NOT NULL,
                    target_cluster_id INTEGER NOT NULL
                )'''
			)
			conn.execute('CREATE INDEX idx_edge_pair ON edge_geometry(source_cluster_id,target_cluster_id)')
			conn.execute('CREATE VIRTUAL TABLE edge_rtree USING rtree(edge_id,min_x,max_x,min_y,max_y)')
			graph_uri=str(self.graph_db).replace('\'','\'\'')
			cluster_uri=str(self.clustering_db).replace('\'','\'\'')
			conn.execute(f'ATTACH DATABASE \'{graph_uri}\' AS graphdb')
			conn.execute(f'ATTACH DATABASE \'{cluster_uri}\' AS clusterdb')
			conn.execute(
				'INSERT INTO nodes SELECT global_cell_id,centroid_x,centroid_y,cluster_id FROM clusterdb.cells'
			)
			conn.execute('INSERT INTO node_rtree SELECT global_cell_id,x,x,y,y FROM nodes')
			conn.execute(
				'''INSERT INTO edge_geometry(
                    source_cell_id,target_cell_id,sx,sy,tx,ty,source_cluster_id,target_cluster_id
                )
                SELECT e.source_cell_id,e.target_cell_id,s.x,s.y,t.x,t.y,s.cluster_id,t.cluster_id
                FROM graphdb.edges e
                JOIN nodes s ON s.global_cell_id=e.source_cell_id
                JOIN nodes t ON t.global_cell_id=e.target_cell_id'''
			)
			conn.execute(
				'''INSERT INTO edge_rtree
                SELECT edge_id,
                       CASE WHEN sx<tx THEN sx ELSE tx END,
                       CASE WHEN sx>tx THEN sx ELSE tx END,
                       CASE WHEN sy<ty THEN sy ELSE ty END,
                       CASE WHEN sy>ty THEN sy ELSE ty END
                FROM edge_geometry'''
			)
			conn.execute('INSERT INTO metadata VALUES(\'fingerprint\',?)',(json.dumps(self._fingerprint(),sort_keys=True),))
			conn.commit()
			conn.execute('DETACH DATABASE graphdb')
			conn.execute('DETACH DATABASE clusterdb')
		finally:
			conn.close()
		if self.index_db.exists():
			self.index_db.unlink()
		temp.replace(self.index_db)


	def bounds(self)->tuple[float,float,float,float]:
		with sqlite3.connect(self.index_db)as conn:
			row=conn.execute('SELECT MIN(x),MIN(y),MAX(x),MAX(y) FROM nodes').fetchone()
		if row is None or row[0]is None:
			raise SpatialGraphViewError('The spatial graph contains no cells.')
		return tuple(float(v)for v in row)# type: ignore[return-value]


	def cluster_names(self)->tuple[tuple[int,str],...]:
		with sqlite3.connect(self.clustering_db)as conn:
			rows=conn.execute(
				'SELECT cluster_id,MAX(cluster_name) FROM cells GROUP BY cluster_id ORDER BY cluster_id'
			).fetchall()
		return tuple((int(row[0]),str(row[1]))for row in rows)


	@staticmethod
	def _pair_sql(cluster_a:int|None,cluster_b:int|None)->tuple[str,list[int]]:
		if cluster_a is None and cluster_b is None:
			return'',[]
		if cluster_a is not None and cluster_b is None:
			return' AND (e.source_cluster_id=? OR e.target_cluster_id=?)',[cluster_a,cluster_a]
		if cluster_a is None and cluster_b is not None:
			return' AND (e.source_cluster_id=? OR e.target_cluster_id=?)',[cluster_b,cluster_b]
		assert cluster_a is not None and cluster_b is not None
		if cluster_a==cluster_b:
			return' AND e.source_cluster_id=? AND e.target_cluster_id=?',[cluster_a,cluster_b]
		return(
			' AND ((e.source_cluster_id=? AND e.target_cluster_id=?) OR '
			'(e.source_cluster_id=? AND e.target_cluster_id=?))',
			[cluster_a,cluster_b,cluster_b,cluster_a],
		)


	def query_region(
		self,
		x:float,
		y:float,
		width:float,
		height:float,
		*,
		cluster_a:int|None=None,
		cluster_b:int|None=None,
		edge_limit:int=50_000,
		node_limit:int=50_000,
	)->SpatialGraphViewData:
		x0,y0=float(x),float(y)
		x1,y1=x0+float(width),y0+float(height)
		edge_limit=max(1,int(edge_limit))
		node_limit=max(1,int(node_limit))
		pair_sql,pair_args=self._pair_sql(cluster_a,cluster_b)
		node_clusters=sorted({v for v in(cluster_a,cluster_b)if v is not None})
		with sqlite3.connect(self.index_db)as conn:
			edge_query=(
				'SELECT e.source_cell_id,e.target_cell_id,e.sx,e.sy,e.tx,e.ty,e.source_cluster_id,e.target_cluster_id '
				'FROM edge_rtree r JOIN edge_geometry e ON e.edge_id=r.edge_id '
				'WHERE r.max_x>=? AND r.min_x<=? AND r.max_y>=? AND r.min_y<=?'+pair_sql+' LIMIT ?'
			)
			edge_args:list[Any]=[x0,x1,y0,y1,*pair_args,edge_limit+1]
			edge_rows=conn.execute(edge_query,edge_args).fetchall()
			node_sql=(
				'SELECT n.global_cell_id,n.x,n.y,n.cluster_id FROM node_rtree r '
				'JOIN nodes n ON n.global_cell_id=r.global_cell_id '
				'WHERE r.max_x>=? AND r.min_x<=? AND r.max_y>=? AND r.min_y<=?'
			)
			node_args:list[Any]=[x0,x1,y0,y1]
			if node_clusters:
				placeholders=','.join('?'for _ in node_clusters)
				node_sql+= f' AND n.cluster_id IN ({placeholders})'
				node_args.extend(node_clusters)
			node_sql+=' LIMIT ?'
			node_args.append(node_limit+1)
			node_rows=conn.execute(node_sql,node_args).fetchall()
		edges_truncated=len(edge_rows)>edge_limit
		nodes_truncated=len(node_rows)>node_limit
		return SpatialGraphViewData(
			nodes=tuple((int(a),float(b),float(c),int(d))for a,b,c,d in node_rows[:node_limit]),
			edges=tuple(
				(int(a),int(b),float(c),float(d),float(e),float(f),int(g),int(h))
				for a,b,c,d,e,f,g,h in edge_rows[:edge_limit]
			),
			nodes_truncated=nodes_truncated,
			edges_truncated=edges_truncated,
		)


	def nearest_cell(self,x:float,y:float,radius:float=10.0)->dict[str,Any]|None:
		radius=max(0.0,float(radius))
		with sqlite3.connect(self.clustering_db)as conn:
			row=conn.execute(
				'''SELECT c.global_cell_id,c.centroid_x,c.centroid_y,c.cluster_id,c.cluster_name
                FROM cell_rtree r JOIN cells c ON c.global_cell_id=r.global_cell_id
                WHERE r.max_x>=? AND r.min_x<=? AND r.max_y>=? AND r.min_y<=?
                ORDER BY ((c.centroid_x-?)*(c.centroid_x-?)+(c.centroid_y-?)*(c.centroid_y-?)) ASC LIMIT 1''',
				(x-radius,x+radius,y-radius,y+radius,x,x,y,y),
			).fetchone()
		if row is None:
			return None
		dx,dy=float(row[1])-float(x),float(row[2])-float(y)
		if dx*dx+dy*dy>radius*radius:
			return None
		return{
			'global_cell_id':int(row[0]),'centroid_x':float(row[1]),'centroid_y':float(row[2]),
			'cluster_id':int(row[3]),'cluster_name':str(row[4]),
		}
__all__=[
	'SPATIAL_VIEW_INDEX_SCHEMA_VERSION',
	'SpatialGraphOverlayIndex',
	'SpatialGraphViewData',
	'SpatialGraphViewError',
]
