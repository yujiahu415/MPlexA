from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any,Iterator,Mapping,Sequence
from.exceptions import CheckpointError,CheckpointMismatchError
from.tiling import Tile,TileGrid
CHECKPOINT_SCHEMA_VERSION=1
_VALID_STATUSES=('pending','running','completed','failed')


def _utc_now()->str:
	return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _canonical_json(value:Any)->str:
	return json.dumps(value,sort_keys=True,separators=(',',':'),default=str)


def _job_family(value:str)->str:
	'''Return the stable processing-job suffix, independent of application branding.'''
	text=str(value).strip()
	return text.split(' ',1)[1]if' 'in text else text


def _normalized_context(value:Any)->Any:
	'''Normalize persisted context fields that are descriptive rather than result-defining.'''
	if isinstance(value,Mapping):
		return{
			str(key):_normalized_context(item)
			for key,item in value.items()
			if str(key)!='threshold_algorithm'
		}
	if isinstance(value,(list,tuple)):
		return[_normalized_context(item)for item in value]
	return value



@dataclass(frozen=True,slots=True)
class CheckpointProgress:
	total:int
	pending:int
	running:int
	completed:int
	failed:int


	@property
	def processed(self)->int:
		return self.completed+self.failed


	@property
	def completion_fraction(self)->float:
		return self.completed/self.total if self.total else 1.0


	@property
	def completion_percent(self)->float:
		return self.completion_fraction*100.0


	def summary(self)->str:
		return(
			f'Completed: {self.completed:,}/{self.total:,} '
			f'({self.completion_percent:.2f}%)\n'
			f'Pending: {self.pending:,}\n'
			f'Running: {self.running:,}\n'
			f'Failed: {self.failed:,}'
		)



class TileCheckpointStore:


	def __init__(
		self,
		path:str|Path,
		grid:TileGrid,
		*,
		job_name:str='multiplex-tile-job',
		context:Mapping[str,Any]|None=None,
		reset_interrupted:bool=False,
	)->None:
		self.path=Path(path).expanduser().resolve()
		self.path.parent.mkdir(parents=True,exist_ok=True)
		self.grid=grid
		self.job_name=str(job_name)
		self.context=dict(context or{})
		self._lock=threading.RLock()
		self._connection=sqlite3.connect(
			self.path,timeout=30.0,check_same_thread=False
		)
		self._connection.row_factory=sqlite3.Row
		self._closed=False
		try:
			self._connection.execute('PRAGMA journal_mode=WAL')
			self._connection.execute('PRAGMA synchronous=NORMAL')
			self._connection.execute('PRAGMA busy_timeout=30000')
			self._create_schema()
			self._validate_or_initialize_metadata()
			self._initialize_tiles()
			if reset_interrupted:
				self.reset_interrupted()
		except Exception:
			self._connection.close()
			self._closed=True
			raise


	def __enter__(self)->'TileCheckpointStore':
		return self


	def __exit__(self,exc_type:Any,exc:Any,traceback:Any)->None:
		self.close()


	def _create_schema(self)->None:
		with self._connection:
			self._connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                '''
			)
			self._connection.execute(
				'''
                CREATE TABLE IF NOT EXISTS tiles (
                    tile_id TEXT PRIMARY KEY,
                    tile_index INTEGER NOT NULL,
                    tile_row INTEGER NOT NULL,
                    tile_column INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending','running','completed','failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    finished_at TEXT,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    output_json TEXT
                )
                '''
			)
			self._connection.execute(
				'CREATE INDEX IF NOT EXISTS idx_tiles_status_index '
				'ON tiles(status, tile_index)'
			)


	def _metadata(self)->dict[str,str]:
		rows=self._connection.execute('SELECT key, value FROM metadata').fetchall()
		return{str(row['key']):str(row['value'])for row in rows}


	def _validate_or_initialize_metadata(self)->None:
		existing=self._metadata()
		expected={
			'checkpoint_schema_version':str(CHECKPOINT_SCHEMA_VERSION),
			'grid_signature':self.grid.signature,
			'grid_config':_canonical_json(self.grid.to_dict()),
			'job_name':self.job_name,
			'context':_canonical_json(self.context),
		}
		if existing:
			if existing.get('checkpoint_schema_version')!=expected['checkpoint_schema_version']:
				raise CheckpointMismatchError(
					'Checkpoint schema version does not match this MPlexA release.'
				)
			if existing.get('grid_signature')!=expected['grid_signature']:
				raise CheckpointMismatchError(
					'Checkpoint belongs to a different tiling grid. Select a new '
					'checkpoint file or restore the original tile settings.'
				)
			if _job_family(existing.get('job_name',''))!=_job_family(expected['job_name']):
				raise CheckpointMismatchError(
					'Checkpoint belongs to a different processing job.'
				)
			try:
				existing_context=json.loads(existing.get('context','{}'))
				expected_context=json.loads(expected['context'])
			except json.JSONDecodeError as error:
				raise CheckpointMismatchError('Checkpoint context metadata are invalid.')from error
			if _canonical_json(_normalized_context(existing_context))!=_canonical_json(_normalized_context(expected_context)):
				raise CheckpointMismatchError(
					'Checkpoint context does not match the selected image/job.'
				)
			return
		expected['created_at']=_utc_now()
		with self._connection:
			self._connection.executemany(
				'INSERT INTO metadata(key, value) VALUES (?, ?)',expected.items()
			)


	def _initialize_tiles(self)->None:
		timestamp=_utc_now()
		rows=[
			(
				tile.tile_id,
				tile.index,
				tile.row,
				tile.column,
				timestamp,
			)
			for tile in self.grid
		]
		with self._connection:
			self._connection.executemany(
				'''
                INSERT OR IGNORE INTO tiles(
                    tile_id, tile_index, tile_row, tile_column, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ''',
				rows,
			)
		count=int(self._connection.execute('SELECT COUNT(*) FROM tiles').fetchone()[0])
		if count!=len(self.grid):
			raise CheckpointMismatchError(
				f'Checkpoint contains {count} tiles but the grid contains {len(self.grid)}.'
			)


	def reset_interrupted(self)->int:
		with self._lock,self._connection:
			cursor=self._connection.execute(
				'''
                UPDATE tiles
                SET status='pending', updated_at=?, finished_at=NULL,
                    error=COALESCE(error, 'Previous run was interrupted.')
                WHERE status='running'
                ''',
				(_utc_now(),),
			)
			return int(cursor.rowcount)


	def claim_next(self,*,include_failed:bool=False)->Tile|None:
		statuses=('pending','failed')if include_failed else('pending',)
		placeholders=','.join('?'for _ in statuses)
		with self._lock:
			try:
				self._connection.execute('BEGIN IMMEDIATE')
				row=self._connection.execute(
					f'SELECT tile_id FROM tiles WHERE status IN ({placeholders}) '
					'ORDER BY tile_index LIMIT 1',
					statuses,
				).fetchone()
				if row is None:
					self._connection.commit()
					return None
				tile_id=str(row['tile_id'])
				timestamp=_utc_now()
				cursor=self._connection.execute(
					f'''
                    UPDATE tiles
                    SET status='running', attempts=attempts+1, started_at=?,
                        finished_at=NULL, updated_at=?, error=NULL
                    WHERE tile_id=? AND status IN ({placeholders})
                    ''',
					(timestamp,timestamp,tile_id,*statuses),
				)
				if cursor.rowcount!=1:
					self._connection.rollback()
					return self.claim_next(include_failed=include_failed)
				self._connection.commit()
				return self.grid.get(tile_id)
			except Exception:
				self._connection.rollback()
				raise


	def mark_running(self,tile_id:str)->None:
		self._transition(tile_id,'running',increment_attempt=True)


	def mark_completed(
		self,tile_id:str,output:Mapping[str,Any]|Sequence[Any]|None=None
	)->None:
		self._transition(tile_id,'completed',output=output)


	def mark_failed(self,tile_id:str,error:str|BaseException)->None:
		self._transition(tile_id,'failed',error=str(error))


	def mark_pending(self,tile_id:str)->None:
		self._transition(tile_id,'pending')


	def _transition(
		self,
		tile_id:str,
		status:str,
		*,
		increment_attempt:bool=False,
		error:str|None=None,
		output:Mapping[str,Any]|Sequence[Any]|None=None,
	)->None:
		if status not in _VALID_STATUSES:
			raise CheckpointError(f'Invalid tile status: {status}')
		self.grid.get(tile_id)# Validate before touching the database.
		timestamp=_utc_now()
		output_json=_canonical_json(output)if output is not None else None
		started_at=timestamp if status=='running'else None
		finished_at=timestamp if status in{'completed','failed'}else None
		attempts_sql='attempts+1'if increment_attempt else'attempts'
		with self._lock,self._connection:
			cursor=self._connection.execute(
				f'''
                UPDATE tiles
                SET status=?, attempts={attempts_sql},
                    started_at=COALESCE(?, started_at), finished_at=?, updated_at=?,
                    error=?, output_json=?
                WHERE tile_id=?
                ''',
				(
					status,
					started_at,
					finished_at,
					timestamp,
					error,
					output_json,
					tile_id,
				),
			)
			if cursor.rowcount!=1:
				raise CheckpointError(f'Tile {tile_id} was not found in checkpoint.')


	def status(self,tile_id:str)->dict[str,Any]:
		self.grid.get(tile_id)
		row=self._connection.execute(
			'SELECT * FROM tiles WHERE tile_id=?',(tile_id,)
		).fetchone()
		if row is None:
			raise CheckpointError(f'Tile {tile_id} was not found in checkpoint.')
		result=dict(row)
		if result.get('output_json'):
			result['output']=json.loads(result['output_json'])
		else:
			result['output']=None
		return result


	def progress(self)->CheckpointProgress:
		counts={status:0 for status in _VALID_STATUSES}
		for row in self._connection.execute(
			'SELECT status, COUNT(*) AS count FROM tiles GROUP BY status'
		):
			counts[str(row['status'])]=int(row['count'])
		return CheckpointProgress(
			total=sum(counts.values()),
			pending=counts['pending'],
			running=counts['running'],
			completed=counts['completed'],
			failed=counts['failed'],
		)


	def iter_tiles(self,statuses:Sequence[str]|None=None)->Iterator[Tile]:
		selected=tuple(statuses or _VALID_STATUSES)
		invalid=set(selected).difference(_VALID_STATUSES)
		if invalid:
			raise CheckpointError(f'Invalid tile statuses: {sorted(invalid)}')
		placeholders=','.join('?'for _ in selected)
		rows=self._connection.execute(
			f'SELECT tile_id FROM tiles WHERE status IN ({placeholders}) '
			'ORDER BY tile_index',
			selected,
		)
		for row in rows:
			yield self.grid.get(str(row['tile_id']))


	def reset_failed(self)->int:
		with self._lock,self._connection:
			cursor=self._connection.execute(
				'''
                UPDATE tiles
                SET status='pending', updated_at=?, finished_at=NULL, error=NULL
                WHERE status='failed'
                ''',
				(_utc_now(),),
			)
			return int(cursor.rowcount)


	def failed_rows(self,limit:int|None=None)->list[dict[str,Any]]:
		sql=(
			'SELECT tile_id, tile_index, tile_row, tile_column, attempts, error '
			'FROM tiles WHERE status=\'failed\' ORDER BY tile_index'
		)
		parameters:tuple[Any,...]=()
		if limit is not None:
			sql+=' LIMIT ?'
			parameters=(max(0,int(limit)),)
		return[dict(row)for row in self._connection.execute(sql,parameters).fetchall()]


	def failed_error_groups(self,limit:int=5)->list[tuple[str,int]]:
		rows=self._connection.execute(
			'''
            SELECT COALESCE(NULLIF(error, ''), '(no error text)') AS error_text,
                   COUNT(*) AS error_count
            FROM tiles
            WHERE status='failed'
            GROUP BY error_text
            ORDER BY error_count DESC, error_text ASC
            LIMIT ?
            ''',
			(max(1,int(limit)),),
		).fetchall()
		return[(str(row['error_text']),int(row['error_count']))for row in rows]


	def close(self)->None:
		with self._lock:
			if not self._closed:
				self._connection.close()
				self._closed=True
