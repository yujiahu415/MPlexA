from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json
import os
import cv2
import numpy as np
from.image_source import open_multiplex_image
from.segmentation import load_tile_predictions
from.tiling import TileGrid



class COCOExportError(RuntimeError):
	'''Raised when Module 2 predictions cannot be exported as COCO JSON.'''



@dataclass(frozen=True,slots=True)
class COCOExportSummary:
	output_path:str
	image_file_name:str
	image_width:int
	image_height:int
	tile_archives:int
	annotations:int
	skipped_masks:int
	categories:tuple[str,...]


	def summary(self)->str:
		return(
			f'COCO annotations exported: {self.annotations:,}\n'
			f'Masks skipped because no valid polygon could be formed: {self.skipped_masks:,}\n'
			f'Categories: {list(self.categories)}\n'
			f'Image: {self.image_file_name} ({self.image_width} x {self.image_height})\n'
			f'Tile archives read: {self.tile_archives:,}\n'
			f'Output: {self.output_path}'
		)


def _category_names(config:dict)->tuple[str,...]:
	mapping=config.get('detector_mapping')
	if not isinstance(mapping,dict):
		detector=config.get('detector')
		if isinstance(detector,dict):
			mapping=detector.get('cell_mapping')
	names:list[str]=[]
	if isinstance(mapping,dict):


		def key_order(item:tuple[object,object])->tuple[int,str]:
			key=str(item[0])
			try:
				return int(key),key
			except ValueError:
				return 10**9,key
		for _,value in sorted(mapping.items(),key=key_order):
			name=str(value).strip()
			if name and name not in names:
				names.append(name)
	if not names:
		names.append('Nucleus')
	return tuple(names)


def _polygon_from_mask(
	mask:np.ndarray,
	*,
	global_x:float,
	global_y:float,
	scale_x:float,
	scale_y:float,
	image_width:int,
	image_height:int,
	simplify_fraction:float,
)->list[int]:
	contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return[]
	contour=max(contours,key=cv2.contourArea)
	if len(contour)<3:
		return[]
	perimeter=float(cv2.arcLength(contour,True))
	epsilon=max(0.0,float(simplify_fraction))*perimeter
	approx=cv2.approxPolyDP(contour,epsilon,True)if epsilon>0 else contour
	points=approx.reshape(-1,2)
	if len(points)<3:
		points=contour.reshape(-1,2)
	if len(points)<3:
		return[]
	converted:list[tuple[int,int]]=[]
	for x,y in points:
		bx=int(round((float(global_x)+float(x))*float(scale_x)))
		by=int(round((float(global_y)+float(y))*float(scale_y)))
		bx=max(0,min(int(image_width)-1,bx))
		by=max(0,min(int(image_height)-1,by))
		point=(bx,by)
		if not converted or converted[-1]!=point:
			converted.append(point)
	if len(converted)>1 and converted[0]==converted[-1]:
		converted.pop()
	if len(set(converted))<3:
		return[]
	flat:list[int]=[]
	for x,y in converted:
		flat.extend((int(x),int(y)))
	return flat


def _polygon_area_and_bbox(segmentation:list[int])->tuple[float,list[int]]:
	points=np.asarray(segmentation,dtype=np.float32).reshape(-1,2)
	area=float(abs(cv2.contourArea(points.reshape(-1,1,2))))
	x_min=int(np.min(points[:,0]));y_min=int(np.min(points[:,1]))
	x_max=int(np.max(points[:,0]));y_max=int(np.max(points[:,1]))
	return area,[x_min,y_min,max(0,x_max-x_min),max(0,y_max-y_min)]


def export_segmentation_to_coco(
	segmentation_directory:str|Path,
	output_path:str|Path,
	*,
	image_path:str|Path|None=None,
	series:int|None=None,
	owned_only:bool=True,
	simplify_fraction:float=0.005,
	on_progress:Callable[[int,int,int],None]|None=None,
)->COCOExportSummary:

	segmentation_dir=Path(segmentation_directory).expanduser().resolve()
	config_path=segmentation_dir/'segmentation_config.json'
	tiles_dir=segmentation_dir/'tiles'
	if not config_path.is_file()or not tiles_dir.is_dir():
		raise COCOExportError('The selected Module 2 output does not contain completed segmentation data.')
	try:
		config=json.loads(config_path.read_text(encoding='utf-8'))
		grid=TileGrid.from_dict(config['grid'])
	except(OSError,ValueError,KeyError,TypeError,json.JSONDecodeError)as error:
		raise COCOExportError(f'Unable to read Module 2 segmentation configuration: {error}')from error
	source=Path(image_path).expanduser().resolve()if image_path is not None else None
	if source is None:
		identity=config.get('image',{})
		if isinstance(identity,dict)and identity.get('path'):
			source=Path(str(identity['path'])).expanduser().resolve()
	if source is None or not source.exists():
		raise COCOExportError('The original multiplex image is required to export COCO coordinates at full resolution.')
	resolved_series=int(config.get('series',0)if series is None else series)
	try:
		with open_multiplex_image(source,series=resolved_series)as image:
			image_width=int(image.metadata.width)
			image_height=int(image.metadata.height)
	except Exception as error:
		raise COCOExportError(f'Unable to read original-image dimensions: {error}')from error
	scale_x=image_width/float(grid.image_width)
	scale_y=image_height/float(grid.image_height)
	category_names=_category_names(config)
	category_id_by_name={name:index+1 for index,name in enumerate(category_names)}
	categories=[
		{'id':index+1,'name':name,'supercategory':'none'}
		for index,name in enumerate(category_names)
	]
	image_entry={'id':0,'width':image_width,'height':image_height,'file_name':source.name}
	info={
		'year':'',
		'version':'1',
		'description':'MPlexA cell-segmentation annotations',
		'contributor':'',
		'url':'',
		'date_created':'',
	}
	destination=Path(output_path).expanduser().resolve()
	if destination.suffix.lower()!='.json':
		destination=destination.with_suffix('.json')
	destination.parent.mkdir(parents=True,exist_ok=True)
	temporary=destination.with_suffix(destination.suffix+'.tmp')
	annotation_id=0
	skipped=0
	archives_read=0
	tile_paths=[tiles_dir/ f'{tile.tile_id}.npz' for tile in grid]
	existing_paths=[path for path in tile_paths if path.is_file()]
	if len(existing_paths)!=len(tile_paths):
		missing=len(tile_paths)-len(existing_paths)
		raise COCOExportError(
			f'Module 2 segmentation is incomplete: {missing} tile archive(s) are missing. '
			'Finish or resume segmentation before exporting COCO annotations.'
		)
	total_tiles=len(existing_paths)
	try:
		with temporary.open('w',encoding='utf-8',newline='')as handle:
			handle.write('{"info":')
			json.dump(info,handle,separators=(',',':'))
			handle.write(',"licenses":[],"categories":')
			json.dump(categories,handle,separators=(',',':'))
			handle.write(',"images":')
			json.dump([image_entry],handle,separators=(',',':'))
			handle.write(',"annotations":[')
			first=True
			for tile_number,archive_path in enumerate(existing_paths,start=1):
				archive=load_tile_predictions(archive_path)
				archives_read+=1
				for index in range(archive.count):
					if owned_only and not bool(archive.owned_by_core[index]):
						continue
					mask=archive.decode_cropped_mask(index)
					gx0,gy0,_,_=(float(value)for value in archive.global_boxes[index])
					segmentation=_polygon_from_mask(
						mask,
						global_x=gx0,
						global_y=gy0,
						scale_x=scale_x,
						scale_y=scale_y,
						image_width=image_width,
						image_height=image_height,
						simplify_fraction=simplify_fraction,
					)
					if len(segmentation)<6:
						skipped+=1
						continue
					class_name=str(archive.class_names[index])if len(archive.class_names)>index else category_names[0]
					category_id=category_id_by_name.get(class_name)
					if category_id is None:
						category_id=1
					area,bbox=_polygon_area_and_bbox(segmentation)
					if area<=0:
						skipped+=1
						continue
					annotation={
						'id':annotation_id,
						'image_id':0,
						'category_id':int(category_id),
						'segmentation':[segmentation],
						'area':area,
						'bbox':bbox,
						'iscrowd':0,
					}
					if not first:
						handle.write(',')
					json.dump(annotation,handle,separators=(',',':'))
					first=False
					annotation_id+=1
				if on_progress is not None:
					on_progress(tile_number,total_tiles,annotation_id)
			handle.write(']}')
		os.replace(temporary,destination)
	except Exception as error:
		temporary.unlink(missing_ok=True)
		if isinstance(error,COCOExportError):
			raise
		raise COCOExportError(f'Unable to export COCO annotations: {error}')from error
	return COCOExportSummary(
		output_path=str(destination),
		image_file_name=source.name,
		image_width=image_width,
		image_height=image_height,
		tile_archives=archives_read,
		annotations=annotation_id,
		skipped_masks=skipped,
		categories=category_names,
	)
