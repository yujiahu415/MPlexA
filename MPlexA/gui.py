import os
import cv2
import numpy as np
import wx
import wx.aui
import wx.lib.agw.hyperlink as hl
from pathlib import Path
import json
import shutil
import sqlite3
import time
import threading
import pandas as pd
from.detector import Detector
from.multiplex import(
	CellRegionConfig,CellRegionGenerator,GlobalMaskReconciler,resolve_global_label_store,resolve_cell_region_label_store,
	IntensityNormalization,MarkerQuantifier,MultiplexImageError,
	QuantificationConfig,QuantificationError,ReconciliationConfig,
	ReconciliationError,SegmentationConfig,SegmentationError,
	TiledDapiSegmenter,TiledThresholdSegmenter,TilingError,
	ThresholdSegmentationConfig,build_detector_tile_grid,build_threshold_tile_grid,choose_pyramid_level_for_pixel_size,discover_mplexa_detectors,
	COCOExportError,export_segmentation_to_coco,
	estimate_intensity_normalization,open_multiplex_image,
	read_detector_metadata,
	CellPhenotyper,PhenotypingConfig,PhenotypingError,
	discover_marker_features,feature_columns_for_metric,rename_clusters,
	SpatialGraphBuilder,SpatialGraphConfig,SpatialGraphError,
	SpatialGraphOverlayIndex,SpatialGraphViewError,
	ChannelDisplaySettings,ClusterOverlayIndex,DEFAULT_CHANNEL_COLORS,
	MultiplexCompositeRenderer,SegmentationOverlayIndex,ViewerError,Viewport,
	label_boundaries_for_viewport,
	ROIChannel,ROIExtractionConfig,ROIExtractionError,
	SquareROIExtractor,plan_square_rois,
)
from MPlexA import __version__
the_absolute_current_path=str(Path(__file__).resolve().parent)



class InitialPanel(wx.Panel):


	def __init__(self,parent):
		super().__init__(parent)
		self.notebook=parent
		self.display_window()


	def display_window(self):
		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)
		self.text_welcome=wx.StaticText(panel,label='Welcome to MPlexA!',style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(self.text_welcome,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,50,0)
		self.text_developers=wx.StaticText(panel,label='\nDeveloped by Yujia Hu\n',style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(self.text_developers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,45,0)
		modules=wx.BoxSizer(wx.HORIZONTAL)
		button_train=wx.Button(panel,label='Training Module',size=(250,40))
		button_train.Bind(wx.EVT_BUTTON,self.panel_train)
		button_train.SetToolTip('Train and test Detectron2 instance-segmentation Detectors.')
		button_analyze=wx.Button(panel,label='Analysis Module',size=(250,40))
		button_analyze.Bind(wx.EVT_BUTTON,self.panel_analyze)
		button_analyze.SetToolTip('Analyze large multiplex images with MPlexA.')
		modules.Add(button_train,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		modules.Add(button_analyze,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(modules,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)
		panel.SetSizer(boxsizer)
		self.Centre()
		self.Show(True)


	def panel_train(self,event):
		panel=PanelLv1_TrainingModule(self.notebook)
		self.notebook.AddPage(panel,'Training Module',select=True)


	def panel_analyze(self,event):
		panel=PanelLv2_MultiplexAnalysis(self.notebook)
		self.notebook.AddPage(panel,'Analysis Module',select=True)



class PanelLv1_TrainingModule(wx.ScrolledWindow):
	'''Unified training workspace matching the section-based Analysis interface.'''


	def __init__(self,parent):
		super().__init__(parent,style=wx.VSCROLL|wx.HSCROLL)
		self.notebook=parent
		self.display_window()


	def display_window(self):
		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,15,0)
		roi_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 1 — Extract ROIs for EZannot')
		annotate=hl.HyperLinkCtrl(panel,0,'Open EZannot for annotation',URL='https://github.com/yujiahu415/EZannot')
		roi_box.Add(annotate,0,wx.LEFT|wx.RIGHT|wx.TOP,20)
		roi_box.Add(0,8,0)
		self.roi_panel=PanelLv2_ExtractROIs(panel,embedded=True)
		roi_box.Add(self.roi_panel,0,wx.EXPAND)
		boxsizer.Add(roi_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,12,0)
		train_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 2 — Train Detector')
		self.train_panel=PanelLv2_TrainDetectors(panel,embedded=True)
		train_box.Add(self.train_panel,0,wx.EXPAND)
		boxsizer.Add(train_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,12,0)
		test_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 3 — Test Detector')
		self.test_panel=PanelLv2_TestDetectors(panel,embedded=True)
		test_box.Add(self.test_panel,0,wx.EXPAND)
		boxsizer.Add(test_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,15,0)
		panel.SetSizer(boxsizer)
		self.SetScrollRate(10,10)
		self.FitInside()
		self.Show(True)



class ROIChannelColorDialog(wx.Dialog):
	'''Select exported source channels and assign each an RGB display color.'''


	def __init__(self,parent,channel_names,selections=None,colors=None):
		super().__init__(parent,title='ROI channels and display colors',size=(650,500))
		self.channel_names=list(channel_names)
		self.colors={index:tuple((colors or{}).get(index,DEFAULT_CHANNEL_COLORS[index%len(DEFAULT_CHANNEL_COLORS)]))for index in range(len(self.channel_names))}
		self.current_index=0
		root=wx.BoxSizer(wx.VERTICAL)
		explanation=wx.StaticText(self,label='Check the channels to include in each exported RGB TIFF. Select a channel, then click its color swatch to change the display color.')
		root.Add(explanation,0,wx.ALL|wx.EXPAND,10)
		body=wx.BoxSizer(wx.HORIZONTAL)
		self.channel_list=wx.CheckListBox(self,choices=self.channel_names)
		selected=set(selections or[])
		for index in range(len(self.channel_names)):
			self.channel_list.Check(index,index in selected)
		self.channel_list.Bind(wx.EVT_LISTBOX,self.select_channel)
		body.Add(self.channel_list,1,wx.ALL|wx.EXPAND,8)
		controls=wx.BoxSizer(wx.VERTICAL)
		self.text_current=wx.StaticText(self,label='')
		controls.Add(self.text_current,0,wx.ALL|wx.EXPAND,5)
		self.button_color=wx.Button(self,label='Display color',size=(160,38))
		self.button_color.Bind(wx.EVT_BUTTON,self.choose_color)
		controls.Add(self.button_color,0,wx.ALL,5)
		button_all=wx.Button(self,label='Select all',size=(160,34));button_all.Bind(wx.EVT_BUTTON,self.select_all)
		controls.Add(button_all,0,wx.ALL,5)
		button_none=wx.Button(self,label='Clear selection',size=(160,34));button_none.Bind(wx.EVT_BUTTON,self.clear_all)
		controls.Add(button_none,0,wx.ALL,5)
		controls.AddStretchSpacer(1)
		body.Add(controls,0,wx.ALL|wx.EXPAND,8)
		root.Add(body,1,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		buttons=self.CreateSeparatedButtonSizer(wx.OK|wx.CANCEL)
		if buttons is not None:
			root.Add(buttons,0,wx.ALL|wx.EXPAND,10)
		self.SetSizer(root)
		if self.channel_names:
			self.channel_list.SetSelection(0)
		self._refresh_color()
		self.CentreOnParent()


	def _refresh_color(self):
		if not self.channel_names:return
		index=max(0,min(self.current_index,len(self.channel_names)-1))
		self.text_current.SetLabel(self.channel_names[index]+' — RGB '+str(self.colors[index]))
		self.button_color.SetBackgroundColour(wx.Colour(*self.colors[index]))
		self.button_color.Refresh()


	def select_channel(self,event):
		selected=self.channel_list.GetSelection()
		if selected!=wx.NOT_FOUND:
			self.current_index=int(selected)
			self._refresh_color()


	def choose_color(self,event):
		if not self.channel_names:return
		data=wx.ColourData();data.SetColour(wx.Colour(*self.colors[self.current_index]))
		dialog=wx.ColourDialog(self,data)
		if dialog.ShowModal()==wx.ID_OK:
			colour=dialog.GetColourData().GetColour()
			self.colors[self.current_index]=(colour.Red(),colour.Green(),colour.Blue())
			self._refresh_color()
		dialog.Destroy()


	def select_all(self,event):
		for index in range(len(self.channel_names)):
			self.channel_list.Check(index,True)


	def clear_all(self,event):
		for index in range(len(self.channel_names)):
			self.channel_list.Check(index,False)


	def selected_indices(self):
		return[index for index in range(len(self.channel_names))if self.channel_list.IsChecked(index)]



class PanelLv2_ExtractROIs(wx.Panel):
	'''Extract square, colored RGB TIFF ROIs for annotation in EZannot.'''


	def __init__(self,parent,embedded=False):
		super().__init__(parent)
		self.embedded=bool(embedded)
		self.notebook=parent
		self.source_paths=[]
		self.image_metadata=None
		self.channel_indices=[]
		self.channel_colors={}
		self.output_directory=None
		self.cancel_event=None
		self.worker_thread=None
		self.display_window()


	def display_window(self):
		root=wx.BoxSizer(wx.VERTICAL)
		if not self.embedded:
			title=wx.StaticText(self,label='Extract square ROIs for EZannot')
			font=title.GetFont();font.SetPointSize(font.GetPointSize()+2);font.MakeBold();title.SetFont(font)
			root.Add(title,0,wx.ALL,12)
		note=wx.StaticText(self,label='Each selected channel is independently min-max rescaled within each ROI to 0-255, mapped to its selected RGB color, and additively composited into an 8-bit RGB TIFF.')
		root.Add(note,0,wx.ALL|wx.EXPAND,10)
		source_row=wx.BoxSizer(wx.HORIZONTAL)
		button_source=wx.Button(self,label='Select source multiplex\nimage(s)',size=(300,44))
		button_source.Bind(wx.EVT_BUTTON,self.select_sources)
		self.text_sources=wx.StaticText(self,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		source_row.Add(button_source,0,wx.LEFT|wx.RIGHT,10)
		source_row.Add(self.text_sources,1,wx.LEFT|wx.RIGHT,10)
		root.Add(source_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		root.Add(0,6,0)
		series_row=wx.BoxSizer(wx.HORIZONTAL)
		button_inspect=wx.Button(self,label='Inspect selected image(s)\nand load channels',size=(300,44))
		button_inspect.Bind(wx.EVT_BUTTON,self.inspect_sources)
		self.spin_series=wx.SpinCtrl(self,min=0,max=999,initial=0,size=(100,-1))
		series_row.Add(button_inspect,0,wx.LEFT|wx.RIGHT,10)
		series_row.Add(wx.StaticText(self,label='Series index:'),0,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		series_row.Add(self.spin_series,0,wx.LEFT|wx.RIGHT,5)
		root.Add(series_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.text_metadata=wx.TextCtrl(self,style=wx.TE_MULTILINE|wx.TE_READONLY,size=(-1,95))
		root.Add(self.text_metadata,0,wx.ALL|wx.EXPAND,10)
		channel_row=wx.BoxSizer(wx.HORIZONTAL)
		button_channels=wx.Button(self,label='Select channels and\ndisplay colors',size=(300,44))
		button_channels.Bind(wx.EVT_BUTTON,self.select_channels)
		self.text_channels=wx.StaticText(self,label='Inspect an image first.',style=wx.ST_ELLIPSIZE_END)
		channel_row.Add(button_channels,0,wx.LEFT|wx.RIGHT,10)
		channel_row.Add(self.text_channels,1,wx.LEFT|wx.RIGHT,10)
		root.Add(channel_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		root.Add(0,6,0)
		geometry=wx.FlexGridSizer(rows=3,cols=2,vgap=8,hgap=12)
		geometry.AddGrowableCol(1,1)
		geometry.Add(wx.StaticText(self,label='Square ROI size (pixels):'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_roi=wx.SpinCtrl(self,min=32,max=32768,initial=1024,size=(140,-1));geometry.Add(self.spin_roi,0)
		geometry.Add(wx.StaticText(self,label='Overlap ratio between adjacent ROIs:'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_overlap=wx.SpinCtrlDouble(self,min=0.0,max=0.95,initial=0.10,inc=0.01,size=(140,-1));self.spin_overlap.SetDigits(4);geometry.Add(self.spin_overlap,0)
		geometry.Add(wx.StaticText(self,label='Image background / edge padding:'),0,wx.ALIGN_CENTER_VERTICAL)
		self.choice_padding=wx.Choice(self,choices=['Darker background — pad with black pixels','Lighter background — pad with white pixels']);self.choice_padding.SetSelection(0);geometry.Add(self.choice_padding,0,wx.EXPAND)
		root.Add(geometry,0,wx.ALL|wx.EXPAND,14)
		plan_row=wx.BoxSizer(wx.HORIZONTAL)
		button_preview=wx.Button(self,label='Preview ROI plan',size=(180,36));button_preview.Bind(wx.EVT_BUTTON,self.preview_plan)
		self.text_plan=wx.StaticText(self,label='')
		plan_row.Add(button_preview,0,wx.LEFT|wx.RIGHT,10);plan_row.Add(self.text_plan,1,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		root.Add(plan_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		root.Add(0,8,0)
		output_row=wx.BoxSizer(wx.HORIZONTAL)
		button_output=wx.Button(self,label='Select output folder',size=(300,40));button_output.Bind(wx.EVT_BUTTON,self.select_output)
		self.text_output=wx.StaticText(self,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		output_row.Add(button_output,0,wx.LEFT|wx.RIGHT,10);output_row.Add(self.text_output,1,wx.LEFT|wx.RIGHT,10)
		root.Add(output_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		root.Add(0,10,0)
		action_row=wx.BoxSizer(wx.HORIZONTAL)
		self.button_extract=wx.Button(self,label='Extract ROIs',size=(200,40));self.button_extract.Bind(wx.EVT_BUTTON,self.run_extraction)
		self.button_cancel=wx.Button(self,label='Cancel',size=(120,40));self.button_cancel.Bind(wx.EVT_BUTTON,self.cancel_extraction);self.button_cancel.Disable()
		action_row.Add(self.button_extract,0,wx.LEFT|wx.RIGHT,10);action_row.Add(self.button_cancel,0,wx.LEFT|wx.RIGHT,10)
		root.Add(action_row,0,wx.ALIGN_CENTER,10)
		self.gauge=wx.Gauge(self,range=1000)
		root.Add(self.gauge,0,wx.ALL|wx.EXPAND,12)
		self.text_status=wx.TextCtrl(self,style=wx.TE_MULTILINE|wx.TE_READONLY,size=(-1,100))
		root.Add(self.text_status,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,12)
		self.SetSizer(root)


	def select_sources(self,event):
		wildcard='Multiplex TIFF images (*.tif;*.tiff;*.qptiff;*.btf;*.tf8)|*.tif;*.tiff;*.qptiff;*.btf;*.tf8|All files (*.*)|*.*'
		dialog=wx.FileDialog(self,'Select one or more source multiplex images','','',wildcard,style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST|wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.source_paths=[Path(path)for path in dialog.GetPaths()]
			self.text_sources.SetLabel(str(len(self.source_paths))+' image(s): '+', '.join(path.name for path in self.source_paths[:4])+(' ...'if len(self.source_paths)>4 else''))
			self.inspect_sources(event)
		dialog.Destroy()


	def inspect_sources(self,event):
		if not self.source_paths:
			wx.MessageBox('Select at least one source image first.','ROI extraction',wx.OK|wx.ICON_ERROR);return
		series=int(self.spin_series.GetValue())
		try:
			metadatas=[]
			for path in self.source_paths:
				with open_multiplex_image(path,series=series)as source:
					metadatas.append(source.metadata)
			first=metadatas[0]
			for metadata in metadatas[1:]:
				if tuple(metadata.channel_names)!=tuple(first.channel_names):
					raise ROIExtractionError('All selected source images must have the same channel names in the same order.')
			self.image_metadata=first
			if not self.channel_indices:
				dapi=next((i for i,name in enumerate(first.channel_names)if'dapi'in name.lower()),0)
				self.channel_indices=[dapi]
			self.channel_colors={index:self.channel_colors.get(index,DEFAULT_CHANNEL_COLORS[index%len(DEFAULT_CHANNEL_COLORS)])for index in range(len(first.channel_names))}
			self.text_metadata.SetValue('Images: '+str(len(metadatas))+'\n'+first.summary(max_channels=16))
			self._update_channel_summary()
			self.preview_plan(event)
		except Exception as error:
			self.image_metadata=None
			wx.MessageBox(str(error),'Could not inspect image(s)',wx.OK|wx.ICON_ERROR)


	def select_channels(self,event):
		if self.image_metadata is None:
			wx.MessageBox('Select and inspect source image(s) first.','ROI extraction',wx.OK|wx.ICON_ERROR);return
		dialog=ROIChannelColorDialog(self,self.image_metadata.channel_names,self.channel_indices,self.channel_colors)
		if dialog.ShowModal()==wx.ID_OK:
			selected=dialog.selected_indices()
			if not selected:
				wx.MessageBox('Select at least one channel.','ROI extraction',wx.OK|wx.ICON_ERROR)
			else:
				self.channel_indices=selected
				self.channel_colors=dict(dialog.colors)
				self._update_channel_summary()
		dialog.Destroy()


	def _update_channel_summary(self):
		if self.image_metadata is None:return
		pieces=[]
		for index in self.channel_indices[:8]:
			pieces.append(self.image_metadata.channel_names[index]+' '+str(self.channel_colors.get(index)))
		self.text_channels.SetLabel('Selected '+str(len(self.channel_indices))+': '+', '.join(pieces)+(' ...'if len(self.channel_indices)>8 else''))


	def _config(self):
		return ROIExtractionConfig(roi_size=int(self.spin_roi.GetValue()),overlap_ratio=float(self.spin_overlap.GetValue()),padding='black'if self.choice_padding.GetSelection()==0 else'white',series=int(self.spin_series.GetValue()))


	def preview_plan(self,event):
		if not self.source_paths:return
		try:
			config=self._config();total=0
			for path in self.source_paths:
				with open_multiplex_image(path,series=config.series)as source:
					total+=len(plan_square_rois(source.metadata.width,source.metadata.height,config))
			self.text_plan.SetLabel('ROIs: '+str(total)+'  |  overlap: '+str(config.overlap_pixels)+' px  |  stride: '+str(config.stride)+' px')
		except Exception as error:
			self.text_plan.SetLabel(str(error))


	def select_output(self,event):
		default=str(self.output_directory or'')
		dialog=wx.DirDialog(self,'Select or create the ROI output folder',default,style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.output_directory=Path(dialog.GetPath());self.text_output.SetLabel(str(self.output_directory))
		dialog.Destroy()


	def _validate(self):
		if not self.source_paths:raise ROIExtractionError('Select at least one source image.')
		if self.image_metadata is None:raise ROIExtractionError('Inspect the selected image(s) first.')
		if not self.channel_indices:raise ROIExtractionError('Select at least one export channel.')
		if self.output_directory is None:raise ROIExtractionError('Select an output folder.')
		channels=[ROIChannel(index,self.image_metadata.channel_names[index],self.channel_colors[index])for index in self.channel_indices]
		return channels,self._config()


	def run_extraction(self,event):
		try:
			channels,config=self._validate()
			extractor=SquareROIExtractor(self.source_paths,self.output_directory,channels,config)
			total=extractor.count_rois()
		except Exception as error:
			wx.MessageBox(str(error),'ROI extraction',wx.OK|wx.ICON_ERROR);return
		self.cancel_event=threading.Event();self.button_extract.Disable();self.button_cancel.Enable();self.gauge.SetValue(0)
		self.text_status.SetValue('Starting extraction of '+str(total)+' ROI(s)...')


		def worker():
			result=None;error=None
			try:
				result=extractor.run(progress_callback=lambda p:wx.CallAfter(self._update_progress,p),cancel_event=self.cancel_event)
			except Exception as exc:
				error=exc
			wx.CallAfter(self._finished,result,error)
		self.worker_thread=threading.Thread(target=worker,daemon=True);self.worker_thread.start()


	def _update_progress(self,progress):
		self.gauge.SetValue(int(round(progress.fraction*1000)))
		self.text_status.SetValue('Image '+str(progress.source_index+1)+'/'+str(progress.source_count)+': '+progress.source_name+'\nROI '+str(progress.completed)+'/'+str(progress.total)+' — '+str(progress.roi_id or'')+'\n'+progress.message)


	def cancel_extraction(self,event):
		if self.cancel_event is not None:
			self.cancel_event.set();self.text_status.SetValue(self.text_status.GetValue()+'\nCancellation requested...')


	def _finished(self,result,error):
		self.button_extract.Enable();self.button_cancel.Disable();self.cancel_event=None
		if error is not None:
			self.text_status.SetValue('ROI extraction failed:\n'+str(error));wx.MessageBox(str(error),'ROI extraction failed',wx.OK|wx.ICON_ERROR);return
		if not result.cancelled:self.gauge.SetValue(1000)
		self.text_status.SetValue(result.summary())
		wx.MessageBox(result.summary(),'ROI extraction',wx.OK|wx.ICON_INFORMATION)



class PanelLv2_TrainDetectors(wx.Panel):
	'''Detector-training controls using the same visual language as ROI extraction.'''


	def __init__(self,parent,embedded=False):
		super().__init__(parent)
		self.embedded=bool(embedded)
		self.notebook=parent
		self.path_to_trainingimages=None
		self.path_to_annotation=None
		self.num_rois=128
		self.inference_size=None
		self.black_background=0
		self.iteration_num=5000
		self.detector_path=os.path.join(the_absolute_current_path,'detectors')
		self.path_to_detector=None
		self.display_window()


	@staticmethod
	def _summary_box(parent,initial):
		box=wx.TextCtrl(parent,style=wx.TE_MULTILINE|wx.TE_READONLY,size=(-1,70))
		box.SetValue(initial)
		return box


	def display_window(self):
		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		if not self.embedded:
			title=wx.StaticText(panel,label='Train Detector')
			font=title.GetFont();font.SetPointSize(font.GetPointSize()+2);font.MakeBold();title.SetFont(font)
			boxsizer.Add(title,0,wx.ALL,12)
		note=wx.StaticText(panel,label=(
			'Select the training images and COCO instance-segmentation annotation, configure the image background and training iterations, '
			'then train a reusable MPlexA Detector.'))
		note.Wrap(900)
		boxsizer.Add(note,0,wx.ALL|wx.EXPAND,10)
		# Same row style as Module 1 "Select output folder".
		images_row=wx.BoxSizer(wx.HORIZONTAL)
		button_selectimages=wx.Button(panel,label='Select training image folder',size=(300,40))
		button_selectimages.Bind(wx.EVT_BUTTON,self.select_images)
		button_selectimages.SetToolTip('The folder that stores all training images.')
		self.text_selectimages=wx.StaticText(panel,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		images_row.Add(button_selectimages,0,wx.LEFT|wx.RIGHT,10)
		images_row.Add(self.text_selectimages,1,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		boxsizer.Add(images_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.summary_trainingimages=self._summary_box(panel,'No training image folder selected.')
		boxsizer.Add(self.summary_trainingimages,0,wx.ALL|wx.EXPAND,10)
		annotation_row=wx.BoxSizer(wx.HORIZONTAL)
		button_selectannotation=wx.Button(panel,label='Select COCO annotation (.json)',size=(300,40))
		button_selectannotation.Bind(wx.EVT_BUTTON,self.select_annotation)
		button_selectannotation.SetToolTip('Select the COCO instance-segmentation annotation for the training images.')
		self.text_selectannotation=wx.StaticText(panel,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		annotation_row.Add(button_selectannotation,0,wx.LEFT|wx.RIGHT,10)
		annotation_row.Add(self.text_selectannotation,1,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		boxsizer.Add(annotation_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.summary_annotation=self._summary_box(panel,'No COCO training annotation selected.')
		boxsizer.Add(self.summary_annotation,0,wx.ALL|wx.EXPAND,10)
		# Same label + inline choice style as Module 1 "Image background / edge padding".
		background_grid=wx.FlexGridSizer(rows=1,cols=2,vgap=8,hgap=12)
		background_grid.AddGrowableCol(1,1)
		background_grid.Add(wx.StaticText(panel,label='Image background:'),0,wx.ALIGN_CENTER_VERTICAL)
		self.choice_background=wx.Choice(panel,choices=[
			'Darker background — black/dark pixels',
			'Lighter background — white/light pixels',
		])
		self.choice_background.SetSelection(0)
		self.choice_background.Bind(wx.EVT_CHOICE,self.background_changed)
		background_grid.Add(self.choice_background,0,wx.EXPAND)
		boxsizer.Add(background_grid,0,wx.ALL|wx.EXPAND,14)
		self.summary_background=self._summary_box(
			panel,
			'Current setting: darker/black background. This setting is used by Detector training when interpreting image borders and background pixels.'
		)
		boxsizer.Add(self.summary_background,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,10)
		# Same label + SpinCtrl style as Module 1 "Square ROI size".
		iterations_grid=wx.FlexGridSizer(rows=1,cols=2,vgap=8,hgap=12)
		iterations_grid.AddGrowableCol(1,1)
		iterations_grid.Add(wx.StaticText(panel,label='Training iterations:'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_iterations=wx.SpinCtrl(panel,min=1,max=1000000,initial=5000,size=(140,-1))
		self.spin_iterations.Bind(wx.EVT_SPINCTRL,self.iterations_changed)
		self.spin_iterations.Bind(wx.EVT_TEXT,self.iterations_changed)
		iterations_grid.Add(self.spin_iterations,0)
		boxsizer.Add(iterations_grid,0,wx.ALL|wx.EXPAND,14)
		self.summary_iterations=self._summary_box(
			panel,
			'Current setting: 5000 training iterations. Higher values may improve convergence but increase training time.'
		)
		boxsizer.Add(self.summary_iterations,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,10)
		action_row=wx.BoxSizer(wx.HORIZONTAL)
		button_train=wx.Button(panel,label='Train Detector',size=(200,40))
		button_train.Bind(wx.EVT_BUTTON,self.train_detector)
		button_train.SetToolTip('English letters, numbers, “_”, or “-” are acceptable for names, but not “@” or “^”.')
		action_row.Add(button_train,0,wx.LEFT|wx.RIGHT,10)
		boxsizer.Add(action_row,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,8,0)
		self.text_training_status=wx.TextCtrl(panel,style=wx.TE_MULTILINE|wx.TE_READONLY,size=(-1,85))
		self.text_training_status.SetValue('Training has not started.')
		boxsizer.Add(self.text_training_status,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,12)
		panel.SetSizer(boxsizer)
		if not self.embedded:
			self.Centre()
		self.Show(True)


	def select_images(self,event):
		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_trainingimages=dialog.GetPath()
			self.text_selectimages.SetLabel(self.path_to_trainingimages)
			image_exts=('.jpg','.jpeg','.png','.tif','.tiff')
			try:
				images=[name for name in os.listdir(self.path_to_trainingimages)if name.lower().endswith(image_exts)]
				self.summary_trainingimages.SetValue(
					'Training image folder:\n'+self.path_to_trainingimages+'\n'
					+str(len(images))+' supported image file(s) found.'
				)
			except Exception as error:
				self.summary_trainingimages.SetValue('Training image folder:\n'+self.path_to_trainingimages+'\nCould not count images: '+str(error))
		dialog.Destroy()


	def select_annotation(self,event):
		wildcard='Annotation File (*.json)|*.json'
		dialog=wx.FileDialog(self,'Select the annotation file (.json)','',wildcard=wildcard,style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_annotation=dialog.GetPath()
			try:
				with open(self.path_to_annotation,encoding='utf-8')as f:
					info=json.load(f)
				classnames=[i['name']for i in info.get('categories',[])if i.get('id',0)>0]
				image_count=len(info.get('images',[]))
				annotation_count=len(info.get('annotations',[]))
				self.text_selectannotation.SetLabel(os.path.basename(self.path_to_annotation))
				self.summary_annotation.SetValue(
					'COCO annotation:\n'+self.path_to_annotation+'\n'
					+'Categories: '+str(classnames)+'\n'
					+'Images: '+str(image_count)+'  |  annotations: '+str(annotation_count)
				)
			except Exception as error:
				self.summary_annotation.SetValue('Could not read annotation:\n'+str(error))
		dialog.Destroy()


	def background_changed(self,event):
		self.black_background=0 if self.choice_background.GetSelection()==0 else 1
		if self.black_background==0:
			text='Current setting: darker/black background. Detector training will treat dark pixels as the image background.'
		else:
			text='Current setting: lighter/white background. Detector training will treat light pixels as the image background.'
		self.summary_background.SetValue(text)


	def iterations_changed(self,event):
		self.iteration_num=int(self.spin_iterations.GetValue())
		self.summary_iterations.SetValue(
			'Current setting: '+str(self.iteration_num)+' training iterations. '
			'Higher values may improve convergence but increase training time.'
		)


	def train_detector(self,event):
		self.iteration_num=int(self.spin_iterations.GetValue())
		self.black_background=0 if self.choice_background.GetSelection()==0 else 1
		if self.path_to_trainingimages is None or self.path_to_annotation is None:
			wx.MessageBox('No training images or annotation file selected.','Error',wx.OK|wx.ICON_ERROR)
			return
		cell_sizes=[
			'Sparse and large (e.g., large tissue areas)',
			'Median (e.g., structures formed by group of cells)',
			'Small (e.g. typical cell bodies)',
			'Extremely small (e.g., dense subcellular structures)',
		]
		dialog=wx.SingleChoiceDialog(self,message='How large are the objects to detect\ncompared to the images?',caption='Object size',choices=cell_sizes)
		if dialog.ShowModal()==wx.ID_OK:
			cell_size=dialog.GetStringSelection()
			if cell_size==cell_sizes[0]:
				self.num_rois=128
			elif cell_size==cell_sizes[1]:
				self.num_rois=256
			elif cell_size==cell_sizes[2]:
				self.num_rois=512
			else:
				self.num_rois=1024
		else:
			dialog.Destroy()
			return
		dialog.Destroy()
		images=[i for i in os.listdir(self.path_to_trainingimages)if i.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]
		if not images:
			wx.MessageBox('No supported training images were found in the selected folder.','Error',wx.OK|wx.ICON_ERROR)
			return
		first_image=cv2.imread(os.path.join(self.path_to_trainingimages,images[0]))
		if first_image is None:
			wx.MessageBox('Could not read the first training image.','Error',wx.OK|wx.ICON_ERROR)
			return
		self.inference_size=int(first_image.shape[1])
		do_nothing=False
		stop=False
		while not stop:
			name_dialog=wx.TextEntryDialog(self,'Enter a name for the Detector to train','Detector name')
			if name_dialog.ShowModal()==wx.ID_OK:
				if name_dialog.GetValue()!='':
					self.path_to_detector=os.path.join(self.detector_path,name_dialog.GetValue())
					if not os.path.isdir(self.path_to_detector):
						stop=True
					else:
						wx.MessageBox('The name already exists.','Error',wx.OK|wx.ICON_ERROR)
			else:
				do_nothing=True
				stop=True
			name_dialog.Destroy()
		if not do_nothing:
			self.text_training_status.SetValue(
				'Training Detector...\n'
				+'Images: '+self.path_to_trainingimages+'\n'
				+'Iterations: '+str(self.iteration_num)
			)
			try:
				detector=Detector()
				detector.train(
					self.path_to_annotation,
					self.path_to_trainingimages,
					self.path_to_detector,
					self.iteration_num,
					self.inference_size,
					self.num_rois,
					black_background=self.black_background,
				)
				self.text_training_status.SetValue('Training completed.\nDetector: '+str(self.path_to_detector))
			except Exception as error:
				self.text_training_status.SetValue('Training failed:\n'+str(error))
				raise



class PanelLv2_TestDetectors(wx.Panel):
	'''Detector-testing controls using the same visual language as ROI extraction.'''


	def __init__(self,parent,embedded=False):
		super().__init__(parent)
		self.embedded=bool(embedded)
		self.notebook=parent
		self.path_to_testingimages=None
		self.path_to_annotation=None
		self.detector_path=os.path.join(the_absolute_current_path,'detectors')
		self.path_to_detector=None
		self.output_path=None
		self.display_window()


	@staticmethod
	def _summary_box(parent,initial):
		box=wx.TextCtrl(parent,style=wx.TE_MULTILINE|wx.TE_READONLY,size=(-1,70))
		box.SetValue(initial)
		return box


	def display_window(self):
		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		if not self.embedded:
			title=wx.StaticText(panel,label='Test Detector')
			font=title.GetFont();font.SetPointSize(font.GetPointSize()+2);font.MakeBold();title.SetFont(font)
			boxsizer.Add(title,0,wx.ALL,12)
		note=wx.StaticText(panel,label=(
			'Select a trained MPlexA Detector and an annotated testing dataset, choose an output folder, then evaluate the Detector '
			'against the ground-truth annotations.'))
		note.Wrap(900)
		boxsizer.Add(note,0,wx.ALL|wx.EXPAND,10)
		detector_row=wx.BoxSizer(wx.HORIZONTAL)
		button_selectdetector=wx.Button(panel,label='Select Detector to test',size=(300,40))
		button_selectdetector.Bind(wx.EVT_BUTTON,self.select_detector)
		button_selectdetector.SetToolTip('The cell names in the testing dataset should match those in the selected Detector.')
		self.text_selectdetector=wx.StaticText(panel,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		detector_row.Add(button_selectdetector,0,wx.LEFT|wx.RIGHT,10)
		detector_row.Add(self.text_selectdetector,1,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		boxsizer.Add(detector_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.summary_detector=self._summary_box(panel,'No Detector selected.')
		boxsizer.Add(self.summary_detector,0,wx.ALL|wx.EXPAND,10)
		images_row=wx.BoxSizer(wx.HORIZONTAL)
		button_selectimages=wx.Button(panel,label='Select testing image folder',size=(300,40))
		button_selectimages.Bind(wx.EVT_BUTTON,self.select_images)
		button_selectimages.SetToolTip('The folder that stores all testing images.')
		self.text_selectimages=wx.StaticText(panel,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		images_row.Add(button_selectimages,0,wx.LEFT|wx.RIGHT,10)
		images_row.Add(self.text_selectimages,1,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		boxsizer.Add(images_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.summary_testingimages=self._summary_box(panel,'No testing image folder selected.')
		boxsizer.Add(self.summary_testingimages,0,wx.ALL|wx.EXPAND,10)
		annotation_row=wx.BoxSizer(wx.HORIZONTAL)
		button_selectannotation=wx.Button(panel,label='Select COCO annotation (.json)',size=(300,40))
		button_selectannotation.Bind(wx.EVT_BUTTON,self.select_annotation)
		button_selectannotation.SetToolTip('Select the COCO instance-segmentation annotation for the testing images.')
		self.text_selectannotation=wx.StaticText(panel,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		annotation_row.Add(button_selectannotation,0,wx.LEFT|wx.RIGHT,10)
		annotation_row.Add(self.text_selectannotation,1,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		boxsizer.Add(annotation_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.summary_annotation=self._summary_box(panel,'No COCO testing annotation selected.')
		boxsizer.Add(self.summary_annotation,0,wx.ALL|wx.EXPAND,10)
		output_row=wx.BoxSizer(wx.HORIZONTAL)
		button_selectoutpath=wx.Button(panel,label='Select testing result folder',size=(300,40))
		button_selectoutpath.Bind(wx.EVT_BUTTON,self.select_outpath)
		button_selectoutpath.SetToolTip('Select the folder that will store testing results.')
		self.text_selectoutpath=wx.StaticText(panel,label='None selected.',style=wx.ST_ELLIPSIZE_END)
		output_row.Add(button_selectoutpath,0,wx.LEFT|wx.RIGHT,10)
		output_row.Add(self.text_selectoutpath,1,wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		boxsizer.Add(output_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		self.summary_output=self._summary_box(panel,'No testing-result output folder selected.')
		boxsizer.Add(self.summary_output,0,wx.ALL|wx.EXPAND,10)
		action_row=wx.BoxSizer(wx.HORIZONTAL)
		button_test=wx.Button(panel,label='Test Detector',size=(200,40))
		button_test.Bind(wx.EVT_BUTTON,self.test_detector)
		button_test.SetToolTip('Test the selected Detector on the annotated, ground-truth testing images.')
		button_delete=wx.Button(panel,label='Delete Detector',size=(160,40))
		button_delete.Bind(wx.EVT_BUTTON,self.remove_detector)
		button_delete.SetToolTip('Permanently delete a Detector. The deletion CANNOT be restored.')
		action_row.Add(button_test,0,wx.LEFT|wx.RIGHT,10)
		action_row.Add(button_delete,0,wx.LEFT|wx.RIGHT,10)
		boxsizer.Add(action_row,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,8,0)
		self.text_testing_status=wx.TextCtrl(panel,style=wx.TE_MULTILINE|wx.TE_READONLY,size=(-1,85))
		self.text_testing_status.SetValue('Testing has not started.')
		boxsizer.Add(self.text_testing_status,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,12)
		panel.SetSizer(boxsizer)
		if not self.embedded:
			self.Centre()
		self.Show(True)


	def select_detector(self,event):
		detectors=[i for i in os.listdir(self.detector_path)if os.path.isdir(os.path.join(self.detector_path,i))]
		detectors=[i for i in detectors if i not in('__pycache__','__init__','__init__.py')]
		detectors.sort()
		if not detectors:
			wx.MessageBox('No trained Detectors were found.','Test a Detector',wx.OK|wx.ICON_INFORMATION)
			return
		dialog=wx.SingleChoiceDialog(self,message='Select a Detector to test',caption='Test a Detector',choices=detectors)
		if dialog.ShowModal()==wx.ID_OK:
			detector=dialog.GetStringSelection()
			self.path_to_detector=os.path.join(self.detector_path,detector)
			try:
				cellmapping=os.path.join(self.path_to_detector,'model_parameters.txt')
				with open(cellmapping,encoding='utf-8')as f:
					model_parameters=f.read()
				params=json.loads(model_parameters)
				cell_names=params.get('cell_names',[])
				self.text_selectdetector.SetLabel(detector)
				self.summary_detector.SetValue(
					'Selected Detector: '+detector+'\n'
					+'Location: '+self.path_to_detector+'\n'
					+'Cell categories: '+str(cell_names)
				)
			except Exception as error:
				self.text_selectdetector.SetLabel(detector)
				self.summary_detector.SetValue('Selected Detector: '+detector+'\nCould not read model parameters: '+str(error))
		dialog.Destroy()


	def select_images(self,event):
		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_testingimages=dialog.GetPath()
			self.text_selectimages.SetLabel(self.path_to_testingimages)
			image_exts=('.jpg','.jpeg','.png','.tif','.tiff')
			try:
				images=[name for name in os.listdir(self.path_to_testingimages)if name.lower().endswith(image_exts)]
				self.summary_testingimages.SetValue(
					'Testing image folder:\n'+self.path_to_testingimages+'\n'
					+str(len(images))+' supported image file(s) found.'
				)
			except Exception as error:
				self.summary_testingimages.SetValue('Testing image folder:\n'+self.path_to_testingimages+'\nCould not count images: '+str(error))
		dialog.Destroy()


	def select_annotation(self,event):
		wildcard='Annotation File (*.json)|*.json'
		dialog=wx.FileDialog(self,'Select the annotation file (.json)','',wildcard=wildcard,style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_annotation=dialog.GetPath()
			try:
				with open(self.path_to_annotation,encoding='utf-8')as f:
					info=json.load(f)
				classnames=[i['name']for i in info.get('categories',[])if i.get('id',0)>0]
				self.text_selectannotation.SetLabel(os.path.basename(self.path_to_annotation))
				self.summary_annotation.SetValue(
					'COCO annotation:\n'+self.path_to_annotation+'\n'
					+'Categories: '+str(classnames)+'\n'
					+'Images: '+str(len(info.get('images',[])))+'  |  annotations: '+str(len(info.get('annotations',[])))
				)
			except Exception as error:
				self.summary_annotation.SetValue('Could not read annotation:\n'+str(error))
		dialog.Destroy()


	def select_outpath(self,event):
		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.output_path=dialog.GetPath()
			self.text_selectoutpath.SetLabel(self.output_path)
			self.summary_output.SetValue('Testing results will be saved to:\n'+self.output_path)
		dialog.Destroy()


	def test_detector(self,event):
		if self.path_to_detector is None or self.path_to_testingimages is None or self.path_to_annotation is None or self.output_path is None:
			wx.MessageBox('No Detector / testing images / annotation file / output path selected.','Error',wx.OK|wx.ICON_ERROR)
			return
		self.text_testing_status.SetValue(
			'Testing Detector...\n'
			+'Detector: '+self.path_to_detector+'\n'
			+'Testing images: '+self.path_to_testingimages
		)
		try:
			detector=Detector()
			detector.test(self.path_to_annotation,self.path_to_testingimages,self.path_to_detector,self.output_path)
			self.text_testing_status.SetValue('Testing completed.\nResults: '+self.output_path)
		except Exception as error:
			self.text_testing_status.SetValue('Testing failed:\n'+str(error))
			raise


	def remove_detector(self,event):
		detectors=[i for i in os.listdir(self.detector_path)if os.path.isdir(os.path.join(self.detector_path,i))]
		detectors=[i for i in detectors if i not in('__pycache__','__init__','__init__.py')]
		detectors.sort()
		if not detectors:
			wx.MessageBox('No trained Detectors were found.','Delete a Detector',wx.OK|wx.ICON_INFORMATION)
			return
		dialog=wx.SingleChoiceDialog(self,message='Select a Detector to delete',caption='Delete a Detector',choices=detectors)
		if dialog.ShowModal()==wx.ID_OK:
			detector=dialog.GetStringSelection()
			dialog1=wx.MessageDialog(self,'Delete '+str(detector)+'?','CANNOT be restored!',wx.YES_NO|wx.ICON_QUESTION)
			if dialog1.ShowModal()==wx.ID_YES:
				shutil.rmtree(os.path.join(self.detector_path,detector))
				if self.path_to_detector==os.path.join(self.detector_path,detector):
					self.path_to_detector=None
					self.text_selectdetector.SetLabel('None selected.')
					self.summary_detector.SetValue('No Detector selected.')
				self.text_testing_status.SetValue('Deleted Detector: '+detector)
			dialog1.Destroy()
		dialog.Destroy()



class PanelLv2_MultiplexAnalysis(wx.ScrolledWindow):
	'''Direct section-based workspace for scalable multiplex-image analysis.'''


	def __init__(self,parent):
		super().__init__(parent,style=wx.VSCROLL|wx.HSCROLL)
		self.notebook=parent
		self.path_to_image=None
		self.series_index=0
		self.image_metadata=None
		self.path_to_detector=None
		self.detector_metadata=None
		self.segmentation_output=None
		self.segmentation_normalization=None
		self.segmentation_normalization_settings=None
		self.segmentation_thread=None
		self.segmentation_cancel_event=None
		self.coco_export_thread=None
		self.active_segmentation_settings=None
		self.active_segmentation_output=None
		self.reconciliation_output=None
		self.reconciliation_thread=None
		self.reconciliation_cancel_event=None
		self.active_reconciliation_output=None
		self.cell_region_output=None
		self.cell_region_thread=None
		self.cell_region_cancel_event=None
		self.active_cell_region_output=None
		self.quantification_channels=[]
		self.quantification_output=None
		self.quantification_thread=None
		self.quantification_cancel_event=None
		self.active_quantification_output=None
		self.clustering_marker_csv=None
		self.clustering_features=[]
		self.clustering_output=None
		self.clustering_thread=None
		self.clustering_cancel_event=None
		self.active_clustering_output=None
		self.spatial_output=None
		self.spatial_thread=None
		self.spatial_cancel_event=None
		self.active_spatial_output=None
		self.viewer_frames=[]
		self.display_window()


	def display_window(self):
		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,15,0)
		image_heading=wx.StaticText(panel,label='Module 1 — Multiplex Image')
		boxsizer.Add(image_heading,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,20)
		file_row=wx.BoxSizer(wx.HORIZONTAL)
		button_image=wx.Button(panel,label='Select TIFF / OME-TIFF / QPTIFF',size=(300,40))
		button_image.Bind(wx.EVT_BUTTON,self.select_image_file)
		wx.Button.SetToolTip(button_image,'Select a large TIFF-family multiplex image. Only metadata is opened initially.')
		button_zarr=wx.Button(panel,label='Select OME-Zarr Folder',size=(220,40))
		button_zarr.Bind(wx.EVT_BUTTON,self.select_zarr_folder)
		wx.Button.SetToolTip(button_zarr,'Select an OME-Zarr image directory.')
		file_row.Add(button_image,0,wx.RIGHT,10)
		file_row.Add(button_zarr,0)
		boxsizer.Add(file_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,8,0)
		series_row=wx.BoxSizer(wx.HORIZONTAL)
		button_series=wx.Button(panel,label='Specify Image Series',size=(300,40))
		button_series.Bind(wx.EVT_BUTTON,self.specify_series)
		wx.Button.SetToolTip(button_series,'Most images use series 0. Change this for files containing multiple image series.')
		self.text_source=wx.StaticText(panel,label='No multiplex image selected.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_MIDDLE)
		series_row.Add(button_series,0,wx.RIGHT,10)
		series_row.Add(self.text_source,1,wx.ALIGN_CENTER_VERTICAL)
		boxsizer.Add(series_row,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,8,0)
		self.metadata_text=wx.TextCtrl(panel,value='Select an image to inspect its metadata.',
			style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL,size=(-1,180))
		boxsizer.Add(self.metadata_text,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,8,0)
		image_actions=wx.BoxSizer(wx.HORIZONTAL)
		button_refresh=wx.Button(panel,label='Refresh Metadata',size=(220,40))
		button_refresh.Bind(wx.EVT_BUTTON,self.refresh_metadata)
		button_validate=wx.Button(panel,label='Validate Lazy Region Reading',size=(260,40))
		button_validate.Bind(wx.EVT_BUTTON,self.validate_lazy_read)
		wx.Button.SetToolTip(button_validate,'Read a small central region from DAPI, or channel 0, without loading the full image.')
		image_actions.Add(button_refresh,0,wx.RIGHT,10)
		image_actions.Add(button_validate,0)
		boxsizer.Add(image_actions,0,wx.LEFT|wx.RIGHT,20)
		boxsizer.Add(0,12,0)
		segmentation_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 2 — Tiled Cell Segmentation')
		method_row=wx.BoxSizer(wx.HORIZONTAL)
		method_row.Add(wx.StaticText(panel,label='Segmentation method'),0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,10)
		self.choice_segmentation_method=wx.Choice(panel,choices=['Detectron2 Detector','Intensity Threshold'],size=(220,-1))
		self.choice_segmentation_method.SetSelection(0)
		self.choice_segmentation_method.Bind(wx.EVT_CHOICE,self.on_segmentation_method_changed)
		method_row.Add(self.choice_segmentation_method,0)
		segmentation_box.Add(method_row,0,wx.ALL,8)
		tiling_settings_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Tiling and resolution')
		tiling_settings=wx.FlexGridSizer(2,4,4,12)
		for label in('Square processing tile size','Overlap X ratio','Overlap Y ratio','Pyramid level'):
			tiling_settings.Add(wx.StaticText(panel,label=label),0,wx.ALIGN_CENTER_VERTICAL)
		self.text_tile_size=wx.TextCtrl(panel,value='Select detector',style=wx.TE_READONLY,size=(180,-1))
		self.text_tile_size.SetToolTip('Detectron2 uses the Detector inference frame size. Intensity Threshold uses its configured square processing tile size.')
		self.spin_overlap_ratio_x=wx.SpinCtrlDouble(panel,min=0,max=0.95,initial=0.10,inc=0.01,size=(120,-1))
		self.spin_overlap_ratio_y=wx.SpinCtrlDouble(panel,min=0,max=0.95,initial=0.10,inc=0.01,size=(120,-1))
		self.spin_overlap_ratio_x.SetDigits(4)
		self.spin_overlap_ratio_y.SetDigits(4)
		self.spin_overlap_ratio_x.SetToolTip('Fraction of tile width shared with the adjacent tile; 0.10 means 10%.')
		self.spin_overlap_ratio_y.SetToolTip('Fraction of tile height shared with the adjacent tile; 0.10 means 10%.')
		self.spin_level=wx.SpinCtrl(panel,min=0,max=0,initial=0,size=(120,-1))
		for control in(self.text_tile_size,self.spin_overlap_ratio_x,self.spin_overlap_ratio_y,self.spin_level):
			tiling_settings.Add(control,0)
		tiling_settings_box.Add(tiling_settings,0,wx.ALL,8)
		tiling_note=wx.StaticText(panel,label=('Tiling is created automatically when segmentation starts. Resume checkpoints are also managed automatically inside the selected segmentation output folder.'))
		tiling_note.Wrap(1000)
		tiling_settings_box.Add(tiling_note,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		segmentation_box.Add(tiling_settings_box,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		detector_row=wx.BoxSizer(wx.HORIZONTAL)
		self.button_detector=wx.Button(panel,label='Select MPlexA Detector',size=(220,40))
		self.button_detector.Bind(wx.EVT_BUTTON,self.select_segmentation_detector)
		self.text_detector=wx.StaticText(panel,label='Detector: not selected.',style=wx.ST_ELLIPSIZE_MIDDLE)
		detector_row.Add(self.button_detector,0,wx.RIGHT,10)
		detector_row.Add(self.text_detector,1,wx.ALIGN_CENTER_VERTICAL)
		segmentation_box.Add(detector_row,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		segmentation_settings=wx.FlexGridSizer(2,6,4,12)
		segmentation_settings.Add(wx.StaticText(panel,label='Segmentation channel'),0,wx.ALIGN_CENTER_VERTICAL)
		segmentation_settings.Add(wx.StaticText(panel,label='Detector score threshold'),0,wx.ALIGN_CENTER_VERTICAL)
		segmentation_settings.Add(wx.StaticText(panel,label='Low percentile'),0,wx.ALIGN_CENTER_VERTICAL)
		segmentation_settings.Add(wx.StaticText(panel,label='High percentile'),0,wx.ALIGN_CENTER_VERTICAL)
		segmentation_settings.Add(wx.StaticText(panel,label='Normalization samples'),0,wx.ALIGN_CENTER_VERTICAL)
		segmentation_settings.Add(wx.StaticText(panel,label='Detector batch size'),0,wx.ALIGN_CENTER_VERTICAL)
		self.choice_dapi_channel=wx.Choice(panel,choices=['Select image first'],size=(180,-1))
		self.choice_dapi_channel.SetSelection(0)
		self.choice_dapi_channel.Bind(wx.EVT_CHOICE,self.clear_segmentation_normalization)
		self.spin_score_threshold=wx.SpinCtrlDouble(panel,min=0,max=1,initial=0.5,inc=0.05,size=(120,-1))
		self.spin_score_threshold.SetDigits(2)
		self.spin_low_percentile=wx.SpinCtrlDouble(panel,min=0,max=99.99,initial=1.0,inc=0.1,size=(120,-1))
		self.spin_low_percentile.SetDigits(2)
		self.spin_high_percentile=wx.SpinCtrlDouble(panel,min=0.01,max=100,initial=99.8,inc=0.1,size=(120,-1))
		self.spin_high_percentile.SetDigits(2)
		self.spin_normalization_samples=wx.SpinCtrl(panel,min=1,max=256,initial=16,size=(120,-1))
		self.spin_segmentation_batch=wx.SpinCtrl(panel,min=1,max=64,initial=1,size=(120,-1))
		segmentation_settings.Add(self.choice_dapi_channel,0)
		segmentation_settings.Add(self.spin_score_threshold,0)
		segmentation_settings.Add(self.spin_low_percentile,0)
		segmentation_settings.Add(self.spin_high_percentile,0)
		segmentation_settings.Add(self.spin_normalization_samples,0)
		segmentation_settings.Add(self.spin_segmentation_batch,0)
		segmentation_box.Add(segmentation_settings,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		threshold_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Intensity Threshold — Adaptive Watershed Settings')
		threshold_settings=wx.FlexGridSizer(2,11,4,10)
		for label in('Mean threshold (0–255)','Foreground','Background radius','Median radius','Gaussian sigma','Min area (px²)','Max area (px²)','Shape split distance','Square tile size','CPU workers (0=Auto)','Requested pixel size (0=manual level)'):
			threshold_settings.Add(wx.StaticText(panel,label=label),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_threshold_value=wx.SpinCtrlDouble(panel,min=0,max=255,initial=25,inc=1,size=(120,-1))
		self.spin_threshold_value.SetDigits(1)
		self.choice_threshold_foreground=wx.Choice(panel,choices=['Bright objects','Dark objects'],size=(150,-1))
		self.choice_threshold_foreground.SetSelection(0)
		self.spin_threshold_background_radius=wx.SpinCtrl(panel,min=0,max=512,initial=15,size=(120,-1))
		self.spin_threshold_median_radius=wx.SpinCtrl(panel,min=0,max=64,initial=0,size=(120,-1))
		self.spin_threshold_sigma=wx.SpinCtrlDouble(panel,min=0,max=20,initial=3.0,inc=0.25,size=(120,-1))
		self.spin_threshold_sigma.SetDigits(2)
		self.spin_threshold_min_area=wx.SpinCtrl(panel,min=1,max=10000000,initial=10,size=(130,-1))
		self.spin_threshold_max_area=wx.SpinCtrl(panel,min=1,max=100000000,initial=1000,size=(140,-1))
		self.spin_threshold_min_distance=wx.SpinCtrl(panel,min=1,max=256,initial=3,size=(130,-1))
		self.spin_threshold_tile_size=wx.SpinCtrl(panel,min=128,max=8192,initial=2048,size=(130,-1))
		self.spin_threshold_tile_size.Bind(wx.EVT_SPINCTRL,self.on_threshold_tile_size_changed)
		self.spin_threshold_workers=wx.SpinCtrl(panel,min=0,max=64,initial=0,size=(130,-1))
		self.spin_threshold_workers.SetToolTip('0 selects a conservative automatic worker count (up to 4). Increase only if sufficient RAM is available.')
		self.spin_threshold_requested_pixel_size=wx.SpinCtrlDouble(panel,min=0,max=10000,initial=0,inc=0.1,size=(150,-1))
		self.spin_threshold_requested_pixel_size.SetDigits(3)
		self.spin_threshold_requested_pixel_size.SetToolTip('0 uses the Pyramid level control. A positive value selects the closest native pyramid level using image physical-pixel metadata; MPlexA does not resample between levels.')
		for control in(self.spin_threshold_value,self.choice_threshold_foreground,self.spin_threshold_background_radius,
			self.spin_threshold_median_radius,self.spin_threshold_sigma,self.spin_threshold_min_area,
			self.spin_threshold_max_area,self.spin_threshold_min_distance,self.spin_threshold_tile_size,self.spin_threshold_workers,
			self.spin_threshold_requested_pixel_size):
			threshold_settings.Add(control,0)
		threshold_box.Add(threshold_settings,0,wx.ALL,6)
		self.checkbox_threshold_background_reconstruction=wx.CheckBox(panel,label='Use opening by reconstruction for background estimation (recommended)')
		self.checkbox_threshold_background_reconstruction.SetValue(True)
		threshold_box.Add(self.checkbox_threshold_background_reconstruction,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,6)
		self.checkbox_threshold_split=wx.CheckBox(panel,label='Split merged objects by shape using distance-transform watershed')
		self.checkbox_threshold_split.SetValue(True)
		threshold_box.Add(self.checkbox_threshold_split,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,6)
		self.checkbox_threshold_refine=wx.CheckBox(panel,label='Refine boundaries after Gaussian/LoG detection')
		self.checkbox_threshold_refine.SetValue(True)
		threshold_box.Add(self.checkbox_threshold_refine,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,6)
		self.checkbox_threshold_fast_archives=wx.CheckBox(panel,label='Fast tile archives (recommended; masks are already bit-packed)')
		self.checkbox_threshold_fast_archives.SetValue(True)
		threshold_box.Add(self.checkbox_threshold_fast_archives,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,6)
		threshold_note=wx.StaticText(panel,label=(
			'Adaptive watershed detection estimates local background, detects regional maxima in a Gaussian/Laplacian response, '
			'filters candidate nuclei by mean intensity, and resolves merged shapes. MPlexA also keeps only detections whose '
			'centroids lie in each overlapping tile\'s central ownership region, preventing tile-edge artifacts from entering Module 3.'))
		threshold_note.Wrap(1100)
		threshold_box.Add(threshold_note,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,6)
		segmentation_box.Add(threshold_box,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.threshold_controls=[self.spin_threshold_value,self.choice_threshold_foreground,
			self.spin_threshold_background_radius,self.spin_threshold_median_radius,self.spin_threshold_sigma,
			self.spin_threshold_min_area,self.spin_threshold_max_area,self.spin_threshold_min_distance,
			self.spin_threshold_tile_size,self.spin_threshold_workers,self.spin_threshold_requested_pixel_size,self.checkbox_threshold_background_reconstruction,
			self.checkbox_threshold_split,self.checkbox_threshold_refine,self.checkbox_threshold_fast_archives]
		self.checkbox_retry_failed=wx.CheckBox(panel,label='Retry tiles currently marked failed when resuming')
		segmentation_box.Add(self.checkbox_retry_failed,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		output_row=wx.BoxSizer(wx.HORIZONTAL)
		button_output=wx.Button(panel,label='Select Segmentation Output',size=(250,40))
		button_output.Bind(wx.EVT_BUTTON,self.select_segmentation_output)
		self.text_segmentation_output=wx.StaticText(panel,label='Output: not selected.',style=wx.ST_ELLIPSIZE_MIDDLE)
		output_row.Add(button_output,0,wx.RIGHT,10)
		output_row.Add(self.text_segmentation_output,1,wx.ALIGN_CENTER_VERTICAL)
		segmentation_box.Add(output_row,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		segmentation_actions=wx.BoxSizer(wx.HORIZONTAL)
		self.button_normalization=wx.Button(panel,label='Estimate Channel Normalization',size=(250,40))
		self.button_normalization.Bind(wx.EVT_BUTTON,self.estimate_dapi_normalization)
		self.button_run_segmentation=wx.Button(panel,label='Start / Resume Segmentation',size=(250,40))
		self.button_run_segmentation.Bind(wx.EVT_BUTTON,self.start_dapi_segmentation)
		self.button_cancel_segmentation=wx.Button(panel,label='Cancel After Current Batch',size=(240,40))
		self.button_cancel_segmentation.Bind(wx.EVT_BUTTON,self.cancel_dapi_segmentation)
		self.button_cancel_segmentation.Disable()
		segmentation_actions.Add(self.button_normalization,0,wx.RIGHT,10)
		segmentation_actions.Add(self.button_run_segmentation,0,wx.RIGHT,10)
		segmentation_actions.Add(self.button_cancel_segmentation,0)
		segmentation_box.Add(segmentation_actions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		segmentation_qc_actions=wx.BoxSizer(wx.HORIZONTAL)
		self.button_view_segmentation=wx.Button(panel,label='Open Segmentation Viewer',size=(250,40))
		self.button_view_segmentation.Bind(wx.EVT_BUTTON,self.open_segmentation_viewer)
		self.button_export_coco=wx.Button(panel,label='Export Segmentation Masks to COCO JSON',size=(330,40))
		self.button_export_coco.Bind(wx.EVT_BUTTON,self.export_segmentation_coco)
		self.button_export_coco.SetToolTip('Export core-owned Module 2 masks as polygon-based COCO instance annotations compatible with EZannot-style annotation files.')
		segmentation_qc_actions.Add(self.button_view_segmentation,0,wx.RIGHT,10)
		segmentation_qc_actions.Add(self.button_export_coco,0)
		segmentation_box.Add(segmentation_qc_actions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		self.gauge_segmentation=wx.Gauge(panel,range=1000,size=(-1,22))
		segmentation_box.Add(self.gauge_segmentation,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.segmentation_text=wx.TextCtrl(panel,value=(
			'Select either Detectron2 Detector or Intensity Threshold. Both methods operate on overlapping square tiles and save the same compact mask archives, so Module 3 and all downstream analyses are unchanged.'),
			style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL,size=(-1,170))
		segmentation_box.Add(self.segmentation_text,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		boxsizer.Add(segmentation_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,10,0)
		reconciliation_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 3 — Global Cell IDs and Chunked Instance Labels')
		reconciliation_input=wx.BoxSizer(wx.HORIZONTAL)
		self.text_reconciliation_input=wx.StaticText(
			panel,label='Module 2 output: use the selected segmentation output.',
			style=wx.ST_ELLIPSIZE_MIDDLE)
		button_reconciliation_output=wx.Button(panel,label='Select Module 3 Output',size=(230,40))
		button_reconciliation_output.Bind(wx.EVT_BUTTON,self.select_reconciliation_output)
		reconciliation_input.Add(button_reconciliation_output,0,wx.RIGHT,10)
		reconciliation_input.Add(self.text_reconciliation_input,1,wx.ALIGN_CENTER_VERTICAL)
		reconciliation_box.Add(reconciliation_input,0,wx.ALL|wx.EXPAND,8)
		reconciliation_settings=wx.FlexGridSizer(2,4,4,12)
		reconciliation_settings.Add(wx.StaticText(panel,label='Mask IoU threshold'),0,wx.ALIGN_CENTER_VERTICAL)
		reconciliation_settings.Add(wx.StaticText(panel,label='Containment threshold'),0,wx.ALIGN_CENTER_VERTICAL)
		reconciliation_settings.Add(wx.StaticText(panel,label='Duplicate-mask strategy'),0,wx.ALIGN_CENTER_VERTICAL)
		reconciliation_settings.Add(wx.StaticText(panel,label='Label chunk size'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_reconciliation_iou=wx.SpinCtrlDouble(panel,min=0,max=1,initial=0.30,inc=0.05,size=(130,-1))
		self.spin_reconciliation_iou.SetDigits(2)
		self.spin_reconciliation_containment=wx.SpinCtrlDouble(panel,min=0,max=1,initial=0.65,inc=0.05,size=(130,-1))
		self.spin_reconciliation_containment.SetDigits(2)
		self.choice_reconciliation_strategy=wx.Choice(
			panel,choices=['Best-quality mask','Union duplicate masks'],size=(190,-1))
		self.choice_reconciliation_strategy.SetSelection(0)
		self.spin_label_chunk_size=wx.SpinCtrl(panel,min=128,max=8192,initial=1024,size=(130,-1))
		reconciliation_settings.Add(self.spin_reconciliation_iou,0)
		reconciliation_settings.Add(self.spin_reconciliation_containment,0)
		reconciliation_settings.Add(self.choice_reconciliation_strategy,0)
		reconciliation_settings.Add(self.spin_label_chunk_size,0)
		reconciliation_box.Add(reconciliation_settings,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		reconciliation_options=wx.BoxSizer(wx.HORIZONTAL)
		self.checkbox_same_class=wx.CheckBox(panel,label='Only reconcile predictions with the same class')
		self.checkbox_same_class.SetValue(True)
		self.checkbox_retry_label_chunks=wx.CheckBox(panel,label='Retry failed label chunks once automatically')
		self.checkbox_retry_label_chunks.SetValue(True)
		reconciliation_options.Add(self.checkbox_same_class,0,wx.RIGHT,25)
		reconciliation_options.Add(self.checkbox_retry_label_chunks,0)
		reconciliation_box.Add(reconciliation_options,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		reconciliation_actions=wx.BoxSizer(wx.HORIZONTAL)
		self.button_run_reconciliation=wx.Button(panel,label='Start / Resume Module 3',size=(240,40))
		self.button_run_reconciliation.Bind(wx.EVT_BUTTON,self.start_reconciliation)
		self.button_cancel_reconciliation=wx.Button(panel,label='Cancel After Current Unit',size=(240,40))
		self.button_cancel_reconciliation.Bind(wx.EVT_BUTTON,self.cancel_reconciliation)
		self.button_cancel_reconciliation.Disable()
		reconciliation_actions.Add(self.button_run_reconciliation,0,wx.RIGHT,10)
		reconciliation_actions.Add(self.button_cancel_reconciliation,0)
		reconciliation_box.Add(reconciliation_actions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		self.gauge_reconciliation=wx.Gauge(panel,range=1000,size=(-1,22))
		reconciliation_box.Add(self.gauge_reconciliation,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.reconciliation_text=wx.TextCtrl(panel,value=(
			'After Module 2 finishes, this stage compares exact masks in overlapping tiles, groups duplicate predictions, '
			'assigns deterministic global cell IDs, and writes a resumable chunked instance-label image. '
			'Best-quality mask prefers the core-owned, non-edge, highest-confidence prediction; union mode combines duplicate masks.'),
			style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL,size=(-1,180))
		reconciliation_box.Add(self.reconciliation_text,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		boxsizer.Add(reconciliation_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,10,0)
		region_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 4A — Cell-Region Generation')
		region_input=wx.BoxSizer(wx.HORIZONTAL)
		button_region_output=wx.Button(panel,label='Select Cell-Region Output',size=(240,40))
		button_region_output.Bind(wx.EVT_BUTTON,self.select_cell_region_output)
		self.text_region_input=wx.StaticText(panel,label='Module 3 output: use the selected global-instance output.',style=wx.ST_ELLIPSIZE_MIDDLE)
		region_input.Add(button_region_output,0,wx.RIGHT,10)
		region_input.Add(self.text_region_input,1,wx.ALIGN_CENTER_VERTICAL)
		region_box.Add(region_input,0,wx.ALL|wx.EXPAND,8)
		region_settings=wx.FlexGridSizer(2,5,4,12)
		region_settings.Add(wx.StaticText(panel,label='Cell-region mode'),0,wx.ALIGN_CENTER_VERTICAL)
		region_settings.Add(wx.StaticText(panel,label='Maximum expansion (px)'),0,wx.ALIGN_CENTER_VERTICAL)
		region_settings.Add(wx.StaticText(panel,label='Membrane channel'),0,wx.ALIGN_CENTER_VERTICAL)
		region_settings.Add(wx.StaticText(panel,label='Membrane smoothing sigma'),0,wx.ALIGN_CENTER_VERTICAL)
		region_settings.Add(wx.StaticText(panel,label='Output chunk size'),0,wx.ALIGN_CENTER_VERTICAL)
		self.choice_region_mode=wx.Choice(panel,choices=[
			'Nuclear masks only','Fixed-distance expansion','Voronoi-constrained expansion','Membrane-guided watershed'],size=(220,-1))
		self.choice_region_mode.SetSelection(0)
		self.choice_region_mode.Bind(wx.EVT_CHOICE,self.update_region_mode_controls)
		self.spin_region_distance=wx.SpinCtrl(panel,min=0,max=10000,initial=12,size=(130,-1))
		self.choice_membrane_channel=wx.Choice(panel,choices=['Select image first'],size=(190,-1))
		self.choice_membrane_channel.SetSelection(0)
		self.spin_membrane_sigma=wx.SpinCtrlDouble(panel,min=0,max=20,initial=1.0,inc=0.25,size=(130,-1))
		self.spin_membrane_sigma.SetDigits(2)
		self.spin_region_chunk_size=wx.SpinCtrl(panel,min=128,max=8192,initial=1024,size=(130,-1))
		region_settings.Add(self.choice_region_mode,0)
		region_settings.Add(self.spin_region_distance,0)
		region_settings.Add(self.choice_membrane_channel,0)
		region_settings.Add(self.spin_membrane_sigma,0)
		region_settings.Add(self.spin_region_chunk_size,0)
		region_box.Add(region_settings,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		self.checkbox_retry_region_chunks=wx.CheckBox(panel,label='Retry failed cell-region chunks when resuming')
		region_box.Add(self.checkbox_retry_region_chunks,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		region_actions=wx.BoxSizer(wx.HORIZONTAL)
		self.button_run_regions=wx.Button(panel,label='Start / Resume Cell Regions',size=(250,40))
		self.button_run_regions.Bind(wx.EVT_BUTTON,self.start_cell_regions)
		self.button_cancel_regions=wx.Button(panel,label='Cancel After Current Chunk',size=(250,40))
		self.button_cancel_regions.Bind(wx.EVT_BUTTON,self.cancel_cell_regions)
		self.button_cancel_regions.Disable()
		region_actions.Add(self.button_run_regions,0,wx.RIGHT,10)
		region_actions.Add(self.button_cancel_regions,0)
		region_box.Add(region_actions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		self.gauge_regions=wx.Gauge(panel,range=1000,size=(-1,22))
		region_box.Add(self.gauge_regions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.region_text=wx.TextCtrl(panel,value=(
			'Nuclear mode preserves the DAPI masks. Fixed expansion grows from nuclear boundaries; Voronoi uses nuclear centroids; '
			'membrane-guided watershed follows gradients in a selected channel. All modes preserve Module 3 global cell IDs.'),
			style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL,size=(-1,150))
		region_box.Add(self.region_text,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		boxsizer.Add(region_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,10,0)
		quant_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 4B — Multiplex Marker Quantification')
		quant_input=wx.BoxSizer(wx.HORIZONTAL)
		button_quant_channels=wx.Button(panel,label='Select Quantification Channels',size=(260,40))
		button_quant_channels.Bind(wx.EVT_BUTTON,self.select_quantification_channels)
		self.text_quant_channels=wx.StaticText(panel,label='Channels: all channels after an image is selected.',style=wx.ST_ELLIPSIZE_MIDDLE)
		quant_input.Add(button_quant_channels,0,wx.RIGHT,10)
		quant_input.Add(self.text_quant_channels,1,wx.ALIGN_CENTER_VERTICAL)
		quant_box.Add(quant_input,0,wx.ALL|wx.EXPAND,8)
		quant_settings=wx.FlexGridSizer(2,5,4,12)
		quant_settings.Add(wx.StaticText(panel,label='Channel batch size'),0,wx.ALIGN_CENTER_VERTICAL)
		quant_settings.Add(wx.StaticText(panel,label='Positive threshold'),0,wx.ALIGN_CENTER_VERTICAL)
		quant_settings.Add(wx.StaticText(panel,label='Cytoplasmic-ring width (px)'),0,wx.ALIGN_CENTER_VERTICAL)
		quant_settings.Add(wx.StaticText(panel,label='Membrane-ring width (px)'),0,wx.ALIGN_CENTER_VERTICAL)
		quant_settings.Add(wx.StaticText(panel,label='Output formats'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_quant_batch=wx.SpinCtrl(panel,min=1,max=256,initial=8,size=(130,-1))
		self.spin_positive_threshold=wx.SpinCtrlDouble(panel,min=-1e12,max=1e12,initial=0,inc=1,size=(150,-1))
		self.spin_positive_threshold.SetDigits(3)
		self.spin_cytoplasmic_ring=wx.SpinCtrl(panel,min=0,max=1000,initial=3,size=(130,-1))
		self.spin_membrane_ring=wx.SpinCtrl(panel,min=0,max=1000,initial=2,size=(130,-1))
		format_panel=wx.Panel(panel)
		format_sizer=wx.BoxSizer(wx.HORIZONTAL)
		self.checkbox_export_csv=wx.CheckBox(format_panel,label='CSV')
		self.checkbox_export_csv.SetValue(True)
		self.checkbox_export_excel=wx.CheckBox(format_panel,label='Excel')
		format_sizer.Add(self.checkbox_export_csv,0,wx.RIGHT,10)
		format_sizer.Add(self.checkbox_export_excel,0)
		format_panel.SetSizer(format_sizer)
		quant_settings.Add(self.spin_quant_batch,0)
		quant_settings.Add(self.spin_positive_threshold,0)
		quant_settings.Add(self.spin_cytoplasmic_ring,0)
		quant_settings.Add(self.spin_membrane_ring,0)
		quant_settings.Add(format_panel,0)
		quant_box.Add(quant_settings,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		quant_output_row=wx.BoxSizer(wx.HORIZONTAL)
		button_quant_output=wx.Button(panel,label='Select Quantification Output',size=(260,40))
		button_quant_output.Bind(wx.EVT_BUTTON,self.select_quantification_output)
		self.text_quant_output=wx.StaticText(panel,label='Output: created inside the cell-region folder.',style=wx.ST_ELLIPSIZE_MIDDLE)
		quant_output_row.Add(button_quant_output,0,wx.RIGHT,10)
		quant_output_row.Add(self.text_quant_output,1,wx.ALIGN_CENTER_VERTICAL)
		quant_box.Add(quant_output_row,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.checkbox_retry_quant_units=wx.CheckBox(panel,label='Retry failed chunk/channel units when resuming')
		quant_box.Add(self.checkbox_retry_quant_units,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		quant_actions=wx.BoxSizer(wx.HORIZONTAL)
		self.button_run_quantification=wx.Button(panel,label='Start / Resume Quantification',size=(270,40))
		self.button_run_quantification.Bind(wx.EVT_BUTTON,self.start_quantification)
		self.button_cancel_quantification=wx.Button(panel,label='Cancel After Current Unit',size=(250,40))
		self.button_cancel_quantification.Bind(wx.EVT_BUTTON,self.cancel_quantification)
		self.button_cancel_quantification.Disable()
		quant_actions.Add(self.button_run_quantification,0,wx.RIGHT,10)
		quant_actions.Add(self.button_cancel_quantification,0)
		quant_box.Add(quant_actions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		self.gauge_quantification=wx.Gauge(panel,range=1000,size=(-1,22))
		quant_box.Add(self.gauge_quantification,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.quantification_text=wx.TextCtrl(panel,value=(
			'MPlexA reads one spatial chunk and a small channel batch at a time. It reports whole-cell mean, sum, maximum, minimum, '
			'standard deviation, positive-pixel fraction, background-corrected mean, nuclear mean, cytoplasmic mean, '
			'cytoplasmic-ring mean, and membrane-ring mean.'),
			style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL,size=(-1,180))
		quant_box.Add(self.quantification_text,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		boxsizer.Add(quant_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,10,0)
		cluster_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 5A — Cell Clustering and Phenotyping')
		cluster_input=wx.BoxSizer(wx.HORIZONTAL)
		button_marker_table=wx.Button(panel,label='Select Marker CSV',size=(200,40))
		button_marker_table.Bind(wx.EVT_BUTTON,self.select_clustering_marker_csv)
		button_cluster_features=wx.Button(panel,label='Select Clustering Markers',size=(230,40))
		button_cluster_features.Bind(wx.EVT_BUTTON,self.select_clustering_features)
		self.text_cluster_input=wx.StaticText(panel,label='Input: use Module 4 marker CSV when available.',style=wx.ST_ELLIPSIZE_MIDDLE)
		cluster_input.Add(button_marker_table,0,wx.RIGHT,10)
		cluster_input.Add(button_cluster_features,0,wx.RIGHT,10)
		cluster_input.Add(self.text_cluster_input,1,wx.ALIGN_CENTER_VERTICAL)
		cluster_box.Add(cluster_input,0,wx.ALL|wx.EXPAND,8)
		cluster_settings=wx.FlexGridSizer(2,7,4,12)
		for label in('Transform','Arcsinh cofactor','PCs','Method','Clusters / resolution','Embedding','Fit sample size'):
			cluster_settings.Add(wx.StaticText(panel,label=label),0,wx.ALIGN_CENTER_VERTICAL)
		self.choice_cluster_transform=wx.Choice(panel,choices=['Arcsinh','None','Signed log1p'],size=(135,-1));self.choice_cluster_transform.SetSelection(0)
		self.spin_cluster_cofactor=wx.SpinCtrlDouble(panel,min=0.01,max=100000,initial=5.0,inc=0.5,size=(120,-1));self.spin_cluster_cofactor.SetDigits(2)
		self.spin_cluster_pcs=wx.SpinCtrl(panel,min=1,max=200,initial=20,size=(90,-1))
		self.choice_cluster_method=wx.Choice(panel,choices=['Leiden (default)','K-means'],size=(165,-1));self.choice_cluster_method.SetSelection(0);self.choice_cluster_method.Bind(wx.EVT_CHOICE,self.on_cluster_method_changed)
		self.spin_cluster_count=wx.SpinCtrlDouble(panel,min=0.01,max=1000,initial=1.0,inc=0.1,size=(120,-1));self.spin_cluster_count.SetDigits(2)
		self.choice_embedding=wx.Choice(panel,choices=['UMAP (default)','PCA'],size=(150,-1));self.choice_embedding.SetSelection(0)
		self.spin_cluster_sample=wx.SpinCtrl(panel,min=100,max=5000000,initial=100000,size=(130,-1))
		for control in(self.choice_cluster_transform,self.spin_cluster_cofactor,self.spin_cluster_pcs,self.choice_cluster_method,self.spin_cluster_count,self.choice_embedding,self.spin_cluster_sample):
			cluster_settings.Add(control,0)
		cluster_box.Add(cluster_settings,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		cluster_output_row=wx.BoxSizer(wx.HORIZONTAL)
		button_cluster_output=wx.Button(panel,label='Select Clustering Output',size=(230,40));button_cluster_output.Bind(wx.EVT_BUTTON,self.select_clustering_output)
		self.text_cluster_output=wx.StaticText(panel,label='Output: created beside marker quantification.',style=wx.ST_ELLIPSIZE_MIDDLE)
		cluster_output_row.Add(button_cluster_output,0,wx.RIGHT,10);cluster_output_row.Add(self.text_cluster_output,1,wx.ALIGN_CENTER_VERTICAL)
		cluster_box.Add(cluster_output_row,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		cluster_actions=wx.BoxSizer(wx.HORIZONTAL)
		self.button_run_clustering=wx.Button(panel,label='Run Cell Clustering',size=(220,40));self.button_run_clustering.Bind(wx.EVT_BUTTON,self.start_clustering)
		self.button_cancel_clustering=wx.Button(panel,label='Cancel Clustering',size=(190,40));self.button_cancel_clustering.Bind(wx.EVT_BUTTON,self.cancel_clustering);self.button_cancel_clustering.Disable()
		button_rename_cluster=wx.Button(panel,label='Rename Phenotype',size=(190,40));button_rename_cluster.Bind(wx.EVT_BUTTON,self.rename_cluster)
		cluster_actions.Add(self.button_run_clustering,0,wx.RIGHT,10);cluster_actions.Add(self.button_cancel_clustering,0,wx.RIGHT,10);cluster_actions.Add(button_rename_cluster,0)
		cluster_box.Add(cluster_actions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		self.gauge_clustering=wx.Gauge(panel,range=1000,size=(-1,22));cluster_box.Add(self.gauge_clustering,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.clustering_text=wx.TextCtrl(panel,value=('Select marker measurements for phenotyping. The default workflow is UMAP + Leiden: PCA first compresses marker space for scalable neighbor construction, Leiden defines phenotypes, and UMAP provides the 2-D embedding. PCA + K-means remains available as an alternative.'),style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL,size=(-1,150))
		cluster_box.Add(self.clustering_text,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		boxsizer.Add(cluster_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,10,0)
		spatial_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 5B — Spatial Interaction Graphs')
		spatial_settings=wx.FlexGridSizer(2,5,4,12)
		for label in('Graph method','Radius','k neighbors','Distance units','Query block'):
			spatial_settings.Add(wx.StaticText(panel,label=label),0,wx.ALIGN_CENTER_VERTICAL)
		self.choice_graph_method=wx.Choice(panel,choices=['Radius','k-nearest neighbors','Delaunay','Direct cell contact'],size=(190,-1));self.choice_graph_method.SetSelection(0)
		self.spin_graph_radius=wx.SpinCtrlDouble(panel,min=0.01,max=100000,initial=30,inc=1,size=(120,-1));self.spin_graph_radius.SetDigits(2)
		self.spin_graph_k=wx.SpinCtrl(panel,min=1,max=1000,initial=6,size=(100,-1))
		self.checkbox_graph_physical=wx.CheckBox(panel,label='Use image physical units');self.checkbox_graph_physical.SetValue(True)
		self.spin_graph_block=wx.SpinCtrl(panel,min=100,max=100000,initial=5000,size=(120,-1))
		for control in(self.choice_graph_method,self.spin_graph_radius,self.spin_graph_k,self.checkbox_graph_physical,self.spin_graph_block):
			spatial_settings.Add(control,0)
		spatial_box.Add(spatial_settings,0,wx.ALL,8)
		spatial_output_row=wx.BoxSizer(wx.HORIZONTAL)
		button_spatial_output=wx.Button(panel,label='Select Spatial Output',size=(220,40));button_spatial_output.Bind(wx.EVT_BUTTON,self.select_spatial_output)
		self.text_spatial_output=wx.StaticText(panel,label='Output: created inside the clustering folder.',style=wx.ST_ELLIPSIZE_MIDDLE)
		spatial_output_row.Add(button_spatial_output,0,wx.RIGHT,10);spatial_output_row.Add(self.text_spatial_output,1,wx.ALIGN_CENTER_VERTICAL)
		spatial_box.Add(spatial_output_row,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		spatial_actions=wx.BoxSizer(wx.HORIZONTAL)
		self.button_run_spatial=wx.Button(panel,label='Build Spatial Graph',size=(220,40));self.button_run_spatial.Bind(wx.EVT_BUTTON,self.start_spatial_graph)
		self.button_cancel_spatial=wx.Button(panel,label='Cancel Graph',size=(180,40));self.button_cancel_spatial.Bind(wx.EVT_BUTTON,self.cancel_spatial_graph);self.button_cancel_spatial.Disable()
		self.button_view_spatial=wx.Button(panel,label='Open Spatial Graph Viewer',size=(235,40));self.button_view_spatial.Bind(wx.EVT_BUTTON,self.open_spatial_graph_viewer)
		spatial_actions.Add(self.button_run_spatial,0,wx.RIGHT,10);spatial_actions.Add(self.button_cancel_spatial,0,wx.RIGHT,10);spatial_actions.Add(self.button_view_spatial,0)
		spatial_box.Add(spatial_actions,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		self.gauge_spatial=wx.Gauge(panel,range=1000,size=(-1,22));spatial_box.Add(self.gauge_spatial,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.spatial_text=wx.TextCtrl(panel,value=('Build radius, kNN, Delaunay, or direct-contact cell graphs. MPlexA exports sparse edges and interaction enrichment, then the Spatial Graph Viewer can pan/zoom the network and filter edges by phenotype pair.'),style=wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL,size=(-1,130))
		spatial_box.Add(self.spatial_text,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		boxsizer.Add(spatial_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,10,0)
		viewer_box=wx.StaticBoxSizer(wx.VERTICAL,panel,label='Module 5C — Multichannel Whole-Slide Viewer')
		viewer_actions=wx.BoxSizer(wx.HORIZONTAL)
		button_viewer=wx.Button(panel,label='Open Multichannel Viewer',size=(250,44));button_viewer.Bind(wx.EVT_BUTTON,self.open_multichannel_viewer)
		viewer_actions.Add(button_viewer,0,wx.RIGHT,10)
		viewer_box.Add(viewer_actions,0,wx.ALL,8)
		viewer_description=wx.StaticText(panel,label=('The viewer reads only the current field of view and pyramid level. Search hundreds of channels, assign colors, adjust contrast/gamma/opacity, pan/zoom, overlay cell boundaries and phenotype centroids, double-click a cell to inspect its selected marker profile, and save the current view.'))
		viewer_description.Wrap(900);viewer_box.Add(viewer_description,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		boxsizer.Add(viewer_box,0,wx.LEFT|wx.RIGHT|wx.EXPAND,20)
		boxsizer.Add(0,10,0)
		boxsizer.Add(0,15,0)
		self.update_region_mode_controls(None)
		self._update_segmentation_method_controls()
		panel.SetSizer(boxsizer)
		self.SetScrollRate(10,10)
		self.FitInside()
		self.Show(True)


	def select_image_file(self,event):
		wildcard='Multiplex TIFF files (*.tif/*.tiff/*.qptiff)|*.tif;*.TIF;*.tiff;*.TIFF;*.qptiff;*.QPTIFF'
		dialog=wx.FileDialog(self,'Select a multiplex image','','',wildcard,style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_image=dialog.GetPath()
			self.segmentation_output=None
			self.segmentation_normalization=None
			self.segmentation_normalization_settings=None
			self.reconciliation_output=None
			self.cell_region_output=None
			self.quantification_output=None
			self.quantification_channels=[]
			self._inspect_selected_image()
		dialog.Destroy()


	def select_zarr_folder(self,event):
		dialog=wx.DirDialog(self,'Select an OME-Zarr image folder','',style=wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_image=dialog.GetPath()
			self.segmentation_output=None
			self.segmentation_normalization=None
			self.segmentation_normalization_settings=None
			self.reconciliation_output=None
			self.cell_region_output=None
			self.quantification_output=None
			self.quantification_channels=[]
			self._inspect_selected_image()
		dialog.Destroy()


	def specify_series(self,event):
		dialog=wx.NumberEntryDialog(self,'Enter the TIFF series or OME-Zarr multiscale index','The first series is 0:',
			'Image series',self.series_index,0,100000)
		if dialog.ShowModal()==wx.ID_OK:
			self.series_index=int(dialog.GetValue())
			self.segmentation_output=None
			self.segmentation_normalization=None
			self.segmentation_normalization_settings=None
			self.reconciliation_output=None
			self.cell_region_output=None
			self.quantification_output=None
			self.quantification_channels=[]
			if self.path_to_image is not None:
				self._inspect_selected_image()
		dialog.Destroy()


	def refresh_metadata(self,event):
		if self.path_to_image is None:
			wx.MessageBox('No multiplex image selected.','Error',wx.OK|wx.ICON_ERROR)
			return
		self._inspect_selected_image()


	def _inspect_selected_image(self):
		try:
			with open_multiplex_image(self.path_to_image,series=self.series_index)as image:
				self.image_metadata=image.metadata
				self.metadata_text.SetValue(image.metadata.summary(max_channels=25))
				if hasattr(self,'checkbox_graph_physical'):
					has_pixel_size=image.metadata.pixel_size_x is not None and float(image.metadata.pixel_size_x)>0
					self.checkbox_graph_physical.Enable(has_pixel_size)
					if not has_pixel_size:
						self.checkbox_graph_physical.SetValue(False)
				self.text_source.SetLabel('Selected: '+str(self.path_to_image)+'; series '+str(self.series_index)+'.')
				maximum_level=max(0,len(image.metadata.levels)-1)
				self.spin_level.SetRange(0,maximum_level)
				if self.spin_level.GetValue()>maximum_level:
					self.spin_level.SetValue(0)
				self.choice_dapi_channel.Clear()
				self.choice_dapi_channel.AppendItems(list(image.metadata.channel_names))
				dapi_index=next((index for index,name in enumerate(image.metadata.channel_names)if'dapi'in name.lower()),0)
				self.choice_dapi_channel.SetSelection(dapi_index)
				self.segmentation_normalization=None
				self.segmentation_normalization_settings=None
				if self.segmentation_output is None:
					image_path=Path(self.path_to_image)
					self.segmentation_output=image_path.parent/(image_path.stem+'_mplexa_segmentation')
					self.text_segmentation_output.SetLabel('Output: '+str(self.segmentation_output))
				if self.reconciliation_output is None and self.segmentation_output is not None:
					self.reconciliation_output=Path(self.segmentation_output)/'global_instances'
				self.text_reconciliation_input.SetLabel(
					'Module 2 output: '+str(self.segmentation_output)+'; Module 3 output: '+str(self.reconciliation_output))
				self.choice_membrane_channel.Clear()
				self.choice_membrane_channel.AppendItems(list(image.metadata.channel_names))
				membrane_index=next((index for index,name in enumerate(image.metadata.channel_names)
					if any(term in name.lower()for term in('membrane','panc','pan-ck','wga'))),0)
				self.choice_membrane_channel.SetSelection(membrane_index)
				self.quantification_channels=list(range(image.metadata.channel_count))
				self.text_quant_channels.SetLabel('Channels: all '+str(image.metadata.channel_count)+' channels selected.')
				if self.cell_region_output is None and self.reconciliation_output is not None:
					self.cell_region_output=Path(self.reconciliation_output)/'cell_regions'
				self.text_region_input.SetLabel(
					'Module 3 output: '+str(self.reconciliation_output)+'; cell regions: '+str(self.cell_region_output))
				if self.quantification_output is None and self.cell_region_output is not None:
					self.quantification_output=Path(self.cell_region_output)/'marker_quantification'
				self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		except(MultiplexImageError,FileNotFoundError,ValueError,OSError)as error:
			self.image_metadata=None
			self.choice_dapi_channel.Clear()
			self.choice_dapi_channel.Append('Unable to read channels')
			self.choice_dapi_channel.SetSelection(0)
			self.choice_membrane_channel.Clear()
			self.choice_membrane_channel.Append('Unable to read channels')
			self.choice_membrane_channel.SetSelection(0)
			self.quantification_channels=[]
			self.metadata_text.SetValue('Unable to inspect image:\n'+str(error))
			wx.MessageBox(str(error),'Unable to open multiplex image',wx.OK|wx.ICON_ERROR)


	def validate_lazy_read(self,event):
		if self.path_to_image is None:
			wx.MessageBox('No multiplex image selected.','Error',wx.OK|wx.ICON_ERROR)
			return
		try:
			with open_multiplex_image(self.path_to_image,series=self.series_index)as image:
				metadata=image.metadata
				channel=next((index for index,name in enumerate(metadata.channel_names)if'dapi'in name.lower()),0)
				region_width=min(512,metadata.width)
				region_height=min(512,metadata.height)
				x=max(0,(metadata.width-region_width)//2)
				y=max(0,(metadata.height-region_height)//2)
				started=time.perf_counter()
				tile=image.read_region(x=x,y=y,width=region_width,height=region_height,channels=channel,level=0)
				elapsed=time.perf_counter()-started
				message=(
					'Lazy region read completed successfully.\n\n'
					+'Channel: '+metadata.channel_names[channel]+' ('+str(channel)+')\n'
					+'Region: x='+str(x)+', y='+str(y)+', width='+str(region_width)+', height='+str(region_height)+'\n'
					+'Returned shape: '+str(tile.shape)+'\n'
					+'Data type: '+str(tile.dtype)+'\n'
					+'Memory returned: '+format(tile.nbytes/(1024*1024),'.2f')+' MB\n'
					+'Read time: '+format(elapsed,'.3f')+' seconds')
				wx.MessageBox(message,'Multiplex image validation',wx.OK|wx.ICON_INFORMATION)
		except(MultiplexImageError,FileNotFoundError,ValueError,OSError)as error:
			wx.MessageBox(str(error),'Lazy read failed',wx.OK|wx.ICON_ERROR)


	def _build_tile_grid(self):
		if self.path_to_image is None or self.image_metadata is None:
			raise TilingError('Select and successfully inspect a multiplex image first.')
		if self._segmentation_method()=='threshold'and float(self.spin_threshold_requested_pixel_size.GetValue())>0:
			level=choose_pyramid_level_for_pixel_size(
				self.image_metadata,float(self.spin_threshold_requested_pixel_size.GetValue()))
			self.spin_level.SetValue(level)
		else:
			level=int(self.spin_level.GetValue())
		if level<0 or level>=len(self.image_metadata.levels):
			raise TilingError('The selected pyramid level is unavailable for this image.')
		level_metadata=self.image_metadata.levels[level]
		width=int(level_metadata.shape[level_metadata.axes.index('X')])
		height=int(level_metadata.shape[level_metadata.axes.index('Y')])
		overlap=(float(self.spin_overlap_ratio_x.GetValue()),float(self.spin_overlap_ratio_y.GetValue()))
		if self._segmentation_method()=='threshold':
			return build_threshold_tile_grid(
				width,height,tile_size=int(self.spin_threshold_tile_size.GetValue()),
				overlap_ratio=overlap,level=level)
		if self.detector_metadata is None:
			raise TilingError(
				'Select a trained MPlexA detector first. Its saved inference frame size '
				'will be used as both tile width and tile height.')
		return build_detector_tile_grid(
			width,height,self.detector_metadata,overlap_ratio=overlap,level=level)


	def _segmentation_method(self):
		return'threshold'if self.choice_segmentation_method.GetSelection()==1 else'detectron2'


	def _update_segmentation_method_controls(self):
		detectron=self._segmentation_method()=='detectron2'
		self.button_detector.Enable(detectron)
		self.text_detector.Enable(detectron)
		self.spin_score_threshold.Enable(detectron)
		self.spin_segmentation_batch.Enable(detectron)
		for control in self.threshold_controls:
			control.Enable(not detectron)
		if detectron:
			if self.detector_metadata is None:
				self.text_tile_size.SetValue('Select detector')
			else:
				frame=int(self.detector_metadata.inferencing_framesize)
				self.text_tile_size.SetValue(str(frame)+' x '+str(frame)+' px')
		else:
			size=int(self.spin_threshold_tile_size.GetValue())
			self.text_tile_size.SetValue(str(size)+' x '+str(size)+' px')


	def on_segmentation_method_changed(self,event):
		self._update_segmentation_method_controls()
		method='Detectron2 Detector'if self._segmentation_method()=='detectron2'else'Intensity Threshold'
		self.segmentation_text.SetValue(
			'Segmentation method: '+method+'. Both methods save the same compact tile-mask archives for Module 3.')
		if event is not None:
			event.Skip()


	def on_threshold_tile_size_changed(self,event):
		if self._segmentation_method()=='threshold':
			size=int(self.spin_threshold_tile_size.GetValue())
			self.text_tile_size.SetValue(str(size)+' x '+str(size)+' px')
		if event is not None:
			event.Skip()


	def clear_segmentation_normalization(self,event):
		self.segmentation_normalization=None
		self.segmentation_normalization_settings=None
		if event is not None:
			event.Skip()


	def select_segmentation_detector(self,event):
		detectors_root=Path(the_absolute_current_path)/'detectors'
		stored_detectors=list(discover_mplexa_detectors(detectors_root))
		choose_directory='Choose from a directory'
		choices=[path.name for path in stored_detectors]+[choose_directory]
		if stored_detectors:
			message=(
				'Select a Detector stored in:\n'+str(detectors_root)+
				'\n\nChoose the last option to use a Detector outside MPlexA/detectors/.')
		else:
			message=(
				'No valid Detectors were found in:\n'+str(detectors_root)+
				'\n\nChoose a Detector from another directory.')
		dialog=wx.SingleChoiceDialog(self,message=message,caption='Select an MPlexA Detector',choices=choices)
		if dialog.ShowModal()!=wx.ID_OK:
			dialog.Destroy()
			return
		selection=dialog.GetStringSelection()
		dialog.Destroy()
		if selection==choose_directory:
			directory_dialog=wx.DirDialog(
				self,'Choose an MPlexA Detector directory','',
				style=wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST)
			if directory_dialog.ShowModal()!=wx.ID_OK:
				directory_dialog.Destroy()
				return
			path=Path(directory_dialog.GetPath())
			directory_dialog.Destroy()
		else:
			path=next((item for item in stored_detectors if item.name==selection),None)
			if path is None:
				wx.MessageBox('The selected Detector could not be resolved.','Unable to select Detector',wx.OK|wx.ICON_ERROR)
				return
		self._load_segmentation_detector(path)


	def _load_segmentation_detector(self,path):
		try:
			metadata=read_detector_metadata(path)
			self.path_to_detector=Path(path)
			self.detector_metadata=metadata
			frame=int(metadata.inferencing_framesize)
			background='black/dark'if metadata.black_background else'white/light'
			self.text_detector.SetLabel(
				'Detector: '+str(path)+'; square frame '+str(frame)+' x '+str(frame)+' px; classes '+
				str(list(metadata.cell_names))+'; expected '+background+' background.')
			self._update_segmentation_method_controls()
		except(SegmentationError,OSError,ValueError)as error:
			wx.MessageBox(str(error),'Unable to load detector metadata',wx.OK|wx.ICON_ERROR)


	def select_segmentation_output(self,event):
		default_path=''
		if self.segmentation_output is not None:
			output=Path(self.segmentation_output)
			default_path=str(output if output.exists()else output.parent)
		elif self.path_to_image is not None:
			default_path=str(Path(self.path_to_image).parent)
		dialog=wx.DirDialog(self,'Select or create a segmentation output folder',default_path,
			style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.segmentation_output=Path(dialog.GetPath())
			self.text_segmentation_output.SetLabel('Output: '+str(self.segmentation_output))
			self.reconciliation_output=self.segmentation_output/'global_instances'
			self.text_reconciliation_input.SetLabel(
				'Module 2 output: '+str(self.segmentation_output)+'; Module 3 output: '+str(self.reconciliation_output))
			self.cell_region_output=self.reconciliation_output/'cell_regions'
			self.quantification_output=self.cell_region_output/'marker_quantification'
			self.text_region_input.SetLabel(
				'Module 3 output: '+str(self.reconciliation_output)+'; cell regions: '+str(self.cell_region_output))
			self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		dialog.Destroy()


	def _normalization_settings(self):
		return(
			int(self.choice_dapi_channel.GetSelection()),
			int(self.spin_level.GetValue()),
			float(self.spin_low_percentile.GetValue()),
			float(self.spin_high_percentile.GetValue()),
			int(self.spin_normalization_samples.GetValue()),
		)


	def _set_segmentation_busy(self,busy):
		self.button_run_segmentation.Enable(not busy)
		self.button_normalization.Enable(not busy)
		self.button_cancel_segmentation.Enable(busy)
		self.button_view_segmentation.Enable(not busy)
		self.button_export_coco.Enable((not busy)and self.coco_export_thread is None)
		self.choice_segmentation_method.Enable(not busy)
		if busy:
			self.button_detector.Disable()
			self.spin_score_threshold.Disable()
			for control in self.threshold_controls:
				control.Disable()
		else:
			self._update_segmentation_method_controls()


	def estimate_dapi_normalization(self,event):
		if self.segmentation_thread is not None and self.segmentation_thread.is_alive():
			wx.MessageBox('A multiplex operation is already running.','Multiplex analysis',wx.OK|wx.ICON_INFORMATION)
			return
		if self.path_to_image is None or self.image_metadata is None:
			wx.MessageBox('Select and inspect a multiplex image first.','Error',wx.OK|wx.ICON_ERROR)
			return
		channel=int(self.choice_dapi_channel.GetSelection())
		if channel<0 or channel>=self.image_metadata.channel_count:
			wx.MessageBox('Select a valid segmentation channel.','Error',wx.OK|wx.ICON_ERROR)
			return
		settings=self._normalization_settings()
		image_path=self.path_to_image
		series_index=self.series_index
		if settings[3]<=settings[2]:
			wx.MessageBox('The high percentile must exceed the low percentile.','Error',wx.OK|wx.ICON_ERROR)
			return
		self._set_segmentation_busy(True)
		self.segmentation_text.SetValue('Estimating image-wide normalization from deterministic spatial samples...')


		def worker():
			try:
				with open_multiplex_image(image_path,series=series_index)as image:
					normalization=estimate_intensity_normalization(
						image,channel=channel,level=settings[1],low_percentile=settings[2],
						high_percentile=settings[3],sample_size=512,max_samples=settings[4])
				wx.CallAfter(self._normalization_finished,normalization,settings,None)
			except Exception as error:
				wx.CallAfter(self._normalization_finished,None,settings,error)
		self.segmentation_thread=threading.Thread(target=worker,daemon=True,name='MPlexA-normalization')
		self.segmentation_thread.start()


	def _normalization_finished(self,normalization,settings,error):
		self._set_segmentation_busy(False)
		if error is not None:
			self.segmentation_text.SetValue('Normalization failed:\n'+str(error))
			wx.MessageBox(str(error),'Normalization failed',wx.OK|wx.ICON_ERROR)
			return
		self.segmentation_normalization=normalization
		self.segmentation_normalization_settings=settings
		self.segmentation_text.SetValue(
			'Image-wide channel normalization estimated successfully.\n\n'
			+'Low value: '+format(normalization.low_value,'.6g')+' at percentile '+str(normalization.low_percentile)+'\n'
			+'High value: '+format(normalization.high_value,'.6g')+' at percentile '+str(normalization.high_percentile)+'\n'
			+'Spatial samples: '+str(normalization.sample_count)+'\n'
			+'Sampled pixels: '+format(normalization.sampled_pixels,',')+'\n\n'
			+'These limits will be reused if the channel, level, percentiles, and sample count remain unchanged.')


	def _validate_segmentation_inputs(self):
		if self.path_to_image is None or self.image_metadata is None:
			raise SegmentationError('Select and inspect a multiplex image first.')
		method=self._segmentation_method()
		if method=='detectron2'and(self.path_to_detector is None or self.detector_metadata is None):
			raise SegmentationError('Select a trained MPlexA detector first.')
		if self.segmentation_output is None:
			raise SegmentationError('Select a segmentation output folder first.')
		channel=int(self.choice_dapi_channel.GetSelection())
		if channel<0 or channel>=self.image_metadata.channel_count:
			raise SegmentationError('Select a valid segmentation channel.')
		low=float(self.spin_low_percentile.GetValue())
		high=float(self.spin_high_percentile.GetValue())
		if high<=low:
			raise SegmentationError('The high percentile must exceed the low percentile.')
		grid=self._build_tile_grid()
		if method=='detectron2':
			frame=int(self.detector_metadata.inferencing_framesize)
			if grid.tile_width!=frame or grid.tile_height!=frame:
				raise SegmentationError('Internal tiling error: the tile size does not match the detector frame size.')
		else:
			size=int(self.spin_threshold_tile_size.GetValue())
			if grid.tile_width!=size or grid.tile_height!=size:
				raise SegmentationError('Internal tiling error: threshold tiles must match the configured square tile size.')
		return grid,channel,method


	def start_dapi_segmentation(self,event):
		for thread in(self.segmentation_thread,self.reconciliation_thread,self.cell_region_thread,self.quantification_thread,self.clustering_thread,self.spatial_thread):
			if thread is not None and thread.is_alive():
				wx.MessageBox('Another multiplex operation is already running.','Multiplex analysis',wx.OK|wx.ICON_INFORMATION)
				return
		try:
			grid,channel,method=self._validate_segmentation_inputs()
		except(SegmentationError,TilingError,ValueError,TypeError)as error:
			wx.MessageBox(str(error),'Unable to start segmentation',wx.OK|wx.ICON_ERROR)
			return
		settings=self._normalization_settings()
		normalization=self.segmentation_normalization if self.segmentation_normalization_settings==settings else None
		if method=='detectron2':
			config=SegmentationConfig(
				channel=channel,score_threshold=float(self.spin_score_threshold.GetValue()),
				batch_size=int(self.spin_segmentation_batch.GetValue()),
				low_percentile=float(self.spin_low_percentile.GetValue()),
				high_percentile=float(self.spin_high_percentile.GetValue()),
				normalization_samples=int(self.spin_normalization_samples.GetValue()),
				retry_failed=bool(self.checkbox_retry_failed.GetValue()))
		else:
			config=ThresholdSegmentationConfig(
				channel=channel,threshold_value=float(self.spin_threshold_value.GetValue()),
				foreground='bright'if self.choice_threshold_foreground.GetSelection()==0 else'dark',
				background_radius=int(self.spin_threshold_background_radius.GetValue()),
				background_by_reconstruction=bool(self.checkbox_threshold_background_reconstruction.GetValue()),
				median_radius=int(self.spin_threshold_median_radius.GetValue()),
				gaussian_sigma=float(self.spin_threshold_sigma.GetValue()),
				min_area=int(self.spin_threshold_min_area.GetValue()),
				max_area=int(self.spin_threshold_max_area.GetValue()),
				split_touching=bool(self.checkbox_threshold_split.GetValue()),
				watershed_min_distance=int(self.spin_threshold_min_distance.GetValue()),
				refine_boundaries=bool(self.checkbox_threshold_refine.GetValue()),
				retain_core_owned_only=True,
				tile_size=int(self.spin_threshold_tile_size.GetValue()),
				batch_size=int(self.spin_segmentation_batch.GetValue()),
				cpu_workers=int(self.spin_threshold_workers.GetValue()),
				fast_archives=bool(self.checkbox_threshold_fast_archives.GetValue()),
				low_percentile=float(self.spin_low_percentile.GetValue()),
				high_percentile=float(self.spin_high_percentile.GetValue()),
				normalization_samples=int(self.spin_normalization_samples.GetValue()),
				retry_failed=bool(self.checkbox_retry_failed.GetValue()))
		image_path=self.path_to_image
		series_index=self.series_index
		detector_path=self.path_to_detector
		output_directory=self.segmentation_output
		self.active_segmentation_settings=settings
		self.active_segmentation_output=output_directory
		self.segmentation_cancel_event=threading.Event()
		cancel_event=self.segmentation_cancel_event
		self._set_segmentation_busy(True)
		self.gauge_segmentation.SetValue(0)
		self.segmentation_text.SetValue(
			('Loading the detector and opening'if method=='detectron2'else'Opening')+
			' the segmentation checkpoint. Completed tiles will be skipped automatically...')


		def progress_callback(progress,summary):
			wx.CallAfter(self._update_segmentation_progress,progress,summary)


		def log_callback(message):
			wx.CallAfter(self._append_segmentation_log,message)


		def worker():
			try:
				if method=='detectron2':
					segmenter=TiledDapiSegmenter(detector_path)
				else:
					segmenter=TiledThresholdSegmenter()
				result=segmenter.run(
					image_path=image_path,series=series_index,grid=grid,
					output_directory=output_directory,config=config,normalization=normalization,
					cancel_event=cancel_event,on_progress=progress_callback,on_log=log_callback)
				wx.CallAfter(self._segmentation_finished,result,None)
			except Exception as error:
				wx.CallAfter(self._segmentation_finished,None,error)
		self.segmentation_thread=threading.Thread(target=worker,daemon=True,name='MPlexA-cell-segmentation')
		self.segmentation_thread.start()


	def cancel_dapi_segmentation(self,event):
		if self.segmentation_cancel_event is not None:
			self.segmentation_cancel_event.set()
			self.button_cancel_segmentation.Disable()
			self._append_segmentation_log('Cancellation requested. MPlexA will stop after the current processing batch.')


	def _update_segmentation_progress(self,progress,summary):
		value=int(round(progress.completion_fraction*1000))
		self.gauge_segmentation.SetValue(max(0,min(1000,value)))
		line=(
			'Completed '+format(progress.completed,',')+'/'+format(progress.total,',')
			+'; pending '+format(progress.pending,',')+'; running '+format(progress.running,',')
			+'; failed '+format(progress.failed,',')+'.')
		if summary is not None:
			line+=' Last tile: '+summary.tile_id+'; '+str(summary.prediction_count)+' prediction(s).'
		self.segmentation_text.SetValue(line+'\n\nOutput: '+str(self.active_segmentation_output or self.segmentation_output))


	def _append_segmentation_log(self,message):
		current=self.segmentation_text.GetValue()
		if current:
			current+='\n'
		self.segmentation_text.SetValue(current+str(message))


	def _segmentation_finished(self,result,error):
		self._set_segmentation_busy(False)
		self.segmentation_cancel_event=None
		if error is not None:
			self.active_segmentation_settings=None
			self.active_segmentation_output=None
			self.segmentation_text.SetValue('Segmentation stopped with an error:\n'+str(error))
			wx.MessageBox(str(error),'Cell segmentation failed',wx.OK|wx.ICON_ERROR)
			return
		self.segmentation_normalization=result.normalization
		self.segmentation_normalization_settings=self.active_segmentation_settings
		self.gauge_segmentation.SetValue(int(round(result.progress.completion_fraction*1000)))
		self.segmentation_text.SetValue(result.summary())
		self.active_segmentation_settings=None
		self.active_segmentation_output=None
		title='Segmentation paused'if result.cancelled else'Cell segmentation completed'
		wx.MessageBox(result.summary(),title,wx.OK|wx.ICON_INFORMATION)


	def open_segmentation_viewer(self,event):
		if self.path_to_image is None or self.image_metadata is None:
			wx.MessageBox('Select and inspect a multiplex image first.','Segmentation viewer',wx.OK|wx.ICON_ERROR)
			return
		if self.segmentation_output is None:
			wx.MessageBox('Select the Module 2 segmentation output first.','Segmentation viewer',wx.OK|wx.ICON_ERROR)
			return
		segmentation=Path(self.segmentation_output)
		if not(segmentation/'segmentation_config.json').is_file()or not(segmentation/'tiles').is_dir():
			wx.MessageBox('The selected Module 2 output does not contain segmentation results yet. Run or resume segmentation first.','Segmentation viewer',wx.OK|wx.ICON_ERROR)
			return
		try:
			frame=SegmentationViewerFrame(self,self.path_to_image,segmentation_directory=segmentation,series=self.series_index)
			self.viewer_frames.append(frame)
			frame.Show(True)
		except Exception as error:
			wx.MessageBox(str(error),'Unable to open segmentation viewer',wx.OK|wx.ICON_ERROR)


	def export_segmentation_coco(self,event):
		if self.segmentation_thread is not None and self.segmentation_thread.is_alive():
			wx.MessageBox('Wait for Module 2 segmentation to finish before exporting COCO annotations.','COCO export',wx.OK|wx.ICON_INFORMATION)
			return
		if self.coco_export_thread is not None and self.coco_export_thread.is_alive():
			wx.MessageBox('A COCO export is already running.','COCO export',wx.OK|wx.ICON_INFORMATION)
			return
		if self.path_to_image is None or self.image_metadata is None:
			wx.MessageBox('Select and inspect a multiplex image first.','COCO export',wx.OK|wx.ICON_ERROR)
			return
		if self.segmentation_output is None:
			wx.MessageBox('Select the Module 2 segmentation output first.','COCO export',wx.OK|wx.ICON_ERROR)
			return
		segmentation=Path(self.segmentation_output)
		if not(segmentation/'segmentation_config.json').is_file()or not(segmentation/'tiles').is_dir():
			wx.MessageBox('The selected Module 2 output does not contain segmentation results yet. Run or resume segmentation first.','COCO export',wx.OK|wx.ICON_ERROR)
			return
		default_name=Path(self.path_to_image).stem+'_mplexa_segmentation.json'
		dialog=wx.FileDialog(self,'Export Module 2 masks as COCO instance-segmentation JSON',
			defaultDir=str(segmentation),defaultFile=default_name,
			wildcard='COCO JSON (*.json)|*.json',style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
		if dialog.ShowModal()!=wx.ID_OK:
			dialog.Destroy()
			return
		output=Path(dialog.GetPath())
		dialog.Destroy()
		self.button_export_coco.Disable()
		self.button_run_segmentation.Disable()
		self.button_normalization.Disable()
		self.button_view_segmentation.Disable()
		self.segmentation_text.SetValue('Exporting core-owned Module 2 masks to COCO instance-segmentation JSON...')


		def progress(done,total,annotations):


			def update():
				fraction=done/max(1,total)
				self.gauge_segmentation.SetValue(int(round(fraction*1000)))
				self.segmentation_text.SetValue(
					f'Exporting COCO annotations: {done:,}/{total:,} tile archives; {annotations:,} instances written.\n'
					f'Output: {output}')
			wx.CallAfter(update)


		def worker():
			try:
				result=export_segmentation_to_coco(
					segmentation,output,image_path=self.path_to_image,series=self.series_index,
					owned_only=True,on_progress=progress)
				wx.CallAfter(self._coco_export_finished,result,None)
			except Exception as error:
				wx.CallAfter(self._coco_export_finished,None,error)
		self.coco_export_thread=threading.Thread(target=worker,name='MPlexA-COCO-export',daemon=True)
		self.coco_export_thread.start()


	def _coco_export_finished(self,result,error):
		self.coco_export_thread=None
		segmentation_idle=self.segmentation_thread is None or not self.segmentation_thread.is_alive()
		self.button_export_coco.Enable(segmentation_idle)
		self.button_run_segmentation.Enable(segmentation_idle)
		self.button_normalization.Enable(segmentation_idle)
		self.button_view_segmentation.Enable(segmentation_idle)
		if error is not None:
			self.segmentation_text.SetValue('COCO export stopped with an error:\n'+str(error))
			wx.MessageBox(str(error),'COCO export failed',wx.OK|wx.ICON_ERROR)
			return
		self.gauge_segmentation.SetValue(1000)
		self.segmentation_text.SetValue(result.summary())
		wx.MessageBox(result.summary(),'COCO export completed',wx.OK|wx.ICON_INFORMATION)


	def select_reconciliation_output(self,event):
		default_path=''
		if self.reconciliation_output is not None:
			output=Path(self.reconciliation_output)
			default_path=str(output if output.exists()else output.parent)
		elif self.segmentation_output is not None:
			default_path=str(Path(self.segmentation_output))
		dialog=wx.DirDialog(self,'Select or create the Module 3 output folder',default_path,
			style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.reconciliation_output=Path(dialog.GetPath())
			self.text_reconciliation_input.SetLabel(
				'Module 2 output: '+str(self.segmentation_output or'not selected')+
				'; Module 3 output: '+str(self.reconciliation_output))
			self.cell_region_output=Path(self.reconciliation_output)/'cell_regions'
			self.quantification_output=Path(self.cell_region_output)/'marker_quantification'
			self.text_region_input.SetLabel(
				'Module 3 output: '+str(self.reconciliation_output)+'; cell regions: '+str(self.cell_region_output))
			self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		dialog.Destroy()


	def _set_reconciliation_busy(self,busy):
		self.button_run_reconciliation.Enable(not busy)
		self.button_cancel_reconciliation.Enable(busy)
		self.button_run_segmentation.Enable(not busy)
		self.button_normalization.Enable(not busy)


	def _validate_reconciliation_inputs(self):
		if self.segmentation_output is None:
			raise ReconciliationError('Select the Module 2 segmentation output first.')
		segmentation=Path(self.segmentation_output)
		if not(segmentation/'segmentation_config.json').is_file():
			raise ReconciliationError(
				'The selected Module 2 output does not contain segmentation_config.json: '+str(segmentation))
		if not(segmentation/'segmentation.sqlite').is_file()or not(segmentation/'tiles').is_dir():
			raise ReconciliationError(
				'The selected folder is missing the Module 2 checkpoint or tile predictions: '+str(segmentation))
		if self.reconciliation_output is None:
			self.reconciliation_output=segmentation/'global_instances'
		strategy='best'if self.choice_reconciliation_strategy.GetSelection()==0 else'union'
		config=ReconciliationConfig(
			iou_threshold=float(self.spin_reconciliation_iou.GetValue()),
			containment_threshold=float(self.spin_reconciliation_containment.GetValue()),
			same_class_only=bool(self.checkbox_same_class.GetValue()),
			mask_strategy=strategy,
			chunk_size=int(self.spin_label_chunk_size.GetValue()),
			retry_failed_chunks=bool(self.checkbox_retry_label_chunks.GetValue()))
		return segmentation,Path(self.reconciliation_output),config


	def start_reconciliation(self,event):
		for thread in(self.segmentation_thread,self.reconciliation_thread,self.cell_region_thread,self.quantification_thread,self.clustering_thread,self.spatial_thread):
			if thread is not None and thread.is_alive():
				wx.MessageBox('Another multiplex operation is already running.','Multiplex analysis',wx.OK|wx.ICON_INFORMATION)
				return
		try:
			segmentation,output,config=self._validate_reconciliation_inputs()
		except(ReconciliationError,ValueError,TypeError)as error:
			wx.MessageBox(str(error),'Unable to start Module 3',wx.OK|wx.ICON_ERROR)
			return
		self.active_reconciliation_output=output
		self.reconciliation_cancel_event=threading.Event()
		cancel_event=self.reconciliation_cancel_event
		self._set_reconciliation_busy(True)
		self.gauge_reconciliation.SetValue(0)
		self.reconciliation_text.SetValue(
			'Opening or creating the reconciliation database. Completed matching and label chunks will be resumed automatically...')


		def progress_callback(progress):
			wx.CallAfter(self._update_reconciliation_progress,progress)


		def log_callback(message):
			wx.CallAfter(self._append_reconciliation_log,message)


		def worker():
			try:
				reconciler=GlobalMaskReconciler(segmentation)
				result=reconciler.run(
					output_directory=output,config=config,cancel_event=cancel_event,
					on_progress=progress_callback,on_log=log_callback)
				wx.CallAfter(self._reconciliation_finished,result,None)
			except Exception as error:
				wx.CallAfter(self._reconciliation_finished,None,error)
		self.reconciliation_thread=threading.Thread(
			target=worker,daemon=True,name='MPlexA-mask-reconciliation')
		self.reconciliation_thread.start()


	def cancel_reconciliation(self,event):
		if self.reconciliation_cancel_event is not None:
			self.reconciliation_cancel_event.set()
			self.button_cancel_reconciliation.Disable()
			self._append_reconciliation_log(
				'Cancellation requested. MPlexA will preserve completed indexing, matching, grouping, and label chunks.')


	def _update_reconciliation_progress(self,progress):
		value=int(round(progress.fraction*1000))
		self.gauge_reconciliation.SetValue(max(0,min(1000,value)))
		line=(progress.stage.capitalize()+': '+format(progress.current,',')+'/'+format(progress.total,','))
		if progress.message:
			line+=' — '+str(progress.message)
		self.reconciliation_text.SetValue(
			line+'\n\nOutput: '+str(self.active_reconciliation_output or self.reconciliation_output))


	def _append_reconciliation_log(self,message):
		current=self.reconciliation_text.GetValue()
		if current:
			current+='\n'
		self.reconciliation_text.SetValue(current+str(message))


	def _reconciliation_finished(self,result,error):
		self._set_reconciliation_busy(False)
		self.reconciliation_cancel_event=None
		if error is not None:
			self.active_reconciliation_output=None
			self.reconciliation_text.SetValue('Module 3 stopped with an error:\n'+str(error))
			wx.MessageBox(str(error),'Module 3 failed',wx.OK|wx.ICON_ERROR)
			return
		self.gauge_reconciliation.SetValue(1000 if not result.cancelled else self.gauge_reconciliation.GetValue())
		self.reconciliation_text.SetValue(result.summary())
		completed_output=Path(result.output_directory)
		self.reconciliation_output=completed_output
		self.cell_region_output=completed_output/'cell_regions'
		self.quantification_output=self.cell_region_output/'marker_quantification'
		self.text_region_input.SetLabel(
			'Module 3 output: '+str(completed_output)+'; cell regions: '+str(self.cell_region_output))
		self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		self.active_reconciliation_output=None
		title='Module 3 paused'if result.cancelled else'Module 3 completed'
		wx.MessageBox(result.summary(),title,wx.OK|wx.ICON_INFORMATION)


	def update_region_mode_controls(self,event):
		mode=int(self.choice_region_mode.GetSelection())if hasattr(self,'choice_region_mode')else 0
		self.spin_region_distance.Enable(mode!=0)
		watershed=mode==3
		self.choice_membrane_channel.Enable(watershed)
		self.spin_membrane_sigma.Enable(watershed)
		if event is not None:
			event.Skip()


	def select_cell_region_output(self,event):
		default_path=''
		if self.cell_region_output is not None:
			path=Path(self.cell_region_output)
			default_path=str(path if path.exists()else path.parent)
		elif self.reconciliation_output is not None:
			default_path=str(self.reconciliation_output)
		dialog=wx.DirDialog(self,'Select or create the cell-region output folder',default_path,
			style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.cell_region_output=Path(dialog.GetPath())
			self.quantification_output=self.cell_region_output/'marker_quantification'
			self.text_region_input.SetLabel(
				'Module 3 output: '+str(self.reconciliation_output or'not selected')+
				'; cell regions: '+str(self.cell_region_output))
			self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		dialog.Destroy()


	def _set_cell_region_busy(self,busy):
		self.button_run_regions.Enable(not busy)
		self.button_cancel_regions.Enable(busy)
		self.button_run_segmentation.Enable(not busy)
		self.button_normalization.Enable(not busy)
		self.button_run_reconciliation.Enable(not busy)
		self.button_run_quantification.Enable(not busy)


	def _validate_cell_region_inputs(self):
		if self.reconciliation_output is None:
			raise QuantificationError('Select or complete the Module 3 output first.')
		module3=Path(self.reconciliation_output)
		global_labels=resolve_global_label_store(module3)
		if not global_labels.is_dir()or not(module3/'reconciliation.sqlite').is_file():
			raise QuantificationError(
				'The selected Module 3 output is missing its global labels or reconciliation.sqlite: '+str(module3))
		if self.cell_region_output is None:
			self.cell_region_output=module3/'cell_regions'
		mode_names=('nuclear','fixed','voronoi','watershed')
		mode_index=int(self.choice_region_mode.GetSelection())
		if mode_index<0 or mode_index>=len(mode_names):
			raise QuantificationError('Select a valid cell-region mode.')
		mode=mode_names[mode_index]
		distance=int(self.spin_region_distance.GetValue())
		membrane_channel=None
		if mode=='watershed':
			if self.path_to_image is None or self.image_metadata is None:
				raise QuantificationError('Membrane-guided watershed requires the selected multiplex image.')
			membrane_channel=int(self.choice_membrane_channel.GetSelection())
			if membrane_channel<0 or membrane_channel>=self.image_metadata.channel_count:
				raise QuantificationError('Select a valid membrane channel.')
		config=CellRegionConfig(
			mode=mode,expansion_distance=distance,membrane_channel=membrane_channel,
			membrane_sigma=float(self.spin_membrane_sigma.GetValue()),
			chunk_size=int(self.spin_region_chunk_size.GetValue()),
			retry_failed_chunks=bool(self.checkbox_retry_region_chunks.GetValue()))
		return module3,Path(self.cell_region_output),config


	def start_cell_regions(self,event):
		for thread in(self.segmentation_thread,self.reconciliation_thread,self.cell_region_thread,self.quantification_thread,self.clustering_thread,self.spatial_thread):
			if thread is not None and thread.is_alive():
				wx.MessageBox('Another multiplex operation is already running.','Multiplex analysis',wx.OK|wx.ICON_INFORMATION)
				return
		try:
			module3,output,config=self._validate_cell_region_inputs()
		except(QuantificationError,ValueError,TypeError)as error:
			wx.MessageBox(str(error),'Unable to start cell-region generation',wx.OK|wx.ICON_ERROR)
			return
		self.active_cell_region_output=output
		self.cell_region_cancel_event=threading.Event()
		cancel_event=self.cell_region_cancel_event
		self._set_cell_region_busy(True)
		self.gauge_regions.SetValue(0)
		self.region_text.SetValue('Opening or creating the cell-region checkpoint. Completed chunks will be skipped automatically...')
		image_path=self.path_to_image if config.mode=='watershed'else None
		series_index=self.series_index


		def progress_callback(progress):
			wx.CallAfter(self._update_cell_region_progress,progress)


		def log_callback(message):
			wx.CallAfter(self._append_cell_region_log,message)


		def worker():
			try:
				generator=CellRegionGenerator(module3)
				result=generator.run(
					output_directory=output,config=config,image_path=image_path,series=series_index,
					cancel_event=cancel_event,on_progress=progress_callback,on_log=log_callback)
				wx.CallAfter(self._cell_regions_finished,result,None)
			except Exception as error:
				wx.CallAfter(self._cell_regions_finished,None,error)
		self.cell_region_thread=threading.Thread(target=worker,daemon=True,name='MPlexA-cell-regions')
		self.cell_region_thread.start()


	def cancel_cell_regions(self,event):
		if self.cell_region_cancel_event is not None:
			self.cell_region_cancel_event.set()
			self.button_cancel_regions.Disable()
			self._append_cell_region_log('Cancellation requested. Completed region chunks will remain resumable.')


	def _update_cell_region_progress(self,progress):
		self.gauge_regions.SetValue(max(0,min(1000,int(round(progress.fraction*1000)))))
		line=('Completed '+format(progress.completed,',')+'/'+format(progress.total,',')+
			'; failed '+format(progress.failed,','))
		if progress.message:
			line+=' — '+str(progress.message)
		self.region_text.SetValue(line+'\n\nOutput: '+str(self.active_cell_region_output or self.cell_region_output))


	def _append_cell_region_log(self,message):
		current=self.region_text.GetValue()
		self.region_text.SetValue((current+'\n'if current else'')+str(message))


	def _cell_regions_finished(self,result,error):
		self._set_cell_region_busy(False)
		self.cell_region_cancel_event=None
		if error is not None:
			self.active_cell_region_output=None
			self.region_text.SetValue('Cell-region generation stopped with an error:\n'+str(error))
			wx.MessageBox(str(error),'Cell-region generation failed',wx.OK|wx.ICON_ERROR)
			return
		self.gauge_regions.SetValue(1000 if not result.cancelled else self.gauge_regions.GetValue())
		self.region_text.SetValue(result.summary())
		self.cell_region_output=Path(result.output_directory)
		self.quantification_output=self.cell_region_output/'marker_quantification'
		self.text_region_input.SetLabel(
			'Module 3 output: '+str(self.reconciliation_output)+'; cell regions: '+str(self.cell_region_output))
		self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		self.active_cell_region_output=None
		title='Cell-region generation paused'if result.cancelled else'Cell-region generation completed'
		wx.MessageBox(result.summary(),title,wx.OK|wx.ICON_INFORMATION)


	def select_quantification_channels(self,event):
		if self.image_metadata is None:
			wx.MessageBox('Select and inspect a multiplex image first.','Error',wx.OK|wx.ICON_ERROR)
			return
		names=list(self.image_metadata.channel_names)
		dialog=wx.MultiChoiceDialog(
			self,'Select channels to quantify. Use Ctrl/Shift for multiple selections.',
			'Multiplex marker channels',names)
		current=self.quantification_channels or list(range(len(names)))
		dialog.SetSelections(current)
		if dialog.ShowModal()==wx.ID_OK:
			selected=list(dialog.GetSelections())
			if not selected:
				wx.MessageBox('At least one channel must be selected.','Marker channels',wx.OK|wx.ICON_ERROR)
			else:
				self.quantification_channels=selected
				shown=[names[index]for index in selected[:8]]
				suffix=' ...'if len(selected)>8 else''
				self.text_quant_channels.SetLabel(
					'Channels ('+str(len(selected))+'): '+', '.join(shown)+suffix)
		dialog.Destroy()


	def select_quantification_output(self,event):
		default_path=''
		if self.quantification_output is not None:
			path=Path(self.quantification_output)
			default_path=str(path if path.exists()else path.parent)
		elif self.cell_region_output is not None:
			default_path=str(self.cell_region_output)
		dialog=wx.DirDialog(self,'Select or create the marker-quantification output folder',default_path,
			style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.quantification_output=Path(dialog.GetPath())
			self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		dialog.Destroy()


	def _set_quantification_busy(self,busy):
		self.button_run_quantification.Enable(not busy)
		self.button_cancel_quantification.Enable(busy)
		self.button_run_segmentation.Enable(not busy)
		self.button_normalization.Enable(not busy)
		self.button_run_reconciliation.Enable(not busy)
		self.button_run_regions.Enable(not busy)


	def _validate_quantification_inputs(self):
		if self.path_to_image is None or self.image_metadata is None:
			raise QuantificationError('Select and inspect the source multiplex image first.')
		if self.cell_region_output is None:
			raise QuantificationError('Generate or select a cell-region output first.')
		regions=Path(self.cell_region_output)
		region_labels=resolve_cell_region_label_store(regions)
		if not(regions/'region_config.json').is_file()or not region_labels.is_dir():
			raise QuantificationError(
				'The selected cell-region folder is missing region_config.json or its cell-region labels: '+str(regions))
		channels=self.quantification_channels or list(range(self.image_metadata.channel_count))
		if not channels:
			raise QuantificationError('Select at least one marker channel.')
		if not self.checkbox_export_csv.GetValue()and not self.checkbox_export_excel.GetValue():
			raise QuantificationError('Select CSV, Excel, or both output formats.')
		if self.quantification_output is None:
			self.quantification_output=regions/'marker_quantification'
		config=QuantificationConfig(
			channels=tuple(channels),channel_batch_size=int(self.spin_quant_batch.GetValue()),
			positive_threshold=float(self.spin_positive_threshold.GetValue()),
			cytoplasmic_ring_width=int(self.spin_cytoplasmic_ring.GetValue()),
			membrane_ring_width=int(self.spin_membrane_ring.GetValue()),
			export_csv=bool(self.checkbox_export_csv.GetValue()),
			export_excel=bool(self.checkbox_export_excel.GetValue()),
			retry_failed_units=bool(self.checkbox_retry_quant_units.GetValue()))
		return regions,Path(self.quantification_output),config


	def start_quantification(self,event):
		for thread in(self.segmentation_thread,self.reconciliation_thread,self.cell_region_thread,self.quantification_thread,self.clustering_thread,self.spatial_thread):
			if thread is not None and thread.is_alive():
				wx.MessageBox('Another multiplex operation is already running.','Multiplex analysis',wx.OK|wx.ICON_INFORMATION)
				return
		try:
			regions,output,config=self._validate_quantification_inputs()
		except(QuantificationError,ValueError,TypeError)as error:
			wx.MessageBox(str(error),'Unable to start marker quantification',wx.OK|wx.ICON_ERROR)
			return
		self.active_quantification_output=output
		self.quantification_cancel_event=threading.Event()
		cancel_event=self.quantification_cancel_event
		self._set_quantification_busy(True)
		self.gauge_quantification.SetValue(0)
		self.quantification_text.SetValue(
			'Opening or creating the transactional quantification database. Completed chunk/channel units will be skipped...')
		image_path=self.path_to_image
		series_index=self.series_index


		def progress_callback(progress):
			wx.CallAfter(self._update_quantification_progress,progress)


		def log_callback(message):
			wx.CallAfter(self._append_quantification_log,message)


		def worker():
			try:
				quantifier=MarkerQuantifier(image_path,regions,series=series_index)
				result=quantifier.run(
					output_directory=output,config=config,cancel_event=cancel_event,
					on_progress=progress_callback,on_log=log_callback)
				wx.CallAfter(self._quantification_finished,result,None)
			except Exception as error:
				wx.CallAfter(self._quantification_finished,None,error)
		self.quantification_thread=threading.Thread(target=worker,daemon=True,name='MPlexA-marker-quantification')
		self.quantification_thread.start()


	def cancel_quantification(self,event):
		if self.quantification_cancel_event is not None:
			self.quantification_cancel_event.set()
			self.button_cancel_quantification.Disable()
			self._append_quantification_log(
				'Cancellation requested. MPlexA will stop after the current chunk/channel transaction.')


	def _update_quantification_progress(self,progress):
		self.gauge_quantification.SetValue(max(0,min(1000,int(round(progress.fraction*1000)))))
		line=(str(progress.stage)+': '+format(progress.completed,',')+'/'+format(progress.total,',')+
			'; failed '+format(progress.failed,','))
		if progress.message:
			line+=' — '+str(progress.message)
		self.quantification_text.SetValue(
			line+'\n\nOutput: '+str(self.active_quantification_output or self.quantification_output))


	def _append_quantification_log(self,message):
		current=self.quantification_text.GetValue()
		self.quantification_text.SetValue((current+'\n'if current else'')+str(message))


	def _quantification_finished(self,result,error):
		self._set_quantification_busy(False)
		self.quantification_cancel_event=None
		if error is not None:
			self.active_quantification_output=None
			self.quantification_text.SetValue('Marker quantification stopped with an error:\n'+str(error))
			wx.MessageBox(str(error),'Marker quantification failed',wx.OK|wx.ICON_ERROR)
			return
		self.gauge_quantification.SetValue(1000 if not result.cancelled else self.gauge_quantification.GetValue())
		self.quantification_text.SetValue(result.summary())
		self.quantification_output=Path(result.output_directory)
		self.text_quant_output.SetLabel('Output: '+str(self.quantification_output))
		marker_csv=self.quantification_output/'cell_marker_measurements.csv'
		if marker_csv.is_file():
			self.clustering_marker_csv=marker_csv
			self.clustering_features=[]
			self.text_cluster_input.SetLabel('Input: '+str(marker_csv))
		self.active_quantification_output=None
		title='Marker quantification paused'if result.cancelled else'Marker quantification completed'
		wx.MessageBox(result.summary(),title,wx.OK|wx.ICON_INFORMATION)


	def _default_marker_csv(self):
		if self.clustering_marker_csv is not None and Path(self.clustering_marker_csv).is_file():
			return Path(self.clustering_marker_csv)
		if self.quantification_output is not None:
			candidate=Path(self.quantification_output)/'cell_marker_measurements.csv'
			if candidate.is_file():
				return candidate
		return None


	def select_clustering_marker_csv(self,event):
		default=self._default_marker_csv()
		directory=str(default.parent)if default is not None else''
		filename=default.name if default is not None else''
		dialog=wx.FileDialog(self,'Select Module 4 marker-measurement CSV',directory,filename,
			'CSV files (*.csv)|*.csv',style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
		if dialog.ShowModal()==wx.ID_OK:
			self.clustering_marker_csv=Path(dialog.GetPath())
			self.clustering_features=[]
			self.clustering_output=None
			self.spatial_output=None
			self.text_cluster_input.SetLabel('Input: '+str(self.clustering_marker_csv))
			self.text_cluster_output.SetLabel('Output: created beside marker quantification.')
			self.text_spatial_output.SetLabel('Output: created inside the clustering folder.')
		dialog.Destroy()


	def select_clustering_features(self,event):
		marker=self._default_marker_csv()
		if marker is None:
			wx.MessageBox('Select or generate the Module 4 marker CSV first.','Cell clustering',wx.OK|wx.ICON_ERROR)
			return
		try:
			features=list(discover_marker_features(marker))
		except Exception as error:
			wx.MessageBox(str(error),'Cell clustering',wx.OK|wx.ICON_ERROR)
			return
		if not features:
			wx.MessageBox('No marker-measurement columns were found in this CSV.','Cell clustering',wx.OK|wx.ICON_ERROR)
			return
		dialog=wx.MultiChoiceDialog(self,'Select marker features used for cell clustering.\nWhole-cell mean intensities are selected by default.','Clustering markers',features)
		selected_existing={name for name in self.clustering_features}
		if selected_existing:
			selections=[index for index,name in enumerate(features)if name in selected_existing]
		else:
			selections=[index for index,name in enumerate(features)if name.endswith('__mean')and'dapi'not in name.lower()]
			if not selections:
				selections=[index for index,name in enumerate(features)if name.endswith('__mean')]
		if selections:
			dialog.SetSelections(selections)
		if dialog.ShowModal()==wx.ID_OK:
			self.clustering_marker_csv=Path(marker)
			self.clustering_features=[features[index]for index in dialog.GetSelections()]
			self.text_cluster_input.SetLabel(
				'Input: '+str(marker)+'; '+str(len(self.clustering_features))+' clustering marker(s) selected.')
		dialog.Destroy()


	def on_cluster_method_changed(self,event):
		if self.choice_cluster_method.GetSelection()==0:# Leiden
			if float(self.spin_cluster_count.GetValue())>=2:
				self.spin_cluster_count.SetValue(1.0)
			self.spin_cluster_count.SetIncrement(0.1)
		else:# K-means
			if float(self.spin_cluster_count.GetValue())<2:
				self.spin_cluster_count.SetValue(12.0)
			self.spin_cluster_count.SetIncrement(1.0)


	def select_clustering_output(self,event):
		default=''
		if self.clustering_output is not None:
			path=Path(self.clustering_output);default=str(path if path.exists()else path.parent)
		elif self.quantification_output is not None:
			default=str(Path(self.quantification_output).parent)
		dialog=wx.DirDialog(self,'Select or create the cell-clustering output folder',default,
			style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.clustering_output=Path(dialog.GetPath())
			self.text_cluster_output.SetLabel('Output: '+str(self.clustering_output))
		dialog.Destroy()


	def _set_clustering_busy(self,busy):
		self.button_run_clustering.Enable(not busy)
		self.button_cancel_clustering.Enable(busy)
		self.button_run_spatial.Enable(not busy)
		self.button_run_quantification.Enable(not busy)
		self.button_run_regions.Enable(not busy)
		self.button_run_reconciliation.Enable(not busy)
		self.button_run_segmentation.Enable(not busy)


	def _validate_clustering_inputs(self):
		marker=self._default_marker_csv()
		if marker is None:
			raise PhenotypingError('Select or generate the Module 4 marker-measurement CSV first.')
		if not self.clustering_features:
			available=list(discover_marker_features(marker))
			self.clustering_features=[name for name in available if name.endswith('__mean')and'dapi'not in name.lower()]
			if not self.clustering_features:
				self.clustering_features=[name for name in available if name.endswith('__mean')]
		if not self.clustering_features:
			raise PhenotypingError('Select at least one clustering marker feature.')
		if self.clustering_output is None:
			self.clustering_output=marker.parent/'cell_clustering'
		transform={0:'arcsinh',1:'none',2:'signed_log1p'}[self.choice_cluster_transform.GetSelection()]
		method='leiden'if self.choice_cluster_method.GetSelection()==0 else'kmeans'
		embedding='umap'if self.choice_embedding.GetSelection()==0 else'pca'
		value=float(self.spin_cluster_count.GetValue())
		config=PhenotypingConfig(
			feature_columns=tuple(self.clustering_features),transform=transform,
			arcsinh_cofactor=float(self.spin_cluster_cofactor.GetValue()),
			n_pcs=int(self.spin_cluster_pcs.GetValue()),method=method,
			n_clusters=max(2,int(round(value))),leiden_resolution=max(0.01,value),
			embedding=embedding,sample_size=int(self.spin_cluster_sample.GetValue()))
		return Path(marker),Path(self.clustering_output),config


	def start_clustering(self,event):
		for thread in(self.segmentation_thread,self.reconciliation_thread,self.cell_region_thread,self.quantification_thread,self.clustering_thread,self.spatial_thread):
			if thread is not None and thread.is_alive():
				wx.MessageBox('Another multiplex operation is already running.','Multiplex analysis',wx.OK|wx.ICON_INFORMATION)
				return
		try:
			marker,output,config=self._validate_clustering_inputs()
		except Exception as error:
			wx.MessageBox(str(error),'Unable to start cell clustering',wx.OK|wx.ICON_ERROR)
			return
		self.active_clustering_output=output
		self.clustering_cancel_event=threading.Event()
		cancel_event=self.clustering_cancel_event
		self._set_clustering_busy(True)
		self.gauge_clustering.SetValue(0)
		self.clustering_text.SetValue('Sampling marker profiles and fitting the phenotyping model...')


		def progress_callback(progress):
			wx.CallAfter(self._update_clustering_progress,progress)


		def worker():
			try:
				result=CellPhenotyper(marker).run(output,config,cancel_event=cancel_event,progress_callback=progress_callback)
				wx.CallAfter(self._clustering_finished,result,None)
			except Exception as error:
				wx.CallAfter(self._clustering_finished,None,error)
		self.clustering_thread=threading.Thread(target=worker,daemon=True,name='MPlexA-cell-clustering')
		self.clustering_thread.start()


	def cancel_clustering(self,event):
		if self.clustering_cancel_event is not None:
			self.clustering_cancel_event.set()
			self.button_cancel_clustering.Disable()
			self.clustering_text.SetValue('Cancellation requested. MPlexA will stop after the current streaming chunk.')


	def _update_clustering_progress(self,progress):
		self.gauge_clustering.SetValue(max(0,min(1000,int(round(progress.fraction*1000)))))
		self.clustering_text.SetValue(
			str(progress.stage)+': '+format(progress.completed,',')+'/'+format(progress.total,',')+
			('\n'+str(progress.message)if progress.message else'')+
			'\n\nOutput: '+str(self.active_clustering_output or self.clustering_output))


	def _clustering_finished(self,result,error):
		self._set_clustering_busy(False)
		self.clustering_cancel_event=None
		self.active_clustering_output=None
		if error is not None:
			self.clustering_text.SetValue('Cell clustering stopped with an error:\n'+str(error))
			wx.MessageBox(str(error),'Cell clustering failed',wx.OK|wx.ICON_ERROR)
			return
		self.gauge_clustering.SetValue(1000)
		self.clustering_output=Path(result.output_directory)
		self.text_cluster_output.SetLabel('Output: '+str(self.clustering_output))
		self.clustering_text.SetValue(result.summary())
		wx.MessageBox(result.summary(),'Cell clustering completed',wx.OK|wx.ICON_INFORMATION)


	def rename_cluster(self,event):
		if self.clustering_output is None or not(Path(self.clustering_output)/'clustering.sqlite').is_file():
			wx.MessageBox('Run cell clustering first.','Rename phenotype',wx.OK|wx.ICON_ERROR)
			return
		database=Path(self.clustering_output)/'clustering.sqlite'
		with sqlite3.connect(database)as connection:
			rows=connection.execute('SELECT cluster_id,cluster_name,COUNT(*) FROM cells GROUP BY cluster_id,cluster_name ORDER BY cluster_id').fetchall()
		choices=[str(row[0])+': '+str(row[1])+' ('+format(int(row[2]),',')+' cells)'for row in rows]
		dialog=wx.SingleChoiceDialog(self,'Select a cluster to rename as a biological phenotype.','Rename phenotype',choices)
		if dialog.ShowModal()!=wx.ID_OK:
			dialog.Destroy();return
		selection=dialog.GetSelection();dialog.Destroy()
		cluster_id=int(rows[selection][0]);current=str(rows[selection][1])
		name_dialog=wx.TextEntryDialog(self,'Enter the phenotype/cell-type name:','Rename phenotype',current)
		if name_dialog.ShowModal()==wx.ID_OK:
			name=name_dialog.GetValue().strip()
			if name:
				rename_clusters(self.clustering_output,{cluster_id:name})
				self.clustering_text.SetValue('Renamed cluster '+str(cluster_id)+' as '+name+'. Spatial graphs and the viewer will use the updated phenotype name.')
		name_dialog.Destroy()


	def select_spatial_output(self,event):
		default=''
		if self.spatial_output is not None:
			path=Path(self.spatial_output);default=str(path if path.exists()else path.parent)
		elif self.clustering_output is not None:
			default=str(self.clustering_output)
		dialog=wx.DirDialog(self,'Select or create the spatial-graph output folder',default,
			style=wx.DD_DEFAULT_STYLE|wx.DD_NEW_DIR_BUTTON)
		if dialog.ShowModal()==wx.ID_OK:
			self.spatial_output=Path(dialog.GetPath())
			self.text_spatial_output.SetLabel('Output: '+str(self.spatial_output))
		dialog.Destroy()


	def _set_spatial_busy(self,busy):
		self.button_run_spatial.Enable(not busy)
		self.button_cancel_spatial.Enable(busy)
		self.button_view_spatial.Enable(not busy)
		self.button_run_clustering.Enable(not busy)
		self.button_run_quantification.Enable(not busy)
		self.button_run_regions.Enable(not busy)
		self.button_run_reconciliation.Enable(not busy)
		self.button_run_segmentation.Enable(not busy)


	def _validate_spatial_inputs(self):
		if self.clustering_output is None:
			raise SpatialGraphError('Run or select a cell-clustering output first.')
		clustering=Path(self.clustering_output)
		if not(clustering/'clustering.sqlite').is_file():
			raise SpatialGraphError('Missing clustering.sqlite in '+str(clustering))
		method_map={0:'radius',1:'knn',2:'delaunay',3:'contact'}
		method=method_map[self.choice_graph_method.GetSelection()]
		if self.spatial_output is None:
			self.spatial_output=clustering/'spatial_graph'
		config=SpatialGraphConfig(
			method=method,radius=float(self.spin_graph_radius.GetValue()),
			k_neighbors=int(self.spin_graph_k.GetValue()),
			use_physical_units=bool(self.checkbox_graph_physical.GetValue()),
			query_block_size=int(self.spin_graph_block.GetValue()))
		label_store=None
		if method=='contact':
			if self.cell_region_output is None:
				raise SpatialGraphError('Direct cell-contact graph requires Module 4 cell-region labels.')
			label_store=resolve_cell_region_label_store(self.cell_region_output)
			if not label_store.is_dir():
				raise SpatialGraphError('Cell-region label store not found: '+str(label_store))
		pixel_size=None
		if self.image_metadata is not None:
			pixel_size=self.image_metadata.pixel_size_x
		return clustering,Path(self.spatial_output),config,label_store,pixel_size


	def start_spatial_graph(self,event):
		for thread in(self.segmentation_thread,self.reconciliation_thread,self.cell_region_thread,self.quantification_thread,self.clustering_thread,self.spatial_thread):
			if thread is not None and thread.is_alive():
				wx.MessageBox('Another multiplex operation is already running.','Multiplex analysis',wx.OK|wx.ICON_INFORMATION)
				return
		try:
			clustering,output,config,label_store,pixel_size=self._validate_spatial_inputs()
		except Exception as error:
			wx.MessageBox(str(error),'Unable to build spatial graph',wx.OK|wx.ICON_ERROR)
			return
		self.active_spatial_output=output
		self.spatial_cancel_event=threading.Event()
		cancel_event=self.spatial_cancel_event
		self._set_spatial_busy(True)
		self.gauge_spatial.SetValue(0)
		self.spatial_text.SetValue('Building sparse cell-cell graph...')


		def progress_callback(progress):
			wx.CallAfter(self._update_spatial_progress,progress)


		def worker():
			try:
				result=SpatialGraphBuilder(clustering).run(
					output,config,label_store_path=label_store,pixel_size=pixel_size,
					cancel_event=cancel_event,progress_callback=progress_callback)
				wx.CallAfter(self._spatial_finished,result,None)
			except Exception as error:
				wx.CallAfter(self._spatial_finished,None,error)
		self.spatial_thread=threading.Thread(target=worker,daemon=True,name='MPlexA-spatial-graph')
		self.spatial_thread.start()


	def cancel_spatial_graph(self,event):
		if self.spatial_cancel_event is not None:
			self.spatial_cancel_event.set()
			self.button_cancel_spatial.Disable()
			self.spatial_text.SetValue('Cancellation requested. MPlexA will stop after the current graph block.')


	def _update_spatial_progress(self,progress):
		self.gauge_spatial.SetValue(max(0,min(1000,int(round(progress.fraction*1000)))))
		self.spatial_text.SetValue(
			str(progress.stage)+': '+format(progress.completed,',')+'/'+format(progress.total,',')+
			('\n'+str(progress.message)if progress.message else'')+
			'\n\nOutput: '+str(self.active_spatial_output or self.spatial_output))


	def _spatial_finished(self,result,error):
		self._set_spatial_busy(False)
		self.spatial_cancel_event=None
		self.active_spatial_output=None
		if error is not None:
			self.spatial_text.SetValue('Spatial graph stopped with an error:\n'+str(error))
			wx.MessageBox(str(error),'Spatial graph failed',wx.OK|wx.ICON_ERROR)
			return
		self.gauge_spatial.SetValue(1000)
		self.spatial_output=Path(result.output_directory)
		self.text_spatial_output.SetLabel('Output: '+str(self.spatial_output))
		self.spatial_text.SetValue(result.summary())
		wx.MessageBox(result.summary(),'Spatial graph completed',wx.OK|wx.ICON_INFORMATION)


	def open_spatial_graph_viewer(self,event):
		if self.spatial_output is None:
			wx.MessageBox('Build or select a spatial-graph output first.','Spatial graph viewer',wx.OK|wx.ICON_ERROR)
			return
		spatial=Path(self.spatial_output)
		if not(spatial/'spatial_graph.sqlite').is_file():
			wx.MessageBox('Missing spatial_graph.sqlite in '+str(spatial),'Spatial graph viewer',wx.OK|wx.ICON_ERROR)
			return
		clustering=Path(self.clustering_output)if self.clustering_output is not None else None
		self.button_view_spatial.Disable()
		self.spatial_text.SetValue('Preparing viewport index for the spatial interaction graph. This is done once per graph/clustering result and is reused on later opens...')


		def worker():
			try:
				index=SpatialGraphOverlayIndex(spatial,clustering_directory=clustering)
				wx.CallAfter(self._spatial_graph_viewer_ready,spatial,index,None)
			except Exception as error:
				wx.CallAfter(self._spatial_graph_viewer_ready,spatial,None,error)
		threading.Thread(target=worker,daemon=True,name='MPlexA-spatial-view-index').start()


	def _spatial_graph_viewer_ready(self,spatial,index,error):
		self.button_view_spatial.Enable(True)
		if error is not None:
			self.spatial_text.SetValue('Unable to prepare spatial graph viewer:\n'+str(error))
			wx.MessageBox(str(error),'Unable to open spatial graph viewer',wx.OK|wx.ICON_ERROR)
			return
		try:
			frame=SpatialGraphViewerFrame(self,spatial,overlay_index=index)
			self.viewer_frames.append(frame);frame.Show(True)
			self.spatial_text.SetValue('Spatial graph viewer ready. Pan/zoom the network and filter edges by phenotype pair.')
		except Exception as error:
			wx.MessageBox(str(error),'Unable to open spatial graph viewer',wx.OK|wx.ICON_ERROR)


	def open_multichannel_viewer(self,event):
		if self.path_to_image is None or self.image_metadata is None:
			wx.MessageBox('Select and inspect a multiplex image first.','Multichannel viewer',wx.OK|wx.ICON_ERROR)
			return
		label_store=None
		if self.cell_region_output is not None:
			candidate=resolve_cell_region_label_store(self.cell_region_output)
			if candidate.is_dir():label_store=candidate
		elif self.reconciliation_output is not None:
			candidate=resolve_global_label_store(self.reconciliation_output)
			if candidate.is_dir():label_store=candidate
		clustering=None
		if self.clustering_output is not None and(Path(self.clustering_output)/'clustering.sqlite').is_file():
			clustering=Path(self.clustering_output)
		try:
			frame=MultiplexViewerFrame(
				self,self.path_to_image,series=self.series_index,label_store_path=label_store,
				clustering_directory=clustering)
			self.viewer_frames.append(frame)
			frame.Show(True)
		except Exception as error:
			wx.MessageBox(str(error),'Unable to open multichannel viewer',wx.OK|wx.ICON_ERROR)



class MultiplexViewerCanvas(wx.Panel):
	'''Pan/zoom canvas that renders multiplex viewports in a worker thread.'''


	def __init__(self,parent,renderer,settings_provider,label_store_path=None,cluster_index=None,segmentation_index=None,segmentation_settings_provider=None):
		super().__init__(parent,style=wx.BORDER_SIMPLE|wx.FULL_REPAINT_ON_RESIZE)
		# wx.AutoBufferedPaintDC requires BG_STYLE_PAINT on Windows/wxPython.
		# Set it once in the canvas constructor before the first paint event.
		self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
		self.renderer=renderer
		self.settings_provider=settings_provider
		self.label_store_path=Path(label_store_path)if label_store_path is not None else None
		self.cluster_index=cluster_index
		self.segmentation_index=segmentation_index
		self.segmentation_settings_provider=segmentation_settings_provider
		self.center_x=renderer.base_width/2
		self.center_y=renderer.base_height/2
		self.zoom=None
		self.bitmap=None
		self.render_token=0
		self.drag_start=None
		self.drag_center=None
		self.last_viewport=None
		self.last_rgb=None
		self.render_later=None
		self.show_boundaries=True
		self.show_clusters=True
		self.status='Preparing viewer...'
		self.Bind(wx.EVT_PAINT,self.on_paint)
		self.Bind(wx.EVT_SIZE,self.on_resize)
		self.Bind(wx.EVT_MOUSEWHEEL,self.on_mousewheel)
		self.Bind(wx.EVT_LEFT_DOWN,self.on_left_down)
		self.Bind(wx.EVT_LEFT_UP,self.on_left_up)
		self.Bind(wx.EVT_MOTION,self.on_motion)
		self.Bind(wx.EVT_LEFT_DCLICK,self.on_double_click)
		self.Bind(wx.EVT_WINDOW_DESTROY,self.on_destroy)
		self._alive=True
		wx.CallAfter(self.fit_whole_image)


	def on_destroy(self,event):
		self._alive=False
		event.Skip()


	def _client_size(self):
		width,height=self.GetClientSize()
		return max(2,int(width)),max(2,int(height))


	def fit_whole_image(self):
		width,height=self._client_size()
		self.zoom=min(width/self.renderer.base_width,height/self.renderer.base_height)
		self.center_x=self.renderer.base_width/2
		self.center_y=self.renderer.base_height/2
		self.request_render()


	def current_viewport(self):
		width,height=self._client_size()
		zoom=max(1e-8,float(self.zoom or 1.0))
		view_width=min(float(self.renderer.base_width),width/zoom)
		view_height=min(float(self.renderer.base_height),height/zoom)
		x=max(0.0,min(float(self.renderer.base_width)-view_width,self.center_x-view_width/2))
		y=max(0.0,min(float(self.renderer.base_height)-view_height,self.center_y-view_height/2))
		self.center_x=x+view_width/2
		self.center_y=y+view_height/2
		return Viewport(x,y,view_width,view_height,width,height)


	def screen_to_image(self,sx,sy):
		viewport=self.current_viewport()
		x=viewport.x+(float(sx)/max(1,viewport.screen_width))*viewport.width
		y=viewport.y+(float(sy)/max(1,viewport.screen_height))*viewport.height
		return x,y


	def request_render(self):
		if not self._alive:
			return
		self.render_token+=1
		token=self.render_token
		if self.render_later is not None and self.render_later.IsRunning():
			self.render_later.Stop()
		self.render_later=wx.CallLater(80,self._start_render,token)


	def _start_render(self,token):
		if not self._alive or token!=self.render_token:
			return
		viewport=self.current_viewport()
		settings=list(self.settings_provider())
		show_boundaries=self.show_boundaries
		show_clusters=self.show_clusters
		segmentation_settings=self.segmentation_settings_provider()if self.segmentation_settings_provider is not None else None
		self.status='Rendering...'
		self.Refresh(False)


		def worker():
			try:
				result=self.renderer.render(viewport,settings)
				rgb=np.ascontiguousarray(result.rgb.copy())
				segmentation_status=''
				if self.segmentation_index is not None and segmentation_settings is not None and segmentation_settings.get('visible',True):
					overlay=self.segmentation_index.render(
						viewport,owned_only=bool(segmentation_settings.get('owned_only',True)),
						max_predictions=int(segmentation_settings.get('max_predictions',50000)))
					if overlay.hidden_for_zoom:
						segmentation_status='; segmentation hidden at this zoom — zoom in'
					else:
						color=np.asarray(segmentation_settings.get('color',(255,255,0)),dtype=np.float32)
						if bool(segmentation_settings.get('fill',False))and np.any(overlay.fill):
							alpha=float(segmentation_settings.get('opacity',0.30))
							pixels=rgb[overlay.fill].astype(np.float32)
							rgb[overlay.fill]=np.clip(pixels*(1.0-alpha)+color[None,:]*alpha,0,255).astype(np.uint8)
						if bool(segmentation_settings.get('boundaries',True))and np.any(overlay.boundaries):
							rgb[overlay.boundaries]=color.astype(np.uint8)
						segmentation_status='; raw masks '+format(overlay.prediction_count,',')+' from '+format(overlay.tile_count,',')+' tile(s)'
						if overlay.truncated:segmentation_status+='; display limited'
				if show_boundaries and self.label_store_path is not None:
					boundary=label_boundaries_for_viewport(self.label_store_path,viewport)
					if boundary is not None:
						rgb[boundary]=np.array([255,255,255],dtype=np.uint8)
				if show_clusters and self.cluster_index is not None:
					cells=self.cluster_index.cells_in_region(viewport.x,viewport.y,viewport.width,viewport.height,limit=100000)
					for cell in cells:
						sx=int(round((float(cell['centroid_x'])-viewport.x)/viewport.width*viewport.screen_width))
						sy=int(round((float(cell['centroid_y'])-viewport.y)/viewport.height*viewport.screen_height))
						if 0<=sx<viewport.screen_width and 0<=sy<viewport.screen_height:
							cluster=int(cell['cluster_id'])
							color=DEFAULT_CHANNEL_COLORS[(cluster-1)%len(DEFAULT_CHANNEL_COLORS)]
							cv2.circle(rgb,(sx,sy),3,color,-1,lineType=cv2.LINE_AA)
				wx.CallAfter(self._render_finished,token,viewport,rgb,result.level,segmentation_status,None)
			except Exception as error:
				wx.CallAfter(self._render_finished,token,viewport,None,None,'',error)
		threading.Thread(target=worker,daemon=True,name='MPlexA-viewer-render').start()


	def _render_finished(self,token,viewport,rgb,level,segmentation_status,error):
		if not self._alive or token!=self.render_token:
			return
		if error is not None:
			self.status='Viewer error: '+str(error)
			self.Refresh(False)
			return
		self.last_viewport=viewport
		self.last_rgb=rgb
		height,width=rgb.shape[:2]
		self.bitmap=wx.Bitmap.FromBuffer(width,height,np.ascontiguousarray(rgb))
		self.status='Pyramid level '+str(level)+'; viewport '+format(int(viewport.width),',')+' × '+format(int(viewport.height),',')+' px'+str(segmentation_status or'')
		self.Refresh(False)


	def on_paint(self,event):
		dc=wx.AutoBufferedPaintDC(self)
		dc.SetBackground(wx.Brush(wx.Colour(20,20,20)))
		dc.Clear()
		if self.bitmap is not None:
			dc.DrawBitmap(self.bitmap,0,0,False)
		dc.SetTextForeground(wx.Colour(255,255,255))
		dc.SetBackgroundMode(wx.BRUSHSTYLE_TRANSPARENT)
		dc.DrawText(self.status,8,8)


	def on_resize(self,event):
		if self.zoom is not None:
			wx.CallAfter(self.request_render)
		event.Skip()


	def on_mousewheel(self,event):
		if self.zoom is None:
			return
		position=event.GetPosition()
		before=self.screen_to_image(position.x,position.y)
		factor=1.25 if event.GetWheelRotation()>0 else 1/1.25
		minimum=min(self._client_size()[0]/self.renderer.base_width,self._client_size()[1]/self.renderer.base_height)*0.25
		maximum=32.0
		self.zoom=max(minimum,min(maximum,self.zoom*factor))
		after=self.screen_to_image(position.x,position.y)
		self.center_x+=before[0]-after[0]
		self.center_y+=before[1]-after[1]
		self.request_render()


	def on_left_down(self,event):
		self.CaptureMouse()
		self.drag_start=event.GetPosition()
		self.drag_center=(self.center_x,self.center_y)


	def on_left_up(self,event):
		if self.HasCapture():self.ReleaseMouse()
		self.drag_start=None
		self.drag_center=None


	def on_motion(self,event):
		if self.drag_start is None or self.drag_center is None or not event.Dragging()or not event.LeftIsDown():
			return
		position=event.GetPosition()
		dx=position.x-self.drag_start.x
		dy=position.y-self.drag_start.y
		zoom=max(1e-8,float(self.zoom or 1.0))
		self.center_x=self.drag_center[0]-dx/zoom
		self.center_y=self.drag_center[1]-dy/zoom
		self.request_render()


	def on_double_click(self,event):
		if self.cluster_index is None:
			return
		x,y=self.screen_to_image(event.GetPosition().x,event.GetPosition().y)
		radius=max(3.0,15.0/max(1e-8,float(self.zoom or 1.0)))
		cell=self.cluster_index.nearest_cell(x,y,radius=radius)
		if cell is None:
			return
		lines=['Cell ID: '+str(cell['global_cell_id']),'Phenotype: '+str(cell['cluster_name']),
			'Position: ('+format(float(cell['centroid_x']),'.1f')+', '+format(float(cell['centroid_y']),'.1f')+')']
		features=cell.get('features',{})
		if features:
			lines.append('')
			for name,value in list(features.items())[:40]:
				lines.append(str(name)+': '+('NA'if value is None else format(float(value),'.4g')))
		wx.MessageBox('\n'.join(lines),'Cell marker profile',wx.OK|wx.ICON_INFORMATION)


	def save_current_view(self,parent):
		if self.last_rgb is None:
			wx.MessageBox('No rendered view is available yet.','Save view',wx.OK|wx.ICON_ERROR)
			return
		dialog=wx.FileDialog(parent,'Save current multiplex view','','multiplex_view.png','PNG image (*.png)|*.png',style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
		if dialog.ShowModal()==wx.ID_OK:
			path=dialog.GetPath()
			if not path.lower().endswith('.png'):path+='.png'
			cv2.imwrite(path,cv2.cvtColor(self.last_rgb,cv2.COLOR_RGB2BGR))
		dialog.Destroy()



class MultiplexViewerFrame(wx.Frame):
	'''Large-image lazy viewer for many-channel MPlexA images.'''


	def __init__(self,parent,image_path,series=0,label_store_path=None,clustering_directory=None,segmentation_directory=None,viewer_title='MPlexA Multichannel Viewer'):
		super().__init__(parent,title=viewer_title,size=(1450,900))
		self.renderer=MultiplexCompositeRenderer(image_path,series=series)
		self.channel_names=list(self.renderer.metadata.channel_names)
		self.filtered_indices=list(range(len(self.channel_names)))
		self.selected_channel=0
		self.settings={}
		dtype=np.dtype(self.renderer.metadata.dtype)
		if np.issubdtype(dtype,np.integer):
			default_max=float(np.iinfo(dtype).max)
		else:
			default_max=1.0
		for index,name in enumerate(self.channel_names):
			color=DEFAULT_CHANNEL_COLORS[index%len(DEFAULT_CHANNEL_COLORS)]
			self.settings[index]=ChannelDisplaySettings(index,color=color,minimum=0,maximum=default_max,gamma=1,opacity=1,visible=False)
		dapi=next((index for index,name in enumerate(self.channel_names)if'dapi'in name.lower()),0)
		self.settings[dapi]=ChannelDisplaySettings(dapi,color=(0,0,255),minimum=0,maximum=default_max,gamma=1,opacity=1,visible=True)
		self.selected_channel=dapi
		self.cluster_index=ClusterOverlayIndex(clustering_directory)if clustering_directory is not None else None
		self.segmentation_index=SegmentationOverlayIndex(segmentation_directory,base_width=self.renderer.base_width,base_height=self.renderer.base_height)if segmentation_directory is not None else None
		self.segmentation_color=(255,255,0)
		root=wx.Panel(self)
		main=wx.BoxSizer(wx.HORIZONTAL)
		controls=wx.Panel(root,size=(330,-1))
		control_sizer=wx.BoxSizer(wx.VERTICAL)
		title=wx.StaticText(controls,label='Channels');font=title.GetFont();font.SetWeight(wx.FONTWEIGHT_BOLD);title.SetFont(font)
		control_sizer.Add(title,0,wx.ALL,8)
		self.search_channels=wx.SearchCtrl(controls,style=wx.TE_PROCESS_ENTER)
		self.search_channels.Bind(wx.EVT_TEXT,self.filter_channels)
		control_sizer.Add(self.search_channels,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.channel_list=wx.CheckListBox(controls,choices=self.channel_names,size=(300,300))
		self.channel_list.Bind(wx.EVT_CHECKLISTBOX,self.toggle_channel)
		self.channel_list.Bind(wx.EVT_LISTBOX,self.select_channel)
		control_sizer.Add(self.channel_list,1,wx.LEFT|wx.RIGHT|wx.EXPAND,8)
		settings_box=wx.StaticBoxSizer(wx.VERTICAL,controls,label='Selected channel display')
		grid=wx.FlexGridSizer(4,2,4,8)
		grid.Add(wx.StaticText(controls,label='Minimum'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_view_min=wx.SpinCtrlDouble(controls,min=-1e12,max=1e12,initial=0,inc=1,size=(150,-1));self.spin_view_min.SetDigits(3);grid.Add(self.spin_view_min,0)
		grid.Add(wx.StaticText(controls,label='Maximum'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_view_max=wx.SpinCtrlDouble(controls,min=-1e12,max=1e12,initial=default_max,inc=1,size=(150,-1));self.spin_view_max.SetDigits(3);grid.Add(self.spin_view_max,0)
		grid.Add(wx.StaticText(controls,label='Gamma'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_view_gamma=wx.SpinCtrlDouble(controls,min=0.05,max=20,initial=1,inc=0.1,size=(150,-1));self.spin_view_gamma.SetDigits(2);grid.Add(self.spin_view_gamma,0)
		grid.Add(wx.StaticText(controls,label='Opacity'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_view_opacity=wx.SpinCtrlDouble(controls,min=0,max=1,initial=1,inc=0.05,size=(150,-1));self.spin_view_opacity.SetDigits(2);grid.Add(self.spin_view_opacity,0)
		settings_box.Add(grid,0,wx.ALL,6)
		row=wx.BoxSizer(wx.HORIZONTAL)
		self.button_view_color=wx.Button(controls,label='Color',size=(80,34));self.button_view_color.Bind(wx.EVT_BUTTON,self.choose_channel_color)
		button_apply=wx.Button(controls,label='Apply',size=(80,34));button_apply.Bind(wx.EVT_BUTTON,self.apply_channel_settings)
		button_auto=wx.Button(controls,label='Auto',size=(80,34));button_auto.Bind(wx.EVT_BUTTON,self.auto_contrast)
		row.Add(self.button_view_color,0,wx.RIGHT,5);row.Add(button_apply,0,wx.RIGHT,5);row.Add(button_auto,0)
		settings_box.Add(row,0,wx.ALL,6)
		control_sizer.Add(settings_box,0,wx.ALL|wx.EXPAND,8)
		if self.segmentation_index is not None:
			seg_box=wx.StaticBoxSizer(wx.VERTICAL,controls,label='Module 2 segmentation overlay')
			self.checkbox_segmentation_visible=wx.CheckBox(controls,label='Show raw segmentation');self.checkbox_segmentation_visible.SetValue(True);self.checkbox_segmentation_visible.Bind(wx.EVT_CHECKBOX,self.toggle_overlay)
			self.checkbox_segmentation_boundaries=wx.CheckBox(controls,label='Show mask outlines');self.checkbox_segmentation_boundaries.SetValue(True);self.checkbox_segmentation_boundaries.Bind(wx.EVT_CHECKBOX,self.toggle_overlay)
			self.checkbox_segmentation_fill=wx.CheckBox(controls,label='Fill masks');self.checkbox_segmentation_fill.SetValue(False);self.checkbox_segmentation_fill.Bind(wx.EVT_CHECKBOX,self.toggle_overlay)
			self.checkbox_segmentation_owned=wx.CheckBox(controls,label='Core-owned detections only');self.checkbox_segmentation_owned.SetValue(True);self.checkbox_segmentation_owned.Bind(wx.EVT_CHECKBOX,self.toggle_overlay)
			for item in(self.checkbox_segmentation_visible,self.checkbox_segmentation_boundaries,self.checkbox_segmentation_fill,self.checkbox_segmentation_owned):seg_box.Add(item,0,wx.LEFT|wx.RIGHT|wx.TOP,6)
			segrow=wx.BoxSizer(wx.HORIZONTAL)
			self.button_segmentation_color=wx.Button(controls,label='Mask color',size=(100,34));self.button_segmentation_color.SetBackgroundColour(wx.Colour(*self.segmentation_color));self.button_segmentation_color.Bind(wx.EVT_BUTTON,self.choose_segmentation_color)
			self.spin_segmentation_opacity=wx.SpinCtrlDouble(controls,min=0,max=1,initial=0.30,inc=0.05,size=(90,-1));self.spin_segmentation_opacity.SetDigits(2);self.spin_segmentation_opacity.Bind(wx.EVT_SPINCTRLDOUBLE,self.toggle_overlay)
			segrow.Add(self.button_segmentation_color,0,wx.RIGHT,6);segrow.Add(wx.StaticText(controls,label='Fill opacity'),0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,5);segrow.Add(self.spin_segmentation_opacity,0)
			seg_box.Add(segrow,0,wx.ALL,6)
			self.spin_segmentation_limit=wx.SpinCtrl(controls,min=1000,max=500000,initial=50000,size=(110,-1));self.spin_segmentation_limit.Bind(wx.EVT_SPINCTRL,self.toggle_overlay)
			limitrow=wx.BoxSizer(wx.HORIZONTAL);limitrow.Add(wx.StaticText(controls,label='Max visible masks'),0,wx.RIGHT|wx.ALIGN_CENTER_VERTICAL,6);limitrow.Add(self.spin_segmentation_limit,0);seg_box.Add(limitrow,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,6)
			control_sizer.Add(seg_box,0,wx.ALL|wx.EXPAND,8)
		self.checkbox_view_boundaries=wx.CheckBox(controls,label='Overlay cell boundaries');self.checkbox_view_boundaries.SetValue(label_store_path is not None);self.checkbox_view_boundaries.Enable(label_store_path is not None);self.checkbox_view_boundaries.Bind(wx.EVT_CHECKBOX,self.toggle_overlay)
		self.checkbox_view_clusters=wx.CheckBox(controls,label='Overlay phenotype centroids');self.checkbox_view_clusters.SetValue(self.cluster_index is not None);self.checkbox_view_clusters.Enable(self.cluster_index is not None);self.checkbox_view_clusters.Bind(wx.EVT_CHECKBOX,self.toggle_overlay)
		control_sizer.Add(self.checkbox_view_boundaries,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		control_sizer.Add(self.checkbox_view_clusters,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		viewer_buttons=wx.BoxSizer(wx.HORIZONTAL)
		button_fit=wx.Button(controls,label='Fit',size=(75,36));button_fit.Bind(wx.EVT_BUTTON,lambda event:self.canvas.fit_whole_image())
		button_save=wx.Button(controls,label='Save PNG',size=(100,36));button_save.Bind(wx.EVT_BUTTON,lambda event:self.canvas.save_current_view(self))
		viewer_buttons.Add(button_fit,0,wx.RIGHT,6);viewer_buttons.Add(button_save,0)
		control_sizer.Add(viewer_buttons,0,wx.ALL,8)
		help_text=wx.StaticText(controls,label='Mouse wheel: zoom\nLeft drag: pan\nSegmentation masks are read lazily only for the visible region.\nDouble-click phenotype centroid: inspect selected marker values')
		control_sizer.Add(help_text,0,wx.ALL,8)
		controls.SetSizer(control_sizer)
		self.canvas=MultiplexViewerCanvas(root,self.renderer,self.active_settings,label_store_path=label_store_path,cluster_index=self.cluster_index,segmentation_index=self.segmentation_index,segmentation_settings_provider=self.segmentation_overlay_settings if self.segmentation_index is not None else None)
		main.Add(controls,0,wx.EXPAND)
		main.Add(self.canvas,1,wx.EXPAND)
		root.SetSizer(main)
		self._refresh_channel_list()
		self._load_selected_controls()
		self.Centre()


	def segmentation_overlay_settings(self):
		if self.segmentation_index is None:
			return None
		return{
			'visible':bool(self.checkbox_segmentation_visible.GetValue()),
			'boundaries':bool(self.checkbox_segmentation_boundaries.GetValue()),
			'fill':bool(self.checkbox_segmentation_fill.GetValue()),
			'owned_only':bool(self.checkbox_segmentation_owned.GetValue()),
			'opacity':float(self.spin_segmentation_opacity.GetValue()),
			'color':tuple(int(value)for value in self.segmentation_color),
			'max_predictions':int(self.spin_segmentation_limit.GetValue()),
		}


	def choose_segmentation_color(self,event):
		data=wx.ColourData();data.SetColour(wx.Colour(*self.segmentation_color))
		dialog=wx.ColourDialog(self,data)
		if dialog.ShowModal()==wx.ID_OK:
			colour=dialog.GetColourData().GetColour()
			self.segmentation_color=(colour.Red(),colour.Green(),colour.Blue())
			self.button_segmentation_color.SetBackgroundColour(wx.Colour(*self.segmentation_color));self.button_segmentation_color.Refresh()
			self.canvas.request_render()
		dialog.Destroy()


	def active_settings(self):
		return[self.settings[index]for index in range(len(self.channel_names))if self.settings[index].visible]


	def _refresh_channel_list(self):
		query=self.search_channels.GetValue().strip().lower()if hasattr(self,'search_channels')else''
		self.filtered_indices=[index for index,name in enumerate(self.channel_names)if query in name.lower()]
		self.channel_list.Set([self.channel_names[index]for index in self.filtered_indices])
		for displayed,index in enumerate(self.filtered_indices):
			self.channel_list.Check(displayed,self.settings[index].visible)
		if self.selected_channel in self.filtered_indices:
			self.channel_list.SetSelection(self.filtered_indices.index(self.selected_channel))


	def filter_channels(self,event):
		self._refresh_channel_list()


	def toggle_channel(self,event):
		displayed=event.GetInt()
		if displayed<0 or displayed>=len(self.filtered_indices):return
		index=self.filtered_indices[displayed]
		setting=self.settings[index]
		self.settings[index]=ChannelDisplaySettings(index,color=setting.color,minimum=setting.minimum,maximum=setting.maximum,gamma=setting.gamma,opacity=setting.opacity,visible=self.channel_list.IsChecked(displayed))
		self.canvas.request_render()


	def select_channel(self,event):
		displayed=event.GetSelection()
		if displayed<0 or displayed>=len(self.filtered_indices):return
		self.selected_channel=self.filtered_indices[displayed]
		self._load_selected_controls()


	def _load_selected_controls(self):
		setting=self.settings[self.selected_channel]
		self.spin_view_min.SetValue(float(setting.minimum));self.spin_view_max.SetValue(float(setting.maximum))
		self.spin_view_gamma.SetValue(float(setting.gamma));self.spin_view_opacity.SetValue(float(setting.opacity))
		self.button_view_color.SetBackgroundColour(wx.Colour(*setting.color))
		self.button_view_color.Refresh()


	def choose_channel_color(self,event):
		setting=self.settings[self.selected_channel]
		data=wx.ColourData();data.SetColour(wx.Colour(*setting.color))
		dialog=wx.ColourDialog(self,data)
		if dialog.ShowModal()==wx.ID_OK:
			colour=dialog.GetColourData().GetColour()
			color=(colour.Red(),colour.Green(),colour.Blue())
			self.settings[self.selected_channel]=ChannelDisplaySettings(self.selected_channel,color=color,minimum=setting.minimum,maximum=setting.maximum,gamma=setting.gamma,opacity=setting.opacity,visible=setting.visible)
			self._load_selected_controls();self.canvas.request_render()
		dialog.Destroy()


	def apply_channel_settings(self,event):
		minimum=float(self.spin_view_min.GetValue());maximum=float(self.spin_view_max.GetValue())
		if maximum<=minimum:
			wx.MessageBox('Display maximum must exceed minimum.','Channel display',wx.OK|wx.ICON_ERROR);return
		old=self.settings[self.selected_channel]
		self.settings[self.selected_channel]=ChannelDisplaySettings(self.selected_channel,color=old.color,minimum=minimum,maximum=maximum,gamma=float(self.spin_view_gamma.GetValue()),opacity=float(self.spin_view_opacity.GetValue()),visible=old.visible)
		self.canvas.request_render()


	def auto_contrast(self,event):
		try:
			low,high=self.renderer.auto_contrast(self.canvas.current_viewport(),self.selected_channel)
			self.spin_view_min.SetValue(low);self.spin_view_max.SetValue(high)
			self.apply_channel_settings(event)
		except Exception as error:
			wx.MessageBox(str(error),'Auto contrast failed',wx.OK|wx.ICON_ERROR)


	def toggle_overlay(self,event):
		self.canvas.show_boundaries=bool(self.checkbox_view_boundaries.GetValue())
		self.canvas.show_clusters=bool(self.checkbox_view_clusters.GetValue())
		self.canvas.request_render()



class SegmentationViewerFrame(MultiplexViewerFrame):
	'''QC viewer for raw Module 2 segmentation predictions before reconciliation.'''


	def __init__(self,parent,image_path,segmentation_directory,series=0):
		super().__init__(parent,image_path,series=series,segmentation_directory=segmentation_directory,
			viewer_title='MPlexA Segmentation Results Viewer')



class SpatialGraphCanvas(wx.Panel):
	'''Pan/zoom canvas for viewport-limited cell-cell interaction graph rendering.'''


	def __init__(self,parent,index,filter_provider,edge_limit_provider,node_radius_provider):
		super().__init__(parent,style=wx.BORDER_SIMPLE|wx.FULL_REPAINT_ON_RESIZE)
		self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
		self.index=index
		self.filter_provider=filter_provider
		self.edge_limit_provider=edge_limit_provider
		self.node_radius_provider=node_radius_provider
		self.min_x,self.min_y,self.max_x,self.max_y=index.bounds()
		if self.max_x<=self.min_x:self.max_x=self.min_x+1.0
		if self.max_y<=self.min_y:self.max_y=self.min_y+1.0
		self.center_x=(self.min_x+self.max_x)/2
		self.center_y=(self.min_y+self.max_y)/2
		self.zoom=None
		self.bitmap=None
		self.last_rgb=None
		self.last_view=None
		self.render_token=0
		self.render_later=None
		self.drag_start=None
		self.drag_center=None
		self.status='Preparing spatial graph...'
		self.show_edges=True
		self.show_nodes=True
		self._alive=True
		self.Bind(wx.EVT_PAINT,self.on_paint)
		self.Bind(wx.EVT_SIZE,self.on_resize)
		self.Bind(wx.EVT_MOUSEWHEEL,self.on_mousewheel)
		self.Bind(wx.EVT_LEFT_DOWN,self.on_left_down)
		self.Bind(wx.EVT_LEFT_UP,self.on_left_up)
		self.Bind(wx.EVT_MOTION,self.on_motion)
		self.Bind(wx.EVT_LEFT_DCLICK,self.on_double_click)
		self.Bind(wx.EVT_WINDOW_DESTROY,self.on_destroy)
		wx.CallAfter(self.fit_graph)


	def on_destroy(self,event):
		self._alive=False
		event.Skip()


	def _client_size(self):
		width,height=self.GetClientSize()
		return max(2,int(width)),max(2,int(height))


	def fit_graph(self):
		width,height=self._client_size()
		graph_width=max(1.0,self.max_x-self.min_x)
		graph_height=max(1.0,self.max_y-self.min_y)
		self.zoom=max(1e-8,min(width/graph_width,height/graph_height)*0.95)
		self.center_x=(self.min_x+self.max_x)/2
		self.center_y=(self.min_y+self.max_y)/2
		self.request_render()


	def current_view(self):
		width,height=self._client_size()
		zoom=max(1e-8,float(self.zoom or 1.0))
		view_width=width/zoom
		view_height=height/zoom
		x=self.center_x-view_width/2
		y=self.center_y-view_height/2
		return(x,y,view_width,view_height,width,height)


	def screen_to_graph(self,sx,sy):
		x,y,w,h,sw,sh=self.current_view()
		return x+(float(sx)/max(1,sw))*w,y+(float(sy)/max(1,sh))*h


	def request_render(self):
		if not self._alive:return
		self.render_token+=1
		token=self.render_token
		if self.render_later is not None and self.render_later.IsRunning():self.render_later.Stop()
		self.render_later=wx.CallLater(70,self._start_render,token)


	def _start_render(self,token):
		if not self._alive or token!=self.render_token:return
		view=self.current_view()
		cluster_a,cluster_b=self.filter_provider()
		edge_limit=max(100,int(self.edge_limit_provider()))
		show_edges=self.show_edges;show_nodes=self.show_nodes
		self.status='Querying spatial graph...';self.Refresh(False)


		def worker():
			try:
				x,y,w,h,sw,sh=view
				data=self.index.query_region(x,y,w,h,cluster_a=cluster_a,cluster_b=cluster_b,
					edge_limit=edge_limit,node_limit=min(max(edge_limit,10000),200000))
				rgb=np.zeros((sh,sw,3),dtype=np.uint8)
				rgb[:]=np.array([22,22,22],dtype=np.uint8)


				def screen(px,py):
					return int(round((px-x)/w*sw)),int(round((py-y)/h*sh))
				if show_edges:
					for _,_,sx,sy,tx,ty,source_cluster,target_cluster in data.edges:
						x1,y1=screen(sx,sy);x2,y2=screen(tx,ty)
						c1=np.asarray(DEFAULT_CHANNEL_COLORS[(max(1,source_cluster)-1)%len(DEFAULT_CHANNEL_COLORS)],dtype=np.float64)
						c2=np.asarray(DEFAULT_CHANNEL_COLORS[(max(1,target_cluster)-1)%len(DEFAULT_CHANNEL_COLORS)],dtype=np.float64)
						color=tuple(int(v)for v in np.clip((c1+c2)*0.38,45,210))
						cv2.line(rgb,(x1,y1),(x2,y2),color,1,lineType=cv2.LINE_AA)
				if show_nodes:
					radius=max(1,int(self.node_radius_provider()))
					for _,px,py,cluster in data.nodes:
						sx,sy=screen(px,py)
						if 0<=sx<sw and 0<=sy<sh:
							color=DEFAULT_CHANNEL_COLORS[(max(1,cluster)-1)%len(DEFAULT_CHANNEL_COLORS)]
							cv2.circle(rgb,(sx,sy),radius,color,-1,lineType=cv2.LINE_AA)
				status=(f'Visible nodes: {len(data.nodes):,}; edges: {len(data.edges):,}')
				if data.nodes_truncated:status+='; node display limited'
				if data.edges_truncated:status+='; edge display limited'
				wx.CallAfter(self._render_finished,token,view,rgb,status,None)
			except Exception as error:
				wx.CallAfter(self._render_finished,token,view,None,'',error)
		threading.Thread(target=worker,daemon=True,name='MPlexA-spatial-viewer-render').start()


	def _render_finished(self,token,view,rgb,status,error):
		if not self._alive or token!=self.render_token:return
		if error is not None:
			self.status='Spatial graph viewer error: '+str(error);self.Refresh(False);return
		self.last_view=view;self.last_rgb=rgb
		height,width=rgb.shape[:2]
		self.bitmap=wx.Bitmap.FromBuffer(width,height,np.ascontiguousarray(rgb))
		self.status=status;self.Refresh(False)


	def on_paint(self,event):
		dc=wx.AutoBufferedPaintDC(self)
		dc.SetBackground(wx.Brush(wx.Colour(22,22,22)));dc.Clear()
		if self.bitmap is not None:dc.DrawBitmap(self.bitmap,0,0,False)
		dc.SetTextForeground(wx.Colour(255,255,255));dc.SetBackgroundMode(wx.BRUSHSTYLE_TRANSPARENT)
		dc.DrawText(self.status,8,8)


	def on_resize(self,event):
		if self.zoom is not None:wx.CallAfter(self.request_render)
		event.Skip()


	def on_mousewheel(self,event):
		if self.zoom is None:return
		position=event.GetPosition();before=self.screen_to_graph(position.x,position.y)
		factor=1.25 if event.GetWheelRotation()>0 else 1/1.25
		self.zoom=max(1e-8,min(100.0,float(self.zoom)*factor))
		after=self.screen_to_graph(position.x,position.y)
		self.center_x+=before[0]-after[0];self.center_y+=before[1]-after[1]
		self.request_render()


	def on_left_down(self,event):
		if not self.HasCapture():self.CaptureMouse()
		self.drag_start=event.GetPosition();self.drag_center=(self.center_x,self.center_y)


	def on_left_up(self,event):
		if self.HasCapture():self.ReleaseMouse()
		self.drag_start=None;self.drag_center=None


	def on_motion(self,event):
		if self.drag_start is None or self.drag_center is None or not event.Dragging()or not event.LeftIsDown():return
		position=event.GetPosition();zoom=max(1e-8,float(self.zoom or 1.0))
		self.center_x=self.drag_center[0]-(position.x-self.drag_start.x)/zoom
		self.center_y=self.drag_center[1]-(position.y-self.drag_start.y)/zoom
		self.request_render()


	def on_double_click(self,event):
		x,y=self.screen_to_graph(event.GetPosition().x,event.GetPosition().y)
		radius=max(3.0,15.0/max(1e-8,float(self.zoom or 1.0)))
		cell=self.index.nearest_cell(x,y,radius=radius)
		if cell is not None:
			wx.MessageBox('Cell ID: '+str(cell['global_cell_id'])+'\nPhenotype: '+str(cell['cluster_name'])+
				'\nPosition: ('+format(cell['centroid_x'],'.1f')+', '+format(cell['centroid_y'],'.1f')+')',
				'Cell in interaction graph',wx.OK|wx.ICON_INFORMATION)


	def save_current_view(self,parent):
		if self.last_rgb is None:
			wx.MessageBox('No rendered graph view is available yet.','Save graph view',wx.OK|wx.ICON_ERROR);return
		dialog=wx.FileDialog(parent,'Save current spatial graph view','','spatial_graph_view.png','PNG image (*.png)|*.png',style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
		if dialog.ShowModal()==wx.ID_OK:
			path=dialog.GetPath();path=path if path.lower().endswith('.png')else path+'.png'
			cv2.imwrite(path,cv2.cvtColor(self.last_rgb,cv2.COLOR_RGB2BGR))
		dialog.Destroy()



class SpatialGraphViewerFrame(wx.Frame):
	'''Interactive phenotype-filterable cell-cell spatial interaction graph viewer.'''


	def __init__(self,parent,spatial_directory,clustering_directory=None,overlay_index=None):
		super().__init__(parent,title='MPlexA Spatial Interaction Graph Viewer',size=(1400,880))
		self.index=overlay_index if overlay_index is not None else SpatialGraphOverlayIndex(spatial_directory,clustering_directory=clustering_directory)
		self.cluster_rows=list(self.index.cluster_names())
		root=wx.Panel(self);main=wx.BoxSizer(wx.HORIZONTAL)
		controls=wx.Panel(root,size=(330,-1));control_sizer=wx.BoxSizer(wx.VERTICAL)
		title=wx.StaticText(controls,label='Cell-cell interaction graph');font=title.GetFont();font.SetWeight(wx.FONTWEIGHT_BOLD);title.SetFont(font)
		control_sizer.Add(title,0,wx.ALL,10)
		control_sizer.Add(wx.StaticText(controls,label='Filter interaction edges by phenotype pair:'),0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		choices=['Any phenotype']+[str(cid)+': '+name for cid,name in self.cluster_rows]
		self.choice_pair_a=wx.Choice(controls,choices=choices);self.choice_pair_a.SetSelection(0);self.choice_pair_a.Bind(wx.EVT_CHOICE,self.filter_changed)
		self.choice_pair_b=wx.Choice(controls,choices=choices);self.choice_pair_b.SetSelection(0);self.choice_pair_b.Bind(wx.EVT_CHOICE,self.filter_changed)
		control_sizer.Add(wx.StaticText(controls,label='Phenotype A'),0,wx.LEFT|wx.RIGHT|wx.TOP,8);control_sizer.Add(self.choice_pair_a,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		control_sizer.Add(wx.StaticText(controls,label='Phenotype B'),0,wx.LEFT|wx.RIGHT|wx.TOP,8);control_sizer.Add(self.choice_pair_b,0,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		self.checkbox_graph_edges=wx.CheckBox(controls,label='Show interaction edges');self.checkbox_graph_edges.SetValue(True);self.checkbox_graph_edges.Bind(wx.EVT_CHECKBOX,self.overlay_changed)
		self.checkbox_graph_nodes=wx.CheckBox(controls,label='Show cell nodes');self.checkbox_graph_nodes.SetValue(True);self.checkbox_graph_nodes.Bind(wx.EVT_CHECKBOX,self.overlay_changed)
		control_sizer.Add(self.checkbox_graph_edges,0,wx.LEFT|wx.RIGHT|wx.TOP,8);control_sizer.Add(self.checkbox_graph_nodes,0,wx.LEFT|wx.RIGHT|wx.BOTTOM,8)
		grid=wx.FlexGridSizer(2,2,5,8)
		grid.Add(wx.StaticText(controls,label='Max visible edges'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_graph_edges=wx.SpinCtrl(controls,min=100,max=1000000,initial=50000,size=(130,-1));self.spin_graph_edges.Bind(wx.EVT_SPINCTRL,self.filter_changed);grid.Add(self.spin_graph_edges,0)
		grid.Add(wx.StaticText(controls,label='Node radius (px)'),0,wx.ALIGN_CENTER_VERTICAL)
		self.spin_graph_node_radius=wx.SpinCtrl(controls,min=1,max=10,initial=3,size=(130,-1));self.spin_graph_node_radius.Bind(wx.EVT_SPINCTRL,self.filter_changed);grid.Add(self.spin_graph_node_radius,0)
		control_sizer.Add(grid,0,wx.ALL,8)
		buttons=wx.BoxSizer(wx.HORIZONTAL)
		button_fit=wx.Button(controls,label='Fit',size=(75,36));button_fit.Bind(wx.EVT_BUTTON,lambda event:self.canvas.fit_graph())
		button_save=wx.Button(controls,label='Save PNG',size=(100,36));button_save.Bind(wx.EVT_BUTTON,lambda event:self.canvas.save_current_view(self))
		buttons.Add(button_fit,0,wx.RIGHT,6);buttons.Add(button_save,0)
		control_sizer.Add(buttons,0,wx.ALL,8)
		legend=wx.StaticText(controls,label='Node colors represent phenotypes. Edge color blends the two endpoint phenotype colors.\n\nMouse wheel: zoom\nLeft drag: pan\nDouble-click a cell: inspect phenotype')
		control_sizer.Add(legend,0,wx.ALL|wx.EXPAND,8)
		cluster_text='\n'.join(str(cid)+': '+name for cid,name in self.cluster_rows[:40])
		if len(self.cluster_rows)>40:cluster_text+='\n...'
		self.legend_text=wx.TextCtrl(controls,value=cluster_text,style=wx.TE_MULTILINE|wx.TE_READONLY,size=(-1,220))
		control_sizer.Add(self.legend_text,1,wx.LEFT|wx.RIGHT|wx.BOTTOM|wx.EXPAND,8)
		controls.SetSizer(control_sizer)
		self.canvas=SpatialGraphCanvas(root,self.index,self.selected_pair,lambda:self.spin_graph_edges.GetValue(),lambda:self.spin_graph_node_radius.GetValue())
		main.Add(controls,0,wx.EXPAND);main.Add(self.canvas,1,wx.EXPAND);root.SetSizer(main)
		self.Centre()


	def selected_pair(self):


		def get(choice):
			selection=choice.GetSelection()
			return None if selection<=0 else int(self.cluster_rows[selection-1][0])
		return get(self.choice_pair_a),get(self.choice_pair_b)


	def filter_changed(self,event):
		self.canvas.request_render()


	def overlay_changed(self,event):
		self.canvas.show_edges=bool(self.checkbox_graph_edges.GetValue())
		self.canvas.show_nodes=bool(self.checkbox_graph_nodes.GetValue())
		self.canvas.request_render()



class MainFrame(wx.Frame):


	def __init__(self):
		super().__init__(None,title=f'MPlexA v{__version__}')
		self.SetSize((1000,600))
		self.aui_manager=wx.aui.AuiManager()
		self.aui_manager.SetManagedWindow(self)
		self.notebook=wx.aui.AuiNotebook(self)
		self.aui_manager.AddPane(self.notebook,wx.aui.AuiPaneInfo().CenterPane())
		panel=InitialPanel(self.notebook)
		self.notebook.AddPage(panel,'Welcome',select=True)
		sizer=wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.notebook,1,wx.EXPAND)
		self.SetSizer(sizer)
		self.aui_manager.Update()
		self.Centre()
		self.Show()


def main_window():
	app=wx.App()
	MainFrame()
	print('MPlexA user interface initialized!')
	app.MainLoop()



if __name__=='__main__':
	main_window()
