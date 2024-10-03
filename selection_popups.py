#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:28:55 2024

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
import probeinterface as prif
from probeinterface.plotting import plot_probe
import pdb
# custom modules
import pyfx
import ephys
import gui_items as gi
from probe_gui import make_probe_plot, ProbeFileSimple

def clean(mode, base, last, init_ddir=''):
    if os.path.exists(init_ddir) and os.path.isdir(init_ddir):
        return init_ddir
    if mode==0: return base
    if mode==1: return last
    if mode==2:
        res = last if (Path(base) in Path(last).parents) else base
        return res
    
    
def validate_raw_ddir(ddir):
    if not os.path.isdir(ddir):
        return False
    files = os.listdir(ddir)
    a = bool('structure.oebin' in files)
    b = bool(len([f for f in files if f.endswith('.xdat.json')]) > 0)
    return bool(a or b)


def validate_processed_ddir(ddir):
    if not os.path.isdir(ddir):
        return False
    files = os.listdir(ddir)
    flist = ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy']
    x = all([bool(f in files) for f in flist])
    return x


class DirectorySelectionWidget(QtWidgets.QWidget):
    def __init__(self, title='', simple=False, qedit_simple=False, parent=None):
        super().__init__(parent)
        self.simple_mode = simple
        
        # row 1 - QLabel and status icon (formerly info button)
        self.ddir_lbl_hbox = QtWidgets.QHBoxLayout()
        self.ddir_lbl_hbox.setContentsMargins(0,0,0,0)
        self.ddir_lbl_hbox.setSpacing(3)
        self.ddir_lbl = QtWidgets.QLabel()
        self.ddir_lbl.setText(title)
        
        # yes/no icons
        self.ddir_icon_btn = gi.StatusIcon(init_state=0)#QtWidgets.QPushButton()  # folder status icon
        self.ddir_lbl_hbox.addWidget(self.ddir_icon_btn)
        self.ddir_lbl_hbox.addWidget(self.ddir_lbl)
        
        # row 2 - QLineEdit, and folder button
        self.qedit_hbox = gi.QEdit_HBox(simple=qedit_simple)  # 3 x QLineEdit items
        self.ddir_btn = QtWidgets.QPushButton()  # file dlg launch button
        self.ddir_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.ddir_btn.setMinimumSize(30,30)
        self.ddir_btn.setIconSize(QtCore.QSize(20,20))
        
        # assemble layout
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setHorizontalSpacing(5)
        self.grid.setVerticalSpacing(8)
        self.grid.addLayout(self.ddir_lbl_hbox,      0, 1)
        #self.grid.addWidget(self.ddir_icon_btn, 1, 0)
        self.grid.addLayout(self.qedit_hbox,    1, 1, 1, 4)
        self.grid.addWidget(self.ddir_btn,      1, 5)
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 2)
        self.grid.setColumnStretch(2, 2)
        self.grid.setColumnStretch(3, 2)
        self.grid.setColumnStretch(4, 2)
        self.grid.setColumnStretch(5, 0)
        
        if self.simple_mode:
            #self.ddir_icon_btn.hide()
            self.grid.setColumnStretch(0,0)
    
    def update_status(self, ddir, x=False):
        try:
            self.qedit_hbox.update_qedit(ddir, x)  # line edit
            self.ddir_icon_btn.new_status(x)  # status icon
        except:
            print('fu')
            #pdb.set_trace()
        #self.ddir_icon_btn.setIcon(self.ddir_icon_btn.icons[int(x)]) 
        
        
class RawDirectorySelectionPopup(QtWidgets.QDialog):
    RAW_DDIR_VALID = False
    PROCESSED_DDIR_VALID = False
    PROBE_CONFIG_VALID = False
    
    
    def __init__(self, mode=2, raw_ddir='', processed_ddir='', probe_ddir='', parent=None):
        # 0==base, 1=last visited, 2=last visited IF it's within base
        super().__init__(parent)
        
        raw_base, processed_base, probe_base, probe_file  = ephys.base_dirs()
        # get most recently entered directory
        qfd = QtWidgets.QFileDialog()
        last_ddir = str(qfd.directory().path())
        
        self.raw_ddir = clean(mode, raw_base, last_ddir, str(raw_ddir))
        self.processed_ddir = clean(mode, processed_base, last_ddir, str(processed_ddir))
        self.probe_ddir = clean(mode, probe_base, last_ddir, str(probe_ddir))
        
        self.gen_layout()
        self.ddir_gbox2.hide()
        self.probe_gbox.hide()
        
        self.update_raw_ddir()
        if len(os.listdir(self.processed_ddir)) == 0:
            self.update_processed_ddir()
        else:
            self.ddw2.update_status(self.processed_ddir, False)
        try:
            self.probe = ephys.read_probe_file(self.probe_ddir)  # get Probe or None
        except:
            pdb.set_trace()
        self.update_probe_obj()
            
    
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(20)
        gbox_ss = 'QGroupBox {background-color : rgba(230,230,230,255);}'# border : 2px ridge gray; border-radius : 4px;}'
        
        ###   SELECT RAW DATA FOLDER
        
        self.ddir_gbox = QtWidgets.QGroupBox()
        self.ddir_gbox.setStyleSheet(gbox_ss)
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        # basic directory selection widget
        self.ddw = DirectorySelectionWidget(title='<b><u>Raw data directory</u></b>')
        self.qedit_hbox = self.ddw.qedit_hbox
        # types of recording files
        self.oe_radio = QtWidgets.QRadioButton('Open Ephys (.oebin)')
        self.nn_radio = QtWidgets.QRadioButton('NeuroNexus (.xdat.json)')
        self.manual_radio = QtWidgets.QRadioButton('Upload custom file')
        for btn in [self.oe_radio, self.nn_radio, self.manual_radio]:
            btn.setAutoExclusive(False)
            btn.setEnabled(False)
            btn.setStyleSheet('QRadioButton {color : black;}'
                              'QRadioButton:disabled {color : black;}')
        self.ddw.grid.addWidget(self.oe_radio, 2, 1, 1, 2)
        self.ddw.grid.addWidget(self.nn_radio, 2, 3, 1, 2)
        ddir_vbox.addWidget(self.ddw)
        # manual file upload with custom parameters 
        custom_bar = QtWidgets.QFrame()
        custom_bar.setFrameShape(QtWidgets.QFrame.Panel)
        custom_bar.setFrameShadow(QtWidgets.QFrame.Sunken)
        frame_hlay = QtWidgets.QHBoxLayout(custom_bar)
        self.manual_upload_btn = QtWidgets.QPushButton()
        self.manual_upload_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ArrowForward))
        frame_hlay.addWidget(self.manual_radio)
        frame_hlay.addWidget(self.manual_upload_btn)
        ddir_vbox.addWidget(custom_bar)
        
        ###   CREATE PROCESSED DATA FOLDER
        
        self.ddir_gbox2 = QtWidgets.QGroupBox()
        self.ddir_gbox2.setStyleSheet('QGroupBox {border-width : 0px; font-weight : bold; text-decoration : underline;}')
        ddir_vbox2 = pyfx.InterWidgets(self.ddir_gbox2, 'v')[2]
        #ddir_vbox2.setContentsMargins(6,10,6,10)
        #ddir_vbox2.setContentsMargins(11,15,11,15)
        # create/overwrite processed data directory
        self.ddw2 = DirectorySelectionWidget(title='<b><u>Save data</u></b>', simple=True)
        #self.ddw2.ddir_lbl.hide()
        self.qedit_hbox2 = self.ddw2.qedit_hbox
        ddir_vbox2.addWidget(self.ddw2)
        
        ###   LOAD/CREATE PROBE FILE
        
        self.probe_gbox = QtWidgets.QGroupBox()
        self.probe_gbox.setStyleSheet('QGroupBox {border-width : 0px; font-weight : bold; text-decoration : underline;}')
        #prbw = QtWidgets.QWidget()
        probe_vbox = pyfx.InterWidgets(self.probe_gbox, 'v')[2] #findme
        #probe_hbox.setContentsMargins(11,15,11,15)
        #probe_hbox = QtWidgets.QHBoxLayout()
        # title and status button
        row0 = QtWidgets.QHBoxLayout()
        row0.setContentsMargins(0,0,0,0)
        row0.setSpacing(3)
        self.prb_icon_btn = gi.StatusIcon(init_state=0)
        probe_lbl = QtWidgets.QLabel('<b><u>Probe(s)</u></b>')
        row0.addWidget(self.prb_icon_btn)
        row0.addWidget(probe_lbl)
        row0.addStretch()
        # displayed probe name
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(5)
        self.probe_qlabel = QtWidgets.QLabel('---')
        self.probe_qlabel.setStyleSheet('QLabel {background-color:white;'
                                        'border:1px solid gray;'
                                        #'border-right:none;'
                                        'padding:10px;}')
        self.probe_qlabel.setAlignment(QtCore.Qt.AlignCenter)
        probe_x = QtWidgets.QLabel('x')
        probe_x.setStyleSheet('QLabel {background-color:transparent;'
                                      #'border:1px solid gray;'
                                      #'border-right:none;'
                                      #'border-left:none;'
                                      'font-size:14pt; '#'font-weight:bold;'
                                      #'padding: 10px 5px;'
                                      '}')
        probe_x.setAlignment(QtCore.Qt.AlignCenter)
        self.probe_n = QtWidgets.QSpinBox()
        self.probe_n.setAlignment(QtCore.Qt.AlignCenter)
        self.probe_n.setMinimum(1)
        #self.probe_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        #self.probe_n.setButtonSymbols(QtWidgets.QAbstractSpinBox.PlusMinus)
        self.probe_n.setStyleSheet('QSpinBox {'
                                   #'background-color:transparent;'
                                   #'border:3px solid black;'
                                  # 'border-color:white;'
                                   'font-size:14pt; font-weight:bold;'
                                   'padding:10px 0px;}')
        self.probe_n.valueChanged.connect(self.update_nchannels)
        
        probe_arrow = QtWidgets.QLabel('\u27A4')  # unicode ➤
        probe_arrow.setAlignment(QtCore.Qt.AlignCenter)
        probe_arrow.setStyleSheet('QLabel {padding: 0px 5px;}')
        self.total_channel_fmt = '<code>{}<br>channels</code>'
        self.total_channel_lbl = QtWidgets.QLabel(self.total_channel_fmt.format('-'))
        self.total_channel_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.total_channel_lbl.setStyleSheet('QLabel {background-color:rgba(255,255,255,150); padding:2px;}')
        row1.addWidget(self.probe_qlabel, stretch=2)
        row1.addWidget(probe_x, stretch=0)
        row1.addWidget(self.probe_n, stretch=0)
        row1.addWidget(probe_arrow, stretch=0)
        row1.addWidget(self.total_channel_lbl, stretch=1)
        # probe buttons
        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(10)
        self.prbf_load = QtWidgets.QPushButton('Load')
        self.prbf_load.setStyleSheet('QPushButton {padding:5px;}')
        #self.prbf_load.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
        self.prbf_view = QtWidgets.QPushButton('View')
        self.prbf_view.setStyleSheet('QPushButton {padding:5px;}')
        self.prbf_view.setEnabled(False)
        self.prbf_make = QtWidgets.QPushButton('New')
        self.prbf_make.setStyleSheet('QPushButton {padding:5px;}')
        
        #row1.addWidget(self.probe_qlabel, stretch=4)
        row2.addWidget(self.prbf_load)
        row2.addWidget(self.prbf_view)
        row2.addWidget(self.prbf_make)
        probe_vbox.addLayout(row0)
        probe_vbox.addLayout(row1)
        #probe_vbox.addWidget(self.probe_qlabel)
        probe_vbox.addLayout(row2)
        
        splitbox = QtWidgets.QHBoxLayout()
        splitbox.setSpacing(10)
        splitbox.setContentsMargins(0,0,0,0)
        ggbox = QtWidgets.QGroupBox()
        ggv = QtWidgets.QVBoxLayout(ggbox)
        gg_lbl = QtWidgets.QLabel('<b>View<br><u>Settings</u></b>')
        gg_lbl.setAlignment(QtCore.Qt.AlignCenter)
        ggv.addWidget(gg_lbl)
        
        
        self.big_btn = gi.ShowHideBtn(text_shown='', init_show=True)
        self.big_btn.setFixedWidth(60)
        
        
        ###   ACTION BUTTONS
        
        # continue button
        bbox = QtWidgets.QHBoxLayout()
        self.continue_btn = QtWidgets.QPushButton('Next')
        self.continue_btn.setEnabled(False)
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        bbox.addWidget(self.cancel_btn)
        bbox.addWidget(self.continue_btn)
        
        # assemble layout
        self.layout.addWidget(self.ddir_gbox)
        self.layout.addWidget(self.ddir_gbox2)
        self.layout.addWidget(self.probe_gbox)
        line0 = pyfx.DividerLine()
        self.layout.addWidget(line0)
        self.layout.addLayout(bbox)
        
        # connect buttons
        self.ddw.ddir_btn.clicked.connect(self.select_ddir)
        self.manual_upload_btn.clicked.connect(self.load_array)
        self.ddw2.ddir_btn.clicked.connect(self.make_ddir)
        self.prbf_load.clicked.connect(self.load_probe_from_file)
        self.prbf_view.clicked.connect(self.view_probe)
        self.prbf_make.clicked.connect(self.probe_config_popup)
        self.continue_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
    
    
    def load_array(self):
        print('Load raw data array!')
        ffilter = 'Data files (*.npy *.mat *.csv)'
        fpath,_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select data file', self.raw_ddir, ffilter)
        if not fpath: return
        
        ext = os.path.splitext(fpath)[1]
        if ext == '.npy':
            data = np.load(fpath)
        
        dlg = gi.RawArrayLoader(data, parent=self)
        dlg.exec()
    
    def load_probe_from_file(self, arg=None, fpath=None):
        """ Load probe data from filepath (if None, user can select a probe file """
        if fpath is None:
            ffilter = 'Probe files (*.json *.mat *.prb *.csv)'
            fpath,_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select probe file', self.probe_ddir, ffilter)
            if not fpath: return
        # try to load data - could be probe, could be None
        self.probe = ephys.read_probe_file(fpath)
        self.update_probe_obj()
        
    def view_probe(self):
        print('view_probe called')
        PG = ephys.make_probe_group(self.probe, self.probe_n.value())
        fig, axs = make_probe_plot(PG)
        fig_popup = gi.MatplotlibPopup(fig, parent=self)
        qrect = pyfx.ScreenRect(perc_height=1.0, perc_width=min(0.2*len(PG.probes), 1.0), 
                                keep_aspect=False)
        fig_popup.setGeometry(qrect)
        #fig_popup.setWindowTitle(self.probe_name)
        fig_popup.show()
        fig_popup.raise_()
        
    def probe_config_popup(self):
        print('probe_config_popup called')
        popup = ProbeFileSimple(parent=self)
        popup.show()
        popup.raise_()
        res = popup.exec()
        if not res:
            return
        # popup saved new probe configuration to file; load it!
        self.probe_ddir = popup.probe_filepath
        self.probe = ephys.read_probe_file(self.probe_ddir)
        self.update_probe_obj()
        
    def update_probe_obj(self):
        x = bool(self.probe is not None)
        self.prb_icon_btn.new_status(x)
        self.update_nchannels()
        if x:
            self.probe_qlabel.setText(self.probe.name)
            self.probe.create_auto_shape('tip', margin=30)
            #self.probe_df = self.probe.to_dataframe()
            #self.probe_df['chanMap'] = np.array(self.probe.device_channel_indices)
        else:
            self.probe_qlabel.setText('---')
        self.prbf_view.setEnabled(bool(x))
        self.PROBE_CONFIG_VALID = bool(x)
        self.continue_btn.setEnabled(bool(self.RAW_DDIR_VALID and self.PROCESSED_DDIR_VALID and self.PROBE_CONFIG_VALID))
    
    def update_nchannels(self):
        if self.probe is None:
            nch = '-'
        else:
            nch = self.probe.get_contact_count() * self.probe_n.value()
        self.total_channel_lbl.setText(self.total_channel_fmt.format(nch))
        
    def select_ddir(self):
        # open file popup to select raw data folder
        dlg = gi.FileDialog(init_ddir=self.raw_ddir, parent=self)
        res = dlg.exec()
        if res:
            self.raw_ddir = str(dlg.directory().path())
            self.update_raw_ddir()
            
            
    def update_raw_ddir(self):
        # check if raw data files are present
        files = os.listdir(self.raw_ddir)
        xdat_files = [f for f in files if f.endswith('.xdat.json')]
        a = bool('structure.oebin' in files)
        b = bool(len(xdat_files) > 0)
        x = bool(a or b)
        
        # update widgets
        try:
            self.ddw.update_status(self.raw_ddir, x)
        except:
            pdb.set_trace()
        self.oe_radio.setChecked(a)  # recording system buttons
        self.nn_radio.setChecked(b)
        
        self.RAW_DDIR_VALID = bool(x)
        self.ddir_gbox2.setVisible(x)
        self.probe_gbox.setVisible(bool(x and self.PROCESSED_DDIR_VALID))
        self.adjustSize()
        self.continue_btn.setEnabled(bool(self.RAW_DDIR_VALID and self.PROCESSED_DDIR_VALID and self.PROBE_CONFIG_VALID))
    
    
    def make_ddir(self):
        # open file popup to create processed data folder
        dlg = gi.FileDialog(init_ddir=self.processed_ddir, load_or_save='save', parent=self)
        res = dlg.exec()
        if res:
            self.processed_ddir = str(dlg.directory().path())
            self.update_processed_ddir()
    
    
    def update_processed_ddir(self):
        nexisting = len(os.listdir(self.processed_ddir))
        if nexisting > 0:
            txt = (f'The directory <code>{self.processed_ddir.split(os.sep)[-1]}</code> contains '
                   f'{nexisting} items.<br><br>I have taken away your overwrite '
                   'privileges for the time being.<br><br>Stop almost deleting important things!!')
            msg = '<center>{}</center>'.format(txt)
            res = QtWidgets.QMessageBox.warning(self, 'fuck you', msg, 
                                                QtWidgets.QMessageBox.NoButton, QtWidgets.QMessageBox.Close)
            if res == QtWidgets.QMessageBox.Yes:
                return True
            return False
            
        # update widgets
        self.ddw2.update_status(self.processed_ddir, True)
        
        self.PROCESSED_DDIR_VALID = True
        self.probe_gbox.setVisible(bool(self.RAW_DDIR_VALID))
        self.adjustSize()
        self.continue_btn.setEnabled(bool(self.RAW_DDIR_VALID and self.PROCESSED_DDIR_VALID and self.PROBE_CONFIG_VALID))
    
        
    def accept(self):
        print(f'Raw data folder: {self.raw_ddir}')
        print(f'Save folder: {self.processed_ddir}')
        super().accept()


class ProcessedDirectorySelectionPopup(QtWidgets.QDialog):
    def __init__(self, init_ddir='', go_to_last=False, parent=None):
        super().__init__(parent)
        #self.info = None
        self.current_probe = -1
        
        if go_to_last == True:
            qfd = QtWidgets.QFileDialog()
            self.ddir = qfd.directory().path()
        else:
            if init_ddir == '':
                _, self.ddir, _, _ = ephys.base_dirs()
            else:
                self.ddir = init_ddir
        
        self.gen_layout()
        
        if os.path.isdir(self.ddir):
            self.update_ddir(self.ddir)
            
    
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(20)
        
        ###   SELECT PROCESSED DATA FOLDER
        
        self.ddir_gbox = QtWidgets.QGroupBox()
        #ddir_gbox.setStyleSheet(gbox_ss)
        ddir_vbox = pyfx.InterWidgets(self.ddir_gbox, 'v')[2]
        # basic directory selection widget
        self.ddw = DirectorySelectionWidget(title='<b><u>Processed Data Folder</u></b>')
        self.qedit_hbox = self.ddw.qedit_hbox
        self.probe_dropdown = QtWidgets.QComboBox()
        self.probe_dropdown.hide()
        self.info_view_btn = QtWidgets.QPushButton('More Info')
        self.info_view_btn.hide()
        self.ddw.grid.addWidget(self.probe_dropdown, 2, 0, 1, 3)
        self.ddw.grid.addWidget(self.info_view_btn, 2, 3, 1, 3)
        ddir_vbox.addWidget(self.ddw)
        
        ###   ACTION BUTTONS
        
        self.ab = gi.AnalysisBtns()
        intraline = pyfx.DividerLine()
        self.ddw.grid.addWidget(intraline, 3, 0, 1, 6)
        self.ddw.grid.addWidget(self.ab.option1_widget, 4, 1)
        self.ddw.grid.addWidget(self.ab.option2_widget, 5, 1)
        self.ddw.grid.setRowMinimumHeight(2, 20)

        # assemble layout
        self.layout.addWidget(self.ddir_gbox)
        
        # connect buttons
        self.ddw.ddir_btn.clicked.connect(self.select_ddir)
        self.probe_dropdown.currentTextChanged.connect(self.update_probe)
        self.info_view_btn.clicked.connect(self.show_info_popup)
    
    def show_info_popup(self):
        print('show_info_popup called --> under construction')
        #info_popup = InfoView(info=self.info, parent=self)
        #info_popup.show()
        #info_popup.raise_()
    
    # def info_popup(self):
    #     #dlg = QtWidgets.QDialog(self)
    #     #layout = QtWidgets.QVBoxLayout(dlg)
    #     self.
    # def emit_signal(self):
    #     if self.ab.option1_btn.isChecked():
    #         self.launch_ch_selection_signal.emit()
            
    #     elif self.ab.option2_btn.isChecked():
    #         self.launch_ds_class_signal.emit()
        
    def select_ddir(self):
        # open file popup to select processed data folder
        dlg = gi.FileDialog(init_ddir=self.ddir, parent=self)
        res = dlg.exec()
        if res:
            self.update_ddir(str(dlg.directory().path()))
    
    def update_ddir(self, ddir):
        print('update_ddir called')
        self.ddir = ddir
        x = validate_processed_ddir(self.ddir)  # valid data folder?
        self.ddw.update_status(self.ddir, x)    # update folder path/icon style
        # reset probe dropdown, populate with probes in $info
        self.probe_dropdown.blockSignals(True)
        for i in reversed(range(self.probe_dropdown.count())):
            self.probe_dropdown.removeItem(i)
        self.probe_dropdown.setVisible(x)
        self.info_view_btn.setVisible(x)
        if x:
            # read in recording info, clear and update probe dropdown menu
            #self.info = ephys.load_recording_info(self.ddir)
            probe_group = prif.io.read_probeinterface(Path(self.ddir, 'probe_group'))
            items = [f'probe {i}' for i in range(len(probe_group.probes))]
            self.probe_dropdown.addItems(items)
        #else:
        #    self.info = None
        # probe index (e.g. 0,1,...) if directory is valid, otherwise -1
        self.probe_dropdown.blockSignals(False)
        #self.current_probe = self.probe_dropdown.currentIndex()
        self.update_probe()
    
    
    def update_probe(self):
        print('update_probe called')
        self.current_probe = self.probe_dropdown.currentIndex()
        self.ab.ddir_toggled(self.ddir, self.current_probe)  # update action widgets


class thing(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Set base folders')
        qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
        self.setGeometry(qrect)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        
        self.BASE_FOLDERS = [str(x) for x in ephys.base_dirs()]
        qlabel_ss = ('QLabel {background-color:white;'
                             'border:1px solid gray; border-radius:4px; padding:5px;}')
        fmt = '<code>{}</code>'
        
        self.btn_list = []
        for i in range(4):
            btn = QtWidgets.QPushButton()
            btn.setIcon(QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
            btn.setMinimumSize(30,30)
            btn.setIconSize(QtCore.QSize(20,20))
            self.btn_list.append(btn)
        
        ###   RAW BASE FOLDER
        self.raw_w = QtWidgets.QWidget()
        raw_vlay, raw_row0, raw_row1 = self.create_hbox_rows()
        self.raw_w.setLayout(raw_vlay)
        raw_header = QtWidgets.QLabel(fmt.format('RAW DATA'))
        raw_row0.addWidget(raw_header)
        raw_row0.addStretch()
        self.raw_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[0]))
        self.raw_qlabel.setStyleSheet(qlabel_ss)
        self.raw_btn = self.btn_list[0]
        raw_row1.addWidget(self.raw_qlabel, stretch=2)
        raw_row1.addWidget(self.raw_btn, stretch=0)
        layout.addWidget(self.raw_w)
        
        ###   PROCESSED BASE FOLDER
        self.processed_w = QtWidgets.QWidget()
        processed_vlay, processed_row0, processed_row1 = self.create_hbox_rows()
        self.processed_w.setLayout(processed_vlay)
        processed_header = QtWidgets.QLabel(fmt.format('PROCESSED DATA'))
        processed_row0.addWidget(processed_header)
        processed_row0.addStretch()
        self.processed_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[1]))
        self.processed_qlabel.setStyleSheet(qlabel_ss)
        self.processed_btn = self.btn_list[1]
        processed_row1.addWidget(self.processed_qlabel, stretch=2)
        processed_row1.addWidget(self.processed_btn, stretch=0)
        layout.addWidget(self.processed_w)
        
        ###   PROBE BASE FOLDER
        self.probe_w = QtWidgets.QWidget()
        probe_vlay, probe_row0, probe_row1 = self.create_hbox_rows()
        self.probe_w.setLayout(probe_vlay)
        probe_header = QtWidgets.QLabel(fmt.format('PROBE FILES'))
        probe_row0.addWidget(probe_header)
        probe_row0.addStretch()
        self.probe_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[2]))
        self.probe_qlabel.setStyleSheet(qlabel_ss)
        self.probe_btn = self.btn_list[2]
        probe_row1.addWidget(self.probe_qlabel, stretch=2)
        probe_row1.addWidget(self.probe_btn, stretch=0)
        layout.addWidget(self.probe_w)
        
        ###   DEFAULT PROBE FILE
        self.probefile_w = QtWidgets.QWidget()
        probefile_vlay, probefile_row0, probefile_row1 = self.create_hbox_rows()
        self.probefile_w.setLayout(probefile_vlay)
        probefile_header = QtWidgets.QLabel(fmt.format('DEFAULT PROBE'))
        probefile_row0.addWidget(probefile_header)
        probefile_row0.addStretch()
        self.probefile_qlabel = QtWidgets.QLabel(fmt.format(self.BASE_FOLDERS[3]))
        self.probefile_qlabel.setStyleSheet(qlabel_ss)
        self.probefile_btn = self.btn_list[3] #findme
        probefile_row1.addWidget(self.probefile_qlabel, stretch=2)
        probefile_row1.addWidget(self.probefile_btn, stretch=0)
        layout.addWidget(self.probefile_w)
        
        self.raw_btn.clicked.connect(lambda: self.choose_base_ddir(0))
        self.processed_btn.clicked.connect(lambda: self.choose_base_ddir(1))
        self.probe_btn.clicked.connect(lambda: self.choose_base_ddir(2))
        self.probefile_btn.clicked.connect(self.choose_probe_file)
        
        bbox = QtWidgets.QHBoxLayout()
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setStyleSheet('QPushButton {padding : 10px;}')
        bbox.addWidget(self.save_btn)
        layout.addLayout(bbox)
        
        self.save_btn.clicked.connect(self.save_base_folders)
        # rows = [(*x, *create_filepath_row(*x)) for x in ephys.base_dirs(keys=True)]
        
        # for i,(k,p,w,_,_,btn) in enumerate(rows):
        #     layout.addWidget(w)
            
        #     btn.clicked.connect(lambda: self.choose_base_ddir(i,p))
        #     self.BASE_FOLDERS.append(str(p))
            
        # get base folder widgets, put code tags around font
        
        
        # fx2: x -> p,fx(*x) -> (p,w,_,_,z)
        # fx: (k,v) -> (<k>,<v>) -> func() -> (w,x,y,z)
        #fx2 = lambda x: (p, fx(k,p))
    
    def choose_probe_file(self):
        init_ddir = str(self.BASE_FOLDERS[2])
        ffilter = 'Probe files (*.json *.prb *.mat)'
        dlg = gi.FileDialog(init_ddir=init_ddir, is_directory=False, parent=self)
        dlg.setNameFilter(ffilter)
        dlg.setWindowTitle('Default probe file')
        res = dlg.exec()
        if res:
            f = dlg.selectedFiles()[0]
            prb = ephys.read_probe_file(f)
            if prb is None:
                msgbox = gi.MsgboxError('The following is not a valid probe file:', 
                                        sub_msg=f'<nobr><code>{f}</code></nobr>')
                msgbox.sub_label.setWordWrap(False)
                msgbox.exec()
                return
            self.BASE_FOLDERS[3] = str(f)
            self.update_base_ddir(3)
        
        
    def choose_base_ddir(self, i):
        init_ddir = str(self.BASE_FOLDERS[i])
        fmt = 'Base folder for %s'
        titles = [fmt % x for x in ['raw data', 'processed data', 'probe files']]
        # when activated, initialize at ddir and save new base folder at index i
        dlg = gi.FileDialog(init_ddir=init_ddir, parent=self)
        dlg.setWindowTitle(titles[i])
        res = dlg.exec()
        if res:
            self.BASE_FOLDERS[i] = str(dlg.directory().path())
            self.update_base_ddir(i)
            
    
    def update_base_ddir(self, i):
        fmt = '<code>{}</code>'
        if i==0:
            self.raw_qlabel.setText(fmt.format(self.BASE_FOLDERS[0]))
        elif i==1:
            self.processed_qlabel.setText(fmt.format(self.BASE_FOLDERS[1]))
        elif i==2:
            self.probe_qlabel.setText(fmt.format(self.BASE_FOLDERS[2]))
        elif i==3:
            self.probefile_qlabel.setText(fmt.format(self.BASE_FOLDERS[3]))
    
    
    def save_base_folders(self):
        ddir_list = list(self.BASE_FOLDERS)
        ephys.write_base_dirs(ddir_list)
        self.accept()
        # msgbox = MsgboxSave(parent=self)
        # res = msgbox.exec()
        # if res == QtWidgets.QMessageBox.Yes:
        #     self.accept()
    
    def create_hbox_rows(self):
        vlay = QtWidgets.QVBoxLayout()
        vlay.setSpacing(2)
        row0 = QtWidgets.QHBoxLayout()
        row0.setContentsMargins(0,0,0,0)
        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(0,0,0,0)
        row1.setSpacing(5)
        vlay.addLayout(row0)
        vlay.addLayout(row1)
        return vlay, row0, row1
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(True)
    
    #ddir = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/NN_JG008')
    #popup = InfoView(ddir=ddir)
    
    nn_raw = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/raw_data/'
                'JG008_071124n1_neuronexus')
    #popup = RawDirectorySelectionPopup()
    #popup = ProbeFileSimple()
    popup = thing()
    #popup = AuxDialog(n=6)
    
    popup.show()
    popup.raise_()
    
    sys.exit(app.exec())