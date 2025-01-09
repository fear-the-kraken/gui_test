#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:20:23 2024

@author: amandaschott
"""
import os
from pathlib import Path
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

def make_probe_plot(probes):
    if probes.__class__ == prif.ProbeGroup:
        PG = probes
    elif probes.__class__ == prif.Probe:
        PG = prif.ProbeGroup()
        PG.add_probe(probes)
    else:
        raise Exception('Error: Probe plot input must be a Probe or ProbeGroup')
    
    fig, axs = plt.subplots(nrows=1, ncols=len(PG.probes), layout='tight')
    if type(axs) not in (list, np.ndarray):
        axs = [axs]
    for i,ax in enumerate(axs):
        P = PG.probes[i]
        plot_probe(P, with_contact_id=False, 
                   with_device_index=True, title=False, 
                   probe_shape_kwargs=dict(ec='black',lw=3), ax=ax)
        rads = list(map(lambda x: x['radius'], P.contact_shape_params))
        xshift = P.contact_positions[:,0] + np.array(rads) + 30
        kw = dict(ha='left', clip_on=False, 
                  bbox=dict(ec='none', fc='white', alpha=0.3))
        _ = [txt.set(x=xs, **kw) for txt,xs in zip(ax.texts,xshift)]
        ax.spines[['right','top','bottom']].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.label.set_fontsize('large')
    return fig, axs


class ProbeFileSimple(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gen_layout()
        
        xc = np.array([63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 
                       63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 
                       63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 63.5, 
                       63.5, 63.5, 63.5, 63.5, 63.5])
        yc = np.array([1290, 1250, 1210, 1170, 1130, 1090, 1050, 1010,  970,  
                        930,  890,  850,  810,  770,  730,  690,  650,  610,  
                        570,  530,  490,  450,  410,  370,  330,  290,  250,  
                        210,  170,  130,   90,   50])
        shank = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        chMap = np.array([10, 11,  2,  3, 24, 25, 16, 17,  8,  9,  0,  1, 19, 
                          18, 26, 27,  5, 4, 12, 20, 13, 21, 29,  7,  6, 15, 
                          23, 31, 14, 22, 30, 28])
        
        self.name_input.setText('A1x32-Edge-10mm-40-177')
        self.xcoor_input.setPlainText(str(list(xc)))
        self.ycoor_input.setPlainText(str(list(yc)))
        self.shk_input.setPlainText(str(list(shank)))
        self.chMap_input.setPlainText(str(list(chMap)))
        
    def gen_layout(self):
        """ Layout for popup window """
        self.setWindowTitle('Create probe file')
        self.setMinimumWidth(250)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(20)
        
        # probe name
        row0 = QtWidgets.QHBoxLayout()
        row0.setSpacing(10)
        self.name_lbl = QtWidgets.QLabel('<big><u>Probe name</u></big>')
        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setObjectName('probe_text')
        self.name_input.setText('Probe_0')
        row0.addWidget(self.name_lbl)
        row0.addWidget(self.name_input)
        
        # probe geometry
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(15)
        row1_vb0 = QtWidgets.QVBoxLayout()
        row1_vb0.setSpacing(1)
        # electrode configuration
        self.config_lbl = QtWidgets.QLabel('Configuration')
        self.config_cbox = QtWidgets.QComboBox()
        self.config_cbox.addItems(['Linear','Polytrode','Tetrode'])
        row1_vb0.addWidget(self.config_lbl)
        row1_vb0.addWidget(self.config_cbox)
        row1.addLayout(row1_vb0)
        row1_vb1 = QtWidgets.QVBoxLayout()
        row1_vb1.setSpacing(1)
        # number of electrode columns
        self.ncol_lbl = QtWidgets.QLabel('# electrode columns')
        self.ncol_sbox = QtWidgets.QSpinBox()
        self.ncol_sbox.setMinimum(1)
        self.ncol_sbox.setEnabled(False)
        row1_vb1.addWidget(self.ncol_lbl)
        row1_vb1.addWidget(self.ncol_sbox)
        row1.addLayout(row1_vb1)
        
        # x and y-coordinates
        row2 = QtWidgets.QVBoxLayout()
        row2.setSpacing(2)
        self.x_lbl = QtWidgets.QLabel('<b>Enter X-coordinates for each electrode</b>')
        self.xcoor_input = QtWidgets.QTextEdit()
        self.xcoor_input.setObjectName('probe_text')
        row2.addWidget(self.x_lbl)
        row2.addWidget(self.xcoor_input)
        row3 = QtWidgets.QVBoxLayout()
        row3.setSpacing(2)
        self.y_lbl = QtWidgets.QLabel('<b>Enter Y-coordinates for each electrode</b>')
        self.ycoor_input = QtWidgets.QTextEdit()
        self.ycoor_input.setObjectName('probe_text')
        row3.addWidget(self.y_lbl)
        row3.addWidget(self.ycoor_input)
        
        # shank IDs
        row4 = QtWidgets.QVBoxLayout()
        row4.setSpacing(2)
        self.shk_lbl = QtWidgets.QLabel('<b>Electrode shank IDs (multi-shank probes only)</b>')
        self.shk_input = QtWidgets.QTextEdit()
        self.shk_input.setObjectName('probe_text')
        row4.addWidget(self.shk_lbl)
        row4.addWidget(self.shk_input)
        
        # headstage channel map
        row5 = QtWidgets.QVBoxLayout()
        row5.setSpacing(2)
        self.chMap_lbl = QtWidgets.QLabel('<b>Channel map (probe to headstage)</b>')
        self.chMap_input = QtWidgets.QTextEdit()
        self.chMap_input.setObjectName('probe_text')
        row5.addWidget(self.chMap_lbl)
        row5.addWidget(self.chMap_input)
        
        # action buttons
        bbox = QtWidgets.QHBoxLayout()
        self.makeprobe_btn = QtWidgets.QPushButton('Configure probe')
        self.makeprobe_btn.setEnabled(False)
        self.save_prbf_btn = QtWidgets.QPushButton('Save probe file')
        self.save_prbf_btn.setEnabled(False)
        bbox.addWidget(self.makeprobe_btn)
        bbox.addWidget(self.save_prbf_btn)
        
        self.layout.addLayout(row0)
        self.line0 = pyfx.DividerLine(lw=3, mlw=3)
        self.layout.addWidget(self.line0)
        #self.layout.addLayout(row1)
        self.layout.addLayout(row2)
        self.layout.addLayout(row3)
        self.layout.addLayout(row4)
        self.layout.addLayout(row5)
        self.layout.addLayout(bbox)
        
        self.name_input.textChanged.connect(self.check_probe)
        self.xcoor_input.textChanged.connect(self.check_probe)
        self.ycoor_input.textChanged.connect(self.check_probe)
        self.shk_input.textChanged.connect(self.check_probe)
        self.chMap_input.textChanged.connect(self.check_probe)
        self.makeprobe_btn.clicked.connect(self.construct_probe)
        self.save_prbf_btn.clicked.connect(self.save_probe_file)
    
        self.setStyleSheet('QTextEdit { border : 2px solid gray; }')
        
        
    def check_probe(self):
        print('check_probe called')
        self.makeprobe_btn.setEnabled(False)
        self.save_prbf_btn.setEnabled(False)
        probe_name = self.name_input.text()
        xdata      = ''.join(self.xcoor_input.toPlainText().split())
        ydata      = ''.join(self.ycoor_input.toPlainText().split())
        shk_data   = ''.join(self.shk_input.toPlainText().split())
        cmap_data  = ''.join(self.chMap_input.toPlainText().split())
        
        try: 
            xc = np.array(eval(xdata), dtype='float')   # x-coordinates
            yc = np.array(eval(ydata), dtype='float')   # y-coordinates
            if shk_data  == '' : shk = np.ones_like(xc, dtype='int')    # shank IDs
            else               : shk = np.array(eval(shk_data), dtype='int')
            if cmap_data == '' : cmap = np.arange(xc.size, dtype='int') # channel map
            else               : cmap = np.array(eval(cmap_data), dtype='int')
        except:
            #pdb.set_trace()
            return  # failed to convert text to array
        print('got arrays')
        n = [xc.size, yc.size, shk.size, cmap.size]
        if len(np.unique(n)) > 1: return              # mismatched array lengths
        if cmap.size != np.unique(cmap).size: return  # duplicates in channel map 
        if len(probe_name.split()) > 1: return        # spaces in probe name
        if probe_name == '': return                   # probe name left blank
        
        self.probe_name = probe_name
        self.xc   = xc
        self.yc   = yc
        self.shk  = shk
        self.cmap = cmap
        self.makeprobe_btn.setEnabled(True)
        self.save_prbf_btn.setEnabled(True)
        
    
    def construct_probe(self, arg=None, pplot=True):
        print('construct_probe called')
        # create dataframe, sort by shank > x-coor > y-coor
        pdf = pd.DataFrame(dict(chanMap=self.cmap, xc=self.xc, yc=self.yc, shank=self.shk))
        df = pdf.sort_values(['shank','xc','yc'], ascending=[True,True,False]).reset_index(drop=True)
        #ncoors = np.array(df[['xc','yc']].nunique())  # (ncols, nrows)
        #ndim = max((ncoors > 1).astype('int').sum(), 1)
        
        # initialize probe data object
        self.probe = prif.Probe(ndim=2, name=self.probe_name)
        self.probe.set_contacts(np.array(df[['xc','yc']]), shank_ids=df.shank, 
                                contact_ids=df.index.values)
        self.probe.set_device_channel_indices(df.chanMap)
        self.probe.create_auto_shape('tip', margin=30)
        #self.probe_df = self.probe.to_dataframe()
        #self.probe_df['chanMap'] = np.array(self.probe.device_channel_indices)
        if pplot:
            fig, axs = make_probe_plot(self.probe)
            fig_popup = gi.MatplotlibPopup(fig, parent=self)
            qrect = pyfx.ScreenRect(perc_height=0.9, perc_width=0.2, keep_aspect=False)
            fig_popup.setGeometry(qrect)
            fig_popup.setWindowTitle(self.probe_name)
            fig_popup.show()
            fig_popup.raise_()
            
            #self.show_probe_plot()
            
    
    # def show_probe_plot(self):
    #     fig, ax = plt.subplots(layout='tight')
    #     plot_probe(self.probe, with_contact_id=False, with_device_index=True, 
    #                title=False, probe_shape_kwargs=dict(ec='black',lw=3), ax=ax)
    #     rads = list(map(lambda x: x['radius'], self.probe.contact_shape_params))
    #     xshift = self.probe.contact_positions[:,0] + np.array(rads) + 30
    #     kw = dict(ha='left', clip_on=False, 
    #               bbox=dict(ec='none', fc='white', alpha=0.3))
    #     _ = [txt.set(x=xs, **kw) for txt,xs in zip(ax.texts,xshift)]
    #     ax.spines[['right','top','bottom']].set_visible(False)
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.label.set_fontsize('large')
    #     #ax.set_xlim(right=ax.get_xlim()[1] * 1.5)
        
    #     fig_popup = MatplotlibPopup(fig, parent=self)
    #     qrect = pyfx.ScreenRect(perc_height=0.9, perc_width=0.2, keep_aspect=False)
    #     fig_popup.setGeometry(qrect)
    #     fig_popup.setWindowTitle(self.probe_name)
    #     fig_popup.show()
    #     fig_popup.raise_()
    #     print('show_probe_plot called')
    
    def save_probe_file(self, arg=None, extension='.json'):
        # create probe object, select file name/location
        self.construct_probe(pplot=False)
        filename = f'{self.probe_name}_config{extension}'
        fpath,_ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save probe file',
                                                        str(Path(os.getcwd(), filename)))
        if not fpath: return
        
        res = ephys.write_probe_file(self.probe, fpath)
        self.probe_filepath = fpath
        if res:
            # pop-up messagebox appears when save is complete
            msg = 'Probe configuration saved!\nExit window?'
            msgbox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, '', msg, 
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, self)
            # set check icon
            chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
            px_size = msgbox.findChild(QtWidgets.QLabel, 'qt_msgboxex_icon_label').pixmap().size()
            msgbox.setIconPixmap(chk_icon.pixmap(px_size))
            res = msgbox.exec()
            if res == QtWidgets.QMessageBox.Yes:
                self.accept()

