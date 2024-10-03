#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:45:34 2024

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

btn_ss = ('QPushButton {'
          'background-color : rgba%s;'  # light
          'border : 4px outset rgb(128,128,128);'
          'border-radius : 11px;'
          'min-width : 15px;'
          'max-width : 15px;'
          'min-height : 15px;'
          'max-height : 15px;'
          '}'
          
          'QPushButton:disabled {'
          'background-color : rgb(220,220,220);'#'rgba%s;'
          #'border : 4px outset rgb(128,128,128);'
          '}'
          
          'QPushButton:pressed {'
          'background-color : rgba%s;'  # dark
          '}'
          
          # 'QPushButton:checked {'
          # 'background-color : rgba%s;'  # dark
          # #'outline : 2px solid red;'
          # 'border : 3px solid red;'
          # 'border-radius : 2px;'
          # '}'
          )

def iter2str(v):
    if not hasattr(v, '__iter__'): 
        return str(v)
    return ', '.join(np.array(v,dtype='str').tolist())

def val2str(v):
    if np.ndim(v) < 2:
        txt = iter2str(v)
    elif np.ndim(v) == 2:
        txt = ', '.join([f'({x})' for x in map(iter2str, v)])
    return txt

def info2text2(info):
    keys, vals = zip(*info.items())
    vstr = [*map(str,vals)]
    klens = [*map(len, keys)]
    kmax=max(klens)
    padk = [*map(lambda k: k + '_'*(kmax-len(k)), keys)]
    rows = ['<tr><td align="center"><pre>'+pdk+' : '+'</pre></td><td><pre>'+v+'</pre></td></tr>' for pdk,v in zip(padk,vstr)]
    text = '<body><h2><pre>Recording Info</pre></h2><hr><table>' + ''.join(rows) + '</table></body>'
    return text
    #text = ''.join(rows)
    
def info2text(info, rich=True):
    sep = '<br>' if rich else os.linesep
    fmt = ('<tr>'
               '<td align="center"; style="background-color:#f2f2f2; border:2px solid #e6e6e6; white-space:nowrap;"><tt>%s</tt></td>'
               '<td align="center"><font size="4"></font></td>'
               '<td style="background-color:#f2f2f2; border:2px solid #e6e6e6;"><tt>%s</tt></td>'
           '</tr>')
    
    div_row = '<tr><td colspan="3"><hr></td></tr>'
    
    gen_rows = [fmt % (k,info[k]) for k in ['raw_data_path','recording_system','units']]
    gen_rows[1] = gen_rows[1].replace('recording_system','system')
    # probe rows
    keys = ['ports', 'nprobes', 'probe_nch']
    vals = [val2str(info[k]) for k in keys]
    probe_rows = [fmt % (a,b) for a,b in zip(keys,vals)]
    # recording rows
    rec_rows = [fmt % ('fs',        '%s Hz'        % info['fs']),  #    f"{info['nchannels']} Hz"  f'%s Hz'
                fmt % ('nchannels', '%s primary el.' % info['nchannels']),
                fmt % ('nsamples',  '%.1E bins' % info['nsamples']),
                fmt % ('tstart',    '%.2f s' % info['tstart']),
                fmt % ('tend',      '%.2f s' % info['tend']),
                fmt % ('duration',  '%.2f s' % info['dur'])]
        
    # info popup title '<hr width="70%">'
    #'<p style="line-height:30%; vertical-align:middle;">'
    #'<hr align="center"; width="70%"></p>'
    #<div> <p align="center"; style="background-color:green;">' + ; width="60%"
    header = (#'<hr height="5px"; style="background-color:yellow; color:blue; border:5px solid orange;">'
              #'<hr width="60%">'
              # '<p style="background-color:red; border:5px solid black;">'
              # #'nekkid'
              # #'***'
              # '<td style="border:2px solid green;"><hr></td>'
              # '</p>'
              '<h2 align="center"; style="background-color:#f2f2f2; border:2px solid red; padding:100px;"><tt>Recording Info</tt></h2>'
              #'<p style="background-color:none;">'
              #'<hr style="border:5px solid black">')
              #'---')
              )
    
    info_text = ('<body style="background-color:#e6e6e6;">' + header + 
                 '<br style="line-height:30%">' + 
                 '<table border-collapse="collapse"; cellspacing="0"; cellpadding="3">' +
                ''.join(gen_rows)   + str(div_row)     + #'</table>')#str(div_row) +
                ''.join(probe_rows) + str(div_row)     + #'</table>')
                ''.join(rec_rows)   + '</table>' + '</body')
    
    return info_text


def unique_fname(ddir, base_name):
    existing_files = os.listdir(ddir)
    fname = str(base_name)
    i = 1
    while fname in existing_files:
        new_name = fname + f' ({i})'
        if new_name not in existing_files:
            fname = str(new_name)
            break
        i += 1
    return fname


class CSlider(matplotlib.widgets.Slider):
    """ Slider with enable/disable function """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handle._markersize = 20
        self._handle._markeredgewidth = 2
        self.nsteps = 500
    
    def key_step(self, event):
        if event.key == 'right':
            self.set_val(self.val + self.nsteps)
        elif event.key == 'left':
            self.set_val(self.val - self.nsteps)
        
        
    def enable(self, x):
        if x:
            self.track.set_facecolor('lightgray')
            self.track.set_alpha(1)
            self.poly.set_facecolor('indigo')
            self.poly.set_alpha(1)
            self._handle.set_markeredgecolor('darkgray')
            self._handle.set_markerfacecolor('white')
            self._handle.set_alpha(1)
            self.valtext.set_alpha(1)
            self.label.set_alpha(1)
            
        else:
            self.track.set_facecolor('lightgray')
            self.track.set_alpha(0.3)
            self.poly.set_facecolor('lightgray')
            self.poly.set_alpha(0.5)
            self._handle.set_markeredgecolor('darkgray')
            self._handle.set_markerfacecolor('gainsboro')
            self._handle.set_alpha(0.5)
            self.valtext.set_alpha(0.2)
            self.label.set_alpha(0.2)


class EventArrows(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('QPushButton {font-weight:bold; padding:2px;}')
        self.left = QtWidgets.QPushButton('\u2190') # unicode ← and →
        self.right = QtWidgets.QPushButton('\u2192')
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.setSpacing(1)
        hbox.setContentsMargins(0,0,0,0)
        hbox.addWidget(self.left)
        hbox.addWidget(self.right)
        self.bgrp = QtWidgets.QButtonGroup(self)
        self.bgrp.addButton(self.left, 0)
        self.bgrp.addButton(self.right, 1)
    
    
class ShowHideBtn(QtWidgets.QPushButton):
    def __init__(self, text_shown='\u00BB', text_hidden='\u00AB', init_show=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.TEXTS = [text_hidden, text_shown]
        #self.SHOWN_TEXT = text_shown
        #self.HIDDEN_TEXT = text_hidden
        # set checked/visible or unchecked/hidden
        self.setChecked(init_show)
        self.setText(self.TEXTS[int(init_show)])
        
        
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                        QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(policy)
        self.toggled.connect(self.update_state)

        self.setStyleSheet('QPushButton {'
                            'background-color : gainsboro;'
                            'border : 3px outset gray;'
                            'border-radius : 2px;'
                            'color : rgb(50,50,50);'
                            'font-size : 30pt;'
                            'font-weight : normal;'
                            'max-width : 30px;'
                            'padding : 4px;'
                            '}'
                            
                            'QPushButton:pressed {'
                            'background-color : dimgray;'
                            'border : 3px inset gray;'
                            'color : whitesmoke;'
                            '}')
        
    def update_state(self, show):
        self.setText(self.TEXTS[int(show)])


class ReverseSpinBox(QtWidgets.QSpinBox):
    """ Spin box with reversed increments (down=+1) to match LFP channels """
    def stepEnabled(self):
        if self.wrapping() or self.isReadOnly():
            return super().stepEnabled()
        ret = QtWidgets.QAbstractSpinBox.StepNone
        if self.value() > self.minimum():
            ret |= QtWidgets.QAbstractSpinBox.StepUpEnabled
        if self.value() < self.maximum():
            ret |= QtWidgets.QAbstractSpinBox.StepDownEnabled
        return ret

    def stepBy(self, steps):
        return super().stepBy(-steps)


class Msgbox(QtWidgets.QMessageBox):
    def __init__(self, msg='', sub_msg='', title='', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        # get icon label
        self.icon_label = self.findChild(QtWidgets.QLabel, 'qt_msgboxex_icon_label')
        # set main text
        self.setText(msg)
        self.label = self.findChild(QtWidgets.QLabel, 'qt_msgbox_label')
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        # set sub text
        if sub_msg != '':
            self.setInformativeText(sub_msg)
            self.sub_label = self.findChild(QtWidgets.QLabel, 'qt_msgbox_informativelabel')
            
    
class MsgboxSave(Msgbox):
    def __init__(self, msg='Save successful!', sub_msg='Exit window?', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Information)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        # set check icon
        chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        px_size = self.icon_label.pixmap().size()
        self.setIconPixmap(chk_icon.pixmap(px_size))


class MsgboxError(Msgbox):
    def __init__(self, msg='Something went wrong!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        
        
class AuxDialog(QtWidgets.QDialog):
    def __init__(self, n, parent=None):
        super().__init__(parent)
        
        #flags = self.windowFlags() | QtCore.Qt.FramelessWindowHint
        #self.setWindowFlags(flags)
        self.setWindowTitle('AUX channels')
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(10)
        qlabel = QtWidgets.QLabel('Set AUX file names (leave blank to ignore)')
        # load list of previously saved aux files
        self.auxf = list(np.load('.aux_files.npy'))
        completer = QtWidgets.QCompleter(self.auxf, self)
        grid = QtWidgets.QGridLayout()
        self.qedits = []
        # create QLineEdit for each AUX channel
        for i in range(n):
            lbl = QtWidgets.QLabel(f'AUX {i}')
            qedit = QtWidgets.QLineEdit()
            qedit.setCompleter(completer)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(qedit, i, 1)
            self.qedits.append(qedit)
        # action button
        bbox = QtWidgets.QHBoxLayout()
        self.continue_btn = QtWidgets.QPushButton('Continue')
        self.continue_btn.clicked.connect(self.accept)
        self.clear_btn = QtWidgets.QPushButton('Clear all')
        self.clear_btn.clicked.connect(self.clear_files)
        bbox.addWidget(self.continue_btn)
        bbox.addWidget(self.clear_btn)
        # set up layout
        self.layout.addWidget(qlabel)
        line = pyfx.DividerLine()
        self.layout.addWidget(line)
        self.layout.addLayout(grid)
        self.layout.addLayout(bbox)
    
    def update_files(self):
        for i,qedit in enumerate(self.qedits):
            txt = qedit.text()
            if txt != '':
                if not txt.endswith('.npy'):
                    txt += '.npy'
            self.aux_files[i] = txt
    
    def clear_files(self):
        for qedit in self.qedits:
            qedit.setText('')
    
    def accept(self):
        self.aux_files = []
        for qedit in self.qedits:
            txt = qedit.text()
            if txt.endswith('.npy'):
                txt = txt[0:-4]
            if txt not in self.auxf:
                self.auxf.append(txt)
            fname = txt + ('' if txt == '' else '.npy')
            self.aux_files.append(fname)
        np.save('.aux_files.npy', self.auxf)
        super().accept()
        
            

class FileDialog(QtWidgets.QFileDialog):
    
    def __init__(self, init_ddir='', load_or_save='load', overwrite_mode=0, 
                 is_directory=True, parent=None):
        super().__init__(parent)
        self.load_or_save = load_or_save
        self.overwrite_mode = overwrite_mode  # 0=add to folder, 1=overwrite files
        options = self.Options()
        options |= self.DontUseNativeDialog
        
        self.setViewMode(self.List)
        self.setAcceptMode(self.AcceptOpen)  # open file
        if is_directory:
            self.setFileMode(self.Directory)     # allow directories only
        else:
            self.setFileMode(self.ExistingFile)  # allow existing files
        
        try: self.setDirectory(init_ddir)
        except: print('init_ddir argument in FileDialog is invalid')
        self.setOptions(options)
        self.connect_signals()
    
    
    def connect_signals(self):
        self.lineEdit = self.findChild(QtWidgets.QLineEdit)
        self.stackedWidget = self.findChild(QtWidgets.QStackedWidget)
        self.view = self.stackedWidget.findChild(QtWidgets.QListView)
        self.view.selectionModel().selectionChanged.connect(self.updateText)
    
    
    def updateText(self, selected, deselected):
        if selected.indexes() == []:
            return
        # update contents of the line edit widget with the selected files
        txt = self.view.selectionModel().currentIndex().data()
        self.lineEdit.setText(txt)
    
    
    def overwrite_msgbox(self, ddir):
        txt = (f'The directory <code>{ddir.split(os.sep)[-1]}</code> contains '
               f'<code>{len(os.listdir(ddir))}</code> items.<br><br>Overwrite existing files?')
        msg = '<center>{}</center>'.format(txt)
        res = QtWidgets.QMessageBox.warning(self, 'Overwrite Warning', msg, 
                                            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if res == QtWidgets.QMessageBox.Yes:
            return True
        return False
        
    
    def accept(self):
        ddir = self.directory().path()
        if self.load_or_save == 'save' and len(os.listdir(ddir)) > 0:
            # show overwrite warning for existing directory files
            res = self.overwrite_msgbox(ddir)
            if not res: return
        QtWidgets.QDialog.accept(self)
    
    
    
class QEdit_HBox(QtWidgets.QHBoxLayout):
    def __init__(self, simple=False, colors=['gray','darkgreen'], parent=None):
        super().__init__(parent)
        self.simple_mode = simple
        self.c0, self.c1 = colors
        
        self.setContentsMargins(0,0,0,0)
        self.setSpacing(0)
        
        # ellipsis (...)
        self.ellipsis = QtWidgets.QLineEdit()
        self.ellipsis.setAlignment(QtCore.Qt.AlignCenter)
        self.ellipsis.setTextMargins(0,4,0,4)
        self.ellipsis.setReadOnly(True)
        self.ellipsis.setText('...')
        self.ellipsis.setMaximumWidth(20)
        # base path to directory (resizable)
        self.path = QtWidgets.QLineEdit()
        self.path.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.path.setTextMargins(0,4,0,4)
        self.path.setReadOnly(True)
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                        QtWidgets.QSizePolicy.Fixed)
        self.path.setSizePolicy(policy)
        # directory name (gets size priority according to text length)
        self.folder = QtWidgets.QLineEdit()
        self.folder.setTextMargins(0,4,0,4)
        self.folder.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.folder.setReadOnly(True)
        policy2 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed,
                                        QtWidgets.QSizePolicy.Fixed)
        self.folder.setSizePolicy(policy2)
        # set attributes for all QLineEdit items
        self.qedits = [self.ellipsis, self.path, self.folder]
        ss = ('QLineEdit {'
              f'border : 2px groove {self.c0};'
              'border-left : %s;'
              'border-right : %s;'
              'font-weight : %s;'
              'padding : 0px;}')
        self.ellipsis.setStyleSheet(ss % (f'2px groove {self.c0}', 'none', 'normal'))
        self.folder.setStyleSheet(ss % ('none', f'2px groove {self.c0}', 'bold'))
        if self.simple_mode:
            self.path.setStyleSheet(ss % (f'2px groove {self.c0}', f'2px groove {self.c0}', 'normal'))
        else:
            self.path.setStyleSheet(ss % ('none', 'none', 'normal'))
        
        self.addWidget(self.path)
        if not self.simple_mode:
            self.insertWidget(0, self.ellipsis)
            self.addWidget(self.folder)
    
    
    def update_qedit(self, ddir, x=False):
        # update QLineEdit text
        if self.simple_mode:
            self.path.setText(ddir)
            return
        
        dirs = ddir.split(os.sep)
        folder_txt = dirs.pop(-1)
        path_txt = os.sep.join(dirs) + os.sep
        self.qedits[1].setText(path_txt)
        self.qedits[2].setText(folder_txt)
        fm = self.qedits[2].fontMetrics()
        width = fm.horizontalAdvance(folder_txt) + int(fm.maxWidth()/2)
        self.qedits[2].setFixedWidth(width)
        
        c0, c1 = [self.c0, self.c1] if x else [self.c1, self.c0]
        for qedit in self.qedits:
            qedit.setStyleSheet(qedit.styleSheet().replace(c0, c1))
            

class StatusIcon(QtWidgets.QPushButton):
    def __init__(self, init_state=0):
        super().__init__()
        self.icons = [QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogNoButton),
                      QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton)]
        self.new_status(init_state)
        self.setStyleSheet('QPushButton,'
                            'QPushButton:default,'
                            'QPushButton:hover,'
                            'QPushButton:selected,'
                            'QPushButton:disabled,'
                            'QPushButton:pressed {'
                            'background-color: none;'
                               'border: none;'
                               'color: none;}')
    def new_status(self, x):
        self.setIcon(self.icons[int(x)])  # status icon
        


class RawArrayLoader(QtWidgets.QDialog):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        
        self.gen_layout()
        
        self.fs_val.setValue(1000)
    
    def gen_layout(self):
        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.dims_gbox = QtWidgets.QGroupBox('Data Array')
        grid = QtWidgets.QGridLayout(self.dims_gbox)
        nrows, ncols = self.data.shape
        #rows_hbox = QtWidgets.QHBoxLayout()
        self.bgrp = QtWidgets.QButtonGroup()
        self.bgrp.setExclusive(False)
        nrows_lbl = QtWidgets.QLabel(f'<b>{nrows}</b>')
        nrows_lbl.setAlignment(QtCore.Qt.AlignCenter)
        rows_lbl = QtWidgets.QLabel('rows')
        self.rows_bgrp = QtWidgets.QButtonGroup()
        rows_ch_radio = QtWidgets.QRadioButton('Channels')
        rows_t_radio = QtWidgets.QRadioButton('Timepoints')
        #self.rows_bgrp.addButton(rows_ch_radio, 0)
        #self.rows_bgrp.addButton(rows_t_radio, 1)
        #rows_hbox.addWidget(rows_lbl)
        #cols_hbox = QtWidgets.QHBoxLayout()
        ncols_lbl = QtWidgets.QLabel(f'<b>{ncols}</b>')
        ncols_lbl.setAlignment(QtCore.Qt.AlignCenter)
        cols_lbl = QtWidgets.QLabel('columns')
        self.cols_bgrp = QtWidgets.QButtonGroup()
        cols_ch_radio = QtWidgets.QRadioButton('Channels')
        cols_t_radio = QtWidgets.QRadioButton('Timepoints')
        #self.cols_bgrp.addButton(cols_ch_radio, 0)
        #self.cols_bgrp.addButton(cols_t_radio, 1)
        self.bgrp.addButton(rows_ch_radio, 0)
        self.bgrp.addButton(rows_t_radio, 1)
        self.bgrp.addButton(cols_ch_radio, 2)
        self.bgrp.addButton(cols_t_radio, 3)
        #cols_hbox.addWidget(cols_lbl)
        if nrows > ncols:
            rows_t_radio.setChecked(True)
            cols_ch_radio.setChecked(True)
            self.data = np.array(self.data.T)
        else:
            rows_ch_radio.setChecked(True)
            cols_t_radio.setChecked(True)
        
        line0 = pyfx.DividerLine(orientation='v')
        line1 = pyfx.DividerLine(orientation='v')
        hline = pyfx.DividerLine(lw=1, mlw=1)
        grid.addWidget(nrows_lbl,     0, 0)
        grid.addWidget(rows_lbl,      0, 1)
        grid.addWidget(line0,         0, 2)
        grid.addWidget(rows_ch_radio, 0, 3)
        grid.addWidget(rows_t_radio,  0, 4)
        grid.addWidget(hline,         1, 0, 1, 5)
        grid.addWidget(ncols_lbl,     2, 0)
        grid.addWidget(cols_lbl,      2, 1)
        grid.addWidget(line1,         2, 2)
        grid.addWidget(cols_ch_radio, 2, 3)
        grid.addWidget(cols_t_radio,  2, 4)
        
        self.fs_gbox = QtWidgets.QGroupBox('Sampling Rate')
        fs_lay = QtWidgets.QHBoxLayout(self.fs_gbox)
        fs_lbl = QtWidgets.QLabel('fs:')
        self.fs_val = QtWidgets.QDoubleSpinBox()
        self.fs_val.setRange(1, 9999999999)
        self.fs_val.setSuffix(' Hz')
        fs_line = pyfx.DividerLine(orientation='v')
        dur_lbl = QtWidgets.QLabel('Duration:')
        self.dur_val = QtWidgets.QDoubleSpinBox()
        self.dur_val.setRange(1, 9999999999)
        self.dur_val.setDecimals(4)
        self.dur_val.setSuffix(' s')
        self.dur_val.setEnabled(False)
        fs_lay.addWidget(fs_lbl)
        fs_lay.addWidget(self.fs_val)
        #fs_lay.addWidget(fs_line)
        fs_lay.addSpacing(25)
        fs_lay.addWidget(dur_lbl)
        fs_lay.addWidget(self.dur_val)
        
        
        
        
        
        
        #self.layout.addLayout(rows_hbox)
        #self.layout.addLayout(cols_hbox)
        self.layout.addWidget(self.dims_gbox)
        self.layout.addWidget(self.fs_gbox)
        
        self.bgrp.buttonToggled.connect(self.label_dims)
        self.fs_val.valueChanged.connect(lambda x: self.update_fs_dur(x, mode=0))
        self.dur_val.valueChanged.connect(lambda x: self.update_fs_dur(x, mode=1))
        #self.rows_bgrp.buttonToggled.connect(self.label_dims)
        #self.cols_bgrp.buttonToggled.connect(self.label_dims)
        # list of valid file extensions
        # self.ext_gbox = QtWidgets.QGroupBox('Select file type')
        # ext_vlay = QtWidgets.QVBoxLayout(self.ext_gbox)
        # self.ext_bgrp = QtWidgets.QButtonGroup()
        # extensions = ['.npy', '.mat', '.csv']
        # for i,ext in enumerate(extensions):
        #     btn = QtWidgets.QRadioButton(ext)
        #     if i==0:
        #         btn.setChecked(True)
        #     ext_vlay.addWidget(btn)
        #     self.ext_bgrp.addButton(btn)
        # self.layout.addWidget(self.ext_gbox)
    
    def update_fs_dur(self, val, mode):
        nts = self.data.shape[int(self.bgrp.buttons()[3].isChecked())]
        print(nts)
        if mode == 0:  # calculate recording duration from sampling rate
            dur = nts / self.fs_val.value()
            self.dur_val.setValue(dur)
        elif mode == 1:
            fs = nts / self.dur_val.value()
            self.fs_val.setValue(fs)
            
        
        
    def label_dims(self, btn, chk):
        if not chk: return
        if self.bgrp.checkedId()   in [0,3] : chks = [True, False, False, True]
        elif self.bgrp.checkedId() in [1,2] : chks = [False, True, True, False]
        self.bgrp.blockSignals(True)
        for b,x in zip(self.bgrp.buttons(), chks):
            b.setChecked(x)
        self.bgrp.blockSignals(False)
        #if self.bgrp.
        # bgrp = [self.rows_bgrp, self.cols_bgrp][int(btn in self.rows_bgrp.buttons())]
        # print('blocking signals')
        # #bgrp.blockSignals(True)
        # print('reversing checks')
        # bgrp.blockSignals(True)
        # for b in bgrp.buttons():
        #     b.blockSignals(True)
        #     b.toggle()
        #     b.blockSignals(False)
        # bgrp.blockSignals(False)
        # #_ = [b.setChecked(not b.isChecked()) for b in bgrp.buttons()]
        # #bgrp.blockSignals(False)
        
       
        
    

        
        
        
class AnalysisBtns(QtWidgets.QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # View/save event channels; always available for all processed data folders
        self.option1_widget = self.create_widget('Select event channels', 'green')
        self.option1_btn = self.option1_widget.findChild(QtWidgets.QPushButton)
        # Classify dentate spikes; requires event channels and probe DS_DF files
        self.option2_widget = self.create_widget('Classify dentate spikes', 'blue')
        self.option2_btn = self.option2_widget.findChild(QtWidgets.QPushButton)
        
        #self.btn_grp.addButton(self.option1_btn)
        #self.btn_grp.addButton(self.option2_btn)
        self.option1_widget.setEnabled(False)
        self.option2_widget.setEnabled(False)
        
        #self.btn_grp.buttonToggled.connect(self.action_toggled)
    
    def create_widget(self, txt, c):
        widget = QtWidgets.QWidget()
        widget.setContentsMargins(0,0,0,0)
        hlay = QtWidgets.QHBoxLayout(widget)
        hlay.setContentsMargins(0,0,0,0)
        hlay.setSpacing(8)
        btn = QtWidgets.QPushButton()
        #btn.setCheckable(True)
        #clight = pyfx.hue(c, 0.7, 1); cdark = pyfx.hue(c, 0.4, 0)#; cdull = pyfx.hue(clight, 0.8, 0.5, alpha=0.5)
        btn.setStyleSheet(btn_ss % (pyfx.hue(c, 0.7, 1),  pyfx.hue(c, 0.4, 0)))
        lbl = QtWidgets.QLabel(txt)
        hlay.addWidget(btn)
        hlay.addWidget(lbl)
        #self.btn_grp.addButton(btn)
        return widget
        
    
    def ddir_toggled(self, ddir, current_probe=0):
        self.option1_widget.setEnabled(False)
        self.option2_widget.setEnabled(False)
        
        if not os.path.isdir(ddir):
            return
        
        files = os.listdir(ddir)
        # required: basic LFP files
        if all([bool(f in files) for f in ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy']]):
            self.option1_widget.setEnabled(True)  # req: basic LFP data
        
        # required: event channels file, DS_DF file for current probe
        if f'DS_DF_{current_probe}' in files and f'theta_ripple_hil_chan_{current_probe}.npy' in files:
            self.option2_widget.setEnabled(True)


class MatplotlibPopup(QtWidgets.QDialog):
    """ Simple popup window to display Matplotlib figure """
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.canvas_layout = QtWidgets.QVBoxLayout()
        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas_layout.addWidget(self.toolbar)
        self.canvas_layout.addWidget(self.canvas)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.canvas_layout)
        self.setLayout(self.layout)


def create_widget_row(key, val, param_type, description='', **kw):
    """
    key, val - parameter name, value
    param_type - 'num', 'int', 'keyword', 'text', 'bool', 'freq_band'
    """
    def set_val(w, val, **kw):
        if w.minimum() > val: w.setMinimum(val)
        if w.maximum() < val: w.setMaximum(val)
        if 'mmin' in kw    : w.setMinimum(kw['mmin'])
        if 'mmax' in kw    : w.setMaximum(kw['mmax'])
        if 'step' in kw   : w.setSingleStep(kw['step'])
        if 'suffix' in kw : w.setSuffix(kw['suffix'])
        if 'dec' in kw and w.__class__ == QtWidgets.QDoubleSpinBox:
            w.setDecimals(kw['dec'])
        w.setValue(val)
        return w
    
    lbl = QtWidgets.QLabel(key)
    lbl.setToolTip(description)
    if param_type in ['num','int']:
        if param_type == 'int':
            w = QtWidgets.QSpinBox()
            val = int(val)
        else: 
            w = QtWidgets.QDoubleSpinBox()
        w = set_val(w, val, **kw)
    elif param_type == 'keyword':
        w = QtWidgets.QComboBox()
        items = kw.get('opts', [])
        if val not in items: items.insert(0, val)
        w.addItems(items)
        w.setCurrentText(val)
    elif param_type == 'text':
        w = QtWidgets.QLineEdit()
        w.setText(val)
    else:
        # hbox = QtWidgets.QWidget()
        # hlay = QtWidgets.QHBoxLayout(hbox)
        # hlay.setSpacing(0)
        if param_type == 'bool':
            w0 = QtWidgets.QRadioButton('True')
            w0.setChecked(bool(val)==True)
            w1 = QtWidgets.QRadioButton('False')
            w1.setChecked(bool(val)==False)
        elif param_type == 'freq_band':
            w0 = QtWidgets.QDoubleSpinBox()
            w0 = set_val(w0, val[0], **kw)
            w1 = QtWidgets.QDoubleSpinBox()
            w1 = set_val(w1, val[1], **kw)
            # dash = QtWidgets.QLabel(' - ')
            # dash.setAlignment(QtCore.Qt.AlignCenter)
            # hlay.addWidget(dash)
        # hlay.insertWidget(0, w0)
        # hlay.insertWidget(-1, w1)
        w = [w0, w1]
    return lbl, w


def create_fband_row(w0, w1, mid=' - '):
    hlay = QtWidgets.QHBoxLayout()
    hlay.setSpacing(0)
    midw = QtWidgets.QLabel(mid)
    midw.setAlignment(QtCore.Qt.AlignCenter)
    hlay.addWidget(w0)
    hlay.addWidget(midw)
    hlay.addWidget(w1)
    return hlay

class ParamWidgets(object):
    
    def __init__(self, PARAMS):
        D = pd.Series(PARAMS)
        L = {}
        W = {}
        HLAY = {}
        
        # downsampled LFP
        L['lfp_fs'], W['lfp_fs'] = create_widget_row('lfp_fs', D.lfp_fs, 'int', mmax=30000, suffix=' Hz')
        
        # theta, slow gamma, and fast gamma bands, DS bandpass, and SWR bandpass
        for k in ['theta', 'slow_gamma', 'fast_gamma', 'ds_freq', 'swr_freq']:
            L[k], W[k] = create_widget_row(k, D[k], 'freq_band', dec=1, mmax=1000, suffix=' Hz')
            HLAY[k] = create_fband_row(*W[k])
        
        # DS detection params
        L['ds_height_thr'], W['ds_height_thr'] = create_widget_row('ds_height_thr', D['ds_height_thr'], 'num', dec=1, step=0.1, suffix=' STD')
        L['ds_dist_thr'], W['ds_dist_thr'] = create_widget_row('ds_dist_thr', D['ds_dist_thr'], 'num', dec=1, step=0.1, suffix=' s')
        L['ds_prom_thr'], W['ds_prom_thr'] = create_widget_row('ds_prom_thr', D['ds_prom_thr'], 'num', dec=1, step=0.1)
        L['ds_wlen'], W['ds_wlen'] = create_widget_row('ds_wlen', D['ds_wlen'], 'num', dec=3, step=0.005, suffix=' s')
        
        # SWR detection params
        L['swr_ch_bound'], W['swr_ch_bound'] = create_widget_row('swr_ch_bound', D['swr_ch_bound'], 'int', suffix=' channels')
        L['swr_height_thr'], W['swr_height_thr'] = create_widget_row('swr_height_thr', D['swr_height_thr'], 'num', dec=1, step=0.1, suffix=' STD')
        L['swr_min_thr'], W['swr_min_thr'] = create_widget_row('swr_min_thr', D['swr_min_thr'], 'num', dec=1, step=0.1, suffix=' STD')
        L['swr_dist_thr'], W['swr_dist_thr'] = create_widget_row('swr_dist_thr', D['swr_dist_thr'], 'num', dec=1, step=0.1, suffix=' s')
        L['swr_min_dur'], W['swr_min_dur'] = create_widget_row('swr_min_dur', D['swr_min_dur'], 'int', mmax=1000, suffix=' ms')
        L['swr_freq_thr'], W['swr_freq_thr'] = create_widget_row('swr_freq_thr', D['swr_freq_thr'], 'num', mmax=1000, dec=1, step=1, suffix=' Hz')
        L['swr_freq_win'], W['swr_freq_win'] = create_widget_row('swr_freq_win', D['swr_freq_win'], 'int', mmax=1000, suffix=' ms')
        L['swr_maxamp_win'], W['swr_maxamp_win'] = create_widget_row('swr_maxamp_win', D['swr_maxamp_win'], 'int', mmax=1000, suffix=' ms')
        
        # CSD calculation params
        methods = ['standard', 'delta', 'step', 'spline']
        filters = ['gaussian','identity','boxcar','hamming','triangular']
        L['csd_method'], W['csd_method'] = create_widget_row('csd_method', D['csd_method'], 'keyword',opts=methods)
        L['f_type'], W['f_type'] = create_widget_row('f_type', D['f_type'], 'keyword',opts=filters)
        L['f_order'], W['f_order'] = create_widget_row('f_order', D['f_order'], 'int')
        L['f_sigma'], W['f_sigma'] = create_widget_row('f_sigma', D['f_sigma'], 'num', dec=1, step=0.1)
        L['tol'], W['tol'] = create_widget_row('tol', D['tol'], 'num', dec=8, step=0.0000001)
        L['spline_nsteps'], W['spline_nsteps'] = create_widget_row('spline_nsteps', D['spline_nsteps'], 'int')
        L['vaknin_el'], W['vaknin_el'] = create_widget_row('vaknin_el', D['vaknin_el'], 'bool')

        self.LABELS = L
        self.WIDGETS = W
        self.HLAY = HLAY
        
# class ParamView(QtWidgets.QDialog):
#     def __init__(self, params=None, ddir=None, parent=None):
#         super().__init__(parent)
#         qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
#         self.setGeometry(qrect)
        
#         params = None
#         # get info, convert to text string
#         if params is None:
#             params = ephys.read_param_file('default_params.txt')
#         if 'RAW_DATA_FOLDER' in params: rdf = params.pop('RAW_DATA_FOLDER')
#         if 'PROCESSED_DATA_FOLDER' in params: pdf = params.pop('PROCESSED_DATA_FOLDER')
        
        
#         keys, vals = zip(*params.items())
#         vstr = [*map(str,vals)]
#         klen = max(map(len, keys))  # get longest keyword
#         #vlen = max(map(len,vstr)) # get longest value string
        
#         rows3 = ['<pre>' + k + '_'*(klen-len(k)) + ' : ' + v + '</pre>' for k,v in zip(keys,vstr)]
#         ttxt = ''.join(rows3)
        
#         self.textwidget = QtWidgets.QTextEdit(ttxt)
        
        
#         # fmt = f'{{:^{klen}}} : {{:>{vlen}}}'
#         # rows = [fmt.format(k,v) for k,v in zip(keys,vstr)]
#         # TEXT = os.linesep.join(rows)
            
#         # a = 'i am a key'
#         # b = 'i am a value'
#         # TEXT = f'{a:<30} : {b:<60}'
        
#         qfont = QtGui.QFont("Monospace")
#         qfont.setStyleHint(QtGui.QFont.TypeWriter)
#         # qfont.setPointSize(10)
#         # qfm = QtGui.QFontMetrics(qfont)
            
#         # create QTextEdit for displaying file
        
#         #self.textwidget.setAlignment(QtCore.Qt.AlignCenter)
#         #self.textwidget.setFont(qfont)
        
#         # fm = self.textwidget.fontMetrics()
#         # klen2 = max(map(fm.horizontalAdvance, keys))
#         # vlen2 = max(map(fm.horizontalAdvance, vstr))
#         # fmt2 = f'{{:<{klen2}}} {{:<{vlen2}}}'
        
#         # rows3 = [f'{k:>20} : {v}' for k,v in zip(keys,vstr)]
        
#         # rows2 = [fmt2.format(k,v) for k,v in zip(keys,vstr)]
#         # TEXT2 = os.linesep.join(rows2)
        
        
#         # for row in rows3:
#         #     self.textwidget.append(row)
#             #self.textwidget.appendPlainText(row)
        
        
#         layout = QtWidgets.QVBoxLayout(self)
#         layout.addWidget(self.textwidget)
        
    
class InfoView(QtWidgets.QDialog):
    def __init__(self, info=None, ddir=None, parent=None):
        super().__init__(parent)
        qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
        self.setGeometry(qrect)
        
        # get info, convert to text string
        # if info is None:
        #     info = ephys.load_recording_info(ddir)
        #self.info_text = info2text(info)
        self.info_text = 'Sorry - no text here :('
        
        # create QTextEdit for displaying file
        self.textwidget = QtWidgets.QTextEdit(self.info_text)
        self.textwidget.setAlignment(QtCore.Qt.AlignCenter)
        self.textwidget.setReadOnly(True)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.textwidget)


def create_filepath_row(txt, base_dir):
    fmt = '<code>{}</code>'
    w = QtWidgets.QWidget()
    vlay = QtWidgets.QVBoxLayout(w)
    row0 = QtWidgets.QHBoxLayout()
    row0.setContentsMargins(0,0,0,0)
    header = QtWidgets.QLabel(fmt.format(txt))
    row0.addWidget(header)
    row0.addStretch()
    row1 = QtWidgets.QHBoxLayout()
    row1.setContentsMargins(0,0,0,0)
    row1.setSpacing(5)
    qlabel = QtWidgets.QLabel(base_dir.format(base_dir))
    qlabel.setStyleSheet('QLabel {background-color:white;'
                         'border:1px solid gray; border-radius:4px; padding:5px;}')
    btn = QtWidgets.QPushButton()
    btn.setIcon(QtWidgets.QWidget().style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton))
    btn.setMinimumSize(30,30)
    btn.setIconSize(QtCore.QSize(20,20))
    row1.addWidget(qlabel, stretch=2)
    row1.addWidget(btn, stretch=0)
    vlay.addLayout(row0)
    vlay.addLayout(row1)
    return w,header,qlabel,btn

        
        
        
        
# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     app.setStyle('Fusion')
#     app.setQuitOnLastWindowClosed(True)
    
#     #ddir = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/NN_JG008')
#     #popup = InfoView(ddir=ddir)
    
#     nn_raw = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/raw_data/'
#                 'JG008_071124n1_neuronexus')
#     popup = RawDirectorySelectionPopup()
#     #popup = ProbeFileSimple()
#     #popup = thing()
#     #popup = AuxDialog(n=6)
    
#     popup.show()
#     popup.raise_()
    
#     sys.exit(app.exec())
    
