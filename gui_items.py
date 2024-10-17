#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:45:34 2024

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import shutil
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets, QtCore, QtGui
import probeinterface as prif
from probeinterface.plotting import plot_probe
import pdb
# custom modules
import pyfx
import ephys

# widget_list = ('QPushButton, QRadioButton, QLabel, QComboBox, QSpinBox, '
#                'QDoubleSpinBox, QLineEdit, QTextEdit')
basic_widget_style = ('QPushButton, QRadioButton, QCheckBox, QLabel, QComboBox, '
                      'QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit, QAbstractItemView'
                      '{padding : 2px;}'
                      'QWidget:focus {outline : none}')
basic_popup_style = 'QWidget {font-size : 15pt}' #+ basic_widget_style

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

mode_btn_ss = ('QPushButton {'
               'background-color : whitesmoke;'
               'border : 3px outset gray;'
               'border-radius : 2px;'
               'color : black;'
               'padding : 4px;'
               'font-weight : bold;'
               '}'
               
               'QPushButton:pressed {'
               'background-color : gray;'
               'border : 3px inset gray;'
               'color : white;'
               '}'
               
               'QPushButton:checked {'
               'background-color : darkgray;'
               'border : 3px inset gray;'
               'color : black;'
               '}'
               
               'QPushButton:disabled {'
               'background-color : gainsboro;'
               'border : 3px outset darkgray;'
               'color : gray;'
               '}'
               
               'QPushButton:disabled:checked {'
               'background-color : darkgray;'
               'border : 3px inset darkgray;'
               'color : dimgray;'
               '}'
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
    
    def key_step(self, x):
        if x==1:
            self.set_val(self.val + self.nsteps)
        elif x==0:
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
    def __init__(self, text_shown='Hide freq. band power', 
                 text_hidden='Show freq. band power', init_show=False, parent=None):
        #\u00BB , \u00AB
        super().__init__(parent)
        self.setCheckable(True)
        self.TEXTS = [text_hidden, text_shown]
        # set checked/visible or unchecked/hidden
        self.setChecked(init_show)
        self.setText(self.TEXTS[int(init_show)])
        
        self.toggled.connect(self.update_state)

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


class LabeledWidget(QtWidgets.QWidget):
    def __init__(self, widget=QtWidgets.QWidget, txt='', orientation='v', 
                 label_pos=0, **kwargs):
        super().__init__()
        assert orientation in ['h','v'] and label_pos in [0,1]
        self.setContentsMargins(0,0,0,0)
        if orientation == 'h': 
            self.layout = QtWidgets.QHBoxLayout(self)
        else: 
            self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(kwargs.get('spacing', 1))
        self.label = QtWidgets.QLabel(txt)
        self.qw = widget()
        self.layout.addWidget(self.qw, stretch=2)
        self.layout.insertWidget(label_pos, self.label, stretch=0)
    
    def text(self):
        return self.label.text()
    
    def setText(self, txt):
        self.label.setText(txt)
        
class LabeledSpinbox(LabeledWidget):
    def __init__(self, txt='', double=False, **kwargs):
        widget = QtWidgets.QDoubleSpinBox if double else QtWidgets.QSpinBox
        super().__init__(widget, txt, **kwargs)
        if 'suffix' in kwargs: self.qw.setSuffix(kwargs['suffix'])
        if 'minimum' in kwargs: self.qw.setMinimum(kwargs['minimum'])
        if 'maximum' in kwargs: self.qw.setMaximum(kwargs['maximum'])
        if 'range' in kwargs: self.qw.setRange(*kwargs['range'])
        if 'decimals' in kwargs: self.qw.setDecimals(kwargs['decimals'])
    
    def value(self):
        return self.qw.value()
    
    def setValue(self, val):
        self.qw.setValue(val)


class LabeledCombobox(LabeledWidget):
    def __init__(self, txt='', **kwargs):
        super().__init__(QtWidgets.QComboBox, txt, **kwargs)
    
    def addItems(self, items):
        return self.qw.addItems(items)
    
    def currentText(self): 
        return self.qw.currentText()
    
    def currentIndex(self):
        return self.qw.currentIndex()
    
    def setCurrentText(self, txt):
        self.qw.setCurrentText(txt)
    
    def setCurrentIndex(self, idx):
        self.qw.setCurrentIndex(idx)


class LabeledPushbutton(LabeledWidget):
    def __init__(self, txt='', orientation='h', label_pos=1, spacing=10, **kwargs):
        super().__init__(QtWidgets.QPushButton, txt, orientation, label_pos, spacing=spacing, **kwargs)
        if 'btn_txt' in kwargs: self.qw.setText(kwargs['btn_txt'])
        if 'icon' in kwargs: self.qw.setIcon(kwargs['icon'])
        if 'icon_size' in kwargs: self.qw.setIconSize(QtCore.QSize(kwargs['icon_size']))
        if 'ss' in kwargs: self.qw.setStyleSheet(kwargs['ss'])
    
    def isChecked(self):
        return self.qw.isChecked()
    
    def setCheckable(self, x):
        self.qw.setCheckable(x)


class SpinboxRange(QtWidgets.QWidget):
    def __init__(self, double=False, parent=None, **kwargs):
        super().__init__(parent)
        if double:
            self.box0 = QtWidgets.QDoubleSpinBox()
            self.box1 = QtWidgets.QDoubleSpinBox()
        else:
            self.box0 = QtWidgets.QSpinBox()
            self.box1 = QtWidgets.QSpinBox()
        for box in [self.box0, self.box1]:
            if 'suffix' in kwargs: box.setSuffix(kwargs['suffix'])
            if 'minimum' in kwargs: box.setMinimum(kwargs['minimum'])
            if 'maximum' in kwargs: box.setMaximum(kwargs['maximum'])
            if 'decimals' in kwargs: box.setDecimals(kwargs['decimals'])
            
        self.dash = QtWidgets.QLabel(' — ')
        self.dash.setAlignment(QtCore.Qt.AlignCenter)
        
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.box0)
        self.layout.addWidget(self.dash)
        self.layout.addWidget(self.box1)
    
    def get_values(self):
        return [self.box0.value(), self.box1.value()]


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
              'padding : 0px;}'
              
              'QLineEdit:disabled {'
              'border-color : gainsboro;}'
              )
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
        
    
    def ddir_toggled(self, ddir, probe_idx=0):
        self.option1_widget.setEnabled(False)
        self.option2_widget.setEnabled(False)
        
        if not os.path.isdir(ddir):
            return
        
        files = os.listdir(ddir)
        # required: basic LFP files
        if all([bool(f in files) for f in ['lfp_bp.npz', 'lfp_time.npy', 'lfp_fs.npy']]):
            self.option1_widget.setEnabled(True)  # req: basic LFP data
        
        # required: event channels file, DS_DF file for current probe
        if f'DS_DF_{probe_idx}' in files and f'theta_ripple_hil_chan_{probe_idx}.npy' in files:
            self.option2_widget.setEnabled(True)


class TableWidget(QtWidgets.QTableWidget):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.load_df(df)
        self.verticalHeader().hide()
        
        self.itemChanged.connect(self.print_update)
    
    def print_update(self, item):
        print('itemChanged')
        self.df.iloc[item.row(), item.column()] = int(item.text())
        
    
    def init_table(self, selected_columns = []):
        nRows = len(self.df.index)
        nColumns = len(selected_columns) or len(self.df.columns)
        self.setRowCount(nRows)
        self.setColumnCount(nColumns)

        self.setHorizontalHeaderLabels(selected_columns or self.df.columns)
        self.setVerticalHeaderLabels(self.df.index.astype(str))
        
        # Display an empty table
        if self.df.empty:
            self.clearContents()
            return

        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = QtWidgets.QTableWidgetItem(str(self.df.iat[row, col]))
                self.setItem(row, col, item)
        # Enable sorting on the table
        self.setSortingEnabled(True)
        # Enable column moving by drag and drop
        self.horizontalHeader().setSectionsMovable(True)
    
    def load_df(self, df, selected_columns = []):
        self.df = df
        self.init_table(selected_columns)


class TTabWidget(QtWidgets.QTabWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.currentChanged.connect(self.updateGeometry)

    def minimumSizeHint(self):
        return self.sizeHint()

    def sizeHint(self):
        lc = QtCore.QSize(0, 0)
        rc = QtCore.QSize(0, 0)
        opt = QtWidgets.QStyleOptionTabWidgetFrame()
        self.initStyleOption(opt)
        if self.cornerWidget(QtCore.Qt.TopLeftCorner):
            lc = self.cornerWidget(QtCore.Qt.TopLeftCorner).sizeHint()
        if self.cornerWidget(QtCore.Qt.TopRightCorner):
            rc = self.cornerWidget(QtCore.Qt.TopRightCorner).sizeHint()
        layout = self.findChild(QtWidgets.QStackedLayout)
        layoutHint = layout.currentWidget().sizeHint()
        tabHint = self.tabBar().sizeHint()
        if self.tabPosition() in (self.North, self.South):
            size = QtCore.QSize(
                max(layoutHint.width(), tabHint.width() + rc.width() + lc.width()), 
                layoutHint.height() + max(rc.height(), max(lc.height(), tabHint.height()))
            )
        else:
            size = QtCore.QSize(
                layoutHint.width() + max(rc.width(), max(lc.width(), tabHint.width())), 
                max(layoutHint.height(), tabHint.height() + rc.height() + lc.height())
            )
        return size


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
        self.setInformativeText('tmp')  # make sure label shows up in widget children 
        self.setInformativeText(sub_msg)
        self.sub_label = self.findChild(QtWidgets.QLabel, 'qt_msgbox_informativelabel')
        # locate button box
        self.bbox = self.findChild(QtWidgets.QDialogButtonBox, 'qt_msgbox_buttonbox')
        
    @classmethod
    def run(cls, *args, **kwargs):
        pyfx.qapp()
        msgbox = cls(*args, **kwargs)
        msgbox.show()
        msgbox.raise_()
        res = msgbox.exec()
        return res
    
    
class MsgboxSave(Msgbox):
    def __init__(self, msg='Save successful!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Information)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        # set check icon
        chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        px_size = self.icon_label.pixmap().size()
        self.setIconPixmap(chk_icon.pixmap(px_size))


class MsgboxError(Msgbox):
    def __init__(self, msg='Something went wrong!', sub_msg='', title='', 
                 invalid_probe='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        # pop-up messagebox appears when save is complete
        self.setIcon(QtWidgets.QMessageBox.Critical)
        self.setStandardButtons(QtWidgets.QMessageBox.Close)
        if invalid_probe:
            self.invalid_probe_file(invalid_probe)
            
    def invalid_probe_file(self, f):
        self.setText('The following is not a valid probe file:')
        self.setInformativeText(f'<nobr><code>{f}</code></nobr>')
        self.sub_label.setWordWrap(False)


class MsgboxWarning(Msgbox):
    def __init__(self, msg='Warning!', sub_msg='', title='', parent=None):
        super().__init__(msg, sub_msg, title, parent)
        self.setIcon(QtWidgets.QMessageBox.Warning)
        self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        
    def set_overwrite_mode(self, ppath):
        # overwrite items in directory
        if os.path.isdir(ppath) and len(os.listdir(ppath)) > 0:
            n = len(os.listdir(ppath))
            self.setText(f'The directory <code>{os.path.basename(ppath)}</code> contains '
                         f'<code>{n}</code> items.')#'<br><br>Overwrite existing files?')
            self.setInformativeText('Overwrite existing files?')
            self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
                                    | QtWidgets.QMessageBox.Apply)
            merge_btn = self.button(QtWidgets.QMessageBox.Apply)
            merge_btn.setText(merge_btn.text().replace('Apply','Merge'))
        # overwrite file
        elif os.path.isfile(ppath):
            self.setText(f'The file <code>{os.path.basename(ppath)}</code> already exists.')
            self.setInformativeText('Do you want to replace it?')
            self.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        else:
            return False
        # rename "Yes" button to "Overwrite" response buttons
        yes_btn = self.button(QtWidgets.QMessageBox.Yes)
        yes_btn.setText(yes_btn.text().replace('Yes','Overwrite'))
        return True
    
    
    @classmethod
    def unsaved_changes_warning(cls, msg='Unsaved changes', sub_msg='Do you want to save your work?', parent=None):
        msgbox = cls(msg, sub_msg, parent=parent)
        msgbox.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard)
        msgbox.show()
        msgbox.raise_()
        res = msgbox.exec()
        if res == QtWidgets.QMessageBox.Discard:
            return True   # don't worry about changes
        elif res == QtWidgets.QMessageBox.Cancel:
            return False  # abort close attempt
        elif res == QtWidgets.QMessageBox.Save:
            return -1     # save changes and then close
        
        
    @classmethod
    def overwrite_warning(cls, ppath, parent=None):
        msgbox = cls(parent=parent)
        is_ovr = msgbox.set_overwrite_mode(ppath)
        if is_ovr==False: 
            return True  # fake news, no overwrite
        msgbox.show()
        msgbox.raise_()
        res = msgbox.exec()
        if res == QtWidgets.QMessageBox.Yes: 
            if os.path.isdir(ppath):
                shutil.rmtree(ppath)  # delete any existing directory files
                os.makedirs(ppath)
            return True  # continue overwriting
        elif res == QtWidgets.QMessageBox.Apply:
            return True   # add new files to the existing directory contents
        else: return False  # abort save attempt
        
        
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
    array_exts = ['.npy', '.mat']#, '.csv']
    probe_exts = ['.json', '.prb', '.mat']
    
    def __init__(self, init_ddir='', load_or_save='load', overwrite_mode=0, 
                 is_directory=True, is_probe=False, is_array=False, parent=None, **kwargs):
        """
        init_ddir: optional starting directory
        load_or_save: "load" in existing directory/file or "save" new one (not recommended)
        is_directory: allow selection of all files (False) or only directories (True)
        is_probe: add filter for valid probe filetypes
        """
        super().__init__(parent)
        self.load_or_save = load_or_save
        self.overwrite_mode = overwrite_mode  # 0=add to folder, 1=overwrite files
        self.probe_exts = kwargs.get('probe_exts', self.probe_exts)
        self.array_exts = kwargs.get('array_exts', self.array_exts)
        
        options = self.Options()
        options |= self.DontUseNativeDialog
        
        self.setViewMode(self.List)
        self.setAcceptMode(self.AcceptOpen)  # open file
        
        # filter for array/probe files
        fx = lambda llist: ' '.join([*map(lambda x: '*'+x, llist)])
        if is_probe or is_array:
            is_directory = False
            if is_probe:
                ffilter = f'Probe files ({fx(self.probe_exts)})'  #f"Probe files ({' '.join([*map(lambda x: '*'+x, self.probe_exts)])})"
            else:
                ffilter = f'Data files ({fx(self.array_exts)})'
            self.setNameFilter(ffilter)
            if self.load_or_save=='save': 
                self.setAcceptMode(self.AcceptSave)
            
        # allow directories only
        if is_directory:
            self.setFileMode(self.Directory)
        else:
            if self.load_or_save == 'load':
                self.setFileMode(self.ExistingFile)  # allow existing files
            else:
                self.setFileMode(self.AnyFile)       # allow any file name
        
        try: self.setDirectory(init_ddir)
        except: print('init_ddir argument in FileDialog is invalid')
        self.setOptions(options)
        self.connect_signals()
        
        # if is_probe and self.load_or_save == 'save':
        #     print('hi')
            
        #     self.fx = lambda txt: any(map(lambda ext: txt.endswith(ext), self.probe_exts))
        #     self.lineEdit.textChanged.connect(lambda txt: self.btn.setEnabled(self.fx(txt)))
    
    
    def connect_signals(self):
        self.btn = self.findChild(QtWidgets.QPushButton)
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
    
    @classmethod
    def load_array_file(cls, init_ddir=ephys.base_dirs()[0], parent=None, **kwargs):
        pyfx.qapp()
        dlg = cls(init_ddir=init_ddir, is_array=True, parent=parent, **kwargs)
        dlg.show()
        dlg.raise_()
        res = dlg.exec()
        if not res: return
        
        # get filepath, try loading data
        fpath = dlg.selectedFiles()[0]
        data = ephys.read_array_file(fpath)
        if data is None:
            msgbox = MsgboxError('The following is not a valid array:',
                                 sub_msg=f'<nobr><code>{fpath}</code></nobr>')
            msgbox.sub_label.setWordWrap(False)
            msgbox.exec()
            return
        return (data, fpath)
        
        
    @classmethod
    def load_probe_file(cls, init_ddir=ephys.base_dirs()[2], parent=None, **kwargs):
        pyfx.qapp()
        # run file dialog with probe file settings
        dlg = cls(init_ddir, is_probe=True, parent=parent, **kwargs)
        dlg.show()
        dlg.raise_()
        res = dlg.exec()
        if not res: return
        # get filepath, try loading probe object
        fpath = dlg.selectedFiles()[0]
        probe = ephys.read_probe_file(fpath)
        if probe is None:
            msgbox = MsgboxError(invalid_probe=fpath)
            msgbox.exec()
            return
        return probe
    
    @classmethod
    def save_probe_file(cls, probe, init_ddir=ephys.base_dirs()[2], parent=None, **kwargs):
        pyfx.qapp()
        # 
        init_fname = f'{probe.name}_config.json'
        dlg = cls(init_ddir, is_probe=True, load_or_save='save', parent=parent, **kwargs)
        # add validator for supported file extensions
        dlg.fx = lambda txt: any(map(lambda ext: txt.endswith(ext), dlg.probe_exts))
        dlg.lineEdit.textChanged.connect(lambda txt: dlg.btn.setEnabled(dlg.fx(txt)))
        dlg.lineEdit.setText(init_fname)
        dlg.show()
        dlg.raise_()
        res = dlg.exec()
        if res:
            # write probe file to selected location
            fpath = dlg.selectedFiles()[0]
            res = ephys.write_probe_file(probe, fpath)
            if res:
                print('Probe saved!')
            return res
        return False
        # # probe filename convention: [PROBE NAME]_config.[VALID EXTENSION]
        # extension = self.save_exts.currentText()
        # filename = f'{self.probe.name}_config{extension}'
        # fpath = str(Path(ephys.base_dirs()[2], filename))
        # if os.path.exists(fpath):
        #     msgbox = gi.MsgboxWarning(overwrite_file=fpath)
        #     res = msgbox.exec()
        #     if res != QtWidgets.QMessageBox.Yes:
        #         return
        # # save probe file in desired file format
        # res = ephys.write_probe_file(self.probe, fpath)
        # if res:
        #     msgbox = gi.MsgboxSave('Probe saved!<br>Exit window?', parent=self)
        #     res = msgbox.exec()
        #     if res == QtWidgets.QMessageBox.Yes:
        #         self.accept()
        # self.saveAction.setEnabled(False)
        # self.save_exts.setEnabled(False)
        
    
    def accept(self):
        if self.load_or_save == 'save':
            ddir = self.directory().path()
            if self.fileMode()==self.Directory and len(os.listdir(ddir))>0:
                res = MsgboxWarning.overwrite_warning(ddir, parent=self)
                # if res:
                #     shutil.rmtree(self.processed_ddir)
            else:
                res = MsgboxWarning.overwrite_warning(self.selectedFiles()[0], parent=self)
            if not res: return
        QtWidgets.QDialog.accept(self)
    
    


class Popup(QtWidgets.QDialog):
    """ Simple popup window to display any widget(s) """
    def __init__(self, widgets=[], orientation='v', title='', parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        if orientation   == 'v': self.layout = QtWidgets.QVBoxLayout(self)
        elif orientation == 'h': self.layout = QtWidgets.QHBoxLayout(self)
        for widget in widgets:
            self.layout.addWidget(widget)
        
        
    def center_window(self):
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = pyfx.ScreenRect()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())
        


class MatplotlibPopup(Popup):
    """ Simple popup window to display Matplotlib figure """
    def __init__(self, fig, toolbar_pos='top', title='', parent=None):
        super().__init__(widgets=[], orientation='h', title=title, parent=parent)
        # create figure and canvas
        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        # create toolbar
        if toolbar_pos != 'none':
            self.toolbar = NavigationToolbar(self.canvas, self)
        if toolbar_pos in ['top','bottom', 'none']:
            self.canvas_layout = QtWidgets.QVBoxLayout()
        elif toolbar_pos in ['left','right']:
            self.canvas_layout = QtWidgets.QHBoxLayout()
            self.toolbar.setOrientation(QtCore.Qt.Vertical)
            self.toolbar.setMaximumWidth(30)
        # populate layout with canvas and toolbar (if indicated)
        self.canvas_layout.addWidget(self.canvas)
        if toolbar_pos != 'none':
            idx = 0 if toolbar_pos in ['top','left'] else 1
            self.canvas_layout.insertWidget(idx, self.toolbar)
            
        #self.layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.canvas_layout)
        #self.setLayout(self.layout)
    
    def closeEvent(self, event):
        plt.close()
        self.deleteLater()


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


class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    """
    Edit value of QTableView cell using a spinbox widget
    """
    def createEditor(self, parent, option, index):
        """ Create spinbox for table cell """
        spinbox = QtWidgets.QSpinBox(parent)
        spinbox.valueChanged.connect(lambda: self.commitData.emit(spinbox))
        return spinbox
    
    def setEditorData(self, editor, index):
        """ Initialize spinbox value from model data """
        editor.setValue(index.data())
    
    def setModelData(self, editor, model, index):
        """ Update model data with new spinbox value """
        if editor.value() != index.data():
            model.setData(index, editor.value(), QtCore.Qt.EditRole)
            


class QtWaitingSpinner(QtWidgets.QWidget):
    # initialize class variables
    mColor = QtGui.QColor(QtCore.Qt.blue)
    mRoundness = 100.0
    mMinimumTrailOpacity = 31.4159265358979323846
    mTrailFadePercentage = 50.0
    mRevolutionsPerSecond = 1
    mNumberOfLines = 20
    mLineLength = 15
    mLineWidth = 5
    mInnerRadius = 50
    mCurrentCounter = 0
    mIsSpinning = False

    def __init__(self, centerOnParent=True, disableParentWhenSpinning=True, disabledWidget=None, *args, **kwargs):
        QtWidgets.QWidget.__init__(self, *args, **kwargs)
        self.mCenterOnParent = centerOnParent
        self.mDisableParentWhenSpinning = disableParentWhenSpinning
        self.disabledWidget = disabledWidget
        self.initialize()
    
    def initialize(self):
        # connect timer to rotate function
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.rotate)
        self.updateSize()
        self.updateTimer()
        self.hide()


    @QtCore.pyqtSlot()
    def rotate(self):
        self.mCurrentCounter += 1
        if self.mCurrentCounter > self.numberOfLines():
            self.mCurrentCounter = 0
        self.update()

    def updateSize(self):
        # adjust widget size based on input params for spinner radius/arm length
        size = (self.mInnerRadius + self.mLineLength) * 2
        self.setFixedSize(size, size)
        
    def updateTimer(self):
        # set timer to rotate spinner every revolution
        self.timer.setInterval(int(1000 / (self.mNumberOfLines * self.mRevolutionsPerSecond)))

    def updatePosition(self):
        # adjust widget position to stay in the center of parent window
        if self.parentWidget() and self.mCenterOnParent:
            self.move(int(self.parentWidget().width() / 2 - self.width() / 2),
                      int(self.parentWidget().height() / 2 - self.height() / 2))

    def lineCountDistanceFromPrimary(self, current, primary, totalNrOfLines):
        # calculate distance between a given line and the "primary" line
        distance = primary - current
        if distance < 0:
            distance += totalNrOfLines
        return distance

    def currentLineColor(self, countDistance, totalNrOfLines, trailFadePerc, minOpacity, color):
        # adjust color shading on a line by distance from the primary line
        if countDistance == 0:
            return color

        minAlphaF = minOpacity / 100.0

        distanceThreshold = np.ceil((totalNrOfLines - 1) * trailFadePerc / 100.0)
        if countDistance > distanceThreshold:
            color.setAlphaF(minAlphaF)
        # color interpolation
        else:
            alphaDiff = self.mColor.alphaF() - minAlphaF
            gradient = alphaDiff / distanceThreshold + 1.0
            resultAlpha = color.alphaF() - gradient * countDistance
            resultAlpha = min(1.0, max(0.0, resultAlpha))
            color.setAlphaF(resultAlpha)
        return color

    def paintEvent(self, event):
        # initialize painter
        self.updatePosition()
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.transparent)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self.mCurrentCounter > self.mNumberOfLines:
            self.mCurrentCounter = 0
        painter.setPen(QtCore.Qt.NoPen)

        for i in range(self.mNumberOfLines):
            # draw & angle rounded rectangle evenly between lines based on current distance
            painter.save()
            painter.translate(self.mInnerRadius + self.mLineLength,
                              self.mInnerRadius + self.mLineLength)
            rotateAngle = 360.0 * i / self.mNumberOfLines
            painter.rotate(rotateAngle)
            painter.translate(self.mInnerRadius, 0)
            distance = self.lineCountDistanceFromPrimary(i, self.mCurrentCounter,
                                                          self.mNumberOfLines)
            color = self.currentLineColor(distance, self.mNumberOfLines,
                                          self.mTrailFadePercentage, self.mMinimumTrailOpacity, self.mColor)
            painter.setBrush(color)
            painter.drawRoundedRect(QtCore.QRect(0, -self.mLineWidth // 2, self.mLineLength, self.mLineLength),
                                    self.mRoundness, QtCore.Qt.RelativeSize)
            painter.restore()


    def start(self):
        # set spinner visible, disable parent widget if requested
        self.updatePosition()
        self.mIsSpinning = True  # track spinner activity
        self.show()
        
        if self.mDisableParentWhenSpinning:
            if self.parentWidget() and self.disabledWidget is None:
                self.parentWidget.setEnabled(False)
            elif self.disabledWidget is not None:
                self.disabledWidget.setEnabled(False)
        # if self.parentWidget() and self.mDisableParentWhenSpinning:
        #     self.parentWidget().setEnabled(False)
        # start timer
        if not self.timer.isActive():
            self.timer.start()
            self.mCurrentCounter = 0

    def stop(self):
        # hide spinner, re-enable parent widget, stop timer
        self.mIsSpinning = False
        self.hide()
        
        if self.mDisableParentWhenSpinning:
            if self.parentWidget() and self.disabledWidget is None:
                self.parentWidget.setEnabled(True)
            elif self.disabledWidget is not None:
                self.disabledWidget.setEnabled(True)
        
        # if self.parentWidget() and self.mDisableParentWhenSpinning:
        #     self.parentWidget().setEnabled(True)

        if self.timer.isActive():
            self.timer.stop()
            self.mCurrentCounter = 0

    def setNumberOfLines(self, lines):
        self.mNumberOfLines = lines
        self.updateTimer()

    def setLineLength(self, length):
        self.mLineLength = length
        self.updateSize()

    def setLineWidth(self, width):
        self.mLineWidth = width
        self.updateSize()

    def setInnerRadius(self, radius):
        self.mInnerRadius = radius
        self.updateSize()

    def color(self):
        return self.mColor

    def roundness(self):
        return self.mRoundness

    def minimumTrailOpacity(self):
        return self.mMinimumTrailOpacity

    def trailFadePercentage(self):
        return self.mTrailFadePercentage

    def revolutionsPersSecond(self):
        return self.mRevolutionsPerSecond

    def numberOfLines(self):
        return self.mNumberOfLines

    def lineLength(self):
        return self.mLineLength

    def lineWidth(self):
        return self.mLineWidth

    def innerRadius(self):
        return self.mInnerRadius

    def isSpinning(self):
        return self.mIsSpinning

    def setRoundness(self, roundness):
        self.mRoundness = min(0.0, max(100, roundness))

    def setColor(self, color):
        self.mColor = color

    def setRevolutionsPerSecond(self, revolutionsPerSecond):
        self.mRevolutionsPerSecond = revolutionsPerSecond
        self.updateTimer()

    def setTrailFadePercentage(self, trail):
        self.mTrailFadePercentage = trail

    def setMinimumTrailOpacity(self, minimumTrailOpacity):
        self.mMinimumTrailOpacity = minimumTrailOpacity


class SpinnerWindow(QtWidgets.QWidget):
    def __init__(self, parent=None, show_label=True):
        super(SpinnerWindow, self).__init__(parent)
        self.win = parent
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(10)
        # create spinner object
        self.spinner_widget = QtWidgets.QWidget()
        self.spinner = QtWaitingSpinner(disabledWidget=self.win)
        self.spinner.setParent(self.spinner_widget)
        # create label to display updates
        self.spinner_label = QtWidgets.QLabel('')
        self.spinner_label.setAlignment(QtCore.Qt.AlignCenter)
        self.spinner_label.setWordWrap(True)
        self.spinner_label.setStyleSheet('QLabel'
                                         '{'
                                         'background-color : rgba(230,230,255,200);'
                                         'border : 2px double darkblue;'
                                         'border-radius : 8px;'
                                         'color : darkblue;'
                                         'font-size : 12pt;'
                                         'font-weight : 900;'
                                         'padding : 5px'
                                         '}')
        self.spinner_label.setFixedSize(int(self.spinner.width()*2), int(self.spinner.height()*0.75))
        # add label and spinner to layout, set size
        if show_label:
            self.layout.addWidget(self.spinner_label, alignment=QtCore.Qt.AlignHCenter)
        self.layout.addWidget(self.spinner_widget)
        self.setLayout(self.layout)
        self.setFixedSize(int(self.spinner.width()*2), int(self.spinner.height()*2))
        self.hide()
    
    def adjust_labelSize(self, lw=2, lh=0.75, ww=2, wh=2):
        self.spinner_label.setFixedSize(int(self.spinner.width()*lw), int(self.spinner.height()*lh))
    #def adjust_widgetSize(self, w=2, h=2):
        self.setFixedSize(int(self.spinner.width()*ww), int(self.spinner.height()*wh))
    
    def start_spinner(self):
        xpos = int(self.win.width() / 2 - self.width() / 2)
        ypos = int(self.win.height() / 2 - self.height() / 2)
        self.move(xpos, ypos)
        self.show()
        self.spinner.start()
    
    def stop_spinner(self):
        self.spinner.stop()
        self.spinner_label.setText('')
        self.hide()
        
    
    @QtCore.pyqtSlot(str)
    def report_progress_string(self, txt):
        self.spinner_label.setText(txt)
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(True)
    
    #ddir = ('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/NN_JG008')
    #popup = InfoView(ddir=ddir)
    
    arr = np.load('/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/'
                  'stanford/JF512/DownsampleLFP.npy')
    
    # popup = RawArrayLoader(arr)
    
    
    # popup.show()
    # popup.raise_()
    
    # sys.exit(app.exec())
    
