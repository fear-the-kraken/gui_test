#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:25:32 2024

@author: amandaschott
"""
import sys
import os
from pathlib import Path
import numpy as np
from PyQt5 import QtWidgets, QtCore
import probeinterface as prif
import pdb
# set app folder as working directory
app_ddir = Path(__file__).parent
os.chdir(app_ddir)
# import custom modules
import pyfx
import ephys
import data_processing as dp
import gui_items as gi
import selection_popups as sp
from probe_handler import probegod
from channel_selection_gui import ChannelSelectionWindow
from ds_classification_gui import DS_CSDWindow

    
class hippos(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.BASE_FOLDERS = ephys.base_dirs()  # base data directories
        self.PARAMS = ephys.read_param_file()  # default param values
        
        self.gen_layout()
        
        self.show()
        self.center_window()
        
    
    def gen_layout(self):
        """ Set up layout """
        self.setWindowTitle('Hippos')
        self.setContentsMargins(25,25,25,25)
        
        self.centralWidget = QtWidgets.QWidget()
        self.centralLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.centralLayout.setSpacing(20)
        
        mode_btn_ss = ('QPushButton {'
                       'background-color : gainsboro;'
                       'border : 3px outset gray;'
                       'border-radius : 2px;'
                       'color : black;'
                       'padding : 4px;'
                       'font-weight : bold;'
                       '}'
                       
                       'QPushButton:pressed {'
                       'background-color : dimgray;'
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
        
        # create popup window for processed data
        self.analysis_popup = sp.ProcessedDirectorySelectionPopup(go_to_last=False, parent=self)
        self.analysis_popup.ab.option1_btn.clicked.connect(self.ch_selection_popup)
        self.analysis_popup.ab.option2_btn.clicked.connect(self.classify_ds_popup)
        
        # create main buttons
        self.base_folder_btn = QtWidgets.QPushButton('Base folders')
        self.base_folder_btn.setStyleSheet(mode_btn_ss)
        self.view_params_btn = QtWidgets.QPushButton('View parameters')
        self.view_params_btn.setStyleSheet(mode_btn_ss)
        self.probe_btn =  QtWidgets.QPushButton('Create probe')
        self.probe_btn.setStyleSheet(mode_btn_ss)
        self.process_btn = QtWidgets.QPushButton('Process raw data')
        self.process_btn.setStyleSheet(mode_btn_ss)
        self.analyze_btn = QtWidgets.QPushButton('Analyze data')
        self.analyze_btn.setStyleSheet(mode_btn_ss)
        
        
        # connect to functions
        self.process_btn.clicked.connect(self.raw_data_popup)
        self.analyze_btn.clicked.connect(self.processed_data_popup)
        self.probe_btn.clicked.connect(self.probe_popup)
        self.view_params_btn.clicked.connect(self.view_param_popup)
        self.base_folder_btn.clicked.connect(self.base_folder_popup)
        
        self.centralLayout.addWidget(self.base_folder_btn)
        self.centralLayout.addWidget(self.view_params_btn)
        self.centralLayout.addWidget(self.probe_btn)
        self.centralLayout.addWidget(self.process_btn)
        self.centralLayout.addWidget(self.analyze_btn)
        
        self.setCentralWidget(self.centralWidget)
        
    def probe_popup(self):
        probe = probegod.run_probe_window(accept_visible=False, title='Create probe', parent=self)
        
        
    def raw_data_popup(self, init_raw_ddir='base', init_save_ddir='base'):
        """ Select raw data for processing """
        popup = sp.RawDirectorySelectionPopup(init_raw_ddir, init_save_ddir, parent=self)
        res = popup.exec()
        if not res:
            return
        
        # save_ddir = popup.processed_ddir
        # ephys.load_event_dfs(save_ddir, 'ds', -1)
        #pdb.set_trace()
        
        # # get paths to raw data directory, save directory, and probe file
        # raw_ddir = popup.raw_ddir         # raw data directory
        # save_ddir = popup.processed_ddir  # processed data location
        # probe = popup.probe               # channel map for raw signal array
        # nch = probe.get_contact_count()
        # PARAMS = dict(self.PARAMS)
        
        # # load raw files and recording info, make sure probe maps onto raw signals
        # (pri_mx, aux_mx), fs = dp.load_raw_data(raw_ddir) # removed info
        # num_channels = pri_mx.shape[0]
        # dur = pri_mx.shape[1] / fs
        # if num_channels % nch > 0:
        #     msg = f'ERROR: Cannot map {nch} probe contacts to {num_channels} raw data signals'
        #     res = gi.MsgboxError(msg, parent=self)
        #     return
        # lfp_fs = PARAMS.pop('lfp_fs')
        # #info['lfp_fs'] = lfp_fs = PARAMS.pop('lfp_fs')
        # #info['lfp_units'] = 'mV'
        
        # #lfp0 = pri_mx[0:32, :]
        # #lfp1 = pri_mx[32:, :]
        # #lfp_list = [lfp0, lfp1]
        
        # # create probe group for recording
        # probe_group = ephys.make_probe_group(probe, int(num_channels / nch))
        
        # # get LFP array for each probe
        # lfp_list = dp.extract_data_by_probe(pri_mx, probe_group, fs=fs, lfp_fs=lfp_fs)#, fs=info.fs, lfp_fs=lfp_fs,
        #                                     #units=info.units, lfp_units=info.lfp_units)
        # lfp_time = np.linspace(0, dur, int(lfp_list[0].shape[1]))
        
        # # process data by probe, save files in target directory 
        # dp.process_all_probes(lfp_list, lfp_time, lfp_fs, PARAMS, save_ddir)
        # prif.write_probeinterface(Path(save_ddir, 'probe_group'), probe_group)
        
        # # process auxilary channels
        # if aux_mx.size > 0:
        #     aux_dn = dp.extract_data(aux_mx, np.arange(aux_mx.shape[0]), fs=fs, 
        #                              lfp_fs=lfp_fs, units='V', lfp_units='V')
        #     #np.save(Path(save_ddir, 'AUX.npy'), aux_dn)
        #     aux_dlg = gi.AuxDialog(aux_dn.shape[0], parent=self)
        #     res = aux_dlg.exec()
        #     if res:
        #         for i,fname in enumerate(aux_dlg.aux_files):
        #             if fname != '':
        #                 np.save(Path(save_ddir, fname), aux_dn[i])
            
        # # pop-up messagebox appears when processing is complete
        # msgbox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, '',
        #                                'Data processing complete!', QtWidgets.QMessageBox.Ok, self)
        # # set check icon
        # chk_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        # px_size = msgbox.findChild(QtWidgets.QLabel, 'qt_msgboxex_icon_label').pixmap().size()
        # msgbox.setIconPixmap(chk_icon.pixmap(px_size))
        # msgbox.exec()
        
        
    def processed_data_popup(self, _, init_ddir=None, go_to_last=True):
        """ Show processed data options """
        self.analysis_popup.show()
        self.analysis_popup.raise_()
        
    
    def ch_selection_popup(self):
        """ Launch event channel selection window """
        ddir = self.analysis_popup.ddir
        probe_list = self.analysis_popup.probe_group.probes
        iprb = self.analysis_popup.probe_idx
        self.ch_selection_dlg = ChannelSelectionWindow(ddir, probe_list=probe_list, 
                                                       iprb=iprb, parent=self.analysis_popup)
        self.ch_selection_dlg.show()
        self.ch_selection_dlg.raise_()
        _ = self.ch_selection_dlg.exec()
        # check for updated files, enable/disable analysis options
        iprb = int(self.ch_selection_dlg.iprb)
        self.analysis_popup.ab.ddir_toggled(ddir, iprb)
        
    
    def classify_ds_popup(self):
        """ Launch DS analysis window """
        ddir = self.analysis_popup.ddir
        iprb = self.analysis_popup.probe_idx
        PARAMS = ephys.load_recording_params(ddir)
        self.classify_ds_dlg = DS_CSDWindow(ddir, iprb, self.PARAMS, parent=self.analysis_popup)
        self.classify_ds_dlg.show()
        self.classify_ds_dlg.raise_()
        _ = self.classify_ds_dlg.exec()
        # check for updated files, enable/disable analysis options
        self.analysis_popup.ab.ddir_toggled(ddir)
        
        
    def view_param_popup(self):
        """ View default parameters """
        params = ephys.read_param_file()
        keys, vals = zip(*params.items())
        vstr = [*map(str,vals)]
        klens = [*map(len, keys)]; kmax=max(klens)
        padk = [*map(lambda k: k + '_'*(kmax-len(k))+':', keys)]
        html = ['<pre>'+k+v+'</pre>' for k,v in zip(padk,vstr)]
        text = '<h3><tt>DEFAULT PARAMETERS</tt></h3>' + '<hr>' + ''.join(html)
        textwidget = QtWidgets.QTextEdit(text)
        
        # create popup window for text widget
        dlg = QtWidgets.QDialog(self)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(textwidget)
        qrect = pyfx.ScreenRect(perc_height=0.5, perc_width=0.3, keep_aspect=False)
        dlg.setGeometry(qrect)
        dlg.show()
        dlg.raise_()
    
    
    def base_folder_popup(self):
        """ View or change base data directories """
        dlg = sp.BaseFolderPopup(parent=self)
        
        qrect = pyfx.ScreenRect(perc_width=0.5, perc_height=0.5, keep_aspect=False)
        #dlg.setSizeHint(qrect.width(), qrect.height())
        dlg.show()
        dlg.raise_()
        dlg.exec()
        
    
    def center_window(self):
        """ Move GUI to center of screen """
        qrect = self.frameGeometry()  # proxy rectangle for window with frame
        screen_rect = QtWidgets.QDesktopWidget().screenGeometry()
        qrect.moveCenter(screen_rect.center())  # move center of qr to center of screen
        self.move(qrect.topLeft())

def run():
    args = list(sys.argv)
    if len(args) == 1:
        args.append('')
    app = QtWidgets.QApplication(args)
    app.setStyle('Fusion')
    app.setQuitOnLastWindowClosed(True)
    
    if not Path('default_params.txt').exists():
        # create settings text file from hidden params
        _ddict = np.load('.default_params.npy', allow_pickle=True).item()
        ephys.write_param_file(_ddict)
        
    if not Path('default_folders.txt').exists():
        # set data base directories to the app folder
        raw_ddir = str(app_ddir)
        processed_ddir = str(app_ddir)
        probe_ddir = str(Path(app_ddir, 'probe_configs'))  # create probe folder
        if not Path(probe_ddir).exists():
            os.mkdir(probe_ddir)
        # default probe file
        probe_files = ephys.get_probe_filepaths(probe_ddir)
        if len(probe_files) > 0:
            probe_file = str(Path(app_ddir, probe_files[0]))
        else:
            probe_file = ''
        ephys.write_base_dirs([raw_ddir, processed_ddir, probe_ddir, probe_file])
        
        # prompt user to customize default directories
        print('executing thing!')
        dlg = sp.thing()
        dlg.exec()
        
    w = hippos()
    #w = ChannelSelectionWindow(ddir, 0)
    
    w.show()
    w.raise_()
    sys.exit(app.exec())
    
if __name__ == '__main__':
    run()
#%%
# if __name__ == '__main__':
#     args = list(sys.argv)
#     if len(args) == 1:
#         args.append('')
        
#     # TO-DO
#     # add notes section
#     # add behavior data
    
#     #ddir = '/Users/amandaschott/Library/CloudStorage/Dropbox/Farrell_Programs/saved_data/NN_JG023_2probes2/'
    
#     app = QtWidgets.QApplication(args)
#     app.setStyle('Fusion')
#     app.setQuitOnLastWindowClosed(True)
#     startup()
    
#     w = hippos()
#     #w = ChannelSelectionWindow(ddir, 0)
    
#     w.show()
#     w.raise_()
#     sys.exit(app.exec())