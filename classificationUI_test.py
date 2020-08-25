import os
import sys
import cv2
import time
import logging
import tensorflow as tf
import threading
import ctypes
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
from keras import backend as K # Tensor tensor 오류 수정 부분
import keras
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

class MySignal(QObject):
    signal = pyqtSignal()
    def run(self):
        self.signal.emit()

class ClassificationUI(QDialog):
    def __init__(self, trainingTest):
        super().__init__()
        self.plot_losses = TrainingPlot()
        self.trainingTest = trainingTest

        self.change_button_signal = MySignal()
        self.messageBox_signal = MySignal()
        self.messageBox_signal.signal.connect(self.messageBox)

        self.open_other_window_signal = MySignal()
        # self.open_other_window_signal.signal.connect(self.open_other_window)

        self.setWindowTitle("Training")
        self.setGeometry(420, 220, 1200, 600)
        self.setFixedSize(1200, 480)

        groupBox = QGroupBox()

        self.pretrained_radio = QRadioButton("Pretrained Model", self)
        self.pretrained_radio.setChecked(True)

        pretrained_hbox = QHBoxLayout()
        pretrained_hbox.addWidget(self.pretrained_radio)

        self.pretrained_combo = QComboBox()
        self.pretrained_combo.addItems(['VGG16', 'RESNET50'])

        pretrained_hbox.addWidget(self.pretrained_combo)

        self.customed_radio = QRadioButton("Customed Model", self)
        self.customed_radio.toggled.connect(self.buttonClicked)

        customed_hbox = QHBoxLayout()
        customed_hbox.addWidget(self.customed_radio)

        self.customed_button = QPushButton("Set", self)
        self.customed_button.setEnabled(False)
        self.customed_button.setFixedWidth(222)
        self.customed_button.clicked.connect(self.buttonClicked)

        customed_hbox.addWidget(self.customed_button)

        self.default_check = QCheckBox("Default", self)
        self.default_check.stateChanged.connect(self.buttonClicked)
        self.default_check.setFixedWidth(68)
        self.default_check.setEnabled(False)

        customed_hbox.addWidget(self.default_check)

        self.trainingDataSet_label = QLabel("Training Data Set : ", self)
        self.trainingDataSet_label.setFixedSize(600, 20)
        self.trainingDataSet_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        trainingDataSet_hbox = QHBoxLayout()
        trainingDataSet_hbox.addWidget(self.trainingDataSet_label)

        self.trainingDataSet_button = QPushButton("Set Training Data Set Directory")
        self.trainingDataSet_button.clicked.connect(self.buttonClicked)

        self.validationDataSet_label = QLabel("Validation Data Set : ", self)
        self.validationDataSet_label.setFixedSize(600, 20)
        self.validationDataSet_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        validationDataSet_hbox = QHBoxLayout()
        validationDataSet_hbox.addWidget(self.validationDataSet_label)

        self.validationDataSet_button = QPushButton("Set Validation Data Set Directory")
        self.validationDataSet_button.clicked.connect(self.buttonClicked)

        self.modelName_label = QLabel("Model Name : ", self)
        self.modelName_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        modelName_hbox = QHBoxLayout()
        modelName_hbox.addWidget(self.modelName_label)

        self.modelName_line = QLineEdit(self)
        self.modelName_line.setReadOnly(True)
        self.modelName_line.setFixedHeight(27)

        modelName_hbox.addWidget(self.modelName_line)

        self.save_button = QPushButton("Save as..", self)
        self.save_button.setFixedHeight(33)
        self.save_button.clicked.connect(self.buttonClicked)

        modelName_hbox.addWidget(self.save_button)

        model_layout1 = QVBoxLayout()
        model_layout1.addLayout(pretrained_hbox)
        model_layout1.addLayout(customed_hbox)
        model_layout1.addSpacing(36)
        model_layout1.addLayout(trainingDataSet_hbox)
        model_layout1.addWidget(self.trainingDataSet_button)
        model_layout1.addSpacing(36)
        model_layout1.addLayout(validationDataSet_hbox)
        model_layout1.addWidget(self.validationDataSet_button)
        model_layout1.addSpacing(36)
        model_layout1.addLayout(modelName_hbox)

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        groupBox.setLayout(groupBox_layout)

        base_layout.addWidget(groupBox)

        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()

        hbox1.addLayout(model_layout1)

        batch_epoch_gbox = QGridLayout()

        self.batch_label = QLabel("Batch Size : ", self)
        self.batch_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.batch_label.setFixedSize(110, 27)

        batch_epoch_gbox.addWidget(self.batch_label, 0, 1)

        self.batch_line = QLineEdit(self)
        self.batch_line.setReadOnly(False)
        self.batch_line.setFixedSize(60, 27)

        batch_epoch_gbox.addWidget(self.batch_line, 0, 2)

        self.epoch_label = QLabel("Epoch : ", self)
        self.epoch_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.epoch_label.setFixedSize(100, 27)

        batch_epoch_gbox.addWidget(self.epoch_label, 0, 3)

        self.epoch_line = QLineEdit(self)
        self.epoch_line.setReadOnly(False)
        self.epoch_line.setFixedSize(60, 27)

        batch_epoch_gbox.addWidget(self.epoch_line, 0, 4)

        self.batch_epoch_empty1 = QLabel("", self)
        self.batch_epoch_empty2 = QLabel("", self)
        self.batch_epoch_empty3 = QLabel("", self)

        batch_epoch_gbox.addWidget(self.batch_epoch_empty1, 0, 0)
        batch_epoch_gbox.addWidget(self.batch_epoch_empty2, 0, 5)
        batch_epoch_gbox.addWidget(self.batch_epoch_empty3, 0, 6)

        inputShape_gbox = QGridLayout()

        self.inputShape_empty1 = QLabel("", self)
        self.inputShape_empty1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_empty1, 0, 0)

        self.inputShape_empty2 = QLabel("", self)
        self.inputShape_empty2.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_empty2, 0, 1)

        self.inputShape_label = QLabel("Input Shape : ", self)
        self.inputShape_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_label, 0, 2)

        self.inputShape_line1 = QLineEdit(self)
        self.inputShape_line1.setReadOnly(False)
        self.inputShape_line1.setText("224")
        self.inputShape_line1.setFixedSize(100, 27)

        inputShape_gbox.addWidget(self.inputShape_line1, 0, 3)

        self.inputShape_label_x = QLabel(" x ", self)
        self.inputShape_label_x.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_label_x, 0, 4)

        self.inputShape_line2 = QLineEdit(self)
        self.inputShape_line2.setReadOnly(False)
        self.inputShape_line2.setText("224")
        self.inputShape_line2.setFixedSize(100, 27)

        inputShape_gbox.addWidget(self.inputShape_line2, 0, 5)

        self.inputShape_empty3 = QLabel("", self)
        self.inputShape_empty3.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_empty3, 0, 6)

        self.inputShape_combo = QComboBox()
        self.inputShape_combo.addItems(["RGB", "Gray Scale"])
        self.inputShape_combo.setFixedSize(100, 27)

        inputShape_gbox.addWidget(self.inputShape_combo, 0, 7)

        self.inputShape_empty4 = QLabel("", self)
        self.inputShape_empty4.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_empty4, 0, 8)

        self.inputShape_empty5 = QLabel("", self)
        self.inputShape_empty5.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_empty5, 0, 9)

        imageGenerator_hbox = QHBoxLayout()

        self.imageGenerator_check = QCheckBox("Image Data Generator", self)
        self.imageGenerator_check.stateChanged.connect(self.buttonClicked)

        self.imageGenerator_button = QPushButton("Set Image Data Generator Parameters", self)
        self.imageGenerator_button.clicked.connect(self.buttonClicked)
        self.imageGenerator_button.setEnabled(False)

        imageGenerator_hbox.addWidget(self.imageGenerator_check)

        fineTuning_hbox = QHBoxLayout()

        self.fineTuning_check = QCheckBox("Fine Tuning", self)
        self.fineTuning_check.stateChanged.connect(self.buttonClicked)

        self.fineTuning_button = QPushButton("Set Fine Tuning Method", self)
        self.fineTuning_button.setEnabled(False)
        self.fineTuning_button.clicked.connect(self.buttonClicked)

        fineTuning_hbox.addWidget(self.fineTuning_check)

        self.optimizer_label = QLabel("Optimizer : ", self)
        self.optimizer_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        optimizer_learningRate_gbox = QGridLayout()
        optimizer_learningRate_gbox.addWidget(self.optimizer_label, 0, 1)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "Adagrad", "AdaDelta", "SGD", "Nadam", "RMSProp"])
        self.optimizer_combo.setFixedSize(100, 27)

        optimizer_learningRate_gbox.addWidget(self.optimizer_combo, 0, 2)

        self.learningRate_label = QLabel("Learning Rate : ", self)
        self.learningRate_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        optimizer_learningRate_gbox.addWidget(self.learningRate_label, 0, 4)

        self.learningRate_line = QLineEdit(self)
        self.learningRate_line.setReadOnly(False)
        self.learningRate_line.setFixedSize(60, 27)

        optimizer_learningRate_gbox.addWidget(self.learningRate_line, 0, 5)

        self.optimizer_learningRate_empty1 = QLabel("", self)
        self.optimizer_learningRate_empty2 = QLabel("", self)
        self.optimizer_learningRate_empty3 = QLabel("", self)

        optimizer_learningRate_gbox.addWidget(self.optimizer_learningRate_empty1, 0, 0)
        optimizer_learningRate_gbox.addWidget(self.optimizer_learningRate_empty2, 0, 3)
        optimizer_learningRate_gbox.addWidget(self.optimizer_learningRate_empty3, 0, 6)

        start_hbox = QHBoxLayout()

        self.confirm_cancel_empty1 = QLabel("", self)
        self.confirm_cancel_empty1.setFixedWidth(90)
        self.confirm_cancel_empty1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self.confirm)
        self.confirm_button.setFixedSize(120, 40)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(True)
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setFixedSize(120, 40)

        self.confirm_cancel_empty2 = QLabel("", self)
        self.confirm_cancel_empty2.setFixedWidth(90)
        self.confirm_cancel_empty2.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        start_hbox.addWidget(self.confirm_cancel_empty1)
        start_hbox.addWidget(self.confirm_button)
        start_hbox.addWidget(self.cancel_button)
        start_hbox.addWidget(self.confirm_cancel_empty2)

        result_hbox = QHBoxLayout()

        model_layout2 = QVBoxLayout()
        model_layout2.addLayout(batch_epoch_gbox)
        model_layout2.addSpacing(9)
        model_layout2.addLayout(inputShape_gbox)
        model_layout2.addSpacing(18)
        model_layout2.addLayout(imageGenerator_hbox)
        model_layout2.addWidget(self.imageGenerator_button)
        model_layout2.addSpacing(18)
        model_layout2.addLayout(fineTuning_hbox)
        model_layout2.addWidget(self.fineTuning_button)
        model_layout2.addSpacing(18)
        model_layout2.addLayout(optimizer_learningRate_gbox)
        model_layout2.addSpacing(18)
        model_layout2.addLayout(start_hbox)
        model_layout2.addSpacing(18)

        hbox2.addLayout(model_layout2)

        hbox = QHBoxLayout()
        hbox.addLayout(hbox1)
        hbox.addLayout(hbox2)

        groupBox_layout.addLayout(hbox)

        base_layout.addLayout(result_hbox)

        self.training_flag = False
        self.model_path_trained = None

        self.fineTuningSettingDialog = FineTuningSettingDialog()
        self.fineTuningSettingDialog.num_of_freezing.setValue(0)

        self.denseLayerSettingWindow = self.fineTuningSettingDialog.denseLayerSettingWindow
        self.dropoutSettingWindow = self.fineTuningSettingDialog.dropoutSettingWindow

        self.customedModelSettingDialog = CustomedModelSettingDialog()

        self.convolutionLayerSettingWindow_customed = self.customedModelSettingDialog.convolutionLayerSettingWindow
        self.maxpoolingLayerSettingWindow_customed = self.customedModelSettingDialog.maxpoolingLayerSettingWindow
        self.denseLayerSettingWindow_customed = self.customedModelSettingDialog.denseLayerSettingWindow
        self.dropoutSettingWindow_customed = self.customedModelSettingDialog.dropoutSettingWindow

        self.imageDataGeneratorSettingWindow = ImageDataGeneratorSettingDialog()
        self.imageDataGeneratorSettingWindow.rotationRangeText.setText("0")
        self.imageDataGeneratorSettingWindow.widthShiftRangeText.setText("0.0")
        self.imageDataGeneratorSettingWindow.heightShiftRangeText.setText("0.0")
        self.imageDataGeneratorSettingWindow.shearRangeText.setText("0.0")
        self.imageDataGeneratorSettingWindow.zoomRangeText.setText("0.0")
        self.imageDataGeneratorSettingWindow.rescaleText.setText("0.0")

        self.pretrained_radio_bool = True
        self.pretrained_combo_buffer = "VGG16"
        self.customed_radio_bool = False
        self.trainingDataSet_label_buffer = "Training Data set : "
        self.trainingDataSet_directoryName_buffer = ""
        self.validationDataSet_label_buffer = "Validation Data set : "
        self.validationDataSet_directoryName_buffer = ""
        self.modelName_line_buffer = ""
        self.batch_line_buffer = ""
        self.epoch_line_buffer = ""
        self.inputShape_line1_buffer = "224"
        self.inputShape_line2_buffer = "224"
        self.inputShape_combo_buffer = "RGB"
        self.optimizer_combo_buffer = "Adam"
        self.learningRate_line_buffer = ""
        self.default_check_bool = False
        self.imageGenerator_check_bool = False
        self.fineTuning_check_bool = False
        self.num_of_freezing_buffer = 0
        self.horizontalFlip_buffer = False
        self.verticalFlip_buffer = False
        self.rotationRangeText_buffer = "0"
        self.widthShiftRangeText_buffer = "0.0"
        self.heightShiftRangeText_buffer = "0.0"
        self.shearRangeText_buffer = "0.0"
        self.zoomRangeText_buffer = "0.0"
        self.rescaleText_buffer = "0.0"

    def buttonClicked(self):
        if self.customed_radio.isChecked():
            self.pretrained_combo.setEnabled(False)
            self.default_check.setEnabled(True)
            self.fineTuning_check.setChecked(False)
            self.fineTuning_check.setEnabled(False)
            self.fineTuningSettingDialog.num_of_freezing.setValue(0)
            if self.default_check.isChecked() == False:
                self.customed_button.setEnabled(True)
                if self.sender().text() == "Set":
                    self.customedModelSettingDialog.exec_()
            else:
                self.customed_button.setEnabled(False)
        else:
            self.pretrained_combo.setEnabled(True)
            self.fineTuning_check.setEnabled(True)
            self.customed_button.setEnabled(False)
            self.default_check.setEnabled(False)
            self.fineTuningSettingDialog.num_of_freezing.setMaximum(16) \
                if self.pretrained_combo.currentText() == "VGG16" else self.fineTuningSettingDialog.num_of_freezing.setMaximum(50)

        if self.imageGenerator_check.isChecked():
            self.imageGenerator_button.setEnabled(True)
            if self.sender().text() == "Set Image Data Generator Parameters":
                self.imageDataGeneratorSettingWindow.exec_()
        else:
            self.imageGenerator_button.setEnabled(False)

        if self.fineTuning_check.isChecked():
            self.fineTuning_button.setEnabled(True)
            if self.sender().text() == "Set Fine Tuning Method":
                self.fineTuningSettingDialog.exec_()
        else:
            self.fineTuning_button.setEnabled(False)

        if self.sender().text() == "Set Training Data Set Directory":
            self.trainingDataSet_directoryName = QFileDialog.getExistingDirectory(self, "Open Training Data Set Directory")
            self.trainingDataSet_label.setText("Training Data Set : " + self.trainingDataSet_directoryName)

        elif self.sender().text() == "Set Validation Data Set Directory":
            self.validationDataSet_directoryName = QFileDialog.getExistingDirectory(self, "Open Validation Data Set Directory")
            self.validationDataSet_label.setText("Validation Data Set : " + self.validationDataSet_directoryName)

        elif self.sender().text() == "Save as..":
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.model_path, extension = QFileDialog.getSaveFileName(self, "Save", "", "h5 file(*.h5)", options=options)
            self.model_path_splited = self.model_path.split("/")
            self.modelName_line.setText(self.model_path_splited[-1])

        if self.trainingDataSet_label.text() != "Training Data Set : " and self.validationDataSet_label.text() != "Validation Data Set : ":
            self.confirm_button.setEnabled(True)
        else:
            self.confirm_button.setEnabled(False)

    def closeEvent(self, event):
        self.pretrained_radio.setChecked(self.pretrained_radio_bool)
        self.pretrained_combo.setCurrentText(self.pretrained_combo_buffer)
        self.customed_radio.setChecked(self.customed_radio_bool)
        self.trainingDataSet_label.setText(self.trainingDataSet_label_buffer)
        self.validationDataSet_label.setText(self.validationDataSet_label_buffer)
        self.modelName_line.setText(self.modelName_line_buffer)
        self.batch_line.setText(self.batch_line_buffer)
        self.epoch_line.setText(self.epoch_line_buffer)
        self.inputShape_line1.setText(self.inputShape_line1_buffer)
        self.inputShape_line2.setText(self.inputShape_line2_buffer)
        self.inputShape_combo.setCurrentText(self.inputShape_combo_buffer)
        self.optimizer_combo.setCurrentText(self.optimizer_combo_buffer)
        self.learningRate_line.setText(self.learningRate_line_buffer)
        self.default_check.setChecked(self.default_check_bool)
        self.imageGenerator_check.setChecked(self.imageGenerator_check_bool)
        self.fineTuning_check.setChecked(self.fineTuning_check_bool)
        self.fineTuningSettingDialog.num_of_freezing.setValue(self.num_of_freezing_buffer)
        self.imageDataGeneratorSettingWindow.horizontalFlip.setChecked(self.horizontalFlip_buffer)
        self.imageDataGeneratorSettingWindow.verticalFlip.setChecked(self.verticalFlip_buffer)
        self.imageDataGeneratorSettingWindow.rotationRangeText.setText(self.rotationRangeText_buffer)
        self.imageDataGeneratorSettingWindow.widthShiftRangeText.setText(self.widthShiftRangeText_buffer)
        self.imageDataGeneratorSettingWindow.heightShiftRangeText.setText(self.heightShiftRangeText_buffer)
        self.imageDataGeneratorSettingWindow.shearRangeText.setText(self.shearRangeText_buffer)
        self.imageDataGeneratorSettingWindow.zoomRangeText.setText(self.zoomRangeText_buffer)
        self.imageDataGeneratorSettingWindow.rescaleText.setText(self.rescaleText_buffer)

    def cancel(self):
        self.pretrained_radio.setChecked(self.pretrained_radio_bool)
        self.pretrained_combo.setCurrentText(self.pretrained_combo_buffer)
        self.customed_radio.setChecked(self.customed_radio_bool)
        self.trainingDataSet_label.setText(self.trainingDataSet_label_buffer)
        self.validationDataSet_label.setText(self.validationDataSet_label_buffer)
        self.modelName_line.setText(self.modelName_line_buffer)
        self.batch_line.setText(self.batch_line_buffer)
        self.epoch_line.setText(self.epoch_line_buffer)
        self.inputShape_line1.setText(self.inputShape_line1_buffer)
        self.inputShape_line2.setText(self.inputShape_line2_buffer)
        self.inputShape_combo.setCurrentText(self.inputShape_combo_buffer)
        self.optimizer_combo.setCurrentText(self.optimizer_combo_buffer)
        self.learningRate_line.setText(self.learningRate_line_buffer)
        self.default_check.setChecked(self.default_check_bool)
        self.imageGenerator_check.setChecked(self.imageGenerator_check_bool)
        self.fineTuning_check.setChecked(self.fineTuning_check_bool)
        self.fineTuningSettingDialog.num_of_freezing.setValue(self.num_of_freezing_buffer)
        self.imageDataGeneratorSettingWindow.horizontalFlip.setChecked(self.horizontalFlip_buffer)
        self.imageDataGeneratorSettingWindow.verticalFlip.setChecked(self.verticalFlip_buffer)
        self.imageDataGeneratorSettingWindow.rotationRangeText.setText(self.rotationRangeText_buffer)
        self.imageDataGeneratorSettingWindow.widthShiftRangeText.setText(self.widthShiftRangeText_buffer)
        self.imageDataGeneratorSettingWindow.heightShiftRangeText.setText(self.heightShiftRangeText_buffer)
        self.imageDataGeneratorSettingWindow.shearRangeText.setText(self.shearRangeText_buffer)
        self.imageDataGeneratorSettingWindow.zoomRangeText.setText(self.zoomRangeText_buffer)
        self.imageDataGeneratorSettingWindow.rescaleText.setText(self.rescaleText_buffer)

        self.close()

    def setStop(self):
        if self.pretrained_radio.isChecked():
            self.pretrained_combo.setEnabled(True)
        elif self.customed_radio.isChecked():
            self.customed_button.setEnabled(True)

        if self.imageGenerator_check.isChecked():
            self.imageGenerator_button.setEnabled(True)

        if self.fineTuning_check.isChecked():
            self.fineTuning_button.setEnabled(True)
            self.default_check.setEnabled(True)

        self.pretrained_radio.setEnabled(True)
        self.customed_radio.setEnabled(True)
        self.trainingDataSet_button.setEnabled(True)
        self.validationDataSet_button.setEnabled(True)
        self.save_button.setEnabled(True)

        self.modelName_line.setEnabled(True)

        self.batch_line.setEnabled(True)
        self.epoch_line.setEnabled(True)

        self.inputShape_line1.setEnabled(True)
        self.inputShape_line2.setEnabled(True)
        self.inputShape_combo.setEnabled(True)

        self.imageGenerator_check.setEnabled(True)
        self.fineTuning_check.setEnabled(True)

        self.optimizer_combo.setEnabled(True)
        self.learningRate_line.setEnabled(True)

    def setRunning(self):

        self.pretrained_radio.setEnabled(False)
        self.pretrained_combo.setEnabled(False)

        self.customed_radio.setEnabled(False)
        self.customed_button.setEnabled(False)
        self.default_check.setEnabled(False)

        self.trainingDataSet_button.setEnabled(False)
        self.validationDataSet_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.modelName_line.setEnabled(False)

        self.batch_line.setEnabled(False)
        self.epoch_line.setEnabled(False)

        self.inputShape_line1.setEnabled(False)
        self.inputShape_line2.setEnabled(False)
        self.inputShape_combo.setEnabled(False)

        self.imageGenerator_check.setEnabled(False)
        self.imageGenerator_button.setEnabled(False)

        self.fineTuning_check.setEnabled(False)
        self.fineTuning_button.setEnabled(False)

        self.optimizer_combo.setEnabled(False)
        self.learningRate_line.setEnabled(False)

    def print_state(self):
        import numpy as np
        import pandas as pd
        import keras
        import tensorflow as tf
        from IPython.display import display
        import PIL
        from tensorflow.python.client import device_lib

        for i in device_lib.list_local_devices():
            tempi = str(i)
            if "physical_device" in tempi:
                if "GPU" in tempi:
                    print("[ Using GPU ]")
                else:
                    print("[ Using CPU ]")
                print(tempi[tempi.find("physical_device"):])

    # def terminate_thread(self, thread):
    #     if not thread.isAlive():
    #         return
    #
    #     exc = ctypes.py_object(SystemExit)
    #     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
    #         ctypes.c_long(thread.ident), exc)
    #     if res == 0:
    #         raise ValueError("nonexistent thread id")
    #     elif res > 1:
    #         ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
    #         raise SystemError("PyThreadState_SetAsyncExc failed")

    def confirm(self):
        if self.customed_radio.isChecked() == True and self.default_check.isChecked() == False \
                and len(self.customedModelSettingDialog.listWidget) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Set Customized Layers")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.modelName_line.text() == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Set Model Name")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.batch_line.text().isdigit() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Batch Size is expected to be a integer value")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.epoch_line.text().isdigit() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Epoch is expected to be a integer value")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.inputShape_line1.text().isdigit() == False or self.inputShape_line2.text().isdigit() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Input Size is expected to be integer by integer")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.isNumber(self.learningRate_line.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Learning Rate is expected to be a positive float or integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
        elif self.isNumber(self.learningRate_line.text()):
            if float(self.learningRate_line.text()) <= 0.0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Learning Rate is expected to be a positive float or integer value")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                self.trainingTest.tableWidget.setItem(0, 0, QTableWidgetItem("Pretrained model"))
                self.trainingTest.tableWidget.setItem(1, 0, QTableWidgetItem("Fine Tuning"))
                self.trainingTest.tableWidget.setItem(2, 0, QTableWidgetItem("Number of trainable layers"))
                self.trainingTest.tableWidget.item(2, 0).setToolTip("Number of trainable layers")
                self.trainingTest.tableWidget.setItem(3, 0, QTableWidgetItem("Batch size"))
                self.trainingTest.tableWidget.setItem(4, 0, QTableWidgetItem("Epoch"))
                self.trainingTest.tableWidget.setItem(5, 0, QTableWidgetItem("Input size"))
                self.trainingTest.tableWidget.setItem(6, 0, QTableWidgetItem("Input channel"))
                self.trainingTest.tableWidget.setItem(7, 0, QTableWidgetItem("Optimizer"))
                self.trainingTest.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.trainingTest.tableWidget.setItem(9, 0, QTableWidgetItem("Image augmentation"))
                self.trainingTest.tableWidget.setItem(10, 0, QTableWidgetItem("Horizontal flip"))
                self.trainingTest.tableWidget.setItem(11, 0, QTableWidgetItem("Vertical flip"))
                self.trainingTest.tableWidget.setItem(12, 0, QTableWidgetItem("Rotation range"))
                self.trainingTest.tableWidget.setItem(13, 0, QTableWidgetItem("Width shift range"))
                self.trainingTest.tableWidget.setItem(14, 0, QTableWidgetItem("Height shift range"))
                self.trainingTest.tableWidget.setItem(15, 0, QTableWidgetItem("Shear range"))
                self.trainingTest.tableWidget.setItem(16, 0, QTableWidgetItem("Zoom range"))
                self.trainingTest.tableWidget.setItem(17, 0, QTableWidgetItem("Rescale"))
                self.trainingTest.tableWidget.setItem(17, 1, QTableWidgetItem("None"))

                self.trainingTest.tableWidget.setItem(0, 1, QTableWidgetItem(self.pretrained_combo.currentText() if self.pretrained_radio.isChecked() else "None"))
                self.trainingTest.tableWidget.setItem(1, 1, QTableWidgetItem("True" if self.fineTuning_check.isChecked() else "False"))
                if self.fineTuning_check.isChecked():
                    self.trainingTest.tableWidget.setItem(2, 1, QTableWidgetItem(self.fineTuningSettingDialog.num_of_freezing.text()))
                    self.trainingTest.tableWidget.setRowHidden(2, False)
                else:
                    self.trainingTest.tableWidget.setRowHidden(2, True)
                self.trainingTest.tableWidget.setItem(3, 1, QTableWidgetItem(self.batch_line.text()))
                self.trainingTest.tableWidget.setItem(4, 1, QTableWidgetItem(self.epoch_line.text()))
                self.trainingTest.tableWidget.setItem(5, 1, QTableWidgetItem(self.inputShape_line1.text() + "×" + self.inputShape_line2.text()))
                self.trainingTest.tableWidget.setItem(6, 1, QTableWidgetItem(self.inputShape_combo.currentText()))
                self.trainingTest.tableWidget.setItem(7, 1, QTableWidgetItem(self.optimizer_combo.currentText()))
                self.trainingTest.tableWidget.setItem(8, 1, QTableWidgetItem(self.learningRate_line.text()))
                self.trainingTest.tableWidget.setItem(9, 1, QTableWidgetItem("True" if self.imageGenerator_check.isChecked() else "False"))
                if self.imageGenerator_check.isChecked():
                    self.trainingTest.tableWidget.item(9, 0).setBackground(QtGui.QColor(58, 58, 58))
                    self.trainingTest.tableWidget.item(9, 1).setBackground(QtGui.QColor(58, 58, 58))
                    self.trainingTest.tableWidget.setItem(10, 1, QTableWidgetItem("True" if self.imageDataGeneratorSettingWindow.horizontalFlip.isChecked() else "False"))
                    # self.trainingTest.tableWidget.item(10, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(10, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(10, False)
                    self.trainingTest.tableWidget.setItem(11, 1, QTableWidgetItem("True" if self.imageDataGeneratorSettingWindow.verticalFlip.isChecked() else "False"))
                    # self.trainingTest.tableWidget.item(11, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(11, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(11, False)
                    self.trainingTest.tableWidget.setItem(12, 1, QTableWidgetItem(self.imageDataGeneratorSettingWindow.rotationRangeText.text()))
                    # self.trainingTest.tableWidget.item(12, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(12, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(12, False)
                    self.trainingTest.tableWidget.setItem(13, 1, QTableWidgetItem(self.imageDataGeneratorSettingWindow.widthShiftRangeText.text()))
                    # self.trainingTest.tableWidget.item(13, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(13, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(13, False)
                    self.trainingTest.tableWidget.setItem(14, 1, QTableWidgetItem(self.imageDataGeneratorSettingWindow.heightShiftRangeText.text()))
                    # self.trainingTest.tableWidget.item(14, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(14, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(14, False)
                    self.trainingTest.tableWidget.setItem(15, 1, QTableWidgetItem(self.imageDataGeneratorSettingWindow.shearRangeText.text()))
                    # self.trainingTest.tableWidget.item(15, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(15, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(15, False)
                    self.trainingTest.tableWidget.setItem(16, 1, QTableWidgetItem(self.imageDataGeneratorSettingWindow.zoomRangeText.text()))
                    # self.trainingTest.tableWidget.item(16, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(16, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(16, False)
                    self.trainingTest.tableWidget.setItem(17, 1, QTableWidgetItem(self.imageDataGeneratorSettingWindow.rescaleText.text()))
                    # self.trainingTest.tableWidget.item(17, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.trainingTest.tableWidget.item(17, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.trainingTest.tableWidget.setRowHidden(17, False)
                else:
                    self.trainingTest.tableWidget.setRowHidden(10, True)
                    self.trainingTest.tableWidget.setRowHidden(11, True)
                    self.trainingTest.tableWidget.setRowHidden(12, True)
                    self.trainingTest.tableWidget.setRowHidden(13, True)
                    self.trainingTest.tableWidget.setRowHidden(14, True)
                    self.trainingTest.tableWidget.setRowHidden(15, True)
                    self.trainingTest.tableWidget.setRowHidden(16, True)
                    self.trainingTest.tableWidget.setRowHidden(17, True)

                self.trainingTest.category_changed_flag_c = True

                self.pretrained_radio_bool = self.pretrained_radio.isChecked()
                self.pretrained_combo_buffer = self.pretrained_combo.currentText()
                self.customed_radio_bool = self.customed_radio.isChecked()
                self.trainingDataSet_label_buffer = self.trainingDataSet_label.text()
                self.trainingDataSet_directoryName_buffer = self.trainingDataSet_directoryName
                self.validationDataSet_directoryName_buffer = self.validationDataSet_directoryName
                self.validationDataSet_label_buffer = self.validationDataSet_label.text()
                self.modelName_line_buffer = self.modelName_line.text()
                self.batch_line_buffer = self.batch_line.text()
                self.epoch_line_buffer = self.epoch_line.text()
                self.inputShape_line1_buffer = self.inputShape_line1.text()
                self.inputShape_line2_buffer = self.inputShape_line2.text()
                self.inputShape_combo_buffer = self.inputShape_combo.currentText()
                self.optimizer_combo_buffer = self.optimizer_combo.currentText()
                self.learningRate_line_buffer = self.learningRate_line.text()
                self.default_check_bool = self.default_check.isChecked()
                self.imageGenerator_check_bool = self.imageGenerator_check.isChecked()
                self.fineTuning_check_bool = self.fineTuning_check.isChecked()
                self.num_of_freezing_buffer = int(self.fineTuningSettingDialog.num_of_freezing.text())
                self.horizontalFlip_buffer = self.imageDataGeneratorSettingWindow.horizontalFlip.isChecked()
                self.verticalFlip_buffer = self.imageDataGeneratorSettingWindow.verticalFlip.isChecked()
                self.rotationRangeText_buffer = self.imageDataGeneratorSettingWindow.rotationRangeText.text()
                self.widthShiftRangeText_buffer = self.imageDataGeneratorSettingWindow.widthShiftRangeText.text()
                self.heightShiftRangeText_buffer = self.imageDataGeneratorSettingWindow.heightShiftRangeText.text()
                self.shearRangeText_buffer = self.imageDataGeneratorSettingWindow.shearRangeText.text()
                self.zoomRangeText_buffer = self.imageDataGeneratorSettingWindow.zoomRangeText.text()
                self.rescaleText_buffer = self.imageDataGeneratorSettingWindow.rescaleText.text()

                self.close()

    def training(self):

        import keras
        from keras.applications import VGG16
        from keras import models
        from keras import layers
        from keras.preprocessing.image import ImageDataGenerator
        from keras.optimizers import Adam, Adagrad, Adadelta, SGD, Nadam, RMSprop
        from keras.models import Model


        # self.plot_losses.graph_update_signal.signal.connect(self.graph_update)

        classf = open(self.model_path + ".txt", "w")

        image_size1 = int(self.inputShape_line1.text())
        image_size2 = int(self.inputShape_line2.text())

        BATCH_SIZE = int(self.batch_line.text())
        EPOCHS = int(self.epoch_line.text())
        self.TRAIN_DIR = self.trainingDataSet_directoryName
        self.VAL_DIR = self.validationDataSet_directoryName

        for a,b,c in os.walk(self.TRAIN_DIR):
            for name in b:
                classf.write(name + "\n")
            dir_count = len(b)
            break

        classf.close()

        # Load the VGG model
        if self.pretrained_combo.currentText() == "VGG16":
            if self.inputShape_combo.currentText() == "RGB":
                vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size1, image_size2, 3))
            elif self.inputShape_combo.currentText() == "Gray Scale":
                vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size1, image_size2, 1))

        # Freeze the layers except the last 4 layers
        if self.pretrained_combo.currentText() == "VGG16":
            for layer in vgg_conv.layers[:-int(self.fineTuningSettingDialog.num_of_freezing.text())]:
                layer.trainable = False

        # Check the trainable status of the individual layers
        if self.pretrained_combo.currentText() == "VGG16" and self.customed_radio.isChecked() == False:
            for layer in vgg_conv.layers:
                print(layer, layer.trainable)

        # Create the model
        model = models.Sequential()
        model_RES = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False,
                                                         input_shape=(image_size1, image_size2, 3))

        # Add the vgg convolutional base model
        if self.customed_radio.isChecked() == False and self.pretrained_combo.currentText() == "VGG16":
            model.add(vgg_conv)

        # Add new layers
        conv_idx = 0
        maxp_idx = 0
        dense_idx = 0
        dropout_idx = 0
        flatten_flag = True
        inputShape_flag = True

        if self.pretrained_combo.currentText() == "VGG16":
            if self.customed_radio.isChecked() == True and self.default_check.isChecked() == False:
                for i in range(0, len(self.customedModelSettingDialog.listWidget)):
                    if self.customedModelSettingDialog.list[i][:2] == "Co":
                        if inputShape_flag == True:
                            model.add(layers.Conv2D(int(self.convolutionLayerSettingWindow_customed.filters_list[conv_idx]),
                                                    (int(self.convolutionLayerSettingWindow_customed.kernel_size1_list[conv_idx]),
                                                     int(self.convolutionLayerSettingWindow_customed.kernel_size2_list[conv_idx])),
                                                    input_shape=(image_size1, image_size2, 3 if self.inputShape_combo.currentText() == "RGB" else 1),
                                                    kernel_initializer=self.convolutionLayerSettingWindow_customed.kernel_initializer_list[conv_idx],
                                                    activation=self.convolutionLayerSettingWindow_customed.activation_list[conv_idx]))
                            inputShape_flag = False
                        else:
                            model.add(layers.Conv2D(int(self.convolutionLayerSettingWindow_customed.filters_list[conv_idx]),
                                                    (int(self.convolutionLayerSettingWindow_customed.kernel_size1_list[conv_idx]),
                                                     int(self.convolutionLayerSettingWindow_customed.kernel_size2_list[conv_idx])),
                                                    kernel_initializer=self.convolutionLayerSettingWindow_customed.kernel_initializer_list[conv_idx],
                                                    activation=self.convolutionLayerSettingWindow_customed.activation_list[conv_idx]))
                        conv_idx += 1
                    elif self.customedModelSettingDialog.list[i][:2] == "Ma":
                        model.add(layers.MaxPool2D(pool_size=(int(self.maxpoolingLayerSettingWindow_customed.pool_size1_list[maxp_idx]),
                                                              int(self.maxpoolingLayerSettingWindow_customed.pool_size2_list[maxp_idx]))))
                        maxp_idx += 1
                    elif self.customedModelSettingDialog.list[i][:2] == "De":
                        if flatten_flag == True:
                            model.add(layers.Flatten())
                            flatten_flag = False
                        model.add(layers.Dense(
                            units=int(self.denseLayerSettingWindow_customed.units[dense_idx]),
                            kernel_initializer=self.denseLayerSettingWindow_customed.kernel_initializer[dense_idx],
                            activation=self.denseLayerSettingWindow_customed.activation[dense_idx]))
                        dense_idx += 1
                    elif self.customedModelSettingDialog.list[i][:2] == "Dr":
                        model.add(layers.Dropout(float(self.dropoutSettingWindow_customed.dropoutRate[dropout_idx])))
                        dropout_idx += 1
            elif self.customed_radio.isChecked() == True and self.default_check.isChecked() == True:
                model.add(layers.Conv2D(32, (3, 3), input_shape=(image_size1, image_size2, 3 if self.inputShape_combo.currentText() == "RGB" else 1),
                                        kernel_initializer='normal', activation='relu'))
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                model.add(layers.Conv2D(32, (3, 3), kernel_initializer='normal', activation='relu'))
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                model.add(layers.Flatten())
                model.add(layers.Dense(units=1024, kernel_initializer='normal', activation='relu'))
                model.add(layers.Dense(units=512, kernel_initializer='normal', activation='relu'))
                model.add(layers.Dropout(0.5))
                model.add(layers.Dense(units=dir_count, kernel_initializer='normal',
                                       activation='sigmoid' if dir_count == 2 else 'softmax'))
            elif self.fineTuning_check.isChecked() == False or len(self.fineTuningSettingDialog.listWidget) == 0:
                model.add(layers.Flatten())
                model.add(layers.Dense(units=1024, kernel_initializer='normal', activation='relu'))
                model.add(layers.Dense(units=512, kernel_initializer='normal', activation='relu'))
                model.add(layers.Dropout(0.5))
                model.add(layers.Dense(units=dir_count, kernel_initializer='normal',
                                       activation='sigmoid' if dir_count==2 else 'softmax'))
            else:
                model.add(layers.Flatten())
                for i in range(0, len(self.fineTuningSettingDialog.listWidget)):
                    if self.fineTuningSettingDialog.list[i][:2] == "De":
                        model.add(layers.Dense(
                            units=int(self.denseLayerSettingWindow.units[dense_idx]),
                            kernel_initializer=self.denseLayerSettingWindow.kernel_initializer[dense_idx],
                            activation=self.denseLayerSettingWindow.activation[dense_idx]))
                        dense_idx += 1
                    elif self.fineTuningSettingDialog.list[i][:2] == "Dr":
                        model.add(layers.Dropout(float(self.dropoutSettingWindow.dropoutRate[dropout_idx])))
                        dropout_idx += 1

        train_datagen = ImageDataGenerator(
            rescale=float(self.imageDataGeneratorSettingWindow.rescaleText.text()),
            rotation_range=int(self.imageDataGeneratorSettingWindow.rotationRangeText.text()),
            width_shift_range=float(self.imageDataGeneratorSettingWindow.widthShiftRangeText.text()),
            height_shift_range=float(self.imageDataGeneratorSettingWindow.heightShiftRangeText.text()),
            zoom_range=float(self.imageDataGeneratorSettingWindow.zoomRangeText.text()),
            horizontal_flip=self.imageDataGeneratorSettingWindow.horizontalFlip.isChecked(),
            vertical_flip=self.imageDataGeneratorSettingWindow.verticalFlip.isChecked(),
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(
            rescale=float(self.imageDataGeneratorSettingWindow.rescaleText.text()))

        train_generator = train_datagen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(image_size1, image_size2),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

        validation_generator = validation_datagen.flow_from_directory(
            self.VAL_DIR,
            target_size=(image_size1, image_size2),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False)

        model_RES.layers.pop()
        if self.pretrained_combo.currentText() == "RESNET50":
            for layer in model_RES.layers[:-int(self.fineTuningSettingDialog.num_of_freezing.text())]:
                layer.trainable = False
            for layer in model_RES.layers:
                print(layer, layer.trainable)

            last = model_RES.layers[-1].output
            last = layers.GlobalAveragePooling2D()(last)
            if self.fineTuning_check.isChecked() == False or len(self.fineTuningSettingDialog.listWidget) == 0:
                last = layers.Dense(units=1024, kernel_initializer='normal', activation='relu')(last)
                last = layers.Dense(units=512, kernel_initializer='normal', activation='relu')(last)
                last = layers.Dropout(0.5)(last)
                last = layers.Dense(units=dir_count, kernel_initializer='normal',
                                       activation='sigmoid' if dir_count == 2 else 'softmax')(last)
            else:
                for i in range(0, len(self.fineTuningSettingDialog.listWidget)):
                    if self.fineTuningSettingDialog.list[i][:2] == "De":
                        last = layers.Dense(
                            units=int(self.denseLayerSettingWindow.units[dense_idx]),
                            kernel_initializer=self.denseLayerSettingWindow.kernel_initializer[dense_idx],
                            activation=self.denseLayerSettingWindow.activation[dense_idx])(last)
                        dense_idx += 1
                    elif self.fineTuningSettingDialog.list[i][:2] == "Dr":
                        last = layers.Dropout(float(self.dropoutSettingWindow.dropoutRate[dropout_idx]))(last)
                        dropout_idx += 1
            finetuned_model = Model(model_RES.input, last)

        # Compile the model
        if self.pretrained_combo.currentText() == "VGG16":
            if self.optimizer_combo.currentText() == "Adam":
                model.compile(loss='categorical_crossentropy',
                              optimizer=optimizers.Adam(lr=float(self.learningRate_line.text())),
                              metrics=['acc'])
            elif self.optimizer_combo.currentText() == "Adagrad":
                 model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.Adagrad(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "AdaDelta":
                 model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.Adadelta(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "SGD":
                 model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.SGD(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "Nadam":
                 model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.Nadam(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "RMSProp":
                 model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.RMSprop(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
        elif self.pretrained_combo.currentText() == "RESNET50":
            if self.optimizer_combo.currentText() == "Adam":
                finetuned_model.compile(loss='categorical_crossentropy',
                              optimizer=optimizers.Adam(lr=float(self.learningRate_line.text())),
                              metrics=['acc'])
            elif self.optimizer_combo.currentText() == "Adagrad":
                 finetuned_model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.Adagrad(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "AdaDelta":
                 finetuned_model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.Adadelta(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "SGD":
                 finetuned_model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.SGD(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "Nadam":
                 finetuned_model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.Nadam(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])
            elif self.optimizer_combo.currentText() == "RMSProp":
                 finetuned_model.compile(loss='categorical_crossentropy',
                               optimizer=optimizers.RMSprop(lr=float(self.learningRate_line.text())),
                               metrics=['acc'])

        # Train the model
        if self.pretrained_combo.currentText() == "VGG16":
            history = model.fit_generator(
                train_generator,
                steps_per_epoch=train_generator.samples // BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                callbacks = [self.plot_losses])
        elif self.pretrained_combo.currentText() == "RESNET50":
            history = finetuned_model.fit_generator(
                train_generator,
                steps_per_epoch=train_generator.samples // BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                callbacks=[self.plot_losses])

        model.summary() if self.pretrained_combo.currentText() == "VGG16" else finetuned_model.summary()

        # Save the model
        model.save(self.model_path + ".h5") if self.pretrained_combo.currentText() == "VGG16" \
            else finetuned_model.save(self.model_path + ".h5")

        self.training_flag = True
        self.model_path_trained = self.model_path + ".h5"
        self.confirm_button.setEnabled(True)
        self.setStop()
        self.messageBox_signal.run()
        self.change_button_signal.run()

    # @pyqtSlot()
    # def graph_update(self):
    #     self.graph_img = QPixmap('training_plot.jpg')
    #     self.graph_label.setPixmap(self.graph_img)

    @pyqtSlot()
    def messageBox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Do you want to move to Prediction Window?")
        msg.setWindowTitle("Prediction")
        buttonReply = msg.question(self, "Prediction", "Do you want to move to Prediction Window?",
                                   QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if buttonReply == QMessageBox.Yes:
            self.open_other_window_signal.run()

    # def open_other_window(self):
    #     if self.model_path_trained:
    #         self.close()
    #         self.predictionWindow = PredictWindow(self.model_path_trained)
    #         self.predictionWindow.exec()

    def isNumber(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, filename='training_plot.jpg'):
        self.filename = filename
        self.graph_update_signal = MySignal()
        self.graph_update_signal2 = MySignal()


    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        # self.losses.append(logs.get('loss'))
        # self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = range(1, len(self.val_losses)+1)

            # You can chose the style of your preference
            # print(plt.style.available)

            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            self.figure2 = plt.figure()
            # plt.plot(N, self.losses, label = "train_loss")
            # plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch+1))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            self.canvas2 = FigureCanvas(self.figure2)
            self.canvas2.draw()
            size = self.canvas2.size()
            width, height = size.width(), size.height()
            self.im2 = QImage(self.canvas2.buffer_rgba(), width, height, QImage.Format_ARGB32)
            self.graph_update_signal2.run()
            plt.close()
        # pass

    def on_batch_end(self, epoch, logs={}):
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        # self.val_losses.append(logs.get('val_loss'))
        # self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = range(1, len(self.losses)+1)

            # You can chose the style of your preference
            # print(plt.style.available)

            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            self.figure = plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            # plt.plot(N, self.val_losses, label = "val_loss")
            # plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Step {}]".format(epoch+1))
            plt.xlabel("Step #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            self.canvas = FigureCanvas(self.figure)
            self.canvas.draw()
            size = self.canvas.size()
            width, height = size.width(), size.height()
            self.im = QImage(self.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
            self.graph_update_signal.run()
            plt.close()
        # pass

class CustomedModelSettingDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Customed Model Setting Dialog")
        self.setGeometry(750, 360, 639, 339)
        self.initUI()

    def initUI(self):
        self.formerValue = 100000
        self.row = -1

        self.listWidget = QListWidget(self)
        self.listWidget.setFixedSize(456, 172)
        self.listWidget.setGeometry(22, 70, 90, 50)
        self.listWidget.setCurrentRow(0)

        self.label = QLabel("Customed Layers Visualization", self)
        self.label.setGeometry(22, 40, 100, 50)
        self.label.setFixedSize(222, 20)
        self.label.setFont(QFont("Arial", 10))
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.layers_combo = QComboBox(self)
        self.layers_combo.addItems(["Conv2D", "MaxPooling2D", "Dense", "Dropout"])
        self.layers_combo.setGeometry(496, 70, 90, 50)
        self.layers_combo.setFixedSize(124, 27)

        add_button = QPushButton("ADD", self)
        add_button.setFont(QFont("Arial", 11))
        add_button.setGeometry(513, 130, 90, 50)
        add_button.clicked.connect(self.moveLayerSet)

        del_button = QPushButton("DEL", self)
        del_button.setFont(QFont("Arial", 11))
        del_button.setGeometry(513, 193, 90, 50)
        del_button.clicked.connect(self.delete)

        confirm_button = QPushButton("Confirm", self)
        confirm_button.setGeometry(448, 280, 80, 30)
        confirm_button.clicked.connect(self.buttonClicked)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.setGeometry(540, 280, 80, 30)
        cancel_button.clicked.connect(self.buttonClicked)

        self.convolutionLayerSettingWindow = ConvolutionLayerSettingWindow(self)
        self.maxpoolingLayerSettingWindow = MaxPoolingLayerSettingWindow(self)
        self.denseLayerSettingWindow = DenseLayerSettingWindow(self)
        self.dropoutSettingWindow = DropoutSettingWindow(self)
        self.list = list()

    def buttonClicked(self):
        if self.sender().text() == "Confirm":
            self.close()
        elif self.sender().text() == "Cancel":
            self.listWidget.clear()
            self.close()

    def moveLayerSet(self):
        if self.sender().text() == "ADD":
            if self.layers_combo.currentText() == "Dense":
                self.denseLayerSettingWindow.exec_()
            elif self.layers_combo.currentText() == "Dropout":
                if self.row == -1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Dropout should not be set at first")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    self.dropoutSettingWindow.exec_()
            elif self.layers_combo.currentText() == "Conv2D":
                self.convolutionLayerSettingWindow.exec_()
            elif self.layers_combo.currentText() == "MaxPooling2D":
                if self.row == -1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("MaxPooling2D should not be set at first")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    self.maxpoolingLayerSettingWindow.exec_()

    def addLayers(self, string):
        self.listWidget.addItem(string)
        self.list.append(string)
        self.row += 1
        self.show()

    def delete(self):
        if self.row >= 0:

            self.listWidget.takeItem(self.row)

            if self.list[self.row][:2] == "Co":
                self.convolutionLayerSettingWindow.filters_list.pop(self.convolutionLayerSettingWindow.filters_list_index)
                self.convolutionLayerSettingWindow.filters_list_index -= 1
                self.convolutionLayerSettingWindow.kernel_size1_list.pop(self.convolutionLayerSettingWindow.kernel_size1_index)
                self.convolutionLayerSettingWindow.kernel_size1_index -= 1
                self.convolutionLayerSettingWindow.kernel_size2_list.pop(self.convolutionLayerSettingWindow.kernel_size2_index)
                self.convolutionLayerSettingWindow.kernel_size2_index -= 1
                self.convolutionLayerSettingWindow.kernel_initializer_list.pop(self.convolutionLayerSettingWindow.kernel_initializer_list_index)
                self.convolutionLayerSettingWindow.kernel_initializer_list_index -= 1
                self.convolutionLayerSettingWindow.activation_list.pop(self.convolutionLayerSettingWindow.activation_list_index)
                self.convolutionLayerSettingWindow.activation_list_index -= 1
            elif self.list[self.row][:2] == "Ma":
                self.maxpoolingLayerSettingWindow.pool_size1_list.pop(self.maxpoolingLayerSettingWindow.pool_size1_index)
                self.maxpoolingLayerSettingWindow.pool_size1_index -= 1
                self.maxpoolingLayerSettingWindow.pool_size2_list.pop(self.maxpoolingLayerSettingWindow.pool_size2_index)
                self.maxpoolingLayerSettingWindow.pool_size2_index -= 1
            elif self.list[self.row][:2] == "De":
                self.denseLayerSettingWindow.units.pop(self.denseLayerSettingWindow.units_index)
                self.denseLayerSettingWindow.units_index -= 1
                if len(self.denseLayerSettingWindow.units) == 0:
                    self.formerValue = 100000
                else:
                    self.formerValue = self.denseLayerSettingWindow.units[-1]
                self.denseLayerSettingWindow.kernel_initializer.pop(self.denseLayerSettingWindow.kernel_initializer_index)
                self.denseLayerSettingWindow.kernel_initializer_index -= 1
                self.denseLayerSettingWindow.activation.pop(self.denseLayerSettingWindow.activation_index)
                self.denseLayerSettingWindow.activation_index -= 1
            elif self.list[self.row][:2] == "Dr":
                self.dropoutSettingWindow.dropoutRate.pop(self.dropoutSettingWindow.dropoutIndex)
                self.dropoutSettingWindow.dropoutIndex -= 1

            self.list.pop(self.row)
            self.row -= 1
            self.show()

class DenseLayerSettingWindow(QDialog):
    def __init__(self, dialog):
        super().__init__()

        self.setWindowTitle("Dense Layer Configuration")
        self.setGeometry(800, 400, 300, 250)
        self.dialog = dialog

        groupBox = QGroupBox()

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        groupBox.setLayout(groupBox_layout)

        self.output_dim = QLabel("output dimension : ")
        self.init = QLabel("kernel_initializer : ")
        self.activation = QLabel("activation : ")

        self.textField = QLineEdit(self)
        self.textField.setReadOnly(False)
        self.textField.setFixedSize(131, 27)

        self.init_combo = QComboBox()
        self.init_combo.addItems(['normal', 'uniform'])

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(['relu', 'sigmoid', 'softmax'])

        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self.output_dim)
        hlayout1.addWidget(self.textField)
        groupBox_layout.addLayout(hlayout1)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.init)
        hlayout2.addWidget(self.init_combo)
        groupBox_layout.addLayout(hlayout2)

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(self.activation)
        hlayout3.addWidget(self.activation_combo)
        groupBox_layout.addLayout(hlayout3)

        base_layout.addWidget(groupBox)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.buttonClicked)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.buttonClicked)

        hbox = QHBoxLayout()

        hbox.addWidget(self.confirm_button)
        hbox.addWidget(self.cancel_button)

        base_layout.addLayout(hbox)

        self.units = list()
        self.units_index = -1
        self.kernel_initializer = list()
        self.kernel_initializer_index = -1
        self.activation = list()
        self.activation_index = -1

    def buttonClicked(self):
        if self.sender().text() == "Cancel":
            self.close()
        elif self.sender().text() == "Confirm":
            if self.textField.text().isdigit():
                if int(self.textField.text()) < 1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Value is expected to be a positive integer")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                elif int(self.textField.text()) >= int(self.dialog.formerValue):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Value is expected to be a positive integer under Former output_dim value")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    string = "Dense(units = " + self.textField.text() + ", kernel_initializer = " + self.init_combo.currentText() \
                             + ", activation = " + self.activation_combo.currentText() + ")"

                    self.units.append(self.textField.text())
                    self.units_index += 1
                    self.kernel_initializer.append(self.init_combo.currentText())
                    self.kernel_initializer_index += 1
                    self.activation.append(self.activation_combo.currentText())
                    self.activation_index += 1

                    self.close()
                    self.dialog.formerValue = int(self.textField.text())
                    self.dialog.addLayers(string)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Value is expected to be a positive integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

class DropoutSettingWindow(QDialog):
    def __init__(self, dialog):
        super().__init__()

        self.setWindowTitle("Designate Dropout Rate")
        self.setGeometry(800, 400, 250, 100)
        self.dialog = dialog

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom, self)
        groupBox_layout.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        groupBox = QGroupBox()
        groupBox.setLayout(groupBox_layout)

        self.label = QLabel("Designate Dropout Rate", self)
        self.label.setFixedSize(200, 27)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.textField = QLineEdit(self)
        self.textField.setReadOnly(False)
        self.textField.setFixedSize(200, 27)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.buttonClicked)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.buttonClicked)

        groupBox_layout.addWidget(self.label)
        groupBox_layout.addWidget(self.textField)

        hbox = QHBoxLayout()

        hbox.addWidget(self.confirm_button)
        hbox.addWidget(self.cancel_button)

        base_layout.addWidget(groupBox)
        base_layout.addLayout(hbox)

        self.dropoutRate = list()
        self.dropoutIndex = -1

    def buttonClicked(self):
        if self.sender().text() == "Cancel":
            self.close()
        elif self.sender().text() == "Confirm":
            if self.isNumber(self.textField.text()):
                if float(self.textField.text()) >= 1.0 or float(self.textField.text()) <= 0.0:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Dropout rate is expected to be a positive float value between 0 and 1")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    string = "Dropout(" + self.textField.text() + ")"

                    self.dropoutRate.append(self.textField.text())
                    self.dropoutIndex += 1

                    self.close()
                    self.dialog.addLayers(string)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Dropout rate is expected to be a positive float value between 0 and 1")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

    def isNumber(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

class ConvolutionLayerSettingWindow(QDialog):
    def __init__(self, dialog):
        super().__init__()

        self.setWindowTitle("Convolution Layer Configuration")
        self.setGeometry(800, 400, 300, 250)
        self.dialog = dialog

        groupBox = QGroupBox()

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        groupBox.setLayout(groupBox_layout)

        self.output_dim = QLabel("filters : ")
        self.init = QLabel("kernel_initializer : ")
        self.activation = QLabel("activation : ")

        self.textField = QLineEdit(self)
        self.textField.setReadOnly(False)
        self.textField.setFixedSize(121, 27)

        self.init_combo = QComboBox()
        self.init_combo.addItems(['normal', 'uniform'])

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(['relu', 'sigmoid', 'softmax'])

        self.kernelSize = QLabel("kernel_size : ")

        self.kernel_size1 = QLineEdit(self)
        self.kernel_size1.setReadOnly(False)
        self.kernel_size1.setFixedSize(44, 27)

        self.x = QLabel(" x ")

        self.kernel_size2 = QLineEdit(self)
        self.kernel_size2.setReadOnly(False)
        self.kernel_size2.setFixedSize(44, 27)

        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self.output_dim)
        hlayout1.addWidget(self.textField)
        groupBox_layout.addLayout(hlayout1)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.init)
        hlayout2.addWidget(self.init_combo)
        groupBox_layout.addLayout(hlayout2)

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(self.activation)
        hlayout3.addWidget(self.activation_combo)
        groupBox_layout.addLayout(hlayout3)

        self.kernel_size_empty = QLabel("", self)

        glayout4 = QGridLayout()
        glayout4.addWidget(self.kernelSize, 0, 0)
        glayout4.addWidget(self.kernel_size_empty, 0, 1)
        glayout4.addWidget(self.kernel_size_empty, 0, 2)
        glayout4.addWidget(self.kernel_size1, 0, 3)
        glayout4.addWidget(self.x, 0, 4)
        glayout4.addWidget(self.kernel_size2, 0, 5)

        groupBox_layout.addLayout(glayout4)

        base_layout.addWidget(groupBox)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.buttonClicked)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.buttonClicked)

        hbox = QHBoxLayout()

        hbox.addWidget(self.confirm_button)
        hbox.addWidget(self.cancel_button)

        base_layout.addLayout(hbox)

        self.filters_list = list()
        self.kernel_size1_list = list()
        self.kernel_size2_list = list()
        self.kernel_initializer_list = list()
        self.activation_list = list()

        self.filters_list_index = -1
        self.kernel_size1_index = -1
        self.kernel_size2_index = -1
        self.kernel_initializer_list_index = -1
        self.activation_list_index = -1

    def buttonClicked(self):
        if self.sender().text() == "Cancel":
            self.close()
        elif self.sender().text() == "Confirm":
            if self.textField.text().isdigit():
                if int(self.textField.text()) < 1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Filters is expected to be a positive integer")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                elif self.kernel_size1.text().isdigit() == False or self.kernel_size2.text().isdigit() == False:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Kernel Size is expected to be positive integer by positive integer")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                elif int(self.kernel_size1.text()) < 1 or int(self.kernel_size2.text()) < 1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Kernel Size is expected to be positive integer by positive integer")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    string = "Conv2D(" + self.textField.text() + ", " + "(" + self.kernel_size1.text() + "," + self.kernel_size2.text() + ")" + ", " \
                             + "activation = " + self.activation_combo.currentText() + ", " + "kernel_initializer = " + self.init_combo.currentText() + ")"

                    self.filters_list.append(self.textField.text())
                    self.filters_list_index += 1
                    self.kernel_size1_list.append(self.kernel_size1.text())
                    self.kernel_size1_index += 1
                    self.kernel_size2_list.append(self.kernel_size2.text())
                    self.kernel_size2_index += 1
                    self.kernel_initializer_list.append(self.init_combo.currentText())
                    self.kernel_initializer_list_index += 1
                    self.activation_list.append(self.activation_combo.currentText())
                    self.activation_list_index += 1

                    self.dialog.addLayers(string)
                    self.close()

            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Filters is expected to be a positive integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

class MaxPoolingLayerSettingWindow(QDialog):
    def __init__(self, dialog):
        super().__init__()

        self.setWindowTitle("Max Pooling Layer Configuration")
        self.setGeometry(800, 400, 90, 120)
        self.dialog = dialog

        groupBox = QGroupBox()

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        groupBox.setLayout(groupBox_layout)

        self.poolSize = QLabel("Pool Size : ")

        self.pool_size1 = QLineEdit(self)
        self.pool_size1.setReadOnly(False)
        self.pool_size1.setFixedSize(30, 27)

        self.x = QLabel(" x ")

        self.pool_size2 = QLineEdit(self)
        self.pool_size2.setReadOnly(False)
        self.pool_size2.setFixedSize(30, 27)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.poolSize)
        hlayout.addWidget(self.pool_size1)
        hlayout.addWidget(self.x)
        hlayout.addWidget(self.pool_size2)

        groupBox_layout.addLayout(hlayout)

        base_layout.addWidget(groupBox)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.buttonClicked)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.buttonClicked)

        hbox = QHBoxLayout()

        hbox.addWidget(self.confirm_button)
        hbox.addWidget(self.cancel_button)

        base_layout.addLayout(hbox)

        self.pool_size1_list = list()
        self.pool_size2_list = list()

        self.pool_size1_index = -1
        self.pool_size2_index = -1

    def buttonClicked(self):
        if self.sender().text() == "Cancel":
            self.close()
        elif self.sender().text() == "Confirm":
            if self.pool_size1.text().isdigit() or self.pool_size2.text().isdigit():
                if int(self.pool_size1.text()) < 1 or int(self.pool_size2.text()) < 1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Value is expected to be a positive integer")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    string = "MaxPooling2D(pool_size = " + "(" + self.pool_size1.text() + "," + self.pool_size2.text() + ")" + ")"

                    self.pool_size1_list.append(self.pool_size1.text())
                    self.pool_size1_index += 1
                    self.pool_size2_list.append(self.pool_size2.text())
                    self.pool_size2_index += 1

                    self.close()
                    self.dialog.addLayers(string)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Value is expected to be a positive integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

class ImageDataGeneratorSettingDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Data Generator Configuration")
        self.setGeometry(840, 320, 300, 300)

        groupBox = QGroupBox()

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        groupBox.setLayout(groupBox_layout)

        self.rotationRange = QLabel("rotation_range : ")
        self.widthShiftRange = QLabel("width_shift_range : ")
        self.heightShiftRange = QLabel("height_shift_range : ")
        self.shearRange = QLabel("shear_range : ")
        self.zoomRange = QLabel("zoom_range : ")
        self.rescale = QLabel("rescale : ")

        self.rotationRangeText = QLineEdit(self)
        self.rotationRangeText.setReadOnly(False)
        self.rotationRangeText.setFixedSize(131, 27)

        self.widthShiftRangeText = QLineEdit(self)
        self.widthShiftRangeText.setReadOnly(False)
        self.widthShiftRangeText.setFixedSize(131, 27)

        self.heightShiftRangeText = QLineEdit(self)
        self.heightShiftRangeText.setReadOnly(False)
        self.heightShiftRangeText.setFixedSize(131, 27)

        self.shearRangeText = QLineEdit(self)
        self.shearRangeText.setReadOnly(False)
        self.shearRangeText.setFixedSize(131, 27)

        self.zoomRangeText = QLineEdit(self)
        self.zoomRangeText.setReadOnly(False)
        self.zoomRangeText.setFixedSize(131, 27)

        self.rescaleText = QLineEdit(self)
        self.rescaleText.setReadOnly(False)
        self.rescaleText.setFixedSize(131, 27)

        self.horizontalFlip = QCheckBox("Horizontal Flip", self)
        self.verticalFlip = QCheckBox("Vertical Flip", self)

        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self.horizontalFlip)
        hlayout1.addWidget(self.verticalFlip)
        groupBox_layout.addLayout(hlayout1)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.rotationRange)
        hlayout2.addWidget(self.rotationRangeText)
        groupBox_layout.addLayout(hlayout2)

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(self.widthShiftRange)
        hlayout3.addWidget(self.widthShiftRangeText)
        groupBox_layout.addLayout(hlayout3)

        hlayout4 = QHBoxLayout()
        hlayout4.addWidget(self.heightShiftRange)
        hlayout4.addWidget(self.heightShiftRangeText)
        groupBox_layout.addLayout(hlayout4)

        hlayout5 = QHBoxLayout()
        hlayout5.addWidget(self.shearRange)
        hlayout5.addWidget(self.shearRangeText)
        groupBox_layout.addLayout(hlayout5)

        hlayout6 = QHBoxLayout()
        hlayout6.addWidget(self.zoomRange)
        hlayout6.addWidget(self.zoomRangeText)
        groupBox_layout.addLayout(hlayout6)

        hlayout7 = QHBoxLayout()
        hlayout7.addWidget(self.rescale)
        hlayout7.addWidget(self.rescaleText)
        groupBox_layout.addLayout(hlayout7)

        base_layout.addWidget(groupBox)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.buttonClicked)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.buttonClicked)

        hbox = QHBoxLayout()

        hbox.addWidget(self.confirm_button)
        hbox.addWidget(self.cancel_button)

        base_layout.addLayout(hbox)

    def buttonClicked(self):
        if self.sender().text() == "Cancel":
            self.rotationRangeText.setText("0")
            self.widthShiftRangeText.setText("0.0")
            self.heightShiftRangeText.setText("0.0")
            self.shearRangeText.setText("0.0")
            self.zoomRangeText.setText("0.0")
            self.rescaleText.setText("0.0")
            self.close()
        elif self.sender().text() == "Confirm":
            if self.rotationRangeText.text().isdigit() == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("rotation range is expected to be integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.widthShiftRangeText.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("width shift range is expected to be float or integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.heightShiftRangeText.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("height shift range is expected to be float or integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.shearRangeText.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("shear range is expected to be float or integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.zoomRangeText.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("zoom range is expected to be float or integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.rescaleText.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("rescale value is expected to be float or integer")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                self.close()

    def isNumber(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

class FineTuningSettingDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fine Tuning Setting Dialog")
        self.setGeometry(750, 360, 616, 410)
        self.initUI()

    def initUI(self):
        self.formerValue = 100000
        self.row = -1

        self.freezing_layer = QLabel("Num of Trainable Layers : ", self)
        self.freezing_layer.setFixedSize(350, 150)
        self.freezing_layer.setFont(QFont("Arial",15))
        self.freezing_layer.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.num_of_freezing = QSpinBox(self)
        self.num_of_freezing.setGeometry(340, 63, 90, 50)
        self.num_of_freezing.setReadOnly(False)
        self.num_of_freezing.setFixedSize(90, 30)
        self.num_of_freezing.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.listWidget = QListWidget(self)
        self.listWidget.setFixedSize(456, 172)
        self.listWidget.setGeometry(22, 150, 90, 50)
        self.listWidget.setCurrentRow(0)

        self.label = QLabel("Dense Layers Visualization", self)
        self.label.setGeometry(22, 120, 90, 50)
        self.label.setFixedSize(200, 20)
        self.label.setFont(QFont("Arial", 10))
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.dense_dropout_combo = QComboBox(self)
        self.dense_dropout_combo.addItems(["Dense", "Dropout"])
        self.dense_dropout_combo.setGeometry(496, 150, 90, 50)
        self.dense_dropout_combo.setFixedHeight(27)

        add_button = QPushButton("ADD", self)
        add_button.setFont(QFont("Arial", 11))
        add_button.setGeometry(496, 210, 90, 50)
        add_button.clicked.connect(self.moveDenseDropoutLayerSet)

        del_button = QPushButton("DEL", self)
        del_button.setFont(QFont("Arial", 11))
        del_button.setGeometry(496, 273, 90, 50)
        del_button.clicked.connect(self.delete)

        confirm_button = QPushButton("Confirm", self)
        confirm_button.setGeometry(424, 360, 80, 30)
        confirm_button.clicked.connect(self.buttonClicked)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.setGeometry(516, 360, 80, 30)
        cancel_button.clicked.connect(self.buttonClicked)

        self.denseLayerSettingWindow = DenseLayerSettingWindow(self)
        self.dropoutSettingWindow = DropoutSettingWindow(self)
        self.list = list()

    def buttonClicked(self):
        if self.sender().text() == "Confirm":
            self.close()
        elif self.sender().text() == "Cancel":
            self.listWidget.clear()
            self.num_of_freezing.setValue(0)
            self.formerValue = 100000
            self.close()

    def moveDenseDropoutLayerSet(self):
        if self.sender().text() == "ADD":
            if self.dense_dropout_combo.currentText() == "Dense":
                self.denseLayerSettingWindow.exec_()
            elif self.dense_dropout_combo.currentText() == "Dropout":
                if self.row == -1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Dropout should not be set at first")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    self.dropoutSettingWindow.exec_()

    def addLayers(self, string):
        self.listWidget.addItem(string)
        self.list.append(string)
        self.row += 1
        self.show()

    def delete(self):
        if self.row >= 0:
            self.listWidget.takeItem(self.row)

            if self.list[self.row][:2] == "De":
                self.denseLayerSettingWindow.units.pop(self.denseLayerSettingWindow.units_index)
                self.denseLayerSettingWindow.units_index -= 1
                if len(self.denseLayerSettingWindow.units) == 0:
                    self.formerValue = 100000
                else:
                    self.formerValue = self.denseLayerSettingWindow.units[-1]
                self.denseLayerSettingWindow.kernel_initializer.pop(self.denseLayerSettingWindow.kernel_initializer_index)
                self.denseLayerSettingWindow.kernel_initializer_index -= 1
                self.denseLayerSettingWindow.activation.pop(self.denseLayerSettingWindow.activation_index)
                self.denseLayerSettingWindow.activation_index -= 1
            elif self.list[self.row][:2] == "Dr":
                self.dropoutSettingWindow.dropoutRate.pop(self.dropoutSettingWindow.dropoutIndex)
                self.dropoutSettingWindow.dropoutIndex -= 1

            self.list.pop(self.row)
            self.row -= 1
            self.show()

class AddClasses(QDialog):

    def __init__(self, file_path):
        super().__init__()

        self.setWindowTitle("Make Classes Text")
        self.setGeometry(750, 360, 300, 300)
        self.file_path = file_path

        self.initUI()

    def initUI(self):
        self.groupBox = QGroupBox()
        self.groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)

        self.name_hbox = QHBoxLayout()

        self.label_empty1 = QLabel("", self)
        self.label_empty1.setFixedSize(0, 27)
        self.label_empty1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.name_hbox.addWidget(self.label_empty1)

        self.name_label = QLabel("Write down Classes", self)
        self.name_label.setFixedSize(250, 50)
        self.name_label.setFont(QFont("Arial", 15))
        self.name_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.name_hbox.addWidget(self.name_label)

        self.label_empty2 = QLabel("", self)
        self.label_empty2.setFixedSize(0, 27)
        self.label_empty2.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.name_hbox.addWidget(self.label_empty2)

        self.groupBox_layout.addLayout(self.name_hbox)

        self.line_hbox = QHBoxLayout()

        self.label_empty9 = QLabel("", self)
        self.label_empty9.setFixedSize(0, 27)
        self.label_empty9.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.line_hbox.addWidget(self.label_empty9)

        self.line = QLineEdit(self)
        self.line.setReadOnly(False)
        self.line.setFixedSize(300, 27)

        self.line_hbox.addWidget(self.line)

        self.label_empty10 = QLabel("", self)
        self.label_empty10.setFixedSize(0, 27)
        self.label_empty10.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.line_hbox.addWidget(self.label_empty10)

        self.groupBox_layout.addLayout(self.line_hbox)

        self.buttons_hbox = QHBoxLayout()

        self.label_empty3 = QLabel("", self)
        self.label_empty3.setFixedSize(0, 27)
        self.label_empty3.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.buttons_hbox.addWidget(self.label_empty3)

        self.add_button = QPushButton("ADD", self)
        self.add_button.setFont(QFont("Arial", 11))
        self.add_button.setFixedSize(90, 45)
        self.add_button.clicked.connect(self.add_button_clicked)

        self.buttons_hbox.addWidget(self.add_button)

        self.del_button = QPushButton("DEL", self)
        self.del_button.setFont(QFont("Arial", 11))
        self.del_button.setFixedSize(90, 45)
        self.del_button.clicked.connect(self.delete_button_clicked)

        self.buttons_hbox.addWidget(self.del_button)

        self.label_empty4 = QLabel("", self)
        self.label_empty4.setFixedSize(0, 27)
        self.label_empty4.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.buttons_hbox.addWidget(self.label_empty4)

        self.groupBox_layout.addLayout(self.buttons_hbox)

        self.listWidget_hbox = QHBoxLayout()

        self.label_empty5 = QLabel("", self)
        self.label_empty5.setFixedSize(0, 27)
        self.label_empty5.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.listWidget_hbox.addWidget(self.label_empty5)

        self.listWidget = QListWidget(self)
        self.listWidget.setFixedSize(300, 172)
        self.listWidget.setCurrentRow(0)

        self.listWidget_hbox.addWidget(self.listWidget)

        self.label_empty6 = QLabel("", self)
        self.label_empty6.setFixedSize(0, 27)
        self.label_empty6.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.listWidget_hbox.addWidget(self.label_empty6)

        self.groupBox_layout.addLayout(self.listWidget_hbox)

        self.confirm_hbox = QHBoxLayout()

        self.label_empty7 = QLabel("", self)
        self.label_empty7.setFixedSize(0, 27)
        self.label_empty7.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.confirm_hbox.addWidget(self.label_empty7)

        self.confirm_button = QPushButton("Confirm", self)
        self.confirm_button.setFixedSize(90, 45)
        self.confirm_button.setFont(QFont("Arial", 11))
        self.confirm_button.clicked.connect(self.confirm_button_clicked)

        self.confirm_hbox.addWidget(self.confirm_button)

        self.label_empty8 = QLabel("", self)
        self.label_empty8.setFixedSize(0, 27)
        self.label_empty8.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.confirm_hbox.addWidget(self.label_empty8)

        self.groupBox_layout.addLayout(self.confirm_hbox)

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        self.groupBox.setLayout(self.groupBox_layout)

        base_layout.addWidget(self.groupBox)

        self.row = -1
        self.list = list()

    def add_button_clicked(self):
        string = self.line.text()
        self.listWidget.addItem(string)
        self.list.append(string)
        self.row += 1
        self.line.clear()

    def delete_button_clicked(self):
        if self.row >= 0:
            self.listWidget.takeItem(self.row)
            self.list.pop()
            self.row -= 1

    def confirm_button_clicked(self):
        if self.list[0]:
            classf = open(self.file_path, "w")
            for i in range(0, self.row+1):
                classf.write(self.list[i] + "\n")
        classf.close()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    classificationUI = ClassificationUI()
    classificationUI.show()
    sys.exit(app.exec_())