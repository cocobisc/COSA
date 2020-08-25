import os
import cv2
import sys
import json
import random
import logging
import colorsys
import ctypes
import skimage.draw
import tensorflow as tf
import threading
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .mrcnn.config import Config
from .mrcnn import model as modellib, utils
from skimage.measure import find_contours
from keras import backend as K  # Tensor tensor 오류 수정 부분

default_stdout = sys.stdout
default_stderr = sys.stderr

logger = logging.getLogger(__name__)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
graph = tf.get_default_graph()

class MySignal(QObject):
    signal = pyqtSignal()
    def run(self):
        self.signal.emit()

class XStream(QObject):
    _stdout = None
    _stderr = None

    messageWritten = pyqtSignal(str)

    def flush( self ):
        pass

    def fileno( self ):
        return -1

    def write( self, msg ):
        if ( not self.signalsBlocked() ):
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        #if ( not XStream._stdout ):
        XStream._stdout = XStream()
        sys.stdout = XStream._stdout

        return XStream._stdout

    @staticmethod
    def stderr():
        if ( not XStream._stderr ):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr

class LogMessageViewer(QTextBrowser):
    def __init__(self, parent=None):
        super(LogMessageViewer,self).__init__(parent)
        self.setReadOnly(True)

    @QtCore.pyqtSlot(str)
    def appendLogMessage(self, msg):
        horScrollBar = self.horizontalScrollBar()
        verScrollBar = self.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10
        self.insertPlainText(msg)

        if scrollIsAtEnd:
            verScrollBar.setValue(verScrollBar.maximum()) # scroll to the bottom
            horScrollBar.setValue(0) # scroll to the left



# ----------------------------------------------------------------------------------------------- stdout redirection ------------------------------------------------------------------------------------------------------------------------ #



class SegmentationUI(QDialog, Config):

    def __init__(self):
        super().__init__()

        self.messageBox_signal = MySignal()
        self.messageBox_signal.signal.connect(self.messageBox)

        self.open_other_window_signal = MySignal()
        self.open_other_window_signal.signal.connect(self.open_other_window)

        self.setWindowTitle("Segmentation")
        self.setGeometry(420, 220, 1200, 600)
        self.setFixedSize(1200, 682)

        self.groupBox = QGroupBox()

        self.vBox1 = QVBoxLayout()
        self.vBox2 = QVBoxLayout()

        self.hBox1 = QHBoxLayout()

        self.label_empty19 = QLabel("", self)
        self.label_empty19.setFixedSize(60, 27)
        self.label_empty19.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox1.addWidget(self.label_empty19)

        self.sourceName_label = QLabel("Source Name : ", self)
        self.sourceName_label.setFixedSize(246, 27)
        self.sourceName_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox1.addWidget(self.sourceName_label)

        self.sourceName_line = QLineEdit(self)
        self.sourceName_line.setReadOnly(False)
        self.sourceName_line.setFixedSize(120, 27)
        self.sourceName_line.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox1.addWidget(self.sourceName_line)

        self.label_empty20 = QLabel("", self)
        self.label_empty20.setFixedSize(60, 27)
        self.label_empty20.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox1.addWidget(self.label_empty20)

        self.vBox1.addLayout(self.hBox1)

        self.hBox2 = QHBoxLayout()

        self.vBox2.addLayout(self.hBox2)

        self.rootDirectory_label = QLabel("Root Directory : ", self)
        self.rootDirectory_label.setFixedSize(600, 18)
        self.rootDirectory_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.vBox1.addWidget(self.rootDirectory_label)

        self.hBox3 = QHBoxLayout()

        self.backBone_label = QLabel("BackBone for Classification : ", self)
        self.backBone_label.setFixedSize(196, 27)
        self.backBone_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox3.addWidget(self.backBone_label)

        self.backBone_line = QComboBox(self)
        self.backBone_line.addItems(["RESNET 101", "RESNET 50"])
        self.backBone_line.setFixedSize(120, 27)

        self.hBox3.addWidget(self.backBone_line)

        self.label_empty1 = QLabel("", self)
        self.label_empty1.setFixedSize(30, 27)
        self.label_empty1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox3.addWidget(self.label_empty1)

        self.rootDirectory_button = QPushButton("Set Root Directory")
        self.rootDirectory_button.setFixedHeight(27)
        self.rootDirectory_button.clicked.connect(self.openDirectory)

        self.vBox1.addWidget(self.rootDirectory_button)

        self.epochs_label = QLabel("Epochs : ", self)
        self.epochs_label.setFixedSize(64, 27)
        self.epochs_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox3.addWidget(self.epochs_label)

        self.epochs_line = QLineEdit(self)
        self.epochs_line.setReadOnly(False)
        self.epochs_line.setFixedSize(120, 27)

        self.hBox3.addWidget(self.epochs_line)

        self.vBox2.addLayout(self.hBox3)

        self.hBox5 = QHBoxLayout()

        self.label_empty2 = QLabel("", self)
        self.label_empty2.setFixedSize(60, 27)
        self.label_empty2.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox5.addWidget(self.label_empty2)

        self.image_per_gpu_label = QLabel("Num of Image assignments per GPU : ", self)
        self.image_per_gpu_label.setFixedSize(246, 27)
        self.image_per_gpu_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox5.addWidget(self.image_per_gpu_label)

        self.image_per_gpu_line = QLineEdit(self)
        self.image_per_gpu_line.setReadOnly(False)
        self.image_per_gpu_line.setFixedSize(120, 27)

        self.hBox5.addWidget(self.image_per_gpu_line)

        self.label_empty3 = QLabel("", self)
        self.label_empty3.setFixedSize(60, 27)
        self.label_empty3.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox5.addWidget(self.label_empty3)

        self.vBox1.addLayout(self.hBox5)

        self.hBox6 = QHBoxLayout()

        self.label_empty4 = QLabel("", self)
        self.label_empty4.setFixedSize(60, 27)
        self.label_empty4.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox6.addWidget(self.label_empty4)

        self.gpuCount_label = QLabel("Num of GPU to use : ", self)
        self.gpuCount_label.setFixedSize(234, 20)
        self.gpuCount_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox6.addWidget(self.gpuCount_label)

        self.gpuCount_line = QLineEdit(self)
        self.gpuCount_line.setReadOnly(False)
        self.gpuCount_line.setFixedSize(120, 27)

        self.hBox6.addWidget(self.gpuCount_line)

        self.label_empty5 = QLabel("", self)
        self.label_empty5.setFixedSize(60, 27)
        self.label_empty5.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox6.addWidget(self.label_empty5)

        self.vBox2.addLayout(self.hBox6)

        self.hBox7 = QHBoxLayout()

        self.label_empty6 = QLabel("", self)
        self.label_empty6.setFixedSize(60, 27)
        self.label_empty6.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox7.addWidget(self.label_empty6)

        self.steps_per_epoch_label = QLabel("Steps per Epoch for Training : ", self)
        self.steps_per_epoch_label.setFixedSize(246, 20)
        self.steps_per_epoch_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox7.addWidget(self.steps_per_epoch_label)

        self.steps_per_epoch_line = QLineEdit(self)
        self.steps_per_epoch_line.setReadOnly(False)
        self.steps_per_epoch_line.setFixedSize(120, 27)

        self.hBox7.addWidget(self.steps_per_epoch_line)

        self.label_empty7 = QLabel("", self)
        self.label_empty7.setFixedSize(60, 27)
        self.label_empty7.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox7.addWidget(self.label_empty7)

        self.vBox1.addLayout(self.hBox7)

        self.hBox8 = QHBoxLayout()

        self.label_empty8 = QLabel("", self)
        self.label_empty8.setFixedSize(60, 27)
        self.label_empty8.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox8.addWidget(self.label_empty8)

        self.validation_steps_label = QLabel("Num of Steps for Validation : ", self)
        self.validation_steps_label.setFixedSize(234, 27)
        self.validation_steps_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox8.addWidget(self.validation_steps_label)

        self.validation_steps_line = QLineEdit(self)
        self.validation_steps_line.setReadOnly(False)
        self.validation_steps_line.setFixedSize(120, 27)

        self.hBox8.addWidget(self.validation_steps_line)

        self.label_empty9 = QLabel("", self)
        self.label_empty9.setFixedSize(60, 27)
        self.label_empty9.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox8.addWidget(self.label_empty9)

        self.vBox2.addLayout(self.hBox8)

        self.hBox9 = QHBoxLayout()

        self.label_empty14 = QLabel("", self)
        self.label_empty14.setFixedSize(60, 27)
        self.label_empty14.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox9.addWidget(self.label_empty14)

        self.learningRate_label = QLabel("Learning Rate : ", self)
        self.learningRate_label.setFixedSize(234, 27)
        self.learningRate_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox9.addWidget(self.learningRate_label)

        self.learningRate_line = QLineEdit(self)
        self.learningRate_line.setReadOnly(False)
        self.learningRate_line.setFixedSize(120, 27)

        self.hBox9.addWidget(self.learningRate_line)

        self.label_empty14 = QLabel("", self)
        self.label_empty14.setFixedSize(60, 27)
        self.label_empty14.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox9.addWidget(self.label_empty14)

        self.vBox2.addLayout(self.hBox9)

        self.hBox10 = QHBoxLayout()

        self.label_empty12 = QLabel("", self)
        self.label_empty12.setFixedSize(60, 27)
        self.label_empty12.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox10.addWidget(self.label_empty12)

        self.learningMomentum_label = QLabel("Learning Momentum : ", self)
        self.learningMomentum_label.setFixedSize(234, 27)
        self.learningMomentum_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox10.addWidget(self.learningMomentum_label)

        self.learningMomentum_line = QLineEdit(self)
        self.learningMomentum_line.setReadOnly(False)
        self.learningMomentum_line.setFixedSize(120, 27)

        self.hBox10.addWidget(self.learningMomentum_line)

        self.label_empty13 = QLabel("", self)
        self.label_empty13.setFixedSize(60, 27)
        self.label_empty13.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox10.addWidget(self.label_empty13)

        self.vBox2.addLayout(self.hBox10)

        self.hBox11 = QHBoxLayout()

        self.label_empty15 = QLabel("", self)
        self.label_empty15.setFixedSize(60, 27)
        self.label_empty15.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox11.addWidget(self.label_empty15)

        self.layers_label = QLabel("Trainable Layers : ", self)
        self.layers_label.setFixedSize(246, 20)
        self.layers_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox11.addWidget(self.layers_label)

        self.layers_combo = QComboBox()
        self.layers_combo.addItems(['heads', '3+', '4+', '5+', 'all'])
        self.layers_combo.setFixedSize(120, 27)

        self.hBox11.addWidget(self.layers_combo)

        self.label_empty16 = QLabel("", self)
        self.label_empty16.setFixedSize(60, 27)
        self.label_empty16.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox11.addWidget(self.label_empty16)

        self.vBox1.addLayout(self.hBox11)

        self.hBox12 = QHBoxLayout()

        self.label_empty10 = QLabel("", self)
        self.label_empty10.setFixedSize(60, 27)
        self.label_empty10.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox12.addWidget(self.label_empty10)

        self.detection_min_confidence_label = QLabel("Minimum Confidence for Detection : ", self)
        self.detection_min_confidence_label.setFixedSize(234, 27)
        self.detection_min_confidence_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox12.addWidget(self.detection_min_confidence_label)

        self.detection_min_confidence_line = QLineEdit(self)
        self.detection_min_confidence_line.setReadOnly(False)
        self.detection_min_confidence_line.setFixedSize(120, 27)

        self.hBox12.addWidget(self.detection_min_confidence_line)

        self.label_empty11 = QLabel("", self)
        self.label_empty11.setFixedSize(60, 27)
        self.label_empty11.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox12.addWidget(self.label_empty11)

        self.vBox2.addLayout(self.hBox12)

        self.hBox13 = QHBoxLayout()

        self.label_empty17 = QLabel("", self)
        self.label_empty17.setFixedSize(60, 27)
        self.label_empty17.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox13.addWidget(self.label_empty17)

        self.weights_label = QLabel("Select Weights file for training ", self)
        self.weights_label.setFixedSize(246, 20)
        self.weights_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox13.addWidget(self.weights_label)

        self.weights_combo = QComboBox()
        self.weights_combo.addItems(['COCO', 'ImageNet', 'Last'])
        self.weights_combo.setFixedSize(120, 27)

        self.hBox13.addWidget(self.weights_combo)

        self.label_empty18 = QLabel("", self)
        self.label_empty18.setFixedSize(60, 27)
        self.label_empty18.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.hBox13.addWidget(self.label_empty18)

        self.vBox1.addLayout(self.hBox13)

        self.start_hbox = QHBoxLayout()

        self.start_button = QPushButton("Start Training", self)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.startTraining)
        self.start_button.setFixedSize(150, 50)

        self.start_hbox.addWidget(self.start_button)

        self.result_hBox = QHBoxLayout()

        self.console_label = QLabel("console", self)
        self.console_label.setFont(QFont("Arial", 12))
        self.console_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.result_hBox.addWidget(self.console_label)

        self.move_button = QPushButton("Predict with Generated Model", self)
        self.move_button.setFixedWidth(210)
        self.move_button.setEnabled(False)
        self.move_button.clicked.connect(self.open_other_window)

        self.result_hBox.addWidget(self.move_button)

        self.resultField = LogMessageViewer(self)
        self.resultField.setFixedSize(1173, 210)

        self.groupBox_h_layout = QHBoxLayout()
        self.groupBox_h_layout.addLayout(self.vBox1)
        self.groupBox_h_layout.addLayout(self.vBox2)

        self.groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        self.groupBox_layout.addLayout(self.groupBox_h_layout)
        self.groupBox_layout.addLayout(self.start_hbox)

        self.groupBox.setLayout(self.groupBox_layout)

        self.base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        self.base_layout.addWidget(self.groupBox)
        self.base_layout.addLayout(self.result_hBox)
        self.base_layout.addWidget(self.resultField)

        self.ROOT_DIR = None
        self.COCO_WEIGHTS_PATH = None
        self.ImageNet_WEIGHTS_PATH = None
        self.DEFAULT_LOGS_DIR = None

        self.training_flag = False
        self.model_path_trained = None

    def openDirectory(self):
        self.rootDirectoryPath = QFileDialog.getExistingDirectory(self, "Open Root Directory")
        if self.rootDirectoryPath != "":
            self.rootDirectory_label.setText("Root Directory : " + self.rootDirectoryPath)
            self.ROOT_DIR = self.rootDirectoryPath
            sys.path.append(self.ROOT_DIR)
            self.COCO_WEIGHTS_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
            self.DEFAULT_LOGS_DIR = os.path.join(self.ROOT_DIR, "logs")
            self.start_button.setEnabled(True)


    def startTraining(self):
        if self.sourceName_line.text() == "":
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Set Source Name")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.image_per_gpu_line.text().isdigit() == False or int(self.image_per_gpu_line.text()) < 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Image per GPU value is expected to be positive integer bigger than 0")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.steps_per_epoch_line.text().isdigit() == False or int(self.steps_per_epoch_line.text()) < 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Steps per Epoch value is expected to be positive integer bigger than 0")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.epochs_line.text().isdigit() == False or int(self.epochs_line.text()) < 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Epoch value is expected to be positive integer bigger than 0")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.gpuCount_line.text().isdigit() == False or int(self.gpuCount_line.text()) < 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("GPU Count value is expected to be positive integer bigger than 0")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.validation_steps_line.text().isdigit() == False or int(self.validation_steps_line.text()) < 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Validation Steps value is expected to be positive integer bigger than 0")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.isNumber(self.learningRate_line.text()) == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Learning Rate value is expected to be positive float or integer")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif float(self.learningRate_line.text()) <= 0.0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Learning Rate value is expected to be positive float or integer")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.isNumber(self.learningMomentum_line.text()) == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Learning Momentum value is expected to be positive float or integer")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif float(self.learningMomentum_line.text()) < 0.0 or float(self.learningMomentum_line.text()) > 1.0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Learning Momentum value is expected to be 0.0 to 1.0")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.isNumber(self.detection_min_confidence_line.text()) == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Minimum Confidence for Detection value is expected to be positive float or integer")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif float(self.detection_min_confidence_line.text()) < 0.0 \
                or float(self.detection_min_confidence_line.text()) > 1.0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Minimum Confidence for Detection is expected to be 0.0 to 1.0")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            self.resultField.moveCursor(QtGui.QTextCursor.End)
            self.resultField.clear()

            self.start_button.setEnabled(True)

            self.t = threading.Thread(target=self.training)
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            self.t.start()

    def isNumber(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False



    def training(self):
        sys.stdout = default_stdout
        sys.stderr = default_stderr

        self.move_button.setEnabled(False)
        self.start_button.setEnabled(False)

        XStream.stdout().messageWritten.connect(self.resultField.appendLogMessage)
        XStream.stderr().messageWritten.connect(self.resultField.appendLogMessage)

        print('proceeding....\n\n')

        K.clear_session()  # Tensor tensor 오류 수정 부분

        dataset_dir = os.path.join(self.rootDirectoryPath, "train")

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        self.class_names = ['BG']
        for i in range(len(annotations)):
            for j in annotations[i]['regions'].keys():
                a = annotations[i]['regions'][j]['region_attributes']['name']
                if a not in self.class_names:
                    self.class_names.append(a)

        Config.NAME = self.sourceName_line.text()
        Config.NUM_CLASSES = len(self.class_names)
        Config.BACKBONE = "resnet101" if self.backBone_line.currentText() == "RESNET 101" else "resnet50"
        Config.IMAGES_PER_GPU = int(self.image_per_gpu_line.text())
        Config.GPU_COUNT = int(self.gpuCount_line.text())
        Config.STEPS_PER_EPOCH = int(self.steps_per_epoch_line.text())
        Config.VALIDATION_STEPS = int(self.validation_steps_line.text())
        Config.LEARNING_RATE = float(self.learningRate_line.text())
        Config.LEARNING_MOMENTUM = float(self.learningMomentum_line.text())
        Config.DETECTION_MIN_CONFIDENCE = float(self.detection_min_confidence_line.text())

        config = Config()

        dataset_train = Dataset(self.sourceName_line.text())
        dataset_train.load_dataset(self.ROOT_DIR, "train")
        dataset_train.prepare()

        dataset_val = Dataset(self.sourceName_line.text())
        dataset_val.load_dataset(self.ROOT_DIR, "val")
        dataset_val.prepare()

        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=self.DEFAULT_LOGS_DIR)

        # Select weights file to load
        if self.weights_combo.currentText() == "COCO":
            weights_path = self.COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        elif self.weights_combo.currentText() == "Last":
            # Find last trained weights
            weights_path = model.find_last()
        elif self.weights_combo.currentText() == "ImageNet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()

        if self.weights_combo.currentText() == "COCO":
            # Exclude the last layers because they require a matching number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

        model.train(dataset_train, dataset_val,
                    learning_rate=self.LEARNING_RATE,
                    epochs=int(self.epochs_line.text()),
                    layers=self.layers_combo.currentText())  # heads → 3+ → 4+ → 5+ → all

        self.training_flag = True
        self.model_path_trained = model.find_last()
        self.move_button.setEnabled(True)
        self.start_button.setEnabled(True)

        self.messageBox_signal.run()

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

    def open_other_window(self):
        if self.model_path_trained:
            # self.close()
            self.predictionWindow = PredictWindow(self.model_path_trained, os.path.join(self.ROOT_DIR + "/train"))
            self.predictionWindow.exec()

class Dataset(utils.Dataset):

    def __init__(self, sourceName):
        super().__init__()
        self.sourceName = sourceName

    def load_dataset(self, dataset_dir, subset):

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        self.class_names = ['BG']
        for i in range(len(annotations)):
            for j in annotations[i]['regions'].keys():
                a = annotations[i]['regions'][j]['region_attributes']['name']
                if a not in self.class_names:
                    self.class_names.append(a)

        # Add classes. We have only one class to add.
        for i in range(1, len(self.class_names)):
            self.add_class(self.sourceName, i, self.class_names[i])

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stored in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                names = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]
                names = [r['region_attributes'] for r in a['regions']]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the data set is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                self.sourceName,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons, names=names)

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        """Returns:
        masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks."""

        # If not a balloon data set image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != self.sourceName:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        for i, p in enumerate(class_names):
            if p['name'] == self.class_names[1]:
                class_ids[i] = 1
            elif p['name'] == self.class_names[2]:
                class_ids[i] = 2
        class_ids = class_ids.astype(int)

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == self.sourceName:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(QtCore.QPoint(event.pos()))
        super(PhotoViewer, self).mousePressEvent(event)

class PredictWindow(QDialog):

    def __init__(self, model_path, dataset):
        super().__init__()
        self.title = "Predict"

        if not model_path:
            self.model_path = "Model Path : Unselected"
            self.model_select_flag = False
        else:
            self.model_path = model_path
            self.model_select_flag = True
        if not dataset:
            self.data_set_path = "Dataset Path : Unselected"
            self.dataset_select_flag = False
        else:
            self.data_set_path = dataset
            self.dataset_select_flag = True

        self.flag = False
        self.img_cnt = 0
        self.img_path = ''
        self.orig = []
        self.qimg_list = []
        self.model_changed = True
        self.before_model = None
        self.initUI()

    def initUI(self):
        self.viewer = PhotoViewer(self.parent())
        self.viewer.setMinimumSize(700, 700)
        self.qimage = None
        self.update_signal = MySignal()
        self.update_signal.signal.connect(self.img_update)
        self.setGeometry(300, 100, 1100, 800)
        self.setWindowTitle(self.title)
        self.model = QFileSystemModel()
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.model.setRootPath(QDir.currentPath())
        self.tree.expandAll()
        self.tree.setFixedWidth(300)
        self.tree.setAnimated(False)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)
        self.tree.doubleClicked.connect(self.doubleClick)

        for i in range(1, 4):
            self.tree.setColumnHidden(i, True)

        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.label1 = QLabel(self.model_path)
        self.label2 = QLabel(self.data_set_path)

        self.predict_button = QAction(QIcon('UI_image/predict_image.png'), 'Predict', self)
        self.predict_button.setShortcut('Ctrl+P')
        self.predict_button.setStatusTip('Predict application')
        self.predict_button.triggered.connect(self.linktest)

        self.save_button = QAction(QIcon('UI_image/save_button.png'), 'Save', self)
        self.save_button.setShortcut('Ctrl+S')
        self.save_button.setStatusTip('Save application')
        self.save_button.triggered.connect(self.save_Click)

        self.model_path_button = QAction(QIcon('UI_image/model_path_button.png'), 'Model Path', self)
        self.model_path_button.setStatusTip('Model Path Select application')
        self.model_path_button.triggered.connect(self.model_path_select)

        self.data_set_button = QAction(QIcon('UI_image/data_set_button.png'), 'Dataset Path', self)
        self.data_set_button.setStatusTip('Dataset Path Select application')
        self.data_set_button.triggered.connect(self.dataset_button_Click)

        mainwindow = QMainWindow()
        toolbar = QToolBar()
        mainwindow.addToolBar(toolbar)

        toolbar.addAction(self.predict_button)
        toolbar.addAction(self.save_button)
        toolbar.addAction(self.model_path_button)
        toolbar.addAction(self.data_set_button)

        toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        self.scaleSize_list = [600, 500, 350, 350]
        self.lbl_img = QLabel()

        self.viewer.photoClicked.connect(self.photoClicked)

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()
        self.hbox2 = QHBoxLayout()
        self.hbox3 = QHBoxLayout()
        self.gridbox = QGridLayout()
        self.hbox.addWidget(self.label1)
        self.hbox.addStretch(2)
        self.hbox3.addWidget(self.label2)
        self.hbox3.addStretch(2)

        self.hbox2.addWidget(self.tree)
        self.hbox2.addWidget(self.viewer)
        self.hbox2.addWidget(mainwindow)
        self.vbox.addLayout(self.hbox)
        self.vbox.addLayout(self.hbox3)
        self.vbox.addLayout(self.hbox2)

        self.setLayout(self.vbox)

        # self.show()

    def photoClicked(self, pos):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def doubleClick(self, index):
        img_extension = ["bmp", "rle", "dib", "jpg", "jpeg", "gif", "png", "tif", "tiff", "raw"]
        tmp = ""
        file_path = self.model.filePath(index)

        for i in range(len(file_path) - 1, 0, -1):
            if file_path[i] == ".":
                break
            else:
                tmp += file_path[i]

        if tmp[-1::-1] in img_extension:
            self.flag = True
            self.img_path = file_path
            self.viewer.setPhoto(QPixmap(self.img_path))
        elif self.model.isDir(index):
            self.flag = True
        else:
            self.showMessageBox("It is not an image file")

    def linktest(self):
        if self.flag and self.model_select_flag and self.dataset_select_flag:
            self.save_button.setEnabled(False)
            self.predict_button.setEnabled(False)
            self.data_set_button.setEnabled(False)
            self.model_path_button.setEnabled(False)

            self.t = threading.Thread(target=self.Predict_Click)
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            self.t.start()
        elif not self.flag:
            self.showMessageBox('Please select image')
        elif not self.model_select_flag:
            self.showMessageBox('Please select model')
        elif not self.dataset_select_flag:
            self.showMessageBox('Please select DataSet Directory')

    def Predict_Click(self):
        annotations = json.load(open(os.path.join(self.data_set_path + "/via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        self.class_names = ['BG']
        for i in range(len(annotations)):
            for j in annotations[i]['regions'].keys():
                a = annotations[i]['regions'][j]['region_attributes']['name']
                if a not in self.class_names:
                    self.class_names.append(a)

        class InferenceConfig(Config):
            # Give the configuration a recognizable name
            NAME = ""
            NUM_CLASSES = len(self.class_names)  # Background + Class Name
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        if self.model_changed:
            K.clear_session()
            self.weight = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
            self.weight.load_weights(self.model_path, by_name=True)
            self.model_changed = False
        self.draw = self.detect_and_color_splash(self.weight, image_path=self.img_path)
        self.update_signal.run()

        self.save_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.data_set_button.setEnabled(True)
        self.model_path_button.setEnabled(True)

    def model_path_select(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Options()
        # weights_file_name 이용
        self.model_path, _ = QFileDialog.getOpenFileName(self, "load h5 file", "", "h5 File (*.h5)", options=options)
        self.label1.setText("Model Path : " + self.model_path)
        if self.model_path:
            self.model_select_flag = True
            if self.before_model != self.model_path:
                self.before_model = self.model_path
                self.model_changed = True
            else:
                self.model_changed = False
        else:
            self.model_select_flag = False

    def showMessageBox(self, msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Error")
        msgBox.exec_()

    def save_Click(self):
        if self.qimage is None:
            self.showMessageBox('You can save the image after prediction')
            return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, extension = QFileDialog.getSaveFileName(self, "Save", "", "JPEG(*.jpg;*.jpeg;*.jpe;*.jfif);;\
        PNG(*.png);;TIF(*.tif;*.tiff)", options=options)

        if fileName:
            if extension == "JPEG(*.jpg;*.jpeg;*.jpe;*.jfif)":
                self.qimage.save(fileName + ".jpg")
            elif extension == "PNG(*.png)":
                self.qimage.save(fileName + ".png")
            elif extension == "TIF(*.tif;*.tiff)":
                self.qimage.save(fileName + ".tif")

    def dataset_button_Click(self):
        self.data_set_path = QFileDialog.getExistingDirectory(self, "Data Set Directory")
        self.label2.setText("Dataset Path : " + self.data_set_path)
        if self.data_set_path:
            self.dataset_select_flag = True
        else:
            self.dataset_select_flag = False

    @pyqtSlot()
    def img_update(self):
        self.qimage = ImageQt(Image.fromarray(self.draw))
        self.viewer.setPhoto(QPixmap().fromImage(self.qimage))

    # 이미지 불러올때 한글 경로가 있으면 바꿔줘야함
    def hangulFilePathImageRead(self, filePath):
        stream = open(filePath.encode("utf-8"), "rb")
        bytes = bytearray(stream.read())
        numpyArray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def detect_and_color_splash(self, model, image_path=None):
        assert image_path
        # Image or video?
        if image_path:
            # Read image
            image = skimage.io.imread(image_path)
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            return self.masking(image, r['rois'], r['masks'], r['class_ids'], self.class_names,
                                r['scores'], show_bbox=True, show_mask=True, title="Prediction")

    def masking(self, image, boxes, masks, class_ids, class_names,
                scores=None, title="",
                figsize=(16, 16), ax=None,
                show_mask=True, show_bbox=True,
                colors=None, captions=None):
        # Number of instances
        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            auto_show = False
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
            auto_show = True

        masked_image = image.astype(np.uint32).copy()
        masked_image = (masked_image.astype(np.uint8))
        colors = colors or self.random_colors(N)
        for i in range(N):
            color = colors[i]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                for c in range(3):
                    masked_image[y1:y1 + 4, x1:x2, c] = masked_image[y1:y1 + 4, x1:x2, c] * (1 - 0.5) + 0.5 * color[
                        c] * 255
                    masked_image[y2:y2 + 4, x1:x2, c] = masked_image[y2:y2 + 4, x1:x2, c] * (1 - 0.5) + 0.5 * color[
                        c] * 255
                    masked_image[y1:y2, x1:x1 + 4, c] = masked_image[y1:y2, x1:x1 + 4, c] * (1 - 0.5) + 0.5 * color[
                        c] * 255
                    masked_image[y1:y2, x2:x2 + 4, c] = masked_image[y1:y2, x2:x2 + 4, c] * (1 - 0.5) + 0.5 * color[
                        c] * 255

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = self.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            w, h, d = masked_image.shape
            for verts in contours:
                for a in range(len(verts)):
                    for c in range(3):
                        if (int(verts[a][0]) < 0 or int(verts[a][0]) >= w or int(verts[a][1]) < 0 or int(
                                verts[a][1]) >= h):
                            continue
                        masked_image[int(verts[a][0]), int(verts[a][1]), c] = masked_image[int(verts[a][0]), int(
                            verts[a][1]), c] * (1 - 0.5) + 0.5 * color[c] * 255

        for i in range(N):
            y1, x1, y2, x2 = boxes[i]
            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            cv2.putText(masked_image, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        if auto_show:
            return masked_image

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def random_colors(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

if __name__ == '__main__':
    app = QApplication(sys.argv)
    segmentationUI = SegmentationUI()
    segmentationUI.show()
    sys.exit(app.exec_())