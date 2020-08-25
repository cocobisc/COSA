import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

import cv2
import time
import logging
import threading
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
from keras import backend as K  # Tensor tensor 오류 수정 부분
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import evaluate
from ..utils.keras_version import check_keras_version
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import argparse
import warnings
import keras
import keras.preprocessing.image
import tensorflow as tf

class MySignal(QObject):
    signal = pyqtSignal()
    def run(self):
        self.signal.emit()

class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, filename='training_plot.jpg'):
        self.filename = filename
        self.graph_update_signal = MySignal()
        self.graph_update_signal2 = MySignal()
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.regression_losses = []
        self.classification_losses = []
        self.val_losses = []
        self.val_regression_losses = []
        self.val_classification_losses = []
        self.mAP = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.val_regression_losses.append(logs.get('val_regression_loss'))
        self.val_classification_losses.append(logs.get('val_classification_loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.mAP.append(logs.get('mAP'))
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = range(1, len(self.val_losses)+1)

            # You can chose the style of your preference
            # print(plt.style.available)

            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            self.figure2 = plt.figure()

            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_regression_losses, label="val_regression_loss")
            plt.plot(N, self.val_classification_losses, label="val_classification_loss")
            plt.plot(N, self.mAP, label = "mAP")
            plt.title("Each Validation Loss and mAP [Epoch {}]".format(epoch+1))
            plt.xlabel("Epoch #")
            plt.ylabel("Each Loss & mAP")
            plt.legend()
            self.canvas2 = FigureCanvas(self.figure2)
            self.canvas2.draw()
            size = self.canvas2.size()
            width, height = size.width(), size.height()
            self.im2 = QImage(self.canvas2.buffer_rgba(), width, height, QImage.Format_ARGB32)
            self.graph_update_signal2.run()
            plt.close()

    def on_batch_end(self, epoch, logs={}):
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.regression_losses.append(logs.get('regression_loss'))
        self.classification_losses.append(logs.get('classification_loss'))
        self.losses.append(logs.get('loss'))
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            N = range(1, len(self.losses)+1)

            # You can chose the style of your preference
            # print(plt.style.available)

            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            self.figure = plt.figure()

            plt.plot(N, self.losses, label = "loss")
            plt.plot(N, self.regression_losses, label="regression_loss")
            plt.plot(N, self.classification_losses, label="classification_loss")
            plt.title("Each Loss [Step {}]".format(epoch+1))
            plt.xlabel("Step #")
            plt.ylabel("Each Loss")
            plt.legend()
            self.canvas = FigureCanvas(self.figure)
            self.canvas.draw()
            size = self.canvas.size()
            width, height = size.width(), size.height()
            self.im = QImage(self.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
            self.graph_update_signal.run()
            plt.close()

class ImageDataGeneratorSettingDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Data Generator Configuration")
        self.setGeometry(840, 320, 300, 300)

        groupBox = QGroupBox()

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)
        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        groupBox.setLayout(groupBox_layout)

        self.rotation_label        = QLabel('Rotation Range : ')
        self.rotation_min_line     = QLineEdit()
        self.rotation_label_middle = QLabel('~')
        self.rotation_max_line     = QLineEdit()
        self.shearing_label = QLabel('Shearing Range : ')
        self.shearing_min_line = QLineEdit()
        self.shearing_label_middle = QLabel('~')
        self.shearing_max_line = QLineEdit()

        self.translation_label = QLabel('(X, Y) Translation Range : ')
        self.translation_label_open = QLabel('(')
        self.translation_xmin_line = QLineEdit()
        self.translation_label_comma1 = QLabel(',')
        self.translation_label_comma2 = QLabel(',')
        self.translation_ymin_line = QLineEdit()
        self.translation_label_middle = QLabel(') ~ (')
        self.translation_xmax_line = QLineEdit()
        self.translation_ymax_line = QLineEdit()
        self.translation_label_close = QLabel(')')

        self.scaling_label = QLabel('(X, Y) Scaling Range : ')
        self.scaling_label_open = QLabel('(')
        self.scaling_xmin_line = QLineEdit()
        self.scaling_label_comma1 = QLabel(',')
        self.scaling_label_comma2 = QLabel(',')
        self.scaling_ymin_line = QLineEdit()
        self.scaling_label_middle = QLabel(') ~ (')
        self.scaling_xmax_line = QLineEdit()
        self.scaling_ymax_line = QLineEdit()
        self.scaling_label_close = QLabel(')')

        self.flip_label=QLabel('X, Y Flip Chance : ')
        self.flip_x_line=QLineEdit()
        self.flip_label_comma=QLabel(',')
        self.flip_y_line=QLineEdit()

        self.rotation_min_line.setText('-0.1')
        self.rotation_max_line.setText('0.1')
        self.shearing_max_line.setText('0.1')
        self.shearing_min_line.setText('-0.1')
        self.translation_xmin_line.setText('-0.1')
        self.translation_ymin_line.setText('-0.1')
        self.translation_xmax_line.setText('0.1')
        self.translation_ymax_line.setText('0.1')
        self.scaling_xmin_line.setText('0.9')
        self.scaling_ymin_line.setText('0.9')
        self.scaling_xmax_line.setText('1.1')
        self.scaling_ymax_line.setText('1.1')
        self.flip_x_line.setText('0.5')
        self.flip_y_line.setText('0.5')

        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(self.rotation_label)
        hlayout1.addSpacing(70)
        hlayout1.addWidget(self.rotation_min_line)
        hlayout1.addWidget(self.rotation_label_middle)
        hlayout1.addWidget(self.rotation_max_line)
        groupBox_layout.addLayout(hlayout1)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.shearing_label)
        hlayout2.addSpacing(68)
        hlayout2.addWidget(self.shearing_min_line)
        hlayout2.addWidget(self.shearing_label_middle)
        hlayout2.addWidget(self.shearing_max_line)
        groupBox_layout.addLayout(hlayout2)

        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(self.translation_label)
        hlayout3.addSpacing(10)
        hlayout3.addWidget(self.translation_label_open)
        hlayout3.addWidget(self.translation_xmin_line)
        hlayout3.addWidget(self.translation_label_comma1)
        hlayout3.addWidget(self.translation_ymin_line)
        hlayout3.addWidget(self.translation_label_middle)
        hlayout3.addWidget(self.translation_xmax_line)
        hlayout3.addWidget(self.translation_label_comma2)
        hlayout3.addWidget(self.translation_ymax_line)
        hlayout3.addWidget(self.translation_label_close)
        groupBox_layout.addLayout(hlayout3)

        hlayout4 = QHBoxLayout()
        hlayout4.addWidget(self.scaling_label)
        hlayout4.addSpacing(32)
        hlayout4.addWidget(self.scaling_label_open)
        hlayout4.addWidget(self.scaling_xmin_line)
        hlayout4.addWidget(self.scaling_label_comma1)
        hlayout4.addWidget(self.scaling_ymin_line)
        hlayout4.addWidget(self.scaling_label_middle)
        hlayout4.addWidget(self.scaling_xmax_line)
        hlayout4.addWidget(self.scaling_label_comma2)
        hlayout4.addWidget(self.scaling_ymax_line)
        hlayout4.addWidget(self.scaling_label_close)
        groupBox_layout.addLayout(hlayout4)

        hlayout5 = QHBoxLayout()
        hlayout5.addWidget(self.flip_label)
        hlayout5.addSpacing(70)
        hlayout5.addWidget(self.flip_x_line)
        hlayout5.addWidget(self.flip_label_comma)
        hlayout5.addWidget(self.flip_y_line)
        groupBox_layout.addLayout(hlayout5)

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
            self.rotation_min_line.setText('-0.1')
            self.rotation_max_line.setText('0.1')
            self.shearing_max_line.setText('0.1')
            self.shearing_min_line.setText('-0.1')
            self.translation_xmin_line.setText('-0.1')
            self.translation_ymin_line.setText('-0.1')
            self.translation_xmax_line.setText('0.1')
            self.translation_ymax_line.setText('0.1')
            self.scaling_xmin_line.setText('0.9')
            self.scaling_ymin_line.setText('0.9')
            self.scaling_xmax_line.setText('1.1')
            self.scaling_ymax_line.setText('1.1')
            self.flip_x_line.setText('0.5')
            self.flip_y_line.setText('0.5')
            self.close()
        elif self.sender().text() == "Confirm":
            if self.isNumber(self.rotation_min_line.text()) == False or self.isNumber(self.rotation_max_line.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("rotation value is expected to be float")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.shearing_min_line.text()) == False or self.isNumber(self.shearing_max_line.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("shearing value is expected to be float")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.translation_xmax_line.text()) == False or self.isNumber(self.translation_xmin_line.text()) == False or\
                    self.isNumber(self.translation_ymin_line.text()) == False or self.isNumber(self.translation_ymax_line.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("translation value is expected to be float")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.scaling_xmax_line.text()) == False or self.isNumber(self.scaling_xmin_line.text()) == False or\
                    self.isNumber(self.scaling_ymin_line.text()) == False or self.isNumber(self.scaling_ymax_line.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("scaling value is expected to be float")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif self.isNumber(self.flip_x_line.text()) == False or self.isNumber(self.flip_y_line.text()) == False:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Flip Chance Value is expected to be float")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif float(self.rotation_min_line.text()) > float(self.rotation_max_line.text()):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Left Rotation Value is must smaller than Right Value")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif float(self.shearing_min_line.text()) > float(self.shearing_max_line.text()):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Left Shearing Value is must smaller than Right Value")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif float(self.translation_xmin_line.text()) > float(self.translation_xmax_line.text()) or float(self.translation_ymin_line.text()) > float(self.translation_ymax_line.text()):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Left X or Y is must smaller than Right X or Y")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif float(self.scaling_xmin_line.text()) > float(self.scaling_xmax_line.text()) or float(self.scaling_ymin_line.text()) > float(self.scaling_ymax_line.text()):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Left X or Y is must smaller than Right X or Y")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif float(self.scaling_xmax_line.text()) <=0 or float(self.scaling_xmin_line.text()) <=0 or float(self.scaling_ymax_line.text()) <=0 or float(self.scaling_ymin_line.text()) <=0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Scaling Value is must bigger than 0")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif float(self.flip_x_line.text()) < 0 or float(self.flip_x_line.text()) > 1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Flip X Chance is must in 0<=X<=1")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            elif float(self.flip_y_line.text()) < 0 or float(self.flip_y_line.text()) > 1:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Flip Y Chance is must in 0<=Y<=1")
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


class TrainingWindow(QDialog):
    def __init__(self, trainingTest):
        super().__init__()
        self.trainingTest = trainingTest
        self.title = 'Object Detection Training'
        self.plot_losses = TrainingPlot()
        self.initUI()

    def initUI(self):
        self.change_button_signal = MySignal()

        self.setGeometry(700, 200, 800, 600)
        self.setWindowTitle(self.title)
        self.dataset_label = QLabel("Training Data Set : ")
        self.model_path_label = QLabel("Model Save Path : ")
        self.model_name_label = QLabel("Model Name")
        self.batch_size_label = QLabel("Batch Size")
        self.epoch_label = QLabel("Epochs")
        self.step_label = QLabel("Steps")
        self.Learning_rate_label = QLabel("Learning Rate")
        self.anchor_label = QLabel("Anchor")
        self.dataset_dir_button = QPushButton("Set Training Data Set Directory")
        self.dataset_dir_button.clicked.connect(self.setting_dataset)
        self.anchor_checkBox = QCheckBox()
        self.anchor_checkBox.stateChanged.connect(self.anchor_clicked)
        self.anchor_size_label = QLabel("size")
        self.anchor_stride_label = QLabel("stride")
        self.anchor_ratio_label = QLabel("ratio")
        self.anchor_scale_label = QLabel("scale")
        self.anchor_size1_line = QLineEdit()
        self.anchor_size2_line = QLineEdit()
        self.anchor_size3_line = QLineEdit()
        self.anchor_size4_line = QLineEdit()
        self.anchor_size5_line = QLineEdit()
        self.anchor_stride_line = QLineEdit()
        self.anchor_stride2_line = QLineEdit()
        self.anchor_stride3_line = QLineEdit()
        self.anchor_stride4_line = QLineEdit()
        self.anchor_stride5_line = QLineEdit()
        self.anchor_ratio_line = QLineEdit()
        self.anchor_ratio2_line = QLineEdit()
        self.anchor_ratio3_line = QLineEdit()
        self.anchor_scale_line = QLineEdit()
        self.anchor_scale2_line = QLineEdit()
        self.anchor_scale3_line = QLineEdit()
        self.anchor_size1_line.setEnabled(False)
        self.anchor_size2_line.setEnabled(False)
        self.anchor_size3_line.setEnabled(False)
        self.anchor_size4_line.setEnabled(False)
        self.anchor_size5_line.setEnabled(False)
        self.anchor_stride_line.setEnabled(False)
        self.anchor_stride2_line.setEnabled(False)
        self.anchor_stride3_line.setEnabled(False)
        self.anchor_stride4_line.setEnabled(False)
        self.anchor_stride5_line.setEnabled(False)
        self.anchor_ratio_line.setEnabled(False)
        self.anchor_ratio2_line.setEnabled(False)
        self.anchor_ratio3_line.setEnabled(False)
        self.anchor_scale_line.setEnabled(False)
        self.anchor_scale2_line.setEnabled(False)
        self.anchor_scale3_line.setEnabled(False)
        self.anchor_size1_line.setText('32')
        self.anchor_size2_line.setText('64')
        self.anchor_size3_line.setText('128')
        self.anchor_size4_line.setText('256')
        self.anchor_size5_line.setText('512')
        self.anchor_stride_line.setText('8')
        self.anchor_stride2_line.setText('16')
        self.anchor_stride3_line.setText('32')
        self.anchor_stride4_line.setText('64')
        self.anchor_stride5_line.setText('128')
        self.anchor_ratio_line.setText('0.5')
        self.anchor_ratio2_line.setText('1')
        self.anchor_ratio3_line.setText('2')
        self.anchor_scale_line.setText('1')
        self.anchor_scale2_line.setText('1.2')
        self.anchor_scale3_line.setText('1.6')
        self.anchor_size_line_list = []
        self.anchor_size_line_list.append(self.anchor_size1_line)
        self.anchor_size_line_list.append(self.anchor_size2_line)
        self.anchor_size_line_list.append(self.anchor_size3_line)
        self.anchor_size_line_list.append(self.anchor_size4_line)
        self.anchor_size_line_list.append(self.anchor_size5_line)
        self.anchor_stride_line_list = []
        self.anchor_stride_line_list.append(self.anchor_stride_line)
        self.anchor_stride_line_list.append(self.anchor_stride2_line)
        self.anchor_stride_line_list.append(self.anchor_stride3_line)
        self.anchor_stride_line_list.append(self.anchor_stride4_line)
        self.anchor_stride_line_list.append(self.anchor_stride5_line)
        self.anchor_ratio_line_list = []
        self.anchor_ratio_line_list.append(self.anchor_ratio_line)
        self.anchor_ratio_line_list.append(self.anchor_ratio2_line)
        self.anchor_ratio_line_list.append(self.anchor_ratio3_line)
        self.anchor_scale_line_list = []
        self.anchor_scale_line_list.append(self.anchor_scale_line)
        self.anchor_scale_line_list.append(self.anchor_scale2_line)
        self.anchor_scale_line_list.append(self.anchor_scale3_line)

        self.model_path_button = QPushButton("Choose Path and Set Model Name")
        self.model_path_button.clicked.connect(self.saving_browser)
        self.startTraining_button = QPushButton("Confrim")
        self.startTraining_button.setEnabled(False)
        self.startTraining_button.clicked.connect(self.confirm)
        self.startEvaluate_button = QPushButton("Cancel")
        self.startEvaluate_button.setEnabled(True)
        self.startEvaluate_button.setFixedSize(150, 40)
        self.startEvaluate_button.clicked.connect(self.cancel)
        self.model_name_line = QLineEdit()
        self.model_name_line.setReadOnly(True)
        self.batchsize_line = QLineEdit()
        self.batchsize_line.setText("1")
        self.epochs_line = QLineEdit()
        self.epochs_line.setText("10")
        self.steps_line = QLineEdit()
        self.steps_line.setText('1000')
        self.learning_rate_line = QLineEdit()
        self.learning_rate_line.setText("0.00001")
        self.batchsize_line.setFixedWidth(100)
        self.epochs_line.setFixedWidth(100)
        self.steps_line.setFixedWidth(100)
        self.learning_rate_line.setFixedWidth(100)
        self.anchor_scale_line.setFixedWidth(30)
        self.anchor_scale2_line.setFixedWidth(30)
        self.anchor_scale3_line.setFixedWidth(30)
        self.anchor_size1_line.setFixedWidth(30)
        self.anchor_size2_line.setFixedWidth(30)
        self.anchor_size3_line.setFixedWidth(30)
        self.anchor_size4_line.setFixedWidth(30)
        self.anchor_size5_line.setFixedWidth(30)
        self.anchor_ratio_line.setFixedWidth(30)
        self.anchor_ratio2_line.setFixedWidth(30)
        self.anchor_ratio3_line.setFixedWidth(30)
        self.anchor_stride_line.setFixedWidth(30)
        self.anchor_stride2_line.setFixedWidth(30)
        self.anchor_stride3_line.setFixedWidth(30)
        self.anchor_stride4_line.setFixedWidth(30)
        self.anchor_stride5_line.setFixedWidth(30)
        self.image_minSize_label = QLabel('image-min-size')
        self.image_maxSize_label = QLabel('image-max-size')
        self.image_minSize_line = QLineEdit()
        self.image_maxSize_line = QLineEdit()
        self.image_minSize_line.setText('900')
        self.image_minSize_line.setFixedWidth(100)
        self.image_maxSize_line.setText('1333')
        self.image_maxSize_line.setFixedWidth(100)
        self.monitor_label = QLabel("set Monitor")
        self.monitor_comboBox = QComboBox()
        self.monitor_comboBox.addItems(['val_loss', 'mAP'])
        self.hbox8 = QHBoxLayout()
        self.hbox9 = QHBoxLayout()
        self.hbox10 = QHBoxLayout()
        self.hbox11 = QHBoxLayout()
        self.hbox13 = QHBoxLayout()
        self.hbox14 = QHBoxLayout()
        self.hbox15 = QHBoxLayout()
        self.hbox16 = QHBoxLayout()
        self.hbox13.addWidget(self.anchor_label)
        self.hbox13.addWidget(self.anchor_checkBox)

        self.hbox13.addSpacing(10)
        self.hbox13.addWidget(self.anchor_size_label)
        self.hbox13.addWidget(self.anchor_size1_line)
        self.hbox13.addWidget(self.anchor_size2_line)
        self.hbox13.addWidget(self.anchor_size3_line)
        self.hbox13.addWidget(self.anchor_size4_line)
        self.hbox13.addWidget(self.anchor_size5_line)
        self.hbox14.addSpacing(90)
        self.hbox14.addWidget(self.anchor_stride_label)
        self.hbox14.addWidget(self.anchor_stride_line)
        self.hbox14.addWidget(self.anchor_stride2_line)
        self.hbox14.addWidget(self.anchor_stride3_line)
        self.hbox14.addWidget(self.anchor_stride4_line)
        self.hbox14.addWidget(self.anchor_stride5_line)
        self.hbox15.addSpacing(100)
        self.hbox15.addWidget(self.anchor_ratio_label)
        self.hbox15.addWidget(self.anchor_ratio_line)
        self.hbox15.addWidget(self.anchor_ratio2_line)
        self.hbox15.addWidget(self.anchor_ratio3_line)
        self.hbox16.addSpacing(90)
        self.hbox16.addWidget(self.anchor_scale_label)
        self.hbox16.addWidget(self.anchor_scale_line)
        self.hbox16.addWidget(self.anchor_scale2_line)
        self.hbox16.addWidget(self.anchor_scale3_line)
        self.hbox17 = QHBoxLayout()
        self.hbox18 = QHBoxLayout()

        self.hbox17.addWidget(self.image_minSize_label)
        self.hbox17.addWidget(self.image_minSize_line)
        self.hbox18.addWidget(self.image_maxSize_label)
        self.hbox18.addWidget(self.image_maxSize_line)
        self.hbox8.addSpacing(20)
        self.hbox8.addWidget(self.dataset_label)
        self.hbox8.addSpacing(20)

        self.hbox9.addSpacing(20)
        self.hbox9.addWidget(self.dataset_dir_button)
        self.hbox9.addSpacing(20)

        self.hbox10.addSpacing(20)
        self.hbox10.addWidget(self.model_path_label)
        self.hbox10.addSpacing(20)

        self.hbox11.addSpacing(20)
        self.hbox11.addWidget(self.model_path_button)
        self.hbox11.addSpacing(20)

        self.vbox1 = QVBoxLayout()

        self.vbox1.addSpacing(10)
        self.vbox1.addLayout(self.hbox8)
        self.vbox1.addSpacing(10)
        self.vbox1.addLayout(self.hbox9)
        self.vbox1.addSpacing(10)
        self.vbox1.addLayout(self.hbox10)
        self.vbox1.addSpacing(10)
        self.vbox1.addLayout(self.hbox11)

        self.hbox1 = QHBoxLayout()
        self.hbox1.addSpacing(20)
        self.hbox1.addWidget(self.model_name_label)
        self.hbox1.addWidget(self.model_name_line)
        self.hbox1.addSpacing(20)
        self.vbox1.addSpacing(10)
        self.vbox1.addLayout(self.hbox1)

        self.hbox2 = QHBoxLayout()
        self.hbox3 = QHBoxLayout()
        self.hbox4 = QHBoxLayout()
        self.hbox5 = QHBoxLayout()
        self.hbox6 = QHBoxLayout()

        self.vbox2 = QVBoxLayout()
        self.vbox3 = QVBoxLayout()

        self.hbox2.addStretch(1)
        self.hbox2.addLayout(self.vbox2)
        self.hbox2.addStretch(1)
        self.hbox2.addLayout(self.vbox3)
        self.hbox2.addStretch(1)

        self.vbox2.addLayout(self.hbox13)
        self.vbox2.addLayout(self.hbox14)
        self.vbox2.addLayout(self.hbox15)
        self.vbox2.addLayout(self.hbox16)

        imageGenerator_hbox = QHBoxLayout()
        self.imageGenerator_check = QCheckBox("Image Data Generator", self)
        self.imageGenerator_check.stateChanged.connect(self.imageGenerator_checked)
        self.imageGenerator_button = QPushButton("Set Image Data Generator Parameters", self)
        self.imageDataGeneratorSettingWindow = ImageDataGeneratorSettingDialog()
        self.imageGenerator_button.clicked.connect(self.imageGenerator_clicked)
        self.imageGenerator_button.setEnabled(False)

        imageGenerator_hbox.addWidget(self.imageGenerator_check)
        self.vbox2.addLayout(imageGenerator_hbox)
        self.vbox2.addWidget(self.imageGenerator_button)

        self.vbox3.addLayout(self.hbox3)
        self.vbox3.addLayout(self.hbox4)
        self.vbox3.addLayout(self.hbox5)
        self.vbox3.addLayout(self.hbox6)
        self.vbox3.addLayout(self.hbox17)
        self.vbox3.addLayout(self.hbox18)
        self.hbox3.addStretch(1)
        self.hbox3.addWidget(self.batch_size_label)
        self.hbox3.addWidget(self.batchsize_line)
        self.hbox3.alignment()

        self.hbox4.addStretch(1)
        self.hbox4.addWidget(self.step_label)
        self.hbox4.addWidget(self.steps_line)
        self.hbox4.alignment()

        self.hbox5.addStretch(1)
        self.hbox5.addWidget(self.epoch_label)
        self.hbox5.addWidget(self.epochs_line)
        self.hbox5.alignment()

        self.hbox6.addStretch(1)
        self.hbox6.addWidget(self.Learning_rate_label)
        self.hbox6.addWidget(self.learning_rate_line)
        self.hbox6.alignment()

        self.hbox12 = QHBoxLayout()

        self.vbox1.addSpacing(10)
        self.vbox1.addLayout(self.hbox2)
        self.vbox1.addSpacing(10)
        self.monitor_comboBox.setFixedWidth(100)
        self.hbox12.addSpacing(180)
        self.hbox12.addWidget(self.monitor_label)
        self.hbox12.addWidget(self.monitor_comboBox)
        self.hbox12.addSpacing(100)
        self.startTraining_button.setFixedSize(150, 40)
        self.hbox12.addWidget(self.startTraining_button)
        self.hbox12.addSpacing(40)
        self.hbox12.addWidget((self.startEvaluate_button))
        self.vbox1.addLayout(self.hbox12)
        self.vbox1.addSpacing(10)
        self.hbox7 = QVBoxLayout()
        self.vbox1.addLayout(self.hbox7)
        self.setLayout(self.vbox1)

        self.trainingDataSet_directoryName = ""

        self.dataset_label_buffer = "Training Data Set : "
        self.trainingDataSet_directoryName_buffer = ""
        self.model_path_label_buffer = "Model Save Path : "
        self.model_name_line_buffer = ""
        self.anchor_checkBox_bool = False
        self.anchor_size1_line_buffer = "32"
        self.anchor_size2_line_buffer = "64"
        self.anchor_size3_line_buffer = "128"
        self.anchor_size4_line_buffer = "256"
        self.anchor_size5_line_buffer = "512"
        self.anchor_stride_line_buffer = "8"
        self.anchor_stride2_line_buffer = "16"
        self.anchor_stride3_line_buffer = "32"
        self.anchor_stride4_line_buffer = "64"
        self.anchor_stride5_line_buffer = "128"
        self.anchor_ratio_line_buffer = "0.5"
        self.anchor_ratio2_line_buffer = "1"
        self.anchor_ratio3_line_buffer = "2"
        self.anchor_scale_line_buffer = "1"
        self.anchor_scale2_line_buffer = "1.2"
        self.anchor_scale3_line_buffer = "1.6"
        self.image_minSize_line_buffer = "900"
        self.image_maxSize_line_buffer = "1333"
        self.batchsize_line_buffer = "1"
        self.steps_line_buffer = "1000"
        self.epochs_line_buffer = "10"
        self.learning_rate_line_buffer = "0.00001"
        self.monitor_comboBox_buffer = "val_loss"
        self.imageGenerator_check_buffer = False
        self.rotation_min_line_buffer = "-0.1"
        self.rotation_max_line_buffer = "0.1"
        self.shearing_min_line_buffer = "-0.1"
        self.shearing_max_line_buffer = "0.1"
        self.translation_xmin_line_buffer = "-0.1"
        self.translation_ymin_line_buffer = "-0.1"
        self.translation_xmax_line_buffer = "0.1"
        self.translation_ymax_line_buffer = "0.1"
        self.scaling_xmin_line_buffer = "0.9"
        self.scaling_ymin_line_buffer = "0.9"
        self.scaling_xmax_line_buffer = "1.1"
        self.scaling_ymax_line_buffer = "1.1"
        self.flip_x_line_buffer = "0.5"
        self.flip_y_line_buffer = "0.5"

    def closeEvent(self, event):
        self.dataset_label.setText(self.dataset_label_buffer)
        self.trainingDataSet_directoryName = self.trainingDataSet_directoryName_buffer
        self.model_path_label.setText(self.model_path_label_buffer)
        self.model_name_line.setText(self.model_name_line_buffer)
        self.anchor_checkBox.setChecked(self.anchor_checkBox_bool)
        self.anchor_size1_line.setText(self.anchor_size1_line_buffer)
        self.anchor_size2_line.setText(self.anchor_size2_line_buffer)
        self.anchor_size3_line.setText(self.anchor_size3_line_buffer)
        self.anchor_size4_line.setText(self.anchor_size4_line_buffer)
        self.anchor_size5_line.setText(self.anchor_size5_line_buffer)
        self.anchor_stride_line.setText(self.anchor_stride_line_buffer)
        self.anchor_stride2_line.setText(self.anchor_stride2_line_buffer)
        self.anchor_stride3_line.setText(self.anchor_stride3_line_buffer)
        self.anchor_stride4_line.setText(self.anchor_stride4_line_buffer)
        self.anchor_stride5_line.setText(self.anchor_stride5_line_buffer)
        self.anchor_ratio_line.setText(self.anchor_ratio_line_buffer)
        self.anchor_ratio2_line.setText(self.anchor_ratio2_line_buffer)
        self.anchor_ratio3_line.setText(self.anchor_ratio3_line_buffer)
        self.anchor_scale_line.setText(self.anchor_scale_line_buffer)
        self.anchor_scale2_line.setText(self.anchor_scale2_line_buffer)
        self.anchor_scale3_line.setText(self.anchor_scale3_line_buffer)
        self.image_minSize_line.setText(self.image_minSize_line_buffer)
        self.image_maxSize_line.setText(self.image_maxSize_line_buffer)
        self.batchsize_line.setText(self.batchsize_line_buffer)
        self.steps_line.setText(self.steps_line_buffer)
        self.epochs_line.setText(self.epochs_line_buffer)
        self.learning_rate_line.setText(self.learning_rate_line_buffer)
        self.monitor_comboBox.setCurrentText(self.monitor_comboBox_buffer)
        self.imageGenerator_check.setChecked(self.imageGenerator_check_buffer)
        self.imageDataGeneratorSettingWindow.rotation_min_line.setText(self.rotation_min_line_buffer)
        self.imageDataGeneratorSettingWindow.rotation_max_line.setText(self.rotation_max_line_buffer)
        self.imageDataGeneratorSettingWindow.shearing_min_line.setText(self.shearing_min_line_buffer)
        self.imageDataGeneratorSettingWindow.shearing_max_line.setText(self.shearing_max_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_xmin_line.setText(self.translation_xmin_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_xmax_line.setText(self.translation_xmax_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_ymin_line.setText(self.translation_ymin_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_ymax_line.setText(self.translation_ymax_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_xmin_line.setText(self.scaling_xmin_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_ymin_line.setText(self.scaling_ymin_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_xmax_line.setText(self.scaling_xmax_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_ymax_line.setText(self.scaling_ymax_line_buffer)
        self.imageDataGeneratorSettingWindow.flip_x_line.setText(self.flip_x_line_buffer)
        self.imageDataGeneratorSettingWindow.flip_y_line.setText

    def cancel(self):
        self.dataset_label.setText(self.dataset_label_buffer)
        self.trainingDataSet_directoryName = self.trainingDataSet_directoryName_buffer
        self.model_path_label.setText(self.model_path_label_buffer)
        self.model_name_line.setText(self.model_name_line_buffer)
        self.anchor_checkBox.setChecked(self.anchor_checkBox_bool)
        self.anchor_size1_line.setText(self.anchor_size1_line_buffer)
        self.anchor_size2_line.setText(self.anchor_size2_line_buffer)
        self.anchor_size3_line.setText(self.anchor_size3_line_buffer)
        self.anchor_size4_line.setText(self.anchor_size4_line_buffer)
        self.anchor_size5_line.setText(self.anchor_size5_line_buffer)
        self.anchor_stride_line.setText(self.anchor_stride_line_buffer)
        self.anchor_stride2_line.setText(self.anchor_stride2_line_buffer)
        self.anchor_stride3_line.setText(self.anchor_stride3_line_buffer)
        self.anchor_stride4_line.setText(self.anchor_stride4_line_buffer)
        self.anchor_stride5_line.setText(self.anchor_stride5_line_buffer)
        self.anchor_ratio_line.setText(self.anchor_ratio_line_buffer)
        self.anchor_ratio2_line.setText(self.anchor_ratio2_line_buffer)
        self.anchor_ratio3_line.setText(self.anchor_ratio3_line_buffer)
        self.anchor_scale_line.setText(self.anchor_scale_line_buffer)
        self.anchor_scale2_line.setText(self.anchor_scale2_line_buffer)
        self.anchor_scale3_line.setText(self.anchor_scale3_line_buffer)
        self.image_minSize_line.setText(self.image_minSize_line_buffer)
        self.image_maxSize_line.setText(self.image_maxSize_line_buffer)
        self.batchsize_line.setText(self.batchsize_line_buffer)
        self.steps_line.setText(self.steps_line_buffer)
        self.epochs_line.setText(self.epochs_line_buffer)
        self.learning_rate_line.setText(self.learning_rate_line_buffer)
        self.monitor_comboBox.setCurrentText(self.monitor_comboBox_buffer)
        self.imageGenerator_check.setChecked(self.imageGenerator_check_buffer)
        self.imageDataGeneratorSettingWindow.rotation_min_line.setText(self.rotation_min_line_buffer)
        self.imageDataGeneratorSettingWindow.rotation_max_line.setText(self.rotation_max_line_buffer)
        self.imageDataGeneratorSettingWindow.shearing_min_line.setText(self.shearing_min_line_buffer)
        self.imageDataGeneratorSettingWindow.shearing_max_line.setText(self.shearing_max_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_xmin_line.setText(self.translation_xmin_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_xmax_line.setText(self.translation_xmax_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_ymin_line.setText(self.translation_ymin_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_ymax_line.setText(self.translation_ymax_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_xmin_line.setText(self.translation_xmin_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_ymin_line.setText(self.translation_ymin_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_xmax_line.setText(self.translation_xmax_line_buffer)
        self.imageDataGeneratorSettingWindow.translation_ymax_line.setText(self.translation_ymax_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_xmin_line.setText(self.scaling_xmin_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_ymin_line.setText(self.scaling_ymin_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_xmax_line.setText(self.scaling_xmax_line_buffer)
        self.imageDataGeneratorSettingWindow.scaling_ymax_line.setText(self.scaling_ymax_line_buffer)
        self.imageDataGeneratorSettingWindow.flip_x_line.setText(self.flip_x_line_buffer)
        self.imageDataGeneratorSettingWindow.flip_y_line.setText(self.flip_y_line_buffer)

        self.close()

    def imageGenerator_checked(self):
        if self.imageGenerator_check.isChecked():
            self.imageGenerator_button.setEnabled(True)
        else:
            self.imageGenerator_button.setEnabled(False)

    def imageGenerator_clicked(self):
        self.imageDataGeneratorSettingWindow.exec_()

    def anchor_clicked(self):
        if self.anchor_checkBox.isChecked():
            for i in range(0, 5):
                self.anchor_size_line_list[i].setEnabled(True)
                self.anchor_stride_line_list[i].setEnabled(True)
            for i in range(0, 3):
                self.anchor_ratio_line_list[i].setEnabled(True)
                self.anchor_scale_line_list[i].setEnabled(True)
        else:
            for i in range(0, 5):
                self.anchor_size_line_list[i].setEnabled(False)
                self.anchor_stride_line_list[i].setEnabled(False)
            for i in range(0, 3):
                self.anchor_ratio_line_list[i].setEnabled(False)
                self.anchor_scale_line_list[i].setEnabled(False)

    def saving_browser(self):
        t, _t = QFileDialog.getSaveFileName(self, "Save", "", "h5 file(*.h5)")
        if t:
            self.model_path, extension = t, _t
            self.new_model_path = ""
            self.model_path_splited = self.model_path.split("/")
            for i in self.model_path_splited[0:-1]:
                self.new_model_path = self.new_model_path + i + '/'
            self.model_path_label.setText("Model Save Path : " + self.model_path)
            self.model_name_line.setText(self.model_path_splited[-1])
            if self.model_path_label.text() != "Model Save Path : " and self.dataset_label.text() != "Training Data Set : ":
                self.startTraining_button.setEnabled(True)

    def setting_dataset(self):
        t = QFileDialog.getExistingDirectory(self, "Open Training Data Set Directory")
        if t:
            self.trainingDataSet_directoryName = t
            self.dataset_label.setText("Training Data Set : " + self.trainingDataSet_directoryName)
            if self.dataset_label.text() != "Training Data Set : " and self.model_path_label.text() != "Model Save Path : ":
                self.startTraining_button.setEnabled(True)

    # --------------------------------------------------------- state change function ------------------------------------------------------------ #

    def setStop(self):
        self.dataset_dir_button.setEnabled(False)
        self.model_path_button.setEnabled(False)
        self.model_name_line.setEnabled(False)

        self.batchsize_line.setEnabled(False)
        self.learning_rate_line.setEnabled(False)

        self.image_minSize_line.setEnabled(False)
        self.image_maxSize_line.setEnabled(False)

        self.batchsize_line.setEnabled(False)
        self.epochs_line.setEnabled(False)
        self.steps_line.setEnabled(False)
        self.learning_rate_line.setEnabled(False)
        self.monitor_comboBox.setEnabled(False)

        self.anchor_checkBox.setEnabled(False)
        if self.anchor_checkBox.isChecked():
            self.anchor_size1_line.setEnabled(False)
            self.anchor_size2_line.setEnabled(False)
            self.anchor_size3_line.setEnabled(False)
            self.anchor_size4_line.setEnabled(False)
            self.anchor_size5_line.setEnabled(False)
            self.anchor_stride_line.setEnabled(False)
            self.anchor_stride2_line.setEnabled(False)
            self.anchor_stride3_line.setEnabled(False)
            self.anchor_stride4_line.setEnabled(False)
            self.anchor_stride5_line.setEnabled(False)
            self.anchor_ratio_line.setEnabled(False)
            self.anchor_ratio2_line.setEnabled(False)
            self.anchor_ratio3_line.setEnabled(False)
            self.anchor_scale_line.setEnabled(False)
            self.anchor_scale2_line.setEnabled(False)
            self.anchor_scale3_line.setEnabled(False)

        self.startTraining_button.setText("Stop Training")

    def setRunning(self):
        self.dataset_dir_button.setEnabled(True)
        self.model_path_button.setEnabled(True)
        self.model_name_line.setEnabled(True)

        self.batchsize_line.setEnabled(True)
        self.learning_rate_line.setEnabled(True)

        self.image_minSize_line.setEnabled(True)
        self.image_maxSize_line.setEnabled(True)

        self.batchsize_line.setEnabled(True)
        self.epochs_line.setEnabled(True)
        self.steps_line.setEnabled(True)
        self.learning_rate_line.setEnabled(True)
        self.monitor_comboBox.setEnabled(True)

        self.anchor_checkBox.setEnabled(True)
        if self.anchor_checkBox.isChecked():
            self.anchor_size1_line.setEnabled(True)
            self.anchor_size2_line.setEnabled(True)
            self.anchor_size3_line.setEnabled(True)
            self.anchor_size4_line.setEnabled(True)
            self.anchor_size5_line.setEnabled(True)
            self.anchor_stride_line.setEnabled(True)
            self.anchor_stride2_line.setEnabled(True)
            self.anchor_stride3_line.setEnabled(True)
            self.anchor_stride4_line.setEnabled(True)
            self.anchor_stride5_line.setEnabled(True)
            self.anchor_ratio_line.setEnabled(True)
            self.anchor_ratio2_line.setEnabled(True)
            self.anchor_ratio3_line.setEnabled(True)
            self.anchor_scale_line.setEnabled(True)
            self.anchor_scale2_line.setEnabled(True)
            self.anchor_scale3_line.setEnabled(True)

        self.startTraining_button.setText("Start Training")

    def print_state(self):
        from tensorflow.python.client import device_lib

        for i in device_lib.list_local_devices():
            tempi = str(i)
            if "physical_device" in tempi:
                if "GPU" in tempi:
                    print("[ Using GPU ]")
                else:
                    print("[ Using CPU ]")
                print(tempi[tempi.find("physical_device"):])

    def terminate_thread(self, thread):
        if not thread.isAlive():
            return

        exc = ctypes.py_object(SystemExit)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread.ident), exc)
        if res == 0:
            raise ValueError("nonexistent thread id")
        elif res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def confirm(self):
        if self.anchor_checkBox.isChecked() == True:
            for i in range(0, 5):
                if self.anchor_size_line_list[i].text().isdigit() == False:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Size > 1 , Type=int")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return
                elif int(self.anchor_size_line_list[i].text()) <= 1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Size > 1 , Type=int")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

            for i in range(0, 4):
                if int(self.anchor_size_line_list[i].text()) >= int(self.anchor_size_line_list[i + 1].text()):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Anchor size value is expected to be smaller than the former value")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

            for i in range(0, 5):
                if self.anchor_stride_line_list[i].text().isdigit() == False:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Stride > 1 , Type=int")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return
                elif int(self.anchor_size_line_list[i].text()) <= 1:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Stride > 1 , Type=int")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

            for i in range(0, 4):
                if int(self.anchor_stride_line_list[i].text()) >= int(self.anchor_stride_line_list[i + 1].text()):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Anchor stride value is expected to be smaller than the former value")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

            for i in range(0, 3):
                if self.isNumber(self.anchor_ratio_line_list[i].text()) == False:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Ratio > 0, Type=Real Number")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return
                elif float(self.anchor_ratio_line_list[i].text()) <= 0:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Ratio > 0, Type=Real Number")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

            for i in range(0, 2):
                if float(self.anchor_ratio_line_list[i].text()) >= float(self.anchor_ratio_line_list[i + 1].text()):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Anchor ratio value is expected to be smaller than the former value")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

            for i in range(0, 3):
                if self.isNumber(self.anchor_scale_line_list[i].text()) == False:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Scale > 0, Type=Real Number")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return
                elif float(self.anchor_scale_line_list[i].text()) <= 0:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Scale > 0, Type=Real Number")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

            for i in range(0, 2):
                if float(self.anchor_scale_line_list[i].text()) >= float(self.anchor_scale_line_list[i + 1].text()):
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Anchor scale value is expected to be smaller than the former value")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    return

        if self.batchsize_line.text().isdigit() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Batch Size is expected to be a integer value")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.epochs_line.text().isdigit() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Epoch is expected to be a integer value")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.steps_line.text().isdigit() == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Step Value is expected to be a integer value")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.isNumber(self.learning_rate_line.text()) == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Learning Rate is expected to be a positive float or integer")
            msg.setWindowTitle("warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        elif self.isNumber(self.learning_rate_line.text()):
            if float(self.learning_rate_line.text()) <= 0.0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Learning Rate is expected to be a positive float or integer value")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                self.trainingTest.tableWidget.setItem(0, 0, QTableWidgetItem("Customized anchor"))
                self.trainingTest.tableWidget.setItem(1, 0, QTableWidgetItem("Anchor size"))
                self.trainingTest.tableWidget.setItem(2, 0, QTableWidgetItem("Anchor stride"))
                self.trainingTest.tableWidget.setItem(3, 0, QTableWidgetItem("Anchor ratio"))
                self.trainingTest.tableWidget.setItem(4, 0, QTableWidgetItem("Anchor scale"))
                self.trainingTest.tableWidget.setItem(5, 0, QTableWidgetItem("Batch size"))
                self.trainingTest.tableWidget.setItem(6, 0, QTableWidgetItem("Steps"))
                self.trainingTest.tableWidget.setItem(7, 0, QTableWidgetItem("Epochs"))
                self.trainingTest.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.trainingTest.tableWidget.setItem(9, 0, QTableWidgetItem("Maximum resizing image size"))
                self.trainingTest.tableWidget.item(9, 0).setToolTip("Maximum resizing image size")
                self.trainingTest.tableWidget.setItem(10, 0, QTableWidgetItem("Minimum resizing image size"))
                self.trainingTest.tableWidget.item(10, 0).setToolTip("Minimum resizing image size")
                self.trainingTest.tableWidget.setItem(11, 0, QTableWidgetItem("Image augmentation"))
                self.trainingTest.tableWidget.setItem(12, 0, QTableWidgetItem("Rotation range"))
                self.trainingTest.tableWidget.setItem(13, 0, QTableWidgetItem("Shearing range"))
                self.trainingTest.tableWidget.setItem(14, 0, QTableWidgetItem("(X, Y) Translation range"))
                self.trainingTest.tableWidget.item(14, 0).setToolTip("(X, Y) Translation range")
                self.trainingTest.tableWidget.setItem(15, 0, QTableWidgetItem("(X, Y) Scaling range"))
                self.trainingTest.tableWidget.item(15, 0).setToolTip("(X, Y) Scaling range")
                self.trainingTest.tableWidget.setItem(16, 0, QTableWidgetItem("X, Y Flip chance"))
                self.trainingTest.tableWidget.item(16, 0).setToolTip("X, Y Flip chance")

                self.trainingTest.tableWidget.setItem(0, 1, QTableWidgetItem(
                    "True" if self.anchor_checkBox.isChecked() else "False"))
                if self.anchor_checkBox.isChecked():
                    self.trainingTest.tableWidget.setItem(1, 1, QTableWidgetItem(
                        self.anchor_size1_line.text() + ", " + self.anchor_size2_line.text()
                        + ", " + self.anchor_size3_line.text() + ", " + self.anchor_size4_line.text()
                        + ", " + self.anchor_size5_line.text()))
                    self.trainingTest.tableWidget.item(1, 1).setToolTip(
                        self.anchor_size1_line.text() + ", " + self.anchor_size2_line.text()
                        + ", " + self.anchor_size3_line.text() + ", " + self.anchor_size4_line.text()
                        + ", " + self.anchor_size5_line.text())
                    self.trainingTest.tableWidget.setRowHidden(1, False)
                    self.trainingTest.tableWidget.setItem(2, 1, QTableWidgetItem(
                        self.anchor_stride_line.text() + ", " + self.anchor_stride2_line.text()
                        + ", " + self.anchor_stride3_line.text() + ", " + self.anchor_stride4_line.text()
                        + ", " + self.anchor_stride5_line.text()))
                    self.trainingTest.tableWidget.setItem(2, 1, QTableWidgetItem(
                        self.anchor_stride_line.text() + ", " + self.anchor_stride2_line.text()
                        + ", " + self.anchor_stride3_line.text() + ", " + self.anchor_stride4_line.text()
                        + ", " + self.anchor_stride5_line.text()))
                    self.trainingTest.tableWidget.setRowHidden(2, False)
                    self.trainingTest.tableWidget.setItem(3, 1, QTableWidgetItem(
                        self.anchor_ratio_line.text() + ", " + self.anchor_ratio2_line.text()
                        + ", " + self.anchor_ratio3_line.text()))
                    self.trainingTest.tableWidget.setRowHidden(3, False)
                    self.trainingTest.tableWidget.setItem(4, 1, QTableWidgetItem(
                        self.anchor_scale_line.text() + ", " + self.anchor_scale2_line.text()
                        + ", " + self.anchor_scale3_line.text()))
                    self.trainingTest.tableWidget.setRowHidden(4, False)
                else:
                    self.trainingTest.tableWidget.setRowHidden(1, True)
                    self.trainingTest.tableWidget.setRowHidden(2, True)
                    self.trainingTest.tableWidget.setRowHidden(3, True)
                    self.trainingTest.tableWidget.setRowHidden(4, True)

                if self.imageGenerator_check.isChecked():
                    self.trainingTest.tableWidget.setItem(12, 1, QTableWidgetItem(
                        self.imageDataGeneratorSettingWindow.rotation_min_line.text() + " ~ " + self.imageDataGeneratorSettingWindow.rotation_max_line.text()))
                    self.trainingTest.tableWidget.setRowHidden(12, False)
                    self.trainingTest.tableWidget.setItem(13, 1, QTableWidgetItem(
                        self.imageDataGeneratorSettingWindow.shearing_min_line.text() + " ~ " + self.imageDataGeneratorSettingWindow.shearing_max_line.text()))
                    self.trainingTest.tableWidget.setRowHidden(13, False)
                    self.trainingTest.tableWidget.setItem(14, 1, QTableWidgetItem(
                        "(" + self.imageDataGeneratorSettingWindow.translation_xmin_line.text() + ", " + self.imageDataGeneratorSettingWindow.translation_ymin_line.text()
                        + ")" + " ~ " + "(" + self.imageDataGeneratorSettingWindow.translation_xmax_line.text() + ", " + self.imageDataGeneratorSettingWindow.translation_ymax_line.text() + ")"))
                    self.trainingTest.tableWidget.item(14, 1).setToolTip(
                        "(" + self.imageDataGeneratorSettingWindow.translation_xmin_line.text() + ", " + self.imageDataGeneratorSettingWindow.translation_ymin_line.text()
                        + ")" + " ~ " + "(" + self.imageDataGeneratorSettingWindow.translation_xmax_line.text() + ", " + self.imageDataGeneratorSettingWindow.translation_ymax_line.text() + ")")
                    self.trainingTest.tableWidget.setRowHidden(14, False)
                    self.trainingTest.tableWidget.setItem(15, 1, QTableWidgetItem(
                        "(" + self.imageDataGeneratorSettingWindow.scaling_xmin_line.text() + ", " + self.imageDataGeneratorSettingWindow.scaling_ymin_line.text()
                        + ")" + " ~ " + "(" + self.imageDataGeneratorSettingWindow.scaling_xmax_line.text() + ", " + self.imageDataGeneratorSettingWindow.scaling_ymax_line.text() + ")"))
                    self.trainingTest.tableWidget.item(15, 1).setToolTip(
                        "(" + self.imageDataGeneratorSettingWindow.scaling_xmin_line.text() + ", " + self.imageDataGeneratorSettingWindow.scaling_ymin_line.text()
                        + ")" + " ~ " + "(" + self.imageDataGeneratorSettingWindow.scaling_xmax_line.text() + ", " + self.imageDataGeneratorSettingWindow.scaling_ymax_line.text() + ")")
                    self.trainingTest.tableWidget.setRowHidden(15, False)
                    self.trainingTest.tableWidget.setItem(16, 1, QTableWidgetItem(
                        self.imageDataGeneratorSettingWindow.flip_x_line.text() + ", " + self.imageDataGeneratorSettingWindow.flip_y_line.text()))
                    self.trainingTest.tableWidget.setRowHidden(16, False)
                else:
                    self.trainingTest.tableWidget.setRowHidden(12, True)
                    self.trainingTest.tableWidget.setRowHidden(13, True)
                    self.trainingTest.tableWidget.setRowHidden(14, True)
                    self.trainingTest.tableWidget.setRowHidden(15, True)
                    self.trainingTest.tableWidget.setRowHidden(16, True)

                self.trainingTest.tableWidget.setItem(5, 1, QTableWidgetItem(self.batchsize_line.text()))
                self.trainingTest.tableWidget.setItem(6, 1, QTableWidgetItem(self.steps_line.text()))
                self.trainingTest.tableWidget.setItem(7, 1, QTableWidgetItem(self.epochs_line.text()))
                self.trainingTest.tableWidget.setItem(8, 1, QTableWidgetItem(self.learning_rate_line.text()))
                self.trainingTest.tableWidget.setItem(9, 1, QTableWidgetItem(self.image_maxSize_line.text()))
                self.trainingTest.tableWidget.setItem(10, 1, QTableWidgetItem(self.image_minSize_line.text()))
                self.trainingTest.tableWidget.setItem(11, 1, QTableWidgetItem(
                    "True" if self.imageGenerator_check.isChecked() else "False"))

                self.dataset_label_buffer = self.dataset_label.text()
                self.trainingDataSet_directoryName_buffer = self.trainingDataSet_directoryName
                self.model_path_label_buffer = self.model_path_label.text()
                self.model_name_line_buffer = self.model_name_line.text()
                self.anchor_checkBox_bool = self.anchor_checkBox.isChecked()
                self.anchor_size1_line_buffer = self.anchor_size1_line.text()
                self.anchor_size2_line_buffer = self.anchor_size2_line.text()
                self.anchor_size3_line_buffer = self.anchor_size3_line.text()
                self.anchor_size4_line_buffer = self.anchor_size4_line.text()
                self.anchor_size5_line_buffer = self.anchor_size5_line.text()
                self.anchor_stride_line_buffer = self.anchor_stride_line.text()
                self.anchor_stride2_line_buffer = self.anchor_stride2_line.text()
                self.anchor_stride3_line_buffer = self.anchor_stride3_line.text()
                self.anchor_stride4_line_buffer = self.anchor_stride4_line.text()
                self.anchor_stride5_line_buffer = self.anchor_stride5_line.text()
                self.anchor_ratio_line_buffer = self.anchor_ratio_line.text()
                self.anchor_ratio2_line_buffer = self.anchor_ratio2_line.text()
                self.anchor_ratio3_line_buffer = self.anchor_ratio3_line.text()
                self.anchor_scale_line_buffer = self.anchor_scale_line.text()
                self.anchor_scale2_line_buffer = self.anchor_scale2_line.text()
                self.anchor_scale3_line_buffer = self.anchor_scale3_line.text()
                self.image_minSize_line_buffer = self.image_minSize_line.text()
                self.image_maxSize_line_buffer = self.image_maxSize_line.text()
                self.batchsize_line_buffer = self.batchsize_line.text()
                self.steps_line_buffer = self.steps_line.text()
                self.epochs_line_buffer = self.epochs_line.text()
                self.learning_rate_line_buffer = self.learning_rate_line.text()
                self.monitor_comboBox_buffer = self.monitor_comboBox.currentText()
                self.imageGenerator_check_buffer = self.imageGenerator_check.isChecked()
                self.rotation_min_line_buffer = self.imageDataGeneratorSettingWindow.rotation_min_line.text()
                self.rotation_max_line_buffer = self.imageDataGeneratorSettingWindow.rotation_max_line.text()
                self.shearing_min_line_buffer = self.imageDataGeneratorSettingWindow.shearing_min_line.text()
                self.shearing_max_line_buffer = self.imageDataGeneratorSettingWindow.shearing_max_line.text()
                self.translation_xmin_line_buffer = self.imageDataGeneratorSettingWindow.translation_xmin_line.text()
                self.translation_ymin_line_buffer = self.imageDataGeneratorSettingWindow.translation_ymin_line.text()
                self.translation_xmax_line_buffer = self.imageDataGeneratorSettingWindow.translation_xmax_line.text()
                self.translation_ymax_line_buffer = self.imageDataGeneratorSettingWindow.translation_ymax_line.text()
                self.scaling_xmin_line_buffer = self.imageDataGeneratorSettingWindow.scaling_xmin_line.text()
                self.scaling_ymin_line_buffer = self.imageDataGeneratorSettingWindow.scaling_ymin_line.text()
                self.scaling_xmax_line_buffer = self.imageDataGeneratorSettingWindow.scaling_xmax_line.text()
                self.scaling_ymax_line_buffer = self.imageDataGeneratorSettingWindow.scaling_ymax_line.text()
                self.flip_x_line_buffer = self.imageDataGeneratorSettingWindow.flip_x_line.text()
                self.flip_y_line_buffer = self.imageDataGeneratorSettingWindow.flip_y_line.text()

                self.trainingTest.category_changed_flag_o = True

                self.close()

    def isNumber(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

        def get_session():
            """ Construct a modified tf session.
            """
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            return tf.Session(config=config)

        def create_generator(args):
            """ Create generators for evaluation.
            """
            if args.dataset_type == 'coco':
                # import here to prevent unnecessary dependency on cocoapi
                from ..preprocessing.coco import CocoGenerator

                validation_generator = CocoGenerator(
                    args.coco_path,
                    'val2017',
                    image_min_side=args.image_min_side,
                    image_max_side=args.image_max_side,
                    config=args.config,
                    shuffle_groups=False,
                )
            elif args.dataset_type == 'pascal':
                validation_generator = PascalVocGenerator(
                    args.pascal_path,
                    'test',
                    image_min_side=args.image_min_side,
                    image_max_side=args.image_max_side,
                    config=args.config,
                    shuffle_groups=False,
                )
            elif args.dataset_type == 'csv':
                validation_generator = CSVGenerator(
                    args.annotations,
                    args.classes,
                    image_min_side=args.image_min_side,
                    image_max_side=args.image_max_side,
                    config=args.config,
                    shuffle_groups=False,
                )
            else:
                raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

            return validation_generator

        def parse_args(args):
            """ Parse the arguments.
            """
            parser = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
            subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
            subparsers.required = True

            coco_parser = subparsers.add_parser('coco')
            coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

            pascal_parser = subparsers.add_parser('pascal')
            pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

            csv_parser = subparsers.add_parser('csv')
            csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
            csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

            parser.add_argument('model', help='Path to RetinaNet model.')
            parser.add_argument('--convert-model',
                                help='Convert the model to an inference model (ie. the input is a training model).',
                                action='store_true')
            parser.add_argument('--backbone', help='The backbone of the model.', default='resnet50')
            parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
            parser.add_argument('--score-threshold',
                                help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05,
                                type=float)
            parser.add_argument('--iou-threshold',
                                help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5,
                                type=float)
            parser.add_argument('--max-detections', help='Max Detections per image (defaults to 100).', default=100,
                                type=int)
            parser.add_argument('--save-path', help='Path for saving images with detections (doesn\'t work for COCO).')
            parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.',
                                type=int, default=IMAGE_MIN_SIZE)
            parser.add_argument('--image-max-side',
                                help='Rescale the image if the largest side is larger than max_side.', type=int,
                                default=IMAGE_MAX_SIZE)
            parser.add_argument('--config',
                                help='Path to a configuration parameters .ini file (only used with --convert-model).')

            return parser.parse_args(args)

            # parse arguments

        args = []
        args.append('csv')
        args.append(DATASET_DIR + '/annotations.csv')
        args.append(DATASET_DIR + '/classes.csv')
        args.append(MODEL_SAVE_PATH + '.h5')
        args = parse_args(args)
        # make sure keras is the minimum required version
        check_keras_version()
        # optionally choose specific GPU
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        keras.backend.tensorflow_backend.set_session(get_session())
        # make save path if it doesn't exist
        if args.save_path is not None and not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # optionally load config parameters
        if args.config:
            args.config = read_config_file(args.config)
        # create the generator
        generator = create_generator(args)
        # optionally load anchor parameters
        anchor_params = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        # load the model
        print('Loading model, this may take a second...')
        model = models.load_model(args.model, backbone_name=args.backbone)
        # optionally convert the model
        if args.convert_model:
            model = models.convert_model(model, anchor_params=anchor_params)
        # print model summary
        # print(model.summary())
        # start evaluation
        print('Evaluating model(name=' + MODEL_NAME + '.h5)')
        if args.dataset_type == 'coco':
            from ..utils.coco_eval import evaluate_coco
            evaluate_coco(generator, model, args.score_threshold)
        else:
            average_precisions = evaluate(
                generator,
                model,
                iou_threshold=args.iou_threshold,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                save_path=args.save_path
            )
            # print evaluation
            total_instances = []
            precisions = []
            for label, (average_precision, num_annotations) in average_precisions.items():
                print('{:.0f} instances of class'.format(num_annotations),
                      generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
                total_instances.append(num_annotations)
                precisions.append(average_precision)
            if sum(total_instances) == 0:
                print('No test instances found.')
                return
            print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
                sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)))
            print('mAP: {:.4f}'.format(sum(precisions) / sum(x > 0 for x in total_instances)))
            # self.startTraining_button.setEnabled(True)

    def training(self):
        Steps = int(self.steps_line.text())
        BATCH_SIZE = int(self.batchsize_line.text())
        EPOCHS = int(self.epochs_line.text())
        Learning_Rate = float(self.learning_rate_line.text())
        MONITOR = self.monitor_comboBox.currentText()
        DATASET_DIR = self.trainingDataSet_directoryName
        MODEL_SAVE_PATH = self.model_path
        NEW_MODEL_PATH = self.new_model_path
        MODEL_NAME = self.model_name_line.text()
        IMAGE_MIN_SIZE = self.image_minSize_line.text()
        IMAGE_MAX_SIZE = self.image_maxSize_line.text()
        print(MODEL_SAVE_PATH)

        def makedirs(path):
            # Intended behavior: try to create the directory,
            # pass if the directory exists already, fails otherwise.
            # Meant for Python 2.7/3.n compatibility.
            try:
                os.makedirs(path)
            except OSError:
                if not os.path.isdir(path):
                    raise

        def get_session():
            """ Construct a modified tf session.
            """
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            return tf.Session(config=config)

        def model_with_weights(model, weights, skip_mismatch):
            """ Load weights for model.

            Args
                model         : The model to load weights for.
                weights       : The weights to load.
                skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
            """
            if weights is not None:
                model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
            return model

        def create_models(backbone_retinanet, num_classes, weights, multi_gpu=1,
                          freeze_backbone=False, lr=1e-5, config=None):
            """ Creates three models (model, training_model, prediction_model).

            Args
                backbone_retinanet : A function to call to create a retinanet model with a given backbone.
                num_classes        : The number of classes to train.
                weights            : The weights to load into the model.
                multi_gpu          : The number of GPUs to use for training.
                freeze_backbone    : If True, disables learning for the backbone.
                config             : Config parameters, None indicates the default configuration.

            Returns
                model            : The base model. This is also the model that is saved in snapshots.
                training_model   : The training model. If multi_gpu=0, this is identical to model.
                prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
            """

            modifier = freeze_model if freeze_backbone else None

            # load anchor parameters, or pass None (so that defaults will be used)
            anchor_params = None
            num_anchors = None
            if config and 'anchor_parameters' in config:
                anchor_params = parse_anchor_parameters(config)
                num_anchors = anchor_params.num_anchors()

            # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
            # optionally wrap in a parallel model
            if multi_gpu > 1:
                from keras.utils import multi_gpu_model
                with tf.device('/cpu:0'):
                    model = model_with_weights(
                        backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier), weights=weights,
                        skip_mismatch=True)
                training_model = multi_gpu_model(model, gpus=multi_gpu)
            else:
                model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
                                           weights=weights, skip_mismatch=True)
                training_model = model

            # make prediction model
            prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

            # compile model
            training_model.compile(
                loss={
                    'regression': losses.smooth_l1(),
                    'classification': losses.focal()
                },
                optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
            )

            return model, training_model, prediction_model

        def create_callbacks(model, training_model, prediction_model, validation_generator, args):
            """ Creates the callbacks to use during training.

            Args
                model: The base model.
                training_model: The model that is used for training.
                prediction_model: The model that should be used for validation.
                validation_generator: The generator for creating validation data.
                args: parseargs args object.

            Returns:
                A list of callbacks used for training.
            """
            callbacks = []

            tensorboard_callback = None

            if args.tensorboard_dir:
                tensorboard_callback = keras.callbacks.TensorBoard(
                    log_dir=args.tensorboard_dir,
                    histogram_freq=0,
                    batch_size=args.batch_size,
                    write_graph=True,
                    write_grads=False,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None
                )
                callbacks.append(tensorboard_callback)

            if args.evaluation and validation_generator:
                if args.dataset_type == 'coco':
                    from ..callbacks.coco import CocoEval

                    # use prediction model for evaluation
                    evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
                else:
                    evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback,
                                          weighted_average=args.weighted_average)
                evaluation = RedirectModel(evaluation, prediction_model)
                callbacks.append(evaluation)

            # save the model
            if args.snapshots:
                # ensure directory created first; otherwise h5py will error after epoch.
                makedirs(args.snapshot_path)
                if MONITOR == 'val_loss':
                    checkpoint = keras.callbacks.ModelCheckpoint(
                        os.path.join(
                            args.snapshot_path,
                            'snapshot.h5'  # 변경가능
                        ),
                        verbose=1,
                        save_best_only=True,
                        monitor="val_loss",
                        mode='min'
                    )
                # elif MONITOR =='acc':
                #     checkpoint = keras.callbacks.ModelCheckpoint(
                #         os.path.join(
                #             args.snapshot_path,
                #             'snapshot.h5'  # 변경가능
                #         ),
                #         verbose=1,
                #         save_best_only=True,
                #         monitor="acc",
                #         mode='max'
                #     )
                else:
                    checkpoint = keras.callbacks.ModelCheckpoint(
                        os.path.join(
                            args.snapshot_path,
                            'snapshot.h5'  # 변경가능
                        ),
                        verbose=1,
                        save_best_only=True,
                        monitor="mAP",
                        mode='max'
                    )
                checkpoint = RedirectModel(checkpoint, model)
                callbacks.append(checkpoint)

            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.1,
                patience=2,
                verbose=1,
                mode='auto',
                min_delta=0.0001,
                cooldown=0,
                min_lr=0
            ))
            callbacks.append(self.plot_losses)
            return callbacks

        def create_generators(args, preprocess_image):
            """ Create generators for training and validation.

            Args
                args             : parseargs object containing configuration for generators.
                preprocess_image : Function that preprocesses an image for the network.
            """
            common_args = {
                'batch_size': args.batch_size,
                'config': args.config,
                'image_min_side': args.image_min_side,
                'image_max_side': args.image_max_side,
                'preprocess_image': preprocess_image,
            }

            # create random transform generator for augmenting training data
            if args.random_transform:
                transform_generator = random_transform_generator(
                    min_rotation=float(self.imageDataGeneratorSettingWindow.rotation_min_line.text()),
                    max_rotation=float(self.imageDataGeneratorSettingWindow.rotation_max_line.text()),
                    min_translation=(float(self.imageDataGeneratorSettingWindow.translation_xmin_line.text()),
                                     float(self.imageDataGeneratorSettingWindow.translation_ymin_line.text())),
                    max_translation=(float(self.imageDataGeneratorSettingWindow.translation_xmax_line.text()),
                                     float(self.imageDataGeneratorSettingWindow.translation_ymax_line.text())),
                    min_shear=float(self.imageDataGeneratorSettingWindow.shearing_min_line.text()),
                    max_shear=float(self.imageDataGeneratorSettingWindow.shearing_max_line.text()),
                    min_scaling=(float(self.imageDataGeneratorSettingWindow.scaling_xmin_line.text()),
                                 float(self.imageDataGeneratorSettingWindow.scaling_ymin_line.text())),
                    max_scaling=(float(self.imageDataGeneratorSettingWindow.scaling_xmax_line.text()),
                                 float(self.imageDataGeneratorSettingWindow.scaling_ymax_line.text())),
                    flip_x_chance=float(self.imageDataGeneratorSettingWindow.flip_x_line.text()),
                    flip_y_chance=float(self.imageDataGeneratorSettingWindow.flip_y_line.text()),
                )
            else:
                transform_generator = random_transform_generator(flip_x_chance=0.5)

            if args.dataset_type == 'coco':
                # import here to prevent unnecessary dependency on cocoapi
                from ..preprocessing.coco import CocoGenerator

                train_generator = CocoGenerator(
                    args.coco_path,
                    'train2017',
                    transform_generator=transform_generator,
                    **common_args
                )

                validation_generator = CocoGenerator(
                    args.coco_path,
                    'val2017',
                    shuffle_groups=False,
                    **common_args
                )
            elif args.dataset_type == 'pascal':
                train_generator = PascalVocGenerator(
                    args.pascal_path,
                    'trainval',
                    transform_generator=transform_generator,
                    **common_args
                )

                validation_generator = PascalVocGenerator(

                    args.pascal_path,
                    'test',
                    shuffle_groups=False,
                    **common_args
                )
            elif args.dataset_type == 'csv':
                train_generator = CSVGenerator(
                    args.annotations,
                    args.classes,
                    transform_generator=transform_generator,
                    **common_args
                )

                if args.val_annotations:
                    validation_generator = CSVGenerator(
                        args.val_annotations,
                        args.classes,
                        shuffle_groups=False,
                        **common_args
                    )
                else:
                    validation_generator = None
            elif args.dataset_type == 'oid':
                train_generator = OpenImagesGenerator(
                    args.main_dir,
                    subset='train',
                    version=args.version,
                    labels_filter=args.labels_filter,
                    annotation_cache_dir=args.annotation_cache_dir,
                    parent_label=args.parent_label,
                    transform_generator=transform_generator,
                    **common_args
                )

                validation_generator = OpenImagesGenerator(
                    args.main_dir,
                    subset='validation',
                    version=args.version,
                    labels_filter=args.labels_filter,
                    annotation_cache_dir=args.annotation_cache_dir,
                    parent_label=args.parent_label,
                    shuffle_groups=False,
                    **common_args
                )
            elif args.dataset_type == 'kitti':
                train_generator = KittiGenerator(
                    args.kitti_path,
                    subset='train',
                    transform_generator=transform_generator,
                    **common_args
                )

                validation_generator = KittiGenerator(
                    args.kitti_path,
                    subset='val',
                    shuffle_groups=False,
                    **common_args
                )
            else:
                raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

            return train_generator, validation_generator

        def check_args(parsed_args):
            """ Function to check for inherent contradictions within parsed arguments.
            For example, batch_size < num_gpus
            Intended to raise errors prior to backend initialisation.

            Args
                parsed_args: parser.parse_args()

            Returns
                parsed_args
            """

            if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
                raise ValueError(
                    "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(
                        parsed_args.batch_size,
                        parsed_args.multi_gpu))

            if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
                raise ValueError(
                    "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(
                        parsed_args.multi_gpu,
                        parsed_args.snapshot))

            if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
                raise ValueError(
                    "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

            if 'resnet' not in parsed_args.backbone:
                warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(
                    parsed_args.backbone))

            return parsed_args

        def parse_args(args):
            """ Parse the arguments.
            """
            parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
            subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
            subparsers.required = True

            coco_parser = subparsers.add_parser('coco')
            coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

            pascal_parser = subparsers.add_parser('pascal')
            pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

            kitti_parser = subparsers.add_parser('kitti')
            kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

            def csv_list(string):
                return string.split(',')

            oid_parser = subparsers.add_parser('oid')
            oid_parser.add_argument('main_dir', help='Path to dataset directory.')
            oid_parser.add_argument('--version', help='The current dataset version is v4.', default='v4')
            oid_parser.add_argument('--labels-filter', help='A list of labels to filter.', type=csv_list, default=None)
            oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
            oid_parser.add_argument('--parent-label', help='Use the hierarchy children of this label.', default=None)

            csv_parser = subparsers.add_parser('csv')
            csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
            csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
            csv_parser.add_argument('--val-annotations',
                                    help='Path to CSV file containing annotations for validation (optional).',
                                    default=DATASET_DIR + '/annotations_val.csv')

            group = parser.add_mutually_exclusive_group()
            group.add_argument('--snapshot', help='Resume training from a snapshot.')
            group.add_argument('--imagenet-weights',
                               help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
                               action='store_const', const=True, default=True)
            group.add_argument('--weights', help='Initialize the model with weights from a file.')
            group.add_argument('--no-weights', help='Don\'t initialize the model with any weights.',
                               dest='imagenet_weights', action='store_const', const=False)

            parser.add_argument('--backbone', help='Backbone model used by retinanet.', default='resnet50', type=str)
            parser.add_argument('--batch-size', help='Size of the batches.', default=BATCH_SIZE, type=int)
            parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).', default=0)
            parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int,
                                default=0)
            parser.add_argument('--multi-gpu-force',
                                help='Extra flag needed to enable (experimental) multi-gpu support.',
                                action='store_true')
            parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=EPOCHS)  # 변경가능
            parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=Steps)  # 변경가능
            parser.add_argument('--lr', help='Learning rate.', type=float, default=Learning_Rate)  # 변경가능
            parser.add_argument('--snapshot-path',
                                help='Path to store snapshots of models during training (C:/Users/sangsu/Desktop)',
                                default='./snapshots')  # 사용자한테 받아서 변경
            parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
            parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots',
                                action='store_false')
            parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                                action='store_false')
            parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
            parser.add_argument('--random-transform', help='Randomly transform image and annotations.',
                                action='store_true')
            parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.',
                                type=int, default=IMAGE_MIN_SIZE)
            parser.add_argument('--image-max-side',
                                help='Rescale the image if the largest side is larger than max_side.', type=int,
                                default=IMAGE_MAX_SIZE)
            parser.add_argument('--config', help='Path to a configuration parameters .ini file.',
                                default='./config.ini')
            parser.add_argument('--weighted-average',
                                help='Compute the mAP using the weighted average of precisions among classes.',
                                action='store_true')
            parser.add_argument('--compute-val-loss', help='Compute validation loss during training',
                                dest='compute_val_loss', action='store_true', default=True)
            parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
            parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.',
                                dest='class_specific_filter', action='store_false')
            # Fit generator arguments
            parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
            parser.add_argument('--workers', help='Number of generator workers.', type=int, default=0)
            parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.',
                                type=int, default=10)

            return check_args(parser.parse_args(args))

        import configparser
        SIZE1 = self.anchor_size1_line.text()
        SIZE2 = self.anchor_size2_line.text()
        SIZE3 = self.anchor_size3_line.text()
        SIZE4 = self.anchor_size4_line.text()
        SIZE5 = self.anchor_size5_line.text()
        STRIDE1 = self.anchor_stride_line.text()
        STRIDE2 = self.anchor_stride2_line.text()
        STRIDE3 = self.anchor_stride3_line.text()
        STRIDE4 = self.anchor_stride4_line.text()
        STRIDE5 = self.anchor_stride5_line.text()
        RATIO1 = self.anchor_ratio_line.text()
        RATIO2 = self.anchor_ratio2_line.text()
        RATIO3 = self.anchor_ratio3_line.text()
        SCALE1 = self.anchor_scale_line.text()
        SCALE2 = self.anchor_scale2_line.text()
        SCALE3 = self.anchor_scale3_line.text()
        SIZE = SIZE1 + ',' + SIZE2 + ',' + SIZE3 + ',' + SIZE4 + ',' + SIZE5
        STRIDE = STRIDE1 + ',' + STRIDE2 + ',' + STRIDE3 + ',' + STRIDE4 + ',' + STRIDE5
        RATIO = RATIO1 + ',' + RATIO2 + ',' + RATIO3
        SCALE = SCALE1 + ',' + SCALE2 + ',' + SCALE3
        cf = configparser.ConfigParser()
        cf['anchor_parameters'] = {
            'sizes': SIZE,
            'strides': STRIDE,
            'ratios': RATIO,
            'scales': SCALE
        }
        with open('./config.ini', 'w') as f:
            cf.write(f)
        args = []
        args.append('csv')
        args.append(DATASET_DIR + '/annotations.csv')
        args.append(DATASET_DIR + '/classes.csv')
        # args.append('--no-snapshots')
        args = parse_args(args)
        backbone = models.backbone(args.backbone)
        # make sure keras is the minimum required version
        check_keras_version()
        # optionally choose specific GPU
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        keras.backend.tensorflow_backend.set_session(get_session())
        # optionally load config parameters
        anchor_parameters = None
        if args.config:
            args.config = read_config_file(args.config)
            if 'anchor_parameters' in args.config:
                anchor_parameters = parse_anchor_parameters(args.config)
        # create the generators
        train_generator, validation_generator = create_generators(args, backbone.preprocess_image)
        # create the model
        if args.snapshot is not None:
            print('Loading model, this may take a second...')
            model = models.load_model(args.snapshot, backbone_name=args.backbone)
            training_model = model
            anchor_params = None
            if args.config and 'anchor_parameters' in args.config:
                anchor_params = parse_anchor_parameters(args.config)
            prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
        else:
            weights = args.weights
            # default to imagenet if nothing else is specified
            if weights is None and args.imagenet_weights:
                weights = backbone.download_imagenet()
            print('Creating model, this may take a second...')
            model, training_model, prediction_model = create_models(
                backbone_retinanet=backbone.retinanet,
                num_classes=train_generator.num_classes(),
                weights=weights,
                multi_gpu=args.multi_gpu,
                freeze_backbone=args.freeze_backbone,
                lr=args.lr,
                config=args.config
            )
        # print model summary
        print(model.summary())
        # this lets the generator compute backbone layer shapes using the actual backbone model
        if 'vgg' in args.backbone or 'densenet' in args.backbone:
            train_generator.compute_shapes = make_shapes_callback(model)
            if validation_generator:
                validation_generator.compute_shapes = train_generator.compute_shapes
        # create the callbacks
        callbacks = create_callbacks(
            model,
            training_model,
            prediction_model,
            validation_generator,
            args,
        )
        if not args.compute_val_loss:
            validation_generator = None
        # start training
        training_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            workers=args.workers,
            use_multiprocessing=args.multiprocessing,
            max_queue_size=args.max_queue_size,
            validation_data=validation_generator
        )

        model = models.load_model('./snapshots/snapshot.h5')
        models.check_training_model(model)

        # convert the model
        model = models.convert_model(model, nms=args.nms, class_specific_filter=args.class_specific_filter,
                                     anchor_params=anchor_parameters)

        # save model
        model.save(MODEL_SAVE_PATH + '.h5')
        print("<--------- model has been saved. --------->")
        self.startEvaluate_button.setEnabled(True)
        self.startTraining_button.setEnabled(True)
        self.setRunning()
        self.change_button_signal.run()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = TrainingWindow()
    main.show()
    sys.exit(app.exec_())