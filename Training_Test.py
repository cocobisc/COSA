import os
import cv2
import sys
import json
import random
import logging
import colorsys
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
from retinanet.keras_retinanet.bin.test import TrainingWindow as objectDetectionTrainingUI
from obj_dtt_predict import PredictWindow
from classificationUI_test import ClassificationUI
# from classificationUI_test import PredictWindow as classification_predict
from detectionLabeling import labelImg as detection_labeling_window
from segmentationLabeling.labelme import main as segmentation_labeling_window
from segmentationUI.InstanceSegmentation_test import SegmentationUI as segmentatinon_training
from segmentationUI.InstanceSegmentation_test import PredictWindow as segmentation_predict
from PyQt5.QtWidgets import *
from detectionLabeling import labelImg  as detection_labeling_window
from segmentationLabeling.labelme import main as segmentation_labeling_window

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from skimage.measure import find_contours
from keras import backend as K  # Tensor tensor 오류 수정 부분
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

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


# ----------------------------------------------------------------------------------------------- stdout redirection ------------------------------------------------------------------------------------------------------------------------ #

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

class TrainingWindow(QWidget):

    def __init__(self, tabs_height):
        super().__init__()
        self.title = "Training"
        self.initUI(tabs_height)

    def initUI(self, tabs_height):

        tablewidgetheight = tabs_height * 0.91
        viewerwidgetheight = tabs_height * 0.93
        viewerwidgetwidth = QDesktopWidget().screenGeometry().width() * 0.65

        self.category = "Classification"
        self.resultFieldClear_signal = MySignal()
        self.viewer = PhotoViewer(self.parent())
        self.viewer.setMinimumSize(viewerwidgetwidth, viewerwidgetheight)
        self.qimage = None
        # self.setGeometry(300, 100, 1100, 800)
        self.setWindowTitle(self.title)

        self.setting_button = QAction(QIcon('UI_image/settings_rev.png'), 'Parameter\nSettings', self)
        self.setting_button.setStatusTip('Settings application')
        self.setting_button.triggered.connect(self.setting_Click)

        self.annotation_ob = QAction(QIcon('UI_image/anno_object_rev.png'), 'Labelling\n(Object\nDetection)', self)
        self.annotation_ob.setStatusTip('Annotate for Object Detection')
        self.annotation_ob.triggered.connect(self.detectionLabeling)

        self.annotation_seg = QAction(QIcon('UI_image/anno_object_rev.png'), 'Labelling\n(Segmentation)', self)
        self.annotation_seg.setStatusTip('Annotate for Object Detection')
        self.annotation_seg.triggered.connect(self.segmentationLabeling)

        mainwindow = QMainWindow()
        toolbar = QToolBar()
        mainwindow.addToolBar(Qt.LeftToolBarArea, toolbar)
        toolbar.addAction(self.setting_button)
        toolbar.addAction(self.annotation_ob)
        toolbar.addAction(self.annotation_seg)
        toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        toolbar.setMovable(False)

        test_label = QLabel("Parameters Viewer")
        test_label.setStyleSheet(
            "QLabel {"
            "font-size:15px;"
            "}"
        )
        font = test_label.font()
        font.setBold(True)
        test_label.setFont(font)
        mainwindow2 = QMainWindow()
        toolbar2 = QToolBar()
        mainwindow2.addToolBar(toolbar2)
        toolbar2.addWidget(test_label)
        toolbar2.setOrientation(Qt.Vertical)
        toolbar2.setMovable(False)

        self.tableWidget = QTableWidget(18, 2, self)
        self.tableWidget.verticalHeader().hide()
        self.tableWidget.setFixedSize(310, tablewidgetheight)

        self.tableWidget.setHorizontalHeaderLabels(["Hyper Parameters", "Values"])
        self.tableWidget.horizontalHeaderItem(0).setBackground(Qt.darkGray)
        self.tableWidget.horizontalHeaderItem(1).setBackground(Qt.darkGray)
        self.tableWidget.horizontalHeader().setFixedHeight(35)
        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 150)

        self.tableWidget.setItem(0, 0, QTableWidgetItem("Pretrained model"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("VGG16"))
        self.tableWidget.setItem(1, 0, QTableWidgetItem("Fine Tuning"))
        self.tableWidget.setItem(1, 1, QTableWidgetItem("False"))
        self.tableWidget.setItem(2, 0, QTableWidgetItem("Number of trainable layers"))
        self.tableWidget.item(2, 0).setToolTip("Number of trainable layers")
        self.tableWidget.setItem(2, 1, QTableWidgetItem("0"))
        self.tableWidget.setRowHidden(2, True)
        self.tableWidget.setItem(3, 0, QTableWidgetItem("Batch size"))
        self.tableWidget.setItem(3, 1, QTableWidgetItem("10"))
        self.tableWidget.setItem(4, 0, QTableWidgetItem("Epoch"))
        self.tableWidget.setItem(4, 1, QTableWidgetItem("20"))
        self.tableWidget.setItem(5, 0, QTableWidgetItem("Input size"))
        self.tableWidget.setItem(5, 1, QTableWidgetItem("224 × 224"))
        self.tableWidget.setItem(6, 0, QTableWidgetItem("Input channel"))
        self.tableWidget.setItem(6, 1, QTableWidgetItem("RGB"))
        self.tableWidget.setItem(7, 0, QTableWidgetItem("Optimizer"))
        self.tableWidget.setItem(7, 1, QTableWidgetItem("Adam"))
        self.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
        self.tableWidget.setItem(8, 1, QTableWidgetItem("0.001"))
        self.tableWidget.setItem(9, 0, QTableWidgetItem("Image augmentation"))
        self.tableWidget.setItem(9, 1, QTableWidgetItem("False"))
        self.tableWidget.setItem(10, 0, QTableWidgetItem("Horizontal flip"))
        self.tableWidget.setItem(10, 1, QTableWidgetItem("False"))
        self.tableWidget.setRowHidden(10, True)
        self.tableWidget.setItem(11, 0, QTableWidgetItem("Vertical flip"))
        self.tableWidget.setItem(11, 1, QTableWidgetItem("False"))
        self.tableWidget.setRowHidden(11, True)
        self.tableWidget.setItem(12, 0, QTableWidgetItem("Rotation range"))
        self.tableWidget.setItem(12, 1, QTableWidgetItem("0"))
        self.tableWidget.setRowHidden(12, True)
        self.tableWidget.setItem(13, 0, QTableWidgetItem("Width shift range"))
        self.tableWidget.setItem(13, 1, QTableWidgetItem("0.0"))
        self.tableWidget.setRowHidden(13, True)
        self.tableWidget.setItem(14, 0, QTableWidgetItem("Height shift range"))
        self.tableWidget.setItem(14, 1, QTableWidgetItem("0.0"))
        self.tableWidget.setRowHidden(14, True)
        self.tableWidget.setItem(15, 0, QTableWidgetItem("Shear range"))
        self.tableWidget.setItem(15, 1, QTableWidgetItem("0.0"))
        self.tableWidget.setRowHidden(15, True)
        self.tableWidget.setItem(16, 0, QTableWidgetItem("Zoom range"))
        self.tableWidget.setItem(16, 1, QTableWidgetItem("0.0"))
        self.tableWidget.setRowHidden(16, True)
        self.tableWidget.setItem(17, 0, QTableWidgetItem("Rescale"))
        self.tableWidget.setItem(17, 1, QTableWidgetItem("None"))
        self.tableWidget.setRowHidden(17, True)

        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

        toolbar2.addWidget(self.tableWidget)
        toolbar2.setOrientation(Qt.Vertical)
        toolbar2.setMovable(False)

        test_label2  = QLabel("Display")
        font = test_label2.font()
        font.setBold(True)
        test_label2.setFont(font)

        #그래프 초기화면
        self.figure = plt.figure()
        # plt.plot(N, self.losses, label="train_loss")
        # plt.plot(N, self.acc, label="train_acc")
        # plt.plot(N, self.val_losses, label="val_loss")
        # plt.plot(N, self.val_acc, label="val_acc")
        plt.title("Training Loss and Accuracy [Epoch]")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()
        size = self.canvas.size()
        width, height = size.width(), size.height()
        print(width, height)
        self.im = QImage(self.canvas.buffer_rgba(), width, height, QImage.Format_ARGB32)
        plt.close()

        self.mainwindow3 = QMainWindow()
        self.toolbar3 = QToolBar()
        self.mainwindow3.addToolBar(self.toolbar3)
        # self.toolbar3.addWidget(self.viewer)
        self.graph_label = QLabel()
        self.graph_label2 = QLabel()
        self.hbox2 = QHBoxLayout()
        self.hbox2.addWidget(self.graph_label)
        self.hbox2.addWidget(self.graph_label2)
        # pixmap = QPixmap('UI_image/classification image.jpg')
        self.graph_label.setPixmap(QPixmap(self.im))
        self.graph_label2.setPixmap(QPixmap(self.im))

        # self.toolbar3.addWidget(test_label2)
        # self.toolbar3.addWidget(self.graph_label)
        # self.toolbar3.addWidget(self.graph_label2)
        # self.toolbar3.setMovable(False)
        self.toolbar3.setOrientation(Qt.Horizontal)

        self.scaleSize_list = [600, 500, 350, 350]
        self.lbl_img = QLabel()

        self.vbox = QVBoxLayout()
        self.vbox2 = QHBoxLayout()
        self.hbox = QHBoxLayout()

        self.vbox2.addWidget(mainwindow)
        # self.vbox2.addWidget(self.mainwindow3)
        self.vbox2.addLayout(self.hbox2)
        # self.vbox2.setSpacing(0)
        # self.vbox2.setContentsMargins(0, 0, 0, 0)

        self.hbox.addWidget(mainwindow2)
        self.hbox.addStretch(1)
        # self.hbox.addWidget(mainwindow3)
        self.hbox.addLayout(self.vbox2)
        self.hbox.addStretch(1)
        self.tabs_predict = QTabWidget()
        self.tabs_training = QTabWidget()
        # self.vbox.addWidget(mainwindow)
        self.vbox.addLayout(self.hbox)
        # self.vbox.addWidget(mainwindow4)

        self.setLayout(self.vbox)
        # self.show()
        # self.showMaximized()

        # self.classificationUI = ClassificationUI(self)
        # self.objectDetectionUI = objectDetectionTrainingUI(self)
        # self.segmentationUI = segmentatinon_training(self)

        self.classificationUI = ClassificationUI(self)
        self.classificationUI.plot_losses.graph_update_signal.signal.connect(self.classification_graph_change)
        self.classificationUI.plot_losses.graph_update_signal2.signal.connect(self.classification_graph_change2)
        self.objectDetectionUI = objectDetectionTrainingUI(self)
        self.objectDetectionUI.plot_losses.graph_update_signal.signal.connect(self.detection_graph_change)
        self.objectDetectionUI.plot_losses.graph_update_signal2.signal.connect(self.detection_graph_change2)
        self.segmentationUI = segmentatinon_training(self)
        self.segmentationUI.plot_losses.graph_update_signal.signal.connect(self.segmentation_graph_change)
        self.segmentationUI.plot_losses.graph_update_signal2.signal.connect(self.segmentation_graph_change2)

        self.category_changed_flag_c = False
        self.category_changed_flag_o = False
        self.category_changed_flag_s = False

        # graph 그래프 수정 함수

    @pyqtSlot()
    def classification_graph_change(self):
        self.graph_label.setPixmap(QPixmap(self.classificationUI.plot_losses.im).scaled(640, 480))

    def classification_graph_change2(self):
        self.graph_label2.setPixmap(QPixmap(self.classificationUI.plot_losses.im2).scaled(640, 480))

    def detection_graph_change(self):
        self.graph_label.setPixmap(QPixmap(self.objectDetectionUI.plot_losses.im).scaled(640, 480))

    def detection_graph_change2(self):
        self.graph_label2.setPixmap(QPixmap(self.objectDetectionUI.plot_losses.im2).scaled(640, 480))

    def segmentation_graph_change(self):
        self.graph_label.setPixmap(QPixmap(self.segmentationUI.plot_losses.im).scaled(640, 480))

    def segmentation_graph_change2(self):
        self.graph_label2.setPixmap(QPixmap(self.segmentationUI.plot_losses.im2).scaled(640, 480))

    def showMessageBox(self, msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Error")
        msgBox.exec_()

    def detectionLabeling(self):
        detection_labeling_window.get_main_app()

    def segmentationLabeling(self):
        segmentation_labeling_window.segmentation_labeling()

    def setting_Click(self):
        if self.category == "Classification":
            self.classificationUI.show()

        elif self.category == "Object Detection":
            self.objectDetectionUI.show()

        elif self.category == "Segmentation":
            self.segmentationUI.show()

    def category_changed(self, category):
        self.category = category
        if self.category == "Classification":
            if self.category_changed_flag_c == False:
                self.tableWidget.setItem(0, 0, QTableWidgetItem("Pretrained model"))
                self.tableWidget.setItem(0, 1, QTableWidgetItem("VGG16"))
                self.tableWidget.setItem(1, 0, QTableWidgetItem("Fine Tuning"))
                self.tableWidget.setItem(1, 1, QTableWidgetItem("False"))
                self.tableWidget.setItem(2, 0, QTableWidgetItem("Number of trainable layers"))
                self.tableWidget.item(2, 0).setToolTip("Number of trainable layers")
                self.tableWidget.setItem(2, 1, QTableWidgetItem("0"))
                self.tableWidget.setRowHidden(2, True)
                self.tableWidget.setItem(3, 0, QTableWidgetItem("Batch size"))
                self.tableWidget.setItem(3, 1, QTableWidgetItem("10"))
                self.tableWidget.setItem(4, 0, QTableWidgetItem("Epoch"))
                self.tableWidget.setItem(4, 1, QTableWidgetItem("20"))
                self.tableWidget.setItem(5, 0, QTableWidgetItem("Input size"))
                self.tableWidget.setItem(5, 1, QTableWidgetItem("224 × 224"))
                self.tableWidget.setItem(6, 0, QTableWidgetItem("Input channel"))
                self.tableWidget.setItem(6, 1, QTableWidgetItem("RGB"))
                self.tableWidget.setItem(7, 0, QTableWidgetItem("Optimizer"))
                self.tableWidget.setItem(7, 1, QTableWidgetItem("Adam"))
                self.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.tableWidget.setItem(8, 1, QTableWidgetItem("0.001"))
                self.tableWidget.setItem(9, 0, QTableWidgetItem("Image augmentation"))
                self.tableWidget.setItem(9, 1, QTableWidgetItem("False"))
                self.tableWidget.setItem(10, 0, QTableWidgetItem("Horizontal flip"))
                self.tableWidget.setItem(10, 1, QTableWidgetItem("False"))
                self.tableWidget.setRowHidden(10, True)
                self.tableWidget.setItem(11, 0, QTableWidgetItem("Vertical flip"))
                self.tableWidget.setItem(11, 1, QTableWidgetItem("False"))
                self.tableWidget.setRowHidden(11, True)
                self.tableWidget.setItem(12, 0, QTableWidgetItem("Rotation range"))
                self.tableWidget.setItem(12, 1, QTableWidgetItem("0"))
                self.tableWidget.setRowHidden(12, True)
                self.tableWidget.setItem(13, 0, QTableWidgetItem("Width shift range"))
                self.tableWidget.setItem(13, 1, QTableWidgetItem("0.0"))
                self.tableWidget.setRowHidden(13, True)
                self.tableWidget.setItem(14, 0, QTableWidgetItem("Height shift range"))
                self.tableWidget.setItem(14, 1, QTableWidgetItem("0.0"))
                self.tableWidget.setRowHidden(14, True)
                self.tableWidget.setItem(15, 0, QTableWidgetItem("Shear range"))
                self.tableWidget.setItem(15, 1, QTableWidgetItem("0.0"))
                self.tableWidget.setRowHidden(15, True)
                self.tableWidget.setItem(16, 0, QTableWidgetItem("Zoom range"))
                self.tableWidget.setItem(16, 1, QTableWidgetItem("0.0"))
                self.tableWidget.setRowHidden(16, True)
                self.tableWidget.setItem(17, 0, QTableWidgetItem("Rescale"))
                self.tableWidget.setItem(17, 1, QTableWidgetItem("None"))
                self.tableWidget.setRowHidden(17, True)

                self.tableWidget.setRowHidden(0, False)
                self.tableWidget.setRowHidden(1, False)
                self.tableWidget.setRowHidden(3, False)
                self.tableWidget.setRowHidden(4, False)
                self.tableWidget.setRowHidden(5, False)
                self.tableWidget.setRowHidden(6, False)
                self.tableWidget.setRowHidden(7, False)
                self.tableWidget.setRowHidden(8, False)
                self.tableWidget.setRowHidden(9, False)

                self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
            elif self.category_changed_flag_c == True:
                self.tableWidget.setItem(0, 0, QTableWidgetItem("Pretrained model"))
                self.tableWidget.setItem(1, 0, QTableWidgetItem("Fine Tuning"))
                self.tableWidget.setItem(2, 0, QTableWidgetItem("Number of trainable layers"))
                self.tableWidget.item(2, 0).setToolTip("Number of trainable layers")
                self.tableWidget.setItem(3, 0, QTableWidgetItem("Batch size"))
                self.tableWidget.setItem(4, 0, QTableWidgetItem("Epoch"))
                self.tableWidget.setItem(5, 0, QTableWidgetItem("Input size"))
                self.tableWidget.setItem(6, 0, QTableWidgetItem("Input channel"))
                self.tableWidget.setItem(7, 0, QTableWidgetItem("Optimizer"))
                self.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.tableWidget.setItem(9, 0, QTableWidgetItem("Image augmentation"))
                self.tableWidget.setItem(10, 0, QTableWidgetItem("Horizontal flip"))
                self.tableWidget.setItem(11, 0, QTableWidgetItem("Vertical flip"))
                self.tableWidget.setItem(12, 0, QTableWidgetItem("Rotation range"))
                self.tableWidget.setItem(13, 0, QTableWidgetItem("Width shift range"))
                self.tableWidget.setItem(14, 0, QTableWidgetItem("Height shift range"))
                self.tableWidget.setItem(15, 0, QTableWidgetItem("Shear range"))
                self.tableWidget.setItem(16, 0, QTableWidgetItem("Zoom range"))
                self.tableWidget.setItem(17, 0, QTableWidgetItem("Rescale"))

                self.tableWidget.setRowHidden(0, False)
                self.tableWidget.setRowHidden(1, False)
                self.tableWidget.setRowHidden(3, False)
                self.tableWidget.setRowHidden(4, False)
                self.tableWidget.setRowHidden(5, False)
                self.tableWidget.setRowHidden(6, False)
                self.tableWidget.setRowHidden(7, False)
                self.tableWidget.setRowHidden(8, False)
                self.tableWidget.setRowHidden(9, False)

                self.tableWidget.setItem(0, 1, QTableWidgetItem(self.classificationUI.pretrained_combo.currentText() if self.classificationUI.pretrained_radio.isChecked() else "None"))
                self.tableWidget.setItem(1, 1, QTableWidgetItem("True" if self.classificationUI.fineTuning_check.isChecked() else "False"))
                if self.classificationUI.fineTuning_check.isChecked():
                    self.tableWidget.setItem(2, 1, QTableWidgetItem(self.classificationUI.fineTuningSettingDialog.num_of_freezing.text()))
                    self.tableWidget.setRowHidden(2, False)
                else:
                    self.tableWidget.setRowHidden(2, True)
                self.tableWidget.setItem(3, 1, QTableWidgetItem(self.classificationUI.batch_line.text()))
                self.tableWidget.setItem(4, 1, QTableWidgetItem(self.classificationUI.epoch_line.text()))
                self.tableWidget.setItem(5, 1, QTableWidgetItem(self.classificationUI.inputShape_line1.text() + "×" + self.classificationUI.inputShape_line2.text()))
                self.tableWidget.setItem(6, 1, QTableWidgetItem(self.classificationUI.inputShape_combo.currentText()))
                self.tableWidget.setItem(7, 1, QTableWidgetItem(self.classificationUI.optimizer_combo.currentText()))
                self.tableWidget.setItem(8, 1, QTableWidgetItem(self.classificationUI.learningRate_line.text()))
                self.tableWidget.setItem(9, 1, QTableWidgetItem("True" if self.classificationUI.imageGenerator_check.isChecked() else "False"))
                if self.classificationUI.imageGenerator_check.isChecked():
                    self.tableWidget.item(9, 0).setBackground(QtGui.QColor(58, 58, 58))
                    self.tableWidget.item(9, 1).setBackground(QtGui.QColor(58, 58, 58))
                    self.tableWidget.setItem(10, 1, QTableWidgetItem("True" if self.classificationUI.imageDataGeneratorSettingWindow.horizontalFlip.isChecked() else "False"))
                    # self.tableWidget.item(10, 0).setBackground(Qt.darkGray)
                    # self.tableWidget.item(10, 1).setBackground(Qt.darkGray)
                    self.tableWidget.setRowHidden(10, False)
                    self.tableWidget.setItem(11, 1, QTableWidgetItem("True" if self.classificationUI.imageDataGeneratorSettingWindow.verticalFlip.isChecked() else "False"))
                    # self.tableWidget.item(11, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.tableWidget.item(11, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.tableWidget.setRowHidden(11, False)
                    self.tableWidget.setItem(12, 1, QTableWidgetItem(self.classificationUI.imageDataGeneratorSettingWindow.rotationRangeText.text()))
                    # self.tableWidget.item(12, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.tableWidget.item(12, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.tableWidget.setRowHidden(12, False)
                    self.tableWidget.setItem(13, 1, QTableWidgetItem(self.classificationUI.imageDataGeneratorSettingWindow.widthShiftRangeText.text()))
                    # self.tableWidget.item(13, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.tableWidget.item(13, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.tableWidget.setRowHidden(13, False)
                    self.tableWidget.setItem(14, 1, QTableWidgetItem(self.classificationUI.imageDataGeneratorSettingWindow.heightShiftRangeText.text()))
                    # self.tableWidget.item(14, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.tableWidget.item(14, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.tableWidget.setRowHidden(14, False)
                    self.tableWidget.setItem(15, 1, QTableWidgetItem(self.classificationUI.imageDataGeneratorSettingWindow.shearRangeText.text()))
                    # self.tableWidget.item(15, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.tableWidget.item(15, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.tableWidget.setRowHidden(15, False)
                    self.tableWidget.setItem(16, 1, QTableWidgetItem(self.classificationUI.imageDataGeneratorSettingWindow.zoomRangeText.text()))
                    # self.tableWidget.item(16, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.tableWidget.item(16, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.tableWidget.setRowHidden(16, False)
                    self.tableWidget.setItem(17, 1, QTableWidgetItem(self.classificationUI.imageDataGeneratorSettingWindow.rescaleText.text()))
                    # self.tableWidget.item(17, 0).setBackground(QtGui.QColor(255, 255, 198))
                    # self.tableWidget.item(17, 1).setBackground(QtGui.QColor(255, 255, 198))
                    self.tableWidget.setRowHidden(17, False)
                else:
                    self.tableWidget.setRowHidden(10, True)
                    self.tableWidget.setRowHidden(11, True)
                    self.tableWidget.setRowHidden(12, True)
                    self.tableWidget.setRowHidden(13, True)
                    self.tableWidget.setRowHidden(14, True)
                    self.tableWidget.setRowHidden(15, True)
                    self.tableWidget.setRowHidden(16, True)
                    self.tableWidget.setRowHidden(17, True)

        elif self.category == "Object Detection":
            if self.category_changed_flag_o == False:
                self.tableWidget.setItem(0, 0, QTableWidgetItem("Customized anchor"))
                self.tableWidget.setItem(0, 1, QTableWidgetItem("False"))
                self.tableWidget.setItem(1, 0, QTableWidgetItem("Anchor size"))
                self.tableWidget.setItem(1, 1, QTableWidgetItem("32, 64, 128, 256, 512"))
                self.tableWidget.item(1, 1).setToolTip("32, 64, 128, 256, 512")
                self.tableWidget.setRowHidden(1, True)
                self.tableWidget.setItem(2, 0, QTableWidgetItem("Anchor stride"))
                self.tableWidget.setItem(2, 1, QTableWidgetItem("8, 16, 32, 64, 128"))
                self.tableWidget.item(2, 1).setToolTip("8, 16, 32, 64, 128")
                self.tableWidget.setRowHidden(2, True)
                self.tableWidget.setItem(3, 0, QTableWidgetItem("Anchor ratio"))
                self.tableWidget.setItem(3, 1, QTableWidgetItem("0.5, 1, 2"))
                self.tableWidget.item(3, 1).setToolTip("0.5, 1, 2")
                self.tableWidget.setRowHidden(3, True)
                self.tableWidget.setItem(4, 0, QTableWidgetItem("Anchor scale"))
                self.tableWidget.setItem(4, 1, QTableWidgetItem("1, 1.2, 1.6"))
                self.tableWidget.item(4, 1).setToolTip("1, 1.2, 1.6")
                self.tableWidget.setRowHidden(4, True)
                self.tableWidget.setItem(5, 0, QTableWidgetItem("Batch size"))
                self.tableWidget.setItem(5, 1, QTableWidgetItem("1"))
                self.tableWidget.setItem(6, 0, QTableWidgetItem("Steps"))
                self.tableWidget.setItem(6, 1, QTableWidgetItem("1000"))
                self.tableWidget.setItem(7, 0, QTableWidgetItem("Epochs"))
                self.tableWidget.setItem(7, 1, QTableWidgetItem("10"))
                self.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.tableWidget.setItem(8, 1, QTableWidgetItem("0.00001"))
                self.tableWidget.setItem(9, 0, QTableWidgetItem("Maximum resizing image size"))
                self.tableWidget.item(9, 0).setToolTip("Maximum resizing image size")
                self.tableWidget.setItem(9, 1, QTableWidgetItem("1333"))
                self.tableWidget.setItem(10, 0, QTableWidgetItem("Minimum resizing image size"))
                self.tableWidget.item(10, 0).setToolTip("Minimum resizing image size")
                self.tableWidget.setItem(10, 1, QTableWidgetItem("900"))
                self.tableWidget.setItem(11, 0, QTableWidgetItem("Image augmentation"))
                self.tableWidget.setItem(11, 1, QTableWidgetItem("False"))
                self.tableWidget.setItem(12, 0, QTableWidgetItem("Rotation range"))
                self.tableWidget.setItem(12, 1, QTableWidgetItem("-0.1 ~ 0.1"))
                self.tableWidget.setItem(13, 0, QTableWidgetItem("Shearing range"))
                self.tableWidget.setItem(13, 1, QTableWidgetItem("-0.1 ~ 0.1"))
                self.tableWidget.setItem(14, 0, QTableWidgetItem("(X, Y) Translation range"))
                self.tableWidget.item(14, 0).setToolTip("(X, Y) Translation range")
                self.tableWidget.setItem(14, 1, QTableWidgetItem("(-0.1, -0.1) ~ (0.1, 0.1)"))
                self.tableWidget.item(14, 1).setToolTip("(-0.1, -0.1) ~ (0.1, 0.1)")
                self.tableWidget.setItem(15, 0, QTableWidgetItem("(X, Y) Scaling range"))
                self.tableWidget.item(15, 0).setToolTip("(X, Y) Scaling range")
                self.tableWidget.setItem(15, 1, QTableWidgetItem("(0.9, 0.9) ~ (1.1, 1.1)"))
                self.tableWidget.item(15, 1).setToolTip("(0.9, 0.9) ~ (1.1, 1.1)")
                self.tableWidget.setItem(16, 0, QTableWidgetItem("X, Y Flip chance"))
                self.tableWidget.item(16, 0).setToolTip("X, Y Flip chance")
                self.tableWidget.setItem(16, 1, QTableWidgetItem("0.5, 0.5"))

                self.tableWidget.setRowHidden(5, False)
                self.tableWidget.setRowHidden(6, False)
                self.tableWidget.setRowHidden(7, False)
                self.tableWidget.setRowHidden(8, False)
                self.tableWidget.setRowHidden(9, False)
                self.tableWidget.setRowHidden(10, False)
                self.tableWidget.setRowHidden(11, False)
                self.tableWidget.setRowHidden(12, True)
                self.tableWidget.setRowHidden(13, True)
                self.tableWidget.setRowHidden(14, True)
                self.tableWidget.setRowHidden(15, True)
                self.tableWidget.setRowHidden(16, True)
                self.tableWidget.setRowHidden(17, True)

                self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
            elif self.category_changed_flag_o == True:
                self.tableWidget.setItem(0, 0, QTableWidgetItem("Customized anchor"))
                self.tableWidget.setItem(1, 0, QTableWidgetItem("Anchor size"))
                self.tableWidget.setItem(2, 0, QTableWidgetItem("Anchor stride"))
                self.tableWidget.setItem(3, 0, QTableWidgetItem("Anchor ratio"))
                self.tableWidget.setItem(4, 0, QTableWidgetItem("Anchor scale"))
                self.tableWidget.setItem(5, 0, QTableWidgetItem("Batch size"))
                self.tableWidget.setItem(6, 0, QTableWidgetItem("Steps"))
                self.tableWidget.setItem(7, 0, QTableWidgetItem("Epochs"))
                self.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.tableWidget.setItem(9, 0, QTableWidgetItem("Maximum resizing image size"))
                self.tableWidget.item(9, 0).setToolTip("Maximum resizing image size")
                self.tableWidget.setItem(10, 0, QTableWidgetItem("Minimum resizing image size"))
                self.tableWidget.item(10, 0).setToolTip("Minimum resizing image size")
                self.tableWidget.setItem(11, 0, QTableWidgetItem("Image augmentation"))
                self.tableWidget.setItem(12, 0, QTableWidgetItem("Rotation range"))
                self.tableWidget.setItem(13, 0, QTableWidgetItem("Shearing range"))
                self.tableWidget.setItem(14, 0, QTableWidgetItem("(X, Y) Translation range"))
                self.tableWidget.item(14, 0).setToolTip("(X, Y) Translation range")
                self.tableWidget.setItem(15, 0, QTableWidgetItem("(X, Y) Scaling range"))
                self.tableWidget.item(15, 0).setToolTip("(X, Y) Scaling range")
                self.tableWidget.setItem(16, 0, QTableWidgetItem("X, Y Flip chance"))
                self.tableWidget.item(16, 0).setToolTip("X, Y Flip chance")

                self.tableWidget.setItem(0, 1, QTableWidgetItem(
                    "True" if self.objectDetectionUI.anchor_checkBox.isChecked() else "False"))
                if self.objectDetectionUI.anchor_checkBox.isChecked():
                    self.tableWidget.setItem(1, 1, QTableWidgetItem(
                        self.objectDetectionUI.anchor_size1_line.text() + ", " + self.objectDetectionUI.anchor_size2_line.text()
                        + ", " + self.objectDetectionUI.anchor_size3_line.text() + ", " + self.objectDetectionUI.anchor_size4_line.text()
                        + ", " + self.objectDetectionUI.anchor_size5_line.text()))
                    self.tableWidget.item(1, 1).setToolTip(
                        self.objectDetectionUI.anchor_size1_line.text() + ", " + self.objectDetectionUI.anchor_size2_line.text()
                        + ", " + self.objectDetectionUI.anchor_size3_line.text() + ", " + self.objectDetectionUI.anchor_size4_line.text()
                        + ", " + self.objectDetectionUI.anchor_size5_line.text())
                    self.tableWidget.setRowHidden(1, False)
                    self.tableWidget.setItem(2, 1, QTableWidgetItem(
                        self.objectDetectionUI.anchor_stride_line.text() + ", " + self.objectDetectionUI.anchor_stride2_line.text()
                        + ", " + self.objectDetectionUI.anchor_stride3_line.text() + ", " + self.objectDetectionUI.anchor_stride4_line.text()
                        + ", " + self.objectDetectionUI.anchor_stride5_line.text()))
                    self.tableWidget.item(2, 1).setToolTip(
                        self.objectDetectionUI.anchor_stride_line.text() + ", " + self.objectDetectionUI.anchor_stride2_line.text()
                        + ", " + self.objectDetectionUI.anchor_stride3_line.text() + ", " + self.objectDetectionUI.anchor_stride4_line.text()
                        + ", " + self.objectDetectionUI.anchor_stride5_line.text())
                    self.tableWidget.setRowHidden(2, False)
                    self.tableWidget.setItem(3, 1, QTableWidgetItem(
                        self.objectDetectionUI.anchor_ratio_line.text() + ", " + self.objectDetectionUI.anchor_ratio2_line.text()
                        + ", " + self.objectDetectionUI.anchor_ratio3_line.text()))
                    self.tableWidget.setRowHidden(3, False)
                    self.tableWidget.setItem(4, 1, QTableWidgetItem(
                        self.objectDetectionUI.anchor_scale_line.text() + ", " + self.objectDetectionUI.anchor_scale2_line.text()
                        + ", " + self.objectDetectionUI.anchor_scale3_line.text()))
                    self.tableWidget.setRowHidden(4, False)
                else:
                    self.tableWidget.setRowHidden(1, True)
                    self.tableWidget.setRowHidden(2, True)
                    self.tableWidget.setRowHidden(3, True)
                    self.tableWidget.setRowHidden(4, True)

                if self.objectDetectionUI.imageGenerator_check.isChecked():
                    self.tableWidget.setItem(12, 1, QTableWidgetItem(
                        self.objectDetectionUI.imageDataGeneratorSettingWindow.rotation_min_line.text() + " ~ "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.rotation_max_line.text()))
                    self.tableWidget.setRowHidden(12, False)
                    self.tableWidget.setItem(13, 1, QTableWidgetItem(
                        self.objectDetectionUI.imageDataGeneratorSettingWindow.shearing_min_line.text() + " ~ "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.shearing_max_line.text()))
                    self.tableWidget.setRowHidden(13, False)
                    self.tableWidget.setItem(14, 1, QTableWidgetItem(
                        "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_xmin_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_ymin_line.text()
                        + ")" + " ~ " + "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_xmax_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_ymax_line.text() + ")"))
                    self.tableWidget.item(14, 1).setToolTip(
                        "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_xmin_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_ymin_line.text()
                        + ")" + " ~ " + "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_xmax_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.translation_ymax_line.text() + ")")
                    self.tableWidget.setRowHidden(14, False)
                    self.tableWidget.setItem(15, 1, QTableWidgetItem(
                        "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_xmin_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_ymin_line.text()
                        + ")" + " ~ " + "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_xmax_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_ymax_line.text() + ")"))
                    self.tableWidget.item(15, 1).setToolTip(
                        "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_xmin_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_ymin_line.text()
                        + ")" + " ~ " + "(" + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_xmax_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.scaling_ymax_line.text() + ")")
                    self.tableWidget.setRowHidden(15, False)
                    self.tableWidget.setItem(16, 1, QTableWidgetItem(
                        self.objectDetectionUI.imageDataGeneratorSettingWindow.flip_x_line.text() + ", "
                        + self.objectDetectionUI.imageDataGeneratorSettingWindow.flip_y_line.text()))
                    self.tableWidget.setRowHidden(16, False)
                else:
                    self.tableWidget.setRowHidden(12, True)
                    self.tableWidget.setRowHidden(13, True)
                    self.tableWidget.setRowHidden(14, True)
                    self.tableWidget.setRowHidden(15, True)
                    self.tableWidget.setRowHidden(16, True)

                self.tableWidget.setItem(5, 1, QTableWidgetItem(self.objectDetectionUI.batchsize_line.text()))
                self.tableWidget.setItem(6, 1, QTableWidgetItem(self.objectDetectionUI.steps_line.text()))
                self.tableWidget.setItem(7, 1, QTableWidgetItem(self.objectDetectionUI.epochs_line.text()))
                self.tableWidget.setItem(8, 1, QTableWidgetItem(self.objectDetectionUI.learning_rate_line.text()))
                self.tableWidget.setItem(9, 1, QTableWidgetItem(self.objectDetectionUI.image_maxSize_line.text()))
                self.tableWidget.setItem(10, 1, QTableWidgetItem(self.objectDetectionUI.image_minSize_line.text()))
                self.tableWidget.setItem(11, 1, QTableWidgetItem(
                    "True" if self.objectDetectionUI.imageGenerator_check.isChecked() else "False"))

                self.tableWidget.setRowHidden(5, False)
                self.tableWidget.setRowHidden(6, False)
                self.tableWidget.setRowHidden(7, False)
                self.tableWidget.setRowHidden(8, False)
                self.tableWidget.setRowHidden(9, False)
                self.tableWidget.setRowHidden(10, False)
                self.tableWidget.setRowHidden(11, False)
                self.tableWidget.setRowHidden(17, True)

        elif self.category == "Segmentation":
            if self.category_changed_flag_s == False:
                self.tableWidget.setItem(0, 0, QTableWidgetItem("Number of image assignments per GPU"))
                self.tableWidget.item(0, 0).setToolTip("Number of image assignments per GPU")
                self.tableWidget.setItem(0, 1, QTableWidgetItem("1"))
                self.tableWidget.setItem(1, 0, QTableWidgetItem("Steps per epoch for training"))
                self.tableWidget.item(1, 0).setToolTip("Steps per epoch for training")
                self.tableWidget.setItem(1, 1, QTableWidgetItem("100"))
                self.tableWidget.setItem(2, 0, QTableWidgetItem("Trainable Layers"))
                self.tableWidget.setItem(2, 1, QTableWidgetItem("heads"))
                self.tableWidget.setItem(3, 0, QTableWidgetItem("Pretrained weights for training"))
                self.tableWidget.item(3, 0).setToolTip("Pretrained weights for training")
                self.tableWidget.setItem(3, 1, QTableWidgetItem("COCO"))
                self.tableWidget.setItem(4, 0, QTableWidgetItem("Backbone for classification"))
                self.tableWidget.item(4, 0).setToolTip("Backbone for classification")
                self.tableWidget.setItem(4, 1, QTableWidgetItem("RESNET101"))
                self.tableWidget.setItem(5, 0, QTableWidgetItem("Epochs"))
                self.tableWidget.setItem(5, 1, QTableWidgetItem("50"))
                self.tableWidget.setItem(6, 0, QTableWidgetItem("Number of GPU to use"))
                self.tableWidget.item(6, 0).setToolTip("Number of GPU to use")
                self.tableWidget.setItem(6, 1, QTableWidgetItem("1"))
                self.tableWidget.setItem(7, 0, QTableWidgetItem("Number of steps for validation"))
                self.tableWidget.item(7, 0).setToolTip("Number of steps for validation")
                self.tableWidget.setItem(7, 1, QTableWidgetItem("10"))
                self.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.tableWidget.setItem(8, 1, QTableWidgetItem("0.00001"))
                self.tableWidget.setItem(9, 0, QTableWidgetItem("Learning momentum"))
                self.tableWidget.setItem(9, 1, QTableWidgetItem("0.9"))
                self.tableWidget.setItem(10, 0, QTableWidgetItem("Minimum confidence for detection"))
                self.tableWidget.item(10, 0).setToolTip("Minimum confidence for detection")
                self.tableWidget.setItem(10, 1, QTableWidgetItem("0.9"))

                self.tableWidget.setRowHidden(0, False)
                self.tableWidget.setRowHidden(1, False)
                self.tableWidget.setRowHidden(2, False)
                self.tableWidget.setRowHidden(3, False)
                self.tableWidget.setRowHidden(4, False)
                self.tableWidget.setRowHidden(5, False)
                self.tableWidget.setRowHidden(6, False)
                self.tableWidget.setRowHidden(7, False)
                self.tableWidget.setRowHidden(8, False)
                self.tableWidget.setRowHidden(9, False)
                self.tableWidget.setRowHidden(10, False)
                self.tableWidget.setRowHidden(11, True)
                self.tableWidget.setRowHidden(12, True)
                self.tableWidget.setRowHidden(13, True)
                self.tableWidget.setRowHidden(14, True)
                self.tableWidget.setRowHidden(15, True)
                self.tableWidget.setRowHidden(16, True)
                self.tableWidget.setRowHidden(17, True)

                self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)

            elif self.category_changed_flag_s == True:

                self.tableWidget.setItem(0, 0, QTableWidgetItem("Number of image assignments per GPU"))
                self.tableWidget.item(0, 0).setToolTip("Number of image assignments per GPU")
                self.tableWidget.setItem(1, 0, QTableWidgetItem("Steps per epoch for training"))
                self.tableWidget.item(1, 0).setToolTip("Steps per epoch for training")
                self.tableWidget.setItem(2, 0, QTableWidgetItem("Trainable Layers"))
                self.tableWidget.setItem(3, 0, QTableWidgetItem("Pretrained weights for training"))
                self.tableWidget.item(3, 0).setToolTip("Pretrained weights for training")
                self.tableWidget.setItem(4, 0, QTableWidgetItem("Backbone for classification"))
                self.tableWidget.item(4, 0).setToolTip("Backbone for classification")
                self.tableWidget.setItem(5, 0, QTableWidgetItem("Epochs"))
                self.tableWidget.setItem(6, 0, QTableWidgetItem("Number of GPU to use"))
                self.tableWidget.item(6, 0).setToolTip("Number of GPU to use")
                self.tableWidget.setItem(7, 0, QTableWidgetItem("Number of steps for validation"))
                self.tableWidget.item(7, 0).setToolTip("Number of steps for validation")
                self.tableWidget.setItem(8, 0, QTableWidgetItem("Learning rate"))
                self.tableWidget.setItem(9, 0, QTableWidgetItem("Learning momentum"))
                self.tableWidget.setItem(10, 0, QTableWidgetItem("Minimum confidence for detection"))
                self.tableWidget.item(10, 0).setToolTip("Minimum confidence for detection")

                self.tableWidget.setItem(0, 1, QTableWidgetItem(self.segmentationUI.gpuCount_line.text()))
                self.tableWidget.setItem(1, 1, QTableWidgetItem(self.segmentationUI.steps_per_epoch_line.text()))
                self.tableWidget.setItem(2, 1, QTableWidgetItem(self.segmentationUI.layers_combo.currentText()))
                self.tableWidget.setItem(3, 1, QTableWidgetItem(self.segmentationUI.weights_combo.currentText()))
                self.tableWidget.setItem(4, 1, QTableWidgetItem(self.segmentationUI.backBone_line.currentText()))
                self.tableWidget.setItem(5, 1, QTableWidgetItem(self.segmentationUI.epochs_line.text()))
                self.tableWidget.setItem(6, 1, QTableWidgetItem(self.segmentationUI.gpuCount_line.text()))
                self.tableWidget.setItem(7, 1, QTableWidgetItem(self.segmentationUI.validation_steps_line.text()))
                self.tableWidget.setItem(8, 1, QTableWidgetItem(self.segmentationUI.learningRate_line.text()))
                self.tableWidget.setItem(9, 1, QTableWidgetItem(self.segmentationUI.learningMomentum_line.text()))
                self.tableWidget.setItem(10, 1, QTableWidgetItem(self.segmentationUI.detection_min_confidence_line.text()))

                self.tableWidget.setRowHidden(0, False)
                self.tableWidget.setRowHidden(1, False)
                self.tableWidget.setRowHidden(2, False)
                self.tableWidget.setRowHidden(3, False)
                self.tableWidget.setRowHidden(4, False)
                self.tableWidget.setRowHidden(5, False)
                self.tableWidget.setRowHidden(6, False)
                self.tableWidget.setRowHidden(7, False)
                self.tableWidget.setRowHidden(8, False)
                self.tableWidget.setRowHidden(9, False)
                self.tableWidget.setRowHidden(10, False)
                self.tableWidget.setRowHidden(11, True)
                self.tableWidget.setRowHidden(12, True)
                self.tableWidget.setRowHidden(13, True)
                self.tableWidget.setRowHidden(14, True)
                self.tableWidget.setRowHidden(15, True)
                self.tableWidget.setRowHidden(16, True)
                self.tableWidget.setRowHidden(17, True)

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
    app.setStyle(QStyleFactory.create("Fusion"))
    trainingWindow = TrainingWindow()
    trainingWindow.show()
    sys.exit(app.exec_())