import sys
import logging
from retinanet.keras_retinanet.bin.test import TrainingWindow
from obj_dtt_predict import PredictWindow
from classificationUI_test import ClassificationUI
from detectionLabeling import labelImg  as detection_labeling_window
from segmentationLabeling.labelme import main as segmentation_labeling_window
from segmentationUI.InstanceSegmentation_test import SegmentationUI as segmentatinon_training
from segmentationUI.InstanceSegmentation_test import PredictWindow as segmentation_predict
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import qdarkstyle
import os

os.environ['QT_API'] = 'pyqt'

class MainWindow(QDialog):
    def __init__(self):
        super().__init__(parent=None)
        self.title = 'Object Detection Training'
        self.initUI()
        # self.app = QApplication(sys.argv)

    def initUI(self):
        self.uiStack = list()
        self.setGeometry(700, 200, 600, 400)
        self.setFixedSize(600, 550)
        self.setWindowTitle(self.title)
        self.setStyleSheet(open("StyleSheet.qss", "r").read())
        self.help_icon = QPixmap("uiImage/help_icon.png")
        self.main_logo = QPixmap("uiImage/main_logo.png")
        self.main_logo_label = QLabel()
        self.main_logo_label.setPixmap(self.main_logo)
        self.main_vbox = QVBoxLayout()
        self.main_hbox = QHBoxLayout()
        self.tabs = QTabWidget()
        self.tab_home = QWidget()
        self.tab_classification = QWidget()
        self.test_line = QLineEdit()
        self.tab_detection = QWidget()
        self.tab_segmentation = QWidget()
        self.tabs.resize(300, 200)
        self.hbox_classification = QHBoxLayout()
        self.hbox_detection = QHBoxLayout()
        self.hbox_segmentation = QHBoxLayout()
        self.vbox_classification = QVBoxLayout()
        self.vbox_detection = QVBoxLayout()
        self.vbox_segmentation = QVBoxLayout()
        self.tabs.addTab(self.tab_home, "Home")
        self.tabs.addTab(self.tab_classification, "Classification")
        self.tabs.addTab(self.tab_detection, "Detection")
        self.tabs.addTab(self.tab_segmentation, "Segmentation")
        self.classification_training_button = QPushButton('Training')
        self.classification_predict_button = QPushButton('Predict')
        self.detection_training_button = QPushButton('Training')
        self.detection_predict_button = QPushButton('Predict')
        self.detection_labeling_button = QPushButton('Labeling')
        self.segmentation_labeling_button = QPushButton('Labeling')
        self.segmentation_training_button = QPushButton('Training')
        self.segmentation_predict_button = QPushButton('Predict')
        self.classification_training_button.setFixedWidth(120)
        self.classification_training_button.setFixedHeight(70)
        self.classification_predict_button.setFixedWidth(120)
        self.classification_predict_button.setFixedHeight(70)
        self.detection_training_button.setFixedWidth(120)
        self.detection_training_button.setFixedHeight(70)
        self.detection_predict_button.setFixedWidth(120)
        self.detection_predict_button.setFixedHeight(70)
        self.segmentation_training_button.setFixedWidth(120)
        self.segmentation_training_button.setFixedHeight(70)
        self.segmentation_predict_button.setFixedWidth(120)
        self.segmentation_predict_button.setFixedHeight(70)
        self.segmentation_labeling_button.setFixedWidth(120)
        self.segmentation_labeling_button.setFixedHeight(70)
        self.detection_labeling_button.setFixedWidth(120)
        self.detection_labeling_button.setFixedHeight(70)
        # self.detection_training_button.setStyleSheet("border-style: outset; border-width: 2; border-radius: 10; border-color: bei; font: bold 14; min-width: 10; padding: 6px")

        self.classification_image = QPixmap('uiImage/classification image.jpg')
        self.classification_img_label = QLabel()
        self.classification_img_label.setPixmap(QPixmap(self.classification_image))
        self.detection_image = QPixmap('uiImage/detection image.jpg')
        self.detection_img_label = QLabel()
        self.detection_img_label.setPixmap(QPixmap(self.detection_image))
        self.segmentation_image = QPixmap('uiImage/segmentation image.jpg')
        self.segmentation_img_label = QLabel()
        self.segmentation_img_label.setPixmap(QPixmap(self.segmentation_image))
        self.main_label = QLabel()
        self.main_label.setPixmap(QPixmap(self.help_icon))
        self.main_label.setFixedWidth(30)
        self.main_label.setFixedHeight(30)
        self.hbox_classification.addWidget(self.classification_img_label)
        self.vbox_classification.addWidget(self.classification_training_button)
        self.vbox_classification.addWidget(self.classification_predict_button)
        self.hbox_classification.addLayout(self.vbox_classification)
        self.hbox_classification.addSpacing(55)

        self.hbox_detection.addWidget(self.detection_img_label)
        self.vbox_detection.addWidget(self.detection_labeling_button)
        self.vbox_detection.addWidget(self.detection_training_button)
        self.vbox_detection.addWidget(self.detection_predict_button)
        self.hbox_detection.addLayout(self.vbox_detection)
        self.hbox_detection.addSpacing(40)

        self.hbox_segmentation.addWidget(self.segmentation_img_label)
        self.vbox_segmentation.addWidget(self.segmentation_labeling_button)
        self.vbox_segmentation.addWidget(self.segmentation_training_button)
        self.vbox_segmentation.addWidget(self.segmentation_predict_button)
        self.hbox_segmentation.addLayout(self.vbox_segmentation)
        self.hbox_segmentation.addSpacing(40)

        self.tab_home.layout = QVBoxLayout()
        self.main_hbox.addStretch(1)
        self.main_hbox.addWidget(self.main_logo_label)
        self.main_hbox.addStretch(1)
        # self.main_hbox.addSpacing(100)
        self.tab_home.layout.addLayout(self.main_hbox)
        self.tab_home.setLayout(self.tab_home.layout)

        self.tab_classification.layout = QVBoxLayout()
        self.tab_classification.layout.addLayout(self.hbox_classification)
        self.tab_classification.setLayout(self.tab_classification.layout)

        self.tab_detection.layout = QVBoxLayout()
        self.tab_detection.layout.addLayout(self.hbox_detection)
        self.tab_detection.setLayout(self.tab_detection.layout)

        self.tab_segmentation.layout = QVBoxLayout()
        self.tab_segmentation.layout.addLayout((self.hbox_segmentation))
        self.tab_segmentation.setLayout(self.tab_segmentation.layout)

        self.main_vbox.addWidget(self.tabs)
        self.setLayout(self.main_vbox)


        self.detection_training_button.clicked.connect(self.objectDetectionTraining)
        self.detection_predict_button.clicked.connect(self.objectDetectionPredict)
        self.classification_training_button.clicked.connect(self.classificationTraining)
        self.classification_predict_button.clicked.connect(self.classificationPredict)
        self.detection_labeling_button.clicked.connect(self.detectionLabeling)
        self.segmentation_labeling_button.clicked.connect(self.segmentationLabeling)
        self.segmentation_training_button.clicked.connect(self.segmentationTraining)
        self.segmentation_predict_button.clicked.connect(self.segmentationPredict)

    def segmentationLabeling(self):
        self.stackRefresh()

        self.uiStack.append(segmentation_labeling_window.segmentation_labeling())

    def detectionLabeling(self):
        # self.app, _win = detection_labeling_window.get_main_app()
        self.stackRefresh()
        self.uiStack.append(detection_labeling_window.get_main_app())

    def segmentationTraining(self):
        self.stackRefresh()
        self.segmentationTrainingUI = segmentatinon_training()
        self.segmentationTrainingUI.show()
        self.uiStack.append(self.segmentationTrainingUI)

    def segmentationPredict(self):
        self.stackRefresh()
        self.segmentationPredictUI = segmentation_predict(None)
        self.segmentationPredictUI.show()
        # self.segmentationPredictUI.close()
        # sys.exit(self.segmentationPredictUI().exec_())
        self.uiStack.append(self.segmentationPredictUI)

    def classificationTraining(self):
        self.stackRefresh()
        self.classificationTrainingUI = ClassificationUI()
        self.classificationTrainingUI.show()
        self.uiStack.append(self.classificationTrainingUI)

    def classificationPredict(self):
        self.stackRefresh()
        self.classificationPredictUI = classification_predict(None)
        self.classificationPredictUI.show()
        self.uiStack.append(self.classificationPredictUI)

    def objectDetectionTraining(self):
        self.stackRefresh()
        self.detectionTrainingUI = TrainingWindow()
        self.detectionTrainingUI.show()
        self.uiStack.append(self.detectionTrainingUI)

    def objectDetectionPredict(self):
        self.stackRefresh()
        self.detectionPredictUI = PredictWindow()
        self.detectionPredictUI.show()
        self.uiStack.append(self.detectionPredictUI)

    def stackRefresh(self):
        if self.uiStack:
            t = self.uiStack.pop()
            t.close()



if __name__ == '__main__':
    import time
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # Create and display the splash screen
    splash_pix = QPixmap(r'.\uiImage\main_logo.jpg')

    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    # adding progress bar
    # progressBar = QProgressBar(splash)
    # progressBar.resize(625,10)
    # progressBar.setGeometry(0, 200, 425, 10)
    splash.setMask(splash_pix.mask())

    splash.show()

    time.sleep(5)
    main = MainWindow()
    main.show()
    splash.finish(main)

    sys.exit(app.exec_())