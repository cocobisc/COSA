import os
import sys
import cv2
import time
import logging
import tensorflow as tf
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
from keras import backend as K # Tensor tensor 오류 수정 부분
import csv
import PyQt5.Qt
# import keras
import keras

# import keras_retinanet
from retinanet.keras_retinanet import models
from retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from retinanet.keras_retinanet.utils.colors import label_color


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from PIL import Image
from PIL.ImageQt import ImageQt
# default_stdout = sys.stdout
# default_stderr = sys.stderr

logger = logging.getLogger(__name__)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
graph = tf.get_default_graph()

# class XStream(QObject):
#     _stdout = None
#     _stderr = None
#
#     messageWritten = pyqtSignal(str)
#
#     def flush( self ):
#         pass
#
#     def fileno( self ):
#         return -1
#
#     def write( self, msg ):
#         if ( not self.signalsBlocked() ):
#             self.messageWritten.emit(msg)
#
#     @staticmethod
#     def stdout():
#         #if ( not XStream._stdout ):
#         XStream._stdout = XStream()
#         sys.stdout = XStream._stdout
#
#         return XStream._stdout
#
#     @staticmethod
#     def stderr():
#         if ( not XStream._stderr ):
#             XStream._stderr = XStream()
#             sys.stderr = XStream._stderr
#         return XStream._stderr
#
# class LogMessageViewer(QTextBrowser):
#     def __init__(self, parent=None):
#         super(LogMessageViewer,self).__init__(parent)
#         self.setReadOnly(True)
#
#     @QtCore.pyqtSlot(str)
#     def appendLogMessage(self, msg):
#         horScrollBar = self.horizontalScrollBar()
#         verScrollBar = self.verticalScrollBar()
#         scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10
#         self.insertPlainText(msg)
#
#         if scrollIsAtEnd:
#             verScrollBar.setValue(verScrollBar.maximum()) # scroll to the bottom
#             horScrollBar.setValue(0) # scroll to the left



# ----------------------------------------------------------------------------------------------- stdout redirection ------------------------------------------------------------------------------------------------------------------------ #
from PyQt5 import QtCore, QtGui, QtWidgets

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
class MySignal(QObject) :
    signal1 = pyqtSignal()
    def run(self):
        self.signal1.emit()

class PredictWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'Predict'
        self.model_path = 'Model Path : Unselected'
        self.data_set_path = 'Dataset Path : Unselected'
        self.model_select_flag = False
        self.dataset_select_flag=False
        self.flag = False
        self.img_cnt=0
        self.img_path = ''
        self.orig = []
        self.qimg_list = []

        self.initUI()

    def initUI(self):

        self.viewer = PhotoViewer(self.parent())
        self.viewer.setMinimumSize(700, 700)
        # self.viewer.setPhoto(QPixmap('default-image.png'))

        self.qimage = None
        self.update_signal = MySignal()
        self.update_signal.signal1.connect(self.img_update)
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

        # self.model_path_button = QPushButton('   choose model path   ')
        # self.model_path_button.clicked.connect(self.model_path_select)

        # self.data_set_button = QPushButton("   DataSet   ")
        # self.data_set_button.clicked.connect(self.dataset_button_Click)

        self.label1 = QLabel(self.model_path)
        self.label2 = QLabel(self.data_set_path)
        self.threshold_label = QLabel("Threshold")
        self.threshold_line = QLineEdit()
        self.threshold_line.setText('0.5')
        self.threshold_hbox = QHBoxLayout()
        self.threshold_hbox.addWidget(self.threshold_label)
        self.threshold_hbox.addWidget(self.threshold_line)
        # self.resultField = LogMessageViewer()
        # self.resultField.setFixedHeight(200)

        self.predict_button = QAction(QIcon('UI_image/predict_image.png'), 'Predict', self)
        self.predict_button.setShortcut('Ctrl+P')
        self.predict_button.setStatusTip('Predict application')
        self.predict_button.triggered.connect(self.linktest)


        self.save_button = QAction(QIcon('UI_image/save_button.png'), 'Save', self)
        self.save_button.setShortcut('Ctrl+S')
        self.save_button.setStatusTip('Save application')
        self.save_button.triggered.connect(self.save_Click)

        self.model_path_button = QAction(QIcon('UI_image/model_path_button.png'), 'Model Path', self)
        # self.model_path_button.setShortcut('Ctrl+S')
        self.model_path_button.setStatusTip('Model Path Select application')
        self.model_path_button.triggered.connect(self.model_path_select)

        self.data_set_button = QAction(QIcon('UI_image/data_set_button.png'), 'Dataset Path', self)
        # self.data_set_button.setShortcut('Ctrl+S')
        self.data_set_button.setStatusTip('Dataset Path Select application')
        self.data_set_button.triggered.connect(self.dataset_button_Click)




        # self.statusBar()
        mainwindow = QMainWindow()
        toolbar = QToolBar()
        mainwindow.addToolBar(toolbar)
        # toolbar.addAction(self.threshold_hbox)
        # toolbar.addWidget(self.threshold_line)
        toolbar.addAction(self.predict_button)
        toolbar.addAction(self.save_button)
        toolbar.addAction(self.model_path_button)
        toolbar.addAction(self.data_set_button)


        toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        # toolbar.setMinimumSize(10,50)



        self.scaleSize_list = [600,500,350,350]

        self.lbl_img = QLabel()

        # for i in range(0,4):
        #     self.lbl_img.append(QLabel())

        # self.lbl_img.setPixmap(QPixmap('default-image.png').scaled(800,800))


        self.viewer.photoClicked.connect(self.photoClicked)

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()
        self.hbox2 = QHBoxLayout()
        self.hbox3 = QHBoxLayout()
        # self.vbox2 = QVBoxLayout()
        self.gridbox = QGridLayout()
        # self.hbox.addWidget(self.model_path_button)
        self.hbox.addWidget(self.label1)
        self.hbox.addStretch(2)
        # self.hbox.addWidget(self.label3)
        # self.hbox.addWidget(self.save_button)
        # self.hbox.addWidget(self.predict_button)
        # self.hbox.addWidget(mainwindow)
        # self.hbox3.addWidget(self.data_set_button)
        self.hbox3.addWidget(self.label2)
        self.hbox3.addStretch(2)

        # self.gridbox.addWidget(self.lbl_img)
        # self.gridbox.addWidget(self.viewer)

        self.hbox2.addWidget(self.tree)
        # self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.viewer)
        # self.hbox2.addStretch(2)
        self.hbox2.addWidget(mainwindow)
        # self.vbox2.addWidget(self.resultField)
        self.vbox.addLayout(self.hbox)
        self.vbox.addLayout(self.hbox3)
        self.vbox.addLayout(self.hbox2)
        # self.vbox.addLayout(self.vbox2)

        self.setLayout(self.vbox)

        # self.show()

    def photoClicked(self, pos):
        if self.viewer.dragMode()  == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    def doubleClick(self,index):
        img_extension = ["bmp","rle","dib","jpg","jpeg","gif","png","tif","tiff","raw"]
        tmp = ""
        file_path = self.model.filePath(index)

        for i in range(len(file_path) - 1, 0, -1):
            if file_path[i] == ".":
                break
            else:
                tmp += file_path[i]

        if tmp[-1::-1] in img_extension :
            self.flag = True
            self.img_path  = file_path

            # item = self.gridbox.takeAt(0)
            # widget = item.widget()
            # widget.deleteLater()
            # tmp_lbl_img = QLabel()
            # tmp_lbl_img.setPixmap(QPixmap(self.img_path).scaled(800, 800))
            # self.gridbox.addWidget(tmp_lbl_img)
            self.viewer.setPhoto(QPixmap(self.img_path))


        elif self.model.isDir(index) :
            self.flag = True
        else :
            self.showMessageBox("It is not an image file")

    def linktest(self):
        # sys.stdout = default_stdout
        # sys.stderr = default_stderr

        # XStream.stdout().messageWritten.connect(self.resultField.appendLogMessage)
        # XStream.stderr().messageWritten.connect(self.resultField.appendLogMessage)

        threads = []
        if self.flag and self.model_select_flag and self.dataset_select_flag :
            # self.resultField.moveCursor(QtGui.QTextCursor.End)
            # self.resultField.clear()

            self.save_button.setEnabled(False)
            self.predict_button.setEnabled(False)
            self.data_set_button.setEnabled(False)
            self.model_path_button.setEnabled(False)

            # sys.stdout.write('proceeding....\n\n')

            # for i in range(self.img_cnt):
            self.t = threading.Thread(target=self.Predict_Click)
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            self.t.start()
            # self.t.join()

        elif not self.flag :
            self.showMessageBox('Please select image')
        elif not self.model_select_flag :
            self.showMessageBox('Please select model')
        elif not self.dataset_select_flag :
            self.showMessageBox('Please select DataSet Directory')

    def Predict_Click(self):

        keras.backend.tensorflow_backend.set_session(self.get_session())
        model = models.load_model(self.model_path,backbone_name='resnet50')
        # labels_to_names = {20:'pore'}
        labels_to_names = {}
        f = open(self.data_set_path+'/classes.csv', 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for line in rdr:
            # print(line)
            labels_to_names[int(line[1])] = line[0]
        f.close()
        image = read_image_bgr(self.img_path)
        self.draw = image.copy()
        self.draw = cv2.cvtColor(self.draw,cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(self.draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(self.draw, b, caption)
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
        self.label1.setText("Model Path : "+self.model_path)
        self.model_select_flag = True

    def showMessageBox(self,msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Error")
        msgBox.exec_()

    def save_Click(self):
        if self.qimage is None :
            self.showMessageBox('You can save the image after prediction')
            return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, extension = QFileDialog.getSaveFileName(self, "Save","","JPEG(*.jpg;*.jpeg;*.jpe;*.jfif);;\
        PNG(*.png);;TIF(*.tif;*.tiff)",options=options)

        if fileName:
            if extension == "JPEG(*.jpg;*.jpeg;*.jpe;*.jfif)":
                self.qimage.save(fileName+".jpg")
            elif extension == "PNG(*.png)":
                self.qimage.save(fileName+".png")
            elif extension == "TIF(*.tif;*.tiff)":
                self.qimage.save(fileName+".tif")



    def dataset_button_Click(self):
        self.data_set_path = QFileDialog.getExistingDirectory(self, "Data Set Directory")
        self.label2.setText("Dataset Path : "+self.data_set_path)
        # self.model.setRootPath(self.data_set_path)
        self.dataset_select_flag = True

    @pyqtSlot()
    def img_update(self):
        self.qimage = ImageQt(Image.fromarray(self.draw))
        self.viewer.setPhoto(QPixmap().fromImage(self.qimage))


    #이미지 불러올때 한글 경로가 있으면 바꿔줘야함
    def hangulFilePathImageRead(self,filePath):
        stream = open(filePath.encode("utf-8"), "rb")
        bytes = bytearray(stream.read())
        numpyArray = np.asarray(bytes, dtype=np.uint8)
        return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)



class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.wg = PredictWindow()
        self.setCentralWidget(self.wg)
        self.setGeometry(300, 30, 1200, 900)
        exitaction = QAction(QIcon('exit.png'), 'Exit', self)
        exitaction.setShortcut('Ctrl+Q')
        exitaction.setStatusTip('프로그램 종료')
        exitaction.triggered.connect(QCoreApplication.instance().quit)

        # newmodelaction = QAction(QIcon('new.png'), 'New Model', self)
        # newmodelaction.setShortcut('Ctrl+N')
        # newmodelaction.setStatusTip('모델 생성')
        # newmodelaction.triggered.connect(self.newWindowStart)

        self.statusBar()

        # menubar = self.menuBar()
        # menubar.setNativeMenuBar(False)
        # fileMenu = menubar.addMenu('&File')
        # fileMenu.addAction(newmodelaction)
        # fileMenu.addAction(exitaction)

    # def newWindowStart(self):
    #     self.wg.resultField.setEnabled(False)
    #     window = ClassificationUI()
    #     window.exec_()
    #     self.wg.resultField.clear()
    #     self.wg.resultField.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = PredictWindow()
    main.show()
    sys.exit(app.exec_())



