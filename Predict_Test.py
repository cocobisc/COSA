import ctypes
import random
import colorsys
import skimage.draw
from PyQt5 import QtWidgets
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from skimage.measure import find_contours
import sys
import logging
import tensorflow as tf
import threading
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from keras import backend as K # Tensor tensor 오류 수정 부분
import keras
from retinanet.keras_retinanet import models
from retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from retinanet.keras_retinanet.utils.colors import label_color
import cv2
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from keras.preprocessing import image
import time
from keras.models import load_model

class MySignal(QObject):
    signal = pyqtSignal()
    def run(self):
        self.signal.emit()

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

class PredictWindow(QWidget):

    def __init__(self, tabs_height):
        super().__init__()
        self.title = "Predict"
        self.classes = ["BG"]
        self.flag = False
        self.img_path = ''
        self.orig = []
        self.model_changed = True
        self.before_model = None
        self.initUI(tabs_height)
        self.classes_flag = False

    def initUI(self, tabs_height):

        viewerwidgetwidth = QDesktopWidget().screenGeometry().width() * 0.68
        viewerwidgetheight = tabs_height * 0.93
        treewidgetheight = tabs_height * 0.91

        self.change_button_signal = MySignal()
        self.model_select_signal = MySignal()

        self.output_color = (0, 0, 0)
        self.model_select_flag = False
        self.viewer = PhotoViewer(self.parent())
        self.viewer.setFixedSize(viewerwidgetwidth, viewerwidgetheight)
        self.qimage = None
        self.update_signal = MySignal()
        self.resultFieldClear_signal = MySignal()
        self.update_signal.signal.connect(self.img_update)
        self.setGeometry(300, 100, 1100, 800)
        self.setWindowTitle(self.title)
        self.model = QFileSystemModel()
        filters = ["*.bmp", "*.rle", "*.dib", "*.jpg", "*.jpeg", "*.gif", "*.png", "*.tif", "*.tiff", "*.raw"]
        self.model.setNameFilters(filters)
        self.model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.model.setRootPath(QDir.currentPath())
        self.tree.expandAll()
        self.tree.setFixedWidth(300)
        self.tree.setAnimated(False)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)
        self.tree.doubleClicked.connect(self.doubleClick)
        self.category = "Classification"

        for i in range(1, 4):
            self.tree.setColumnHidden(i, True)

        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.tree.setMinimumSize(0,treewidgetheight)

        # if self.model_path != "Model Path : Unselected":
        #     self.label1 = QLabel("Model Path : " + self.model_path)
        # else:
        #     self.label1 = QLabel(self.model_path)

        # self.predict_button = QAction(QIcon('UI_image/predict_image.png'), 'Predict', self)
        # self.predict_button.setShortcut('Ctrl+P')
        # self.predict_button.setStatusTip('Predict application')
        # self.predict_button.triggered.connect(self.linktest)

        self.save_button = QAction(QIcon('UI_image/save_button.png'), 'Save Image', self)
        self.save_button.setShortcut('Ctrl+S')
        self.save_button.setStatusTip('Save application')
        self.save_button.triggered.connect(self.save_Click)

        self.model_path_button = QAction(QIcon('UI_image/model_path_button.png'), 'Load Model', self)
        self.model_path_button.setStatusTip('Model Path Select application')
        self.model_path_button.triggered.connect(self.model_path_select)

        self.choose_color = QAction(QIcon('UI_image/choose_color.png'), 'Choose\nOutput Color', self)
        self.choose_color.setStatusTip('Choose text color on result image')
        self.choose_color.triggered.connect(self.choose_color_dialog)


        mainwindow = QMainWindow()
        toolbar = QToolBar()
        toolbar.setMovable(False)
        mainwindow.addToolBar(Qt.LeftToolBarArea, toolbar)

        test_label  = QLabel("File Browser")
        test_label.setStyleSheet(
            "QLabel {"
            "font-size:15px;"
            "}"
        )
        font = test_label.font()
        font.setBold(True)
        test_label.setFont(font)

        toolbar.addAction(self.model_path_button)
        toolbar.addAction(self.save_button)
        toolbar.addAction(self.choose_color)

        toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        mainwindow2 = QMainWindow()
        toolbar2 = QToolBar()
        mainwindow2.addToolBar(toolbar2)
        toolbar2.addWidget(test_label)
        toolbar2.addWidget(self.tree)
        toolbar2.setOrientation(Qt.Vertical)
        toolbar2.setMovable(False)

        test_label2  = QLabel()
        font = test_label2.font()
        font.setBold(True)
        test_label2.setFont(font)


        mainwindow3 = QMainWindow()
        toolbar3 = QToolBar()
        mainwindow3.addToolBar(toolbar3)
        toolbar3.addWidget(self.viewer)
        toolbar3.setOrientation(Qt.Vertical)
        toolbar3.setContentsMargins(0, 1, 0, 1)
        toolbar3.setMovable(False)


        test_label3  = QLabel("Console")
        font = test_label3.font()
        font.setBold(True)
        test_label3.setFont(font)



        self.scaleSize_list = [600, 500, 350, 350]
        self.lbl_img = QLabel()

        self.viewer.photoClicked.connect(self.photoClicked)

        self.vbox = QVBoxLayout()
        self.vbox2 = QHBoxLayout()
        self.hbox = QHBoxLayout()

        self.vbox2.addWidget(mainwindow)
        self.vbox2.addWidget(mainwindow3)
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
        # self.showMaximized()

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

    def terminate(self):
        self.terminate_thread(self.t)


    def terminate_thread(self, thread):
        if not thread.isAlive():
            self.save_button.setEnabled(True)
            self.model_path_button.setEnabled(True)
            self.change_button_signal.run()

            return
        mb = QMessageBox
        msg = "Training process has been working. Are you sure to STOP?"
        answer = mb.warning(self, 'Attention', msg, mb.Yes | mb.No)
        if answer == mb.Yes:
            if not thread.isAlive():
                return
            self.save_button.setEnabled(True)
            self.model_path_button.setEnabled(True)
            self.change_button_signal.run()

            exc = ctypes.py_object(SystemExit)
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident), exc)
            if res == 0:
                raise ValueError("nonexistent thread id")
            elif res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
                raise SystemError("PyThreadState_SetAsyncExc failed")

            print("\nStopped")
        return



    def linktest(self):
        if self.flag and self.model_select_flag:
            self.save_button.setEnabled(False)
            self.model_path_button.setEnabled(False)
            # self.predict_button.setEnabled(False)

            self.resultFieldClear_signal.run()
            if (self.category == "Classification"):
                self.t = threading.Thread(target=self.Classification_Predict_Click)
            elif (self.category == "Object Detection"):
                self.t = threading.Thread(target=self.Objectdetection_Predict_Click)
            elif(self.category == "Segmentation"):
                self.t = threading.Thread(target=self.Segmentation_Predict_Click)
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            self.change_button_signal.run()
            self.t.start()
        elif not self.flag:
            self.showMessageBox('Please select image')
        elif not self.model_select_flag:
            self.showMessageBox('Please select model')

    def Classification_Predict_Click(self):
        print("Classification predict proceeding.....")
        # if self.model_changed :
        #     K.clear_session()  # Tensor tensor 오류 수정 부분
        #     self.model_changed = False
        model_path = self.model_path

        if self.model_changed:
            keras.backend.tensorflow_backend.set_session(self.get_session())
            self.classification_model = load_model(model_path)
            self.model_changed = False

        # self.orig.append(self.hangulFilePathImageRead(test_path))

        img = image.load_img(self.img_path, target_size=(
        self.classification_model.input_shape[1], self.classification_model.input_shape[2]))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        t0 = time.time()
        preds = self.classification_model.predict(x)
        t1 = time.time()

        if self.classes_flag :
            label = self.classes[np.argmax(preds, axis=1)[0]]
        else :
            label = np.argmax(preds,axis=1)[0]
        proba = np.max(preds)

        self.draw = read_image_bgr(self.img_path)
        self.draw = cv2.cvtColor(self.draw, cv2.COLOR_BGR2RGB)
        print("Image size : ",self.draw.shape[0],self.draw.shape[1])
        print("Label : ",end="")
        print(label)
        print("Probability : ", np.max(preds))

        caption = "{}: {:.2f}%".format(label, proba * 100)
        cv2.putText(self.draw, caption, (int(self.draw.shape[0] * 0.025), int(self.draw.shape[0] * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, (self.draw.shape[0] * 0.00175), self.output_color,
                    int(self.draw.shape[0] * 0.005))
        self.update_signal.run()


        self.save_button.setEnabled(True)
        self.model_path_button.setEnabled(True)
        self.change_button_signal.run()

    def Objectdetection_Predict_Click(self):
        print("Objectdetection predict proceeding.....")
        if self.model_changed:
            keras.backend.tensorflow_backend.set_session(self.get_session())
            self.obd_model = models.load_model(self.model_path, backbone_name='resnet50')
            self.model_changed = False

        # keras.backend.tensorflow_backend.set_session(self.get_session())
        # model = models.load_model(self.model_path, backbone_name='resnet50')

        labels_to_names = {}
        if self.classes_flag:
            for i in range(len(self.classes)):
                labels_to_names[i] = self.classes[i]

        # 여기에 labels_to_names = 추가
        image = read_image_bgr(self.img_path)
        self.draw = image.copy()
        self.draw = cv2.cvtColor(self.draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)
        start = time.time()
        boxes, scores, labels = self.obd_model.predict_on_batch(np.expand_dims(image, axis=0))
        print('Loaded in : ', round(time.time() - start, 2), ' sec')
        print('Threshold : ', 0.5)
        boxes /= scale
        cnt = 0
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            cnt += 1
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)
            b = box.astype(int)
            draw_box(self.draw, b, color=color)

            if self.classes_flag:
                caption = "{} {:.2f}%".format(labels_to_names[label], score * 100)
                print('Box', cnt, ' : ', box, labels_to_names[label], score)
            else:
                caption = "{} {:.2f}%".format(label, score * 100)
                print('Box', cnt, ' : ', box, ', ', label, ', ', score)
            draw_caption(self.draw, b, caption)
        self.update_signal.run()

        self.save_button.setEnabled(True)
        self.model_path_button.setEnabled(True)
        self.change_button_signal.run()

    def Segmentation_Predict_Click(self):
        print("Segmentation predict proceeding.....")

        # self.resultField.moveCursor(QtGui.QTextCursor.End)
        class InferenceConfig(Config):
            # Give the configuration a recognizable name
            NAME = ""
            NUM_CLASSES = len(self.classes)  # Background + Class Name
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        if self.model_changed:
            K.clear_session()
            self.seg_model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
            self.seg_model.load_weights(self.model_path, by_name=True)
            self.model_changed = False
        self.draw = self.detect_and_color_splash(self.seg_model, image_path=self.img_path)
        self.update_signal.run()
        self.save_button.setEnabled(True)
        self.model_path_button.setEnabled(True)

    def choose_color_dialog(self):
        color = QColorDialog.getColor()

        if color.isValid():
            self.output_color = color.getRgb()


    def model_path_select(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Options()
        # weights_file_name 이용
        self.model_path, _ = QFileDialog.getOpenFileName(self, "load h5 file", "", "h5 File (*.h5)", options=options)
        self.model_select_signal.run()


        if self.model_path:
            self.resultFieldClear_signal.run()
            try:
                classf = open(self.model_path[:-2] + "txt", "r")
                print("Model Path : " + self.model_path)
                lines = classf.readlines()
                if self.category == "Segmentation":
                    self.classes = ["BG"]
                else:
                    self.classes = []
                for line in lines:
                    self.classes.append(line[:-1])
                classf.close()
                self.model_select_flag = True
                self.model_changed = True
                self.classes_flag = True
            except FileNotFoundError:
                # self.showMessageBox(msg="You don't have a text file including label information matched to model")
                mb = QMessageBox
                textname = self.model_path.split("/")
                textname = textname[len(textname)-1]
                msg = "Can't find "+textname[:-2]+"txt. You should have it if you want to see the CLASS NAME.\nClick YES, you can add all class name that you already know, unless it will  show numeric class name in order."
                answer = mb.warning(self, 'Warning', msg, mb.Yes | mb.No)
                if answer == mb.Yes:
                    dialog = AddClasses(self.model_path[:-2] + "txt")
                    dialog.exec_()
                    if dialog.flag :
                        print("Model Path : " + self.model_path)
                        classf = open(self.model_path[:-2] + "txt", "r")
                        if self.category == "Segmentation":
                            self.classes = ["BG"]
                        else :
                            self.classes = []
                        lines = classf.readlines()
                        for line in lines:
                            self.classes.append(line[:-1])
                        classf.close()
                        self.classes_flag = True
                        self.model_select_flag = True
                        self.model_changed = True
                    else :
                        self.classes_flag = False
                        if self.category == "Segmentation" :
                            self.model_select_flag = False
                        else :
                            print("Model Path : " + self.model_path)
                            self.model_select_flag = True
                            self.model_changed = True
                elif answer == mb.No :
                    self.classes_flag = False
                    if self.category == "Segmentation":
                        self.model_select_flag = False
                    else:
                        print("Model Path : " + self.model_path)
                        self.model_select_flag = True
                        self.model_changed = True
        else:
            self.model_select_flag = False
            self.model_changed = False
            self.classes_flag = False

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
        image_path
        # Image or video?
        if image_path:
            # Read image
            image = skimage.io.imread(image_path)
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            return self.masking(image, r['rois'], r['masks'], r['class_ids'], self.classes,
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
                # if self.classes_flag :
                #     label = class_names[class_id]
                # else :
                #     label = class_id
                label = class_names[class_id]
                # label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            cv2.putText(masked_image, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        # if auto_show:
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

    def category_changed(self,category):
        self.title = "Predict"
        self.classes = ["BG"]
        self.flag = False
        self.img_path = ''
        self.orig = []
        self.model_changed = True
        self.before_model = None
        self.category = category
        self.viewer.setPhoto(QPixmap(self.img_path))
        self.model_select_flag = False
        self.classes_flag = False

        self.tree.collapseAll()
        index = self.model.index(QDir.currentPath())
        self.tree.expand(index)
        self.tree.scrollTo(index)


class AddClasses(QDialog):

    def __init__(self, file_path):
        super().__init__()

        self.setWindowTitle("Make Classes Text")
        self.setGeometry(750, 360, 300, 300)
        self.file_path = file_path
        self.flag = False
        self.initUI()

    def initUI(self):

        self.model_select_flag = False

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
        self.flag = True
        classf.close()
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    segmentationUI = PredictWindow()
    segmentationUI.show()
    sys.exit(app.exec_())