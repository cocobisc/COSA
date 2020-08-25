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
from keras import backend as K # Tensor tensor 오류 수정 부분
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
import argparse
import warnings
import keras
import keras.preprocessing.image
import tensorflow as tf
default_stdout = sys.stdout
default_stderr = sys.stderr

logger = logging.getLogger(__name__)
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
graph = tf.get_default_graph()

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



class ClassificationUI(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Classification")
        self.setGeometry(420, 220, 1200, 600)
        self.setFixedSize(1200, 630)

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
        self.customed_button.clicked.connect(self.buttonClicked)

        customed_hbox.addWidget(self.customed_button)

        self.trainingDataSet_label = QLabel("Training Data Set : ", self)
        self.trainingDataSet_label.setFixedSize(600, 20)
        self.trainingDataSet_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        trainingDataSet_hbox = QHBoxLayout()
        trainingDataSet_hbox.addWidget(self.trainingDataSet_label)

        self.trainingDataSet_button = QPushButton("Set Training Data Set Directory")
        self.trainingDataSet_button.clicked.connect(self.buttonClicked)

        self.validationDataSet_label = QLabel("Save Model Path : ", self)
        self.validationDataSet_label.setFixedSize(600, 20)
        self.validationDataSet_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        validationDataSet_hbox = QHBoxLayout()
        validationDataSet_hbox.addWidget(self.validationDataSet_label)

        self.validationDataSet_button = QPushButton("Set Save Model Path")
        self.validationDataSet_button.clicked.connect(self.buttonClicked)

        self.modelName_label = QLabel("Model Name : ", self)
        self.modelName_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        modelName_hbox = QHBoxLayout()
        modelName_hbox.addWidget(self.modelName_label)

        self.modelName_line = QLineEdit(self)
        self.modelName_line.setReadOnly(False)
        self.modelName_line.setFixedHeight(27)

        modelName_hbox.addWidget(self.modelName_line)

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

        inputShape_gbox.addWidget(self.inputShape_empty2, d0, 1)

        self.inputShape_label = QLabel("Steps : ", self)
        self.inputShape_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_label, 0, 2)

        self.inputShape_line1 = QLineEdit(self)
        self.inputShape_line1.setReadOnly(False)
        # self.inputShape_line1.setText("224")
        self.inputShape_line1.setFixedSize(100, 27)

        inputShape_gbox.addWidget(self.inputShape_line1, 0, 3)

        self.inputShape_label_x = QLabel(" x ", self)
        self.inputShape_label_x.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        inputShape_gbox.addWidget(self.inputShape_label_x, 0, 4)

        self.inputShape_line2 = QLineEdit(self)
        self.inputShape_line2.setReadOnly(False)
        # self.inputShape_line2.setText("224")
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

        self.start_button = QPushButton("Start Training", self)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.startTraining)
        self.start_button.setFixedSize(150, 50)
        start_hbox.addWidget(self.start_button)

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

        hbox2.addLayout(model_layout2)

        self.resultField = LogMessageViewer(self)
        self.resultField.setFixedSize(1173, 210)


        hbox = QHBoxLayout()
        hbox.addLayout(hbox1)
        hbox.addLayout(hbox2)

        groupBox_layout.addLayout(hbox)

        base_layout.addWidget(self.resultField)

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

    def buttonClicked(self):
        if self.customed_radio.isChecked():
            self.fineTuning_check.setEnabled(False)
            self.customed_button.setEnabled(True)
            if self.sender().text() == "Set":
                self.customedModelSettingDialog.exec_()
        else:
            self.fineTuning_check.setEnabled(True)
            self.customed_button.setEnabled(False)

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

        elif self.sender().text() == "Set Save Model Path":
            self.validationDataSet_directoryName = QFileDialog.getExistingDirectory(self, "Select Model Directory")
            self.validationDataSet_label.setText("Save Model Path : " + self.validationDataSet_directoryName)

        if self.trainingDataSet_label.text() != "Training Data Set : " and self.validationDataSet_label.text() != "Validation Data Set : ":
            self.start_button.setEnabled(True)

    def startTraining(self):
        if self.sender().text() == "Start Training":
            if self.modelName_line.text() == "":
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
                    self.resultField.moveCursor(QtGui.QTextCursor.End)
                    self.resultField.clear()

                    self.start_button.setEnabled(False)

                    self.t = threading.Thread(target=self.training)
                    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                    self.t.start()

    def training(self):
        sys.stdout = default_stdout
        sys.stderr = default_stderr

        XStream.stdout().messageWritten.connect(self.resultField.appendLogMessage)
        XStream.stderr().messageWritten.connect(self.resultField.appendLogMessage)

        print('proceeding....\n\n')

        K.clear_session() # Tensor tensor 오류 수정 부분

        Steps = int(self.inputShape_line1.text())
        BATCH_SIZE = int(self.batch_line.text())
        EPOCHS = int(self.epoch_line.text())
        Learning_Rate=float(self.learningRate_line.text())

        DATASET_DIR = self.trainingDataSet_directoryName
        MODEL_SAVE_PATH = self.validationDataSet_directoryName
        MODEL_NAME = self.modelName_line.text()
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
                checkpoint = keras.callbacks.ModelCheckpoint(
                    os.path.join(
                        args.snapshot_path,
                        'snapshot.h5'  # 변경가능
                    ),
                    verbose=2,
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
                    min_rotation=-0.1,
                    max_rotation=0.1,
                    min_translation=(-0.1, -0.1),
                    max_translation=(0.1, 0.1),
                    min_shear=-0.1,
                    max_shear=0.1,
                    min_scaling=(0.9, 0.9),
                    max_scaling=(1.1, 1.1),
                    flip_x_chance=0.5,
                    flip_y_chance=0.5,
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
                                    help='Path to CSV file containing annotations for validation (optional).')

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
                                default=MODEL_SAVE_PATH)  # 사용자한테 받아서 변경
            parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
            parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots',
                                action='store_false')
            parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation',
                                action='store_false')
            parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
            parser.add_argument('--random-transform', help='Randomly transform image and annotations.',
                                action='store_true')
            parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.',
                                type=int, default=800)
            parser.add_argument('--image-max-side',
                                help='Rescale the image if the largest side is larger than max_side.', type=int,
                                default=1333)
            parser.add_argument('--config', help='Path to a configuration parameters .ini file.')
            parser.add_argument('--weighted-average',
                                help='Compute the mAP using the weighted average of precisions among classes.',
                                action='store_true')
            parser.add_argument('--compute-val-loss', help='Compute validation loss during training',
                                dest='compute_val_loss', action='store_true')
            parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
            parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.',
                                dest='class_specific_filter', action='store_false')
            # Fit generator arguments
            parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
            parser.add_argument('--workers', help='Number of generator workers.', type=int, default=0)
            parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.',
                                type=int, default=10)

            return check_args(parser.parse_args(args))


        args = []
        args.append('csv')
        args.append(DATASET_DIR+'/annotations.csv')
        args.append(DATASET_DIR+'/classes.csv')
        # args.append('--epochs=2')
        # args.append('--workers=0')
        print(type(args))
        # args = "train.py pascal ../../image_set"
        args = parse_args(args)
        print('----------------------------------')
        print(args)
        print(type(args))
        print(args.epochs)
        print('----------------------------------')
        # create object that stores backbone information
        backbone = models.backbone(args.backbone)
        # make sure keras is the minimum required version
        check_keras_version()
        # optionally choose specific GPU
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        keras.backend.tensorflow_backend.set_session(get_session())
        # optionally load config parameters
        anchor_parameters=None
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
        models.check_training_model(training_model)

        # convert the model
        model = models.convert_model(training_model, nms=args.nms, class_specific_filter=args.class_specific_filter,
                                     anchor_params=anchor_parameters)

        # save model
        model.save(MODEL_SAVE_PATH +'/' + MODEL_NAME + '.h5')
        print("model saved....")
    def isNumber(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

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

        self.freezing_layer = QLabel("Num of Freezing Layers : ", self)
        self.freezing_layer.setFixedSize(350, 150)
        self.freezing_layer.setFont(QFont("Arial",15))
        self.freezing_layer.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.num_of_freezing = QSpinBox(self)
        self.num_of_freezing.setMinimum(0)
        self.num_of_freezing.setMaximum(13)
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

class MySignal(QObject) :
    signal1 = pyqtSignal()
    def run(self):
        self.signal1.emit()

class PredictWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.title = 'Predict'
        self.model_path = 'Unselected'
        self.model_select_flag = False
        self.flag = False
        self.img_cnt=0
        self.img_path = []
        self.orig = []
        self.qimg_list = []
        self.initUI()

    def initUI(self):
        self.update_signal = MySignal()
        # self.update_signal.signal1.connect(self.img_update)
        self.setGeometry(300, 150, 600, 600)
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

        self.model_path_button = QPushButton('   choose model path   ')
        self.model_path_button.clicked.connect(self.model_path_select)

        self.label1 = QLabel(self.model_path)

        self.test_button = QPushButton('Predict')
        self.test_button.clicked.connect(self.linktest)

        self.resultField = LogMessageViewer()
        self.resultField.setFixedHeight(200)

        self.label2 = QLabel()
        self.label3 = QLabel(" * 최대 사진 4장까지 한번에 predict 가능 * ")

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.Undo_Click)

        self.scaleSize_list = [600,500,350,350]

        self.lbl_img = []

        for i in range(0,4):
            self.lbl_img.append(QLabel())

        self.lbl_img[0].setPixmap(QPixmap('default-image.png').scaled(600,600))

        self.vbox = QVBoxLayout()
        self.hbox = QHBoxLayout()
        self.hbox2 = QHBoxLayout()
        self.hbox3 = QHBoxLayout()
        self.vbox2 = QVBoxLayout()
        self.gridbox = QGridLayout()
        self.hbox.addWidget(self.model_path_button)
        self.hbox.addWidget(self.label1)
        self.hbox.addStretch(2)
        self.hbox.addWidget(self.label3)
        self.hbox.addWidget(self.undo_button)
        self.hbox.addWidget(self.test_button)
        self.gridbox.addWidget(self.lbl_img[0])
        self.hbox2.addWidget(self.tree)
        self.hbox2.addStretch(1)
        self.hbox2.addLayout(self.gridbox)
        self.hbox2.addStretch(2)
        self.vbox2.addWidget(self.resultField)
        self.vbox.addLayout(self.hbox)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addLayout(self.vbox2)

        self.setLayout(self.vbox)

        self.show()

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
            self.img_cnt += 1
            self.flag = True
            self.img_path .append(file_path)

            if self.img_cnt==5 :
                self.showMessageBox('최대 4장까지만 선택이 가능합니다.')
                self.img_cnt -=1
            elif self.img_cnt == 1 :
                item = self.gridbox.takeAt(0)
                widget = item.widget()
                widget.deleteLater()
                tmp_lbl_img = QLabel()
                tmp_lbl_img.setPixmap(QPixmap(self.img_path[0]).scaled(600, 600))
                self.gridbox.addWidget(tmp_lbl_img)
            else :
                for i in range(self.img_cnt-2,-1,-1):
                    item = self.gridbox.takeAt(i)
                    widget = item.widget()
                    widget.deleteLater()

                for i in range(0, self.img_cnt):
                    tmp_lbl_img = []
                    for i in range(0, self.img_cnt):
                        tmp_lbl_img.append(QLabel())
                        tmp_lbl_img[i].setPixmap(QPixmap(self.img_path[i]).scaled(self.scaleSize_list[self.img_cnt-1],
                                                                              self.scaleSize_list[self.img_cnt-1]))
                for i in range(0, self.img_cnt):
                    self.gridbox.addWidget(tmp_lbl_img[i], i / 2, i % 2)

        elif self.model.isDir(index) :
            self.flag = True
        else :
            self.showMessageBox("It is not an image file")

    def linktest(self):
        sys.stdout = default_stdout
        sys.stderr = default_stderr

        XStream.stdout().messageWritten.connect(self.resultField.appendLogMessage)
        XStream.stderr().messageWritten.connect(self.resultField.appendLogMessage)

        threads = []
        if self.flag and self.model_select_flag :
            self.resultField.moveCursor(QtGui.QTextCursor.End)
            self.resultField.clear()

            self.undo_button.setEnabled(False)
            self.test_button.setEnabled(False)

            sys.stdout.write('proceeding....\n\n')

            for i in range(self.img_cnt):
                self.t = threading.Thread(target=self.Predict_Click, args=(i,))
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                self.t.start()
                self.t.join()

        elif not self.flag :
            self.showMessageBox('Please select image')
        elif not self.model_select_flag :
            self.showMessageBox('Please select model')

    def Predict_Click(self, index):
            print("---------- image ", index+1 , "------------")

            K.clear_session() # Tensor tensor 오류 수정 부분
            classf = open(".\class.txt", "r")
            classes = []
            lines = classf.readlines()

            for line in lines:
                classes.append(line[:-1])

            classf.close()

            model_path = self.model_path

            t0 = time.time()
            model = load_model(model_path)
            t1 = time.time()

            print('Loaded in : ', round(t1 - t0,2) , ' sec')
            test_path = self.img_path[index]
            print('Generating predictions on image:', test_path)
            self.orig.append(self.hangulFilePathImageRead(test_path))

            img = image.load_img(test_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)

            label = classes[np.argmax(preds, axis=1)[0]]
            proba = np.max(preds)
            label = "{}: {:.2f}%".format(label, proba * 100)

            print("lable : " + classes[np.argmax(preds, axis=1)[0]])
            print("probability : " , np.max(preds))

            self.undo_button.setEnabled(True)
            self.test_button.setEnabled(True)

            height, width, channel = self.orig[index].shape
            bytesPerLine = 3 * width
            qImg = QImage(self.orig[index].data, width, height, bytesPerLine, QImage.Format_RGB888)

            self.qimg_list.append(qImg)

    def model_path_select(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Options()
        # weights_file_name 이용
        self.model_path, _ = QFileDialog.getOpenFileName(self, "load h5 file", "", "h5 File (*.h5)", options=options)
        self.label1.setText(self.model_path)
        self.model_select_flag = True

    def showMessageBox(self,msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Error")
        msgBox.exec_()

    def Undo_Click(self):
        if self.img_cnt <1 :
            self.showMessageBox('Please add image file first')
        elif self.img_cnt == 1 :
            self.flag = False
            self.img_path.pop()
            self.img_cnt -= 1
            item = self.gridbox.takeAt(0)
            widget = item.widget()
            widget.deleteLater()
            tmp_lbl_img = QLabel()
            tmp_lbl_img.setPixmap(QPixmap('default-image.png').scaled(600, 600))
            self.gridbox.addWidget(tmp_lbl_img)
        else :
            self.img_path.pop()
            self.img_cnt -= 1
            tmp_lbl_img = []

            for i in range(0,self.img_cnt) :
                tmp_lbl_img.append(QLabel())

            for i in range(self.img_cnt,-1,-1):
                item = self.gridbox.takeAt(i)
                widget = item.widget()
                widget.deleteLater()

            for i in range(0,self.img_cnt):
               tmp_lbl_img[i].setPixmap(QPixmap(self.img_path[i]).scaled(self.scaleSize_list[self.img_cnt-1],self.scaleSize_list[self.img_cnt-1]))

            for i in range(0,self.img_cnt) :
                self.gridbox.addWidget(tmp_lbl_img[i],i/2,i%2)

    def hangulFilePathImageRead(self,filePath):
        stream = open(filePath.encode("utf-8"), "rb")
        bytes = bytearray(stream.read())
        numpyArray = np.asarray(bytes, dtype=np.uint8)

        return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)

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

        newmodelaction = QAction(QIcon('new.png'), 'New Model', self)
        newmodelaction.setShortcut('Ctrl+N')
        newmodelaction.setStatusTip('모델 생성')
        newmodelaction.triggered.connect(self.newWindowStart)

        self.statusBar()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(newmodelaction)
        fileMenu.addAction(exitaction)

    def newWindowStart(self):
        self.wg.resultField.setEnabled(False)
        window = ClassificationUI()
        window.exec_()
        self.wg.resultField.clear()
        self.wg.resultField.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MyApp()
    main.show()
    sys.exit(app.exec_())