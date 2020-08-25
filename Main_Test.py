import os
import sys
import qdarkstyle
from threading import *
from keras import backend as K  # Tensor tensor 오류 수정 부분
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import Predict_Test
import Training_Test
import logging
import tensorflow as tf
import ctypes


os.environ['QT_API'] = 'pyqt'

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

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if (not self.signalsBlocked()):
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if (not XStream._stdout):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout

        return XStream._stdout

    @staticmethod
    def stderr():
        if (not XStream._stderr):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr

        return XStream._stderr


class LogMessageViewer(QTextBrowser):
    def __init__(self, parent=None):
        super(LogMessageViewer, self).__init__(parent)
        self.setReadOnly(True)

    @QtCore.pyqtSlot(str)
    def appendLogMessage(self, msg):
        horScrollBar = self.horizontalScrollBar()
        verScrollBar = self.verticalScrollBar()
        scrollIsAtEnd = verScrollBar.maximum() - verScrollBar.value() <= 10
        self.insertPlainText(msg)

        if scrollIsAtEnd:
            verScrollBar.setValue(verScrollBar.maximum())  # scroll to the bottom
            horScrollBar.setValue(0)  # scroll to the left

class TabBar(QTabBar):

    def tabSizeHint(self, index):
        s = QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()

        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)
            painter.save()

            s = opt.rect.size()
            s.transpose()
            r = QtCore.QRect(QtCore.QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r

            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(-90)
            painter.translate(-c)
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt);
            painter.restore()

class TabWidgetWithWidget(QTabWidget):

    def __init__(self, *args, **kwargs):
        QTabWidget.__init__(self, *args, **kwargs)

        self.setStyleSheet("QTabBar::tab {font: 23px; font-family: Bauhaus 93;}")
        self.setTabBar(TabBar(self))

        self.start_btn = QPushButton("Start", self)
        self.start_btn.setIcon(QIcon('UI_image/startbtn.png'))
        self.start_btn.setIconSize(QSize(35, 35))

        self.stop_btn = QPushButton("Stop", self)
        self.stop_btn.setIcon(QIcon('UI_image/stopbtn.png'))
        self.stop_btn.setIconSize(QSize(35, 35))
        self.stop_btn.setEnabled(False)

        self.combobox = QComboBox(self)
        self.combobox.SizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        view = QListView()  # creat a ListView
        view.setFixedWidth(120)  # set the ListView with fixed Width
        self.combobox.setView(view)  # provide the list view to Combobox object

        # self.combobox.setMaximumWidth(500)  # will be overwritten by style-sheet
        self.combobox.addItems(["Classification", "Object Detection", "Segmentation"])
        self.combobox.setStyleSheet(
            "QComboBox { "
            "background-color: #424242;"
            "font-size:13px;"
            "border: 1px #FF4000;"
            "max-width: 125px;"
            "min-height: 45px; "
            "}"
            "QComboBox QAbstractItemView::item {"
            "font-size: 13px;"
            "min-height: 40px;"
            "max-width: 130px;"
            "}"
            "QListView::item:selected {"
            "color: #9FF781;"
            "}"
            "QComboBox:hover {"
            "background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0d5ca6, stop: 1 #2198c0);"
            "}"
        )
        # self.label = QLabel("Category", self)
        # self.label.setStyleSheet('''
        #     QLabel { font-size:13px; }
        #     ''')

        self.setTabBar(TabBar(self))
        self.setTabPosition(QTabWidget.East)

        self.layout = QVBoxLayout()
        self.layout2 = QVBoxLayout()

        layout_gbox = QVBoxLayout()
        layout_gbox.addWidget(self.combobox)
        gbox2 = QGroupBox("Category")
        gbox2.setLayout(layout_gbox)

        layout_gbox2 = QVBoxLayout()
        self.model_name = QLabel("Not Selected")
        self.model_name.setAlignment(Qt.AlignCenter)
        layout_gbox2.addWidget(self.model_name)
        self.gbox3 = QGroupBox(self)
        self.gbox3.setTitle("Chosen Model")
        self.gbox3.setLayout(layout_gbox2)
        self.gbox3.setStyleSheet("QGroupBox {border: 1px solid #FF8000;}")

        self.tlabel = QLabel()

        self.layout.addWidget(gbox2)
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)
        self.layout.addWidget(self.gbox3)
        self.layout.addWidget(self.tlabel)
        #self.tlabel.setVisible(False)

        self.gbox = QGroupBox(self)
        self.gbox.setLayout(self.layout)

    def resizeEvent(self, event):
        self.start_btn.resize(85, 50)
        self.stop_btn.resize(85, 50)

        self.gbox.move(QDesktopWidget().screenGeometry().width() - 215, 100)
        QTabWidget.resizeEvent(self, event)

class ProxyStyle(QProxyStyle):
    def drawControl(self, element, opt, painter, widget):
        if element == QStyle.CE_TabBarTabLabel:
            ic = self.pixelMetric(QStyle.PM_TabBarIconSize)
            r = QtCore.QRect(opt.rect)
            w =  0 if opt.icon.isNull() else opt.rect.width() + self.pixelMetric(QStyle.PM_TabBarIconSize)
            r.setHeight(opt.fontMetrics.width(opt.text) + w)
            r.moveBottom(opt.rect.bottom())
            opt.rect = r
        QProxyStyle.drawControl(self, element, opt, painter, widget)

class MySignal(QObject):
    signal = pyqtSignal()

    def run(self):
        self.signal.emit()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__(parent=None)
        self.title = os.getcwd() + ' - COSA'
        self.setWindowTitle(self.title)
        self.initUI()

    def initUI(self):

        tabs_width = QDesktopWidget().screenGeometry().width() * 0.987
        tabs_height = (QDesktopWidget().screenGeometry().height()) * 0.66

        resultfield_width = QDesktopWidget().screenGeometry().width() * 0.987
        resultfield_height = QDesktopWidget().screenGeometry().height() * 0.24

        self.resultFieldClear_signal = MySignal()

        self.main_vbox = QVBoxLayout()

        # self.tabs.setStyle(ProxyStyle())


        self.predict_tab = Predict_Test.PredictWindow(tabs_height)
        self.training_tab = Training_Test.TrainingWindow(tabs_height)
        self.tabs = TabWidgetWithWidget()
        self.tabs.setStyle(ProxyStyle())
        self.tabs.currentChanged.connect(self.onChange)
        self.tabs.addTab(self.predict_tab, "Predict")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.combobox.currentTextChanged.connect(self.categorychanged)
        self.tabs.setFixedSize(tabs_width, tabs_height)

        self.tabs.setTabIcon(0, QIcon('UI_image/predict.png'))
        self.tabs.setTabIcon(1, QIcon('UI_image/training.png'))
        self.tabs.setIconSize(QSize(120, 40))

        self.predict_tab.change_button_signal.signal.connect(self.changebtn)
        self.training_tab.classificationUI.change_button_signal.signal.connect(self.changebtn)
        self.training_tab.segmentationUI.change_button_signal.signal.connect(self.changebtn)
        self.training_tab.objectDetectionUI.change_button_signal.signal.connect(self.changebtn)

        self.predict_tab.model_select_signal.signal.connect(self.change_model_name)
        self.tabs.combobox.currentTextChanged.connect(self.combo_change)



        # mainwindow = QMainWindow()
        # toolbar = QToolBar()
        # mainwindow.addToolBar(toolbar)

        # 나중에 수정할 부분(predict/train start가 되도록 하고 stop버튼 추가하기)
        # self.predict_button = QAction(QIcon('UI_image/predict_image.png'), 'Predict', self)
        # self.predict_button.setShortcut('Ctrl+P')
        # self.predict_button.setStatusTip('Predict application')
        # self.predict_button.triggered.connect(self.predict_tab.linktest)
        self.tabs.stop_btn.clicked.connect(self.stop)
        self.tabs.start_btn.clicked.connect(self.start)
        self.predict_tab.resultFieldClear_signal.signal.connect(self.resultFieldClear)

        # toolbar.addSeparator()
        # toolbar.addWidget(self.combobox)
        # toolbar.addAction(self.predict_button)
        # toolbar.setOrientation(Qt.Horizontal)

        # console이 들어갈 부분
        self.resultField = LogMessageViewer(self)
        self.resultField.setStyleSheet('''
            background-color: #151515;
            color: #FFFFFF;
            ''')
        self.resultField.setFixedSize(resultfield_width, resultfield_height)
        self.resultField.setEnabled(False)

        test_label = QLabel("Console")

        font = test_label.font()
        font.setBold(True)
        test_label.setFont(font)
        mainwindow2 = QMainWindow()
        toolbar2 = QToolBar()
        mainwindow2.addToolBar(toolbar2)
        toolbar2.addWidget(test_label)
        toolbar2.addWidget(self.resultField)
        toolbar2.setOrientation(Qt.Vertical)
        toolbar2.setMovable(False)

        XStream.stdout().messageWritten.connect(self.resultField.appendLogMessage)
        XStream.stderr().messageWritten.connect(self.resultField.appendLogMessage)

        # 마지막에 추가하는 부분
        # self.main_vbox.addWidget(mainwindow)
        self.setLayout(self.main_vbox)
        self.main_vbox.addWidget(self.tabs)
        self.main_vbox.addWidget(mainwindow2)
        self.showMaximized()

    def combo_change(self, value):
        if value == 'Classification':
            self.predict_tab.choose_color.setVisible(True)
        else:
            self.predict_tab.choose_color.setVisible(False)

    def categorychanged(self):
        self.predict_tab.category_changed(self.tabs.combobox.currentText())
        self.training_tab.category_changed(self.tabs.combobox.currentText())
        self.resultFieldClear()

    def changebtn(self):
        if self.tabs.start_btn.isEnabled():
            self.tabs.combobox.setEnabled(False)
            self.tabs.start_btn.setEnabled(False)
            self.tabs.stop_btn.setEnabled(True)

        else:
            self.tabs.combobox.setEnabled(True)
            self.tabs.start_btn.setEnabled(True)
            self.tabs.stop_btn.setEnabled(False)

    def start(self):
        self.resultField.moveCursor(QTextCursor.End)
        self.resultField.clear()
        if self.tabs.currentIndex() == 1:
            K.clear_session()  # Tensor tensor 오류 수정 부분
            if self.tabs.combobox.currentText() == "Classification":
                if self.training_tab.classificationUI.trainingDataSet_label.text() == "Training Data Set : ":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Designate Training Data set Directory")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                elif self.training_tab.classificationUI.validationDataSet_label.text() == "Validation Data Set : ":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Designate Validation Data set Directory")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    self.changebtn()
                    self.thread = Thread(target=self.training_tab.classificationUI.training)
                    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                    self.thread.start()
                    self.training_tab.classificationUI.print_state()
                    self.training_tab.resultFieldClear_signal.signal.connect(self.resultFieldClear)

            elif self.tabs.combobox.currentText() == "Object Detection":
                if self.training_tab.objectDetectionUI.dataset_label.text() == "Training Data Set : ":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Designate Data set Directory")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                elif self.training_tab.objectDetectionUI.model_path_label.text() == "Model Save Path : ":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Designate Saving Model Name and Path")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    self.changebtn()
                    self.thread = Thread(target=self.training_tab.objectDetectionUI.training)
                    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                    self.thread.start()
                    self.training_tab.objectDetectionUI.print_state()
                    self.training_tab.resultFieldClear_signal.signal.connect(self.resultFieldClear)

            elif self.tabs.combobox.currentText() == "Segmentation":
                if self.training_tab.segmentationUI.rootDirectory_label.text() == "Root Directory : ":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setText("Designate Data set Directory")
                    msg.setWindowTitle("warning")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                else:
                    self.changebtn()
                    self.thread = Thread(target=self.training_tab.segmentationUI.training)
                    QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
                    self.thread.start()
                    self.training_tab.classificationUI.print_state()
                    self.training_tab.resultFieldClear_signal.signal.connect(self.resultFieldClear)

        elif self.tabs.currentIndex() == 0:
            self.predict_tab.linktest()

    def stop(self):
        if(self.tabs.currentIndex() == 1):
            self.terminate_thread(self.thread)
            K.clear_session()
            self.changebtn()
        else:
            self.predict_tab.terminate()

    def terminate_thread(self, thread):
        if not thread.isAlive():
            return
        mb = QMessageBox
        msg = "Training process has been working. Are you sure to STOP?"
        answer = mb.warning(self, 'Attention', msg, mb.Yes | mb.No)

        if answer == mb.Yes:
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

            print("\nStopped")
            return

    def change_model_name(self):
        self.tabs.model_name.setText(os.path.basename(self.predict_tab.model_path))

    @pyqtSlot()
    def resultFieldClear(self):
        self.resultField.clear()

    @pyqtSlot()
    def onChange(self):
        if(self.tabs.currentIndex() == 0):
            self.tabs.gbox3.setVisible(True)
            self.tabs.tlabel.setVisible(False)
        else:
            self.tabs.gbox3.setVisible(False)
            self.tabs.tlabel.setVisible(True)


if __name__ == '__main__':
    import time

    app = QApplication(sys.argv)
    # app.setStyle(ProxyStyle())
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setWindowIcon(QIcon('UI_image/COSA.png'))
    splash_pix = QPixmap(r'UI_image/main_logo.jpg')
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)

    splash.setMask(splash_pix.mask())
    #splash.show()

    #time.sleep(3)
    main = MainWindow()
    # splash.finish(main)
    # main.show()
    sys.exit(app.exec_())