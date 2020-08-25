import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication

class Setlabel(QDialog) :

    classlist = []
    # checkState = False
    prevLabelText = None

    def __init__(self, classlist):
        super(Setlabel, self).__init__()

        # self.checkState = checkState
        self.setWindowTitle("Edit Class")
        self.setFixedWidth(250)
        self.setFixedHeight(220)
        self.setWindowIcon(QIcon(r"..\\resources\\icons\\app"))

        self.viewer = QListWidget(self)
        self.viewer.resize(140,220)
        self.setClassList(classlist)
        if Setlabel.prevLabelText:
            self.viewer.scrollToItem(self.viewer.item(self.classlist.index(Setlabel.prevLabelText)))
            self.viewer.setCurrentItem(self.viewer.item(self.classlist.index(Setlabel.prevLabelText)))

        self.add_button = QPushButton("+", self)
        self.add_button.setFont(QFont("Arial", 11))
        self.add_button.setGeometry(165, 25, 60, 40)
        self.add_button.clicked.connect(self.moveClassSet)

        self.del_button = QPushButton("-", self)
        self.del_button.setFont(QFont("Arial", 11))
        self.del_button.setGeometry(165, 75, 60, 40)
        self.del_button.clicked.connect(self.moveClassSet)

        self.choose_button = QPushButton("Choose", self)
        self.choose_button.setFont(QFont("Arial", 11))
        self.choose_button.setGeometry(150, 145, 90, 40)
        self.choose_button.clicked.connect(self.chooseClass)

        # self.checkBox = QCheckBox("Use Fixed Label", self)
        # self.checkBox.move(160, 185)
        # self.checkBox.resize(150, 30)
        # self.checkBox.stateChanged.connect(self.checkBoxState)
        # self.checkBox.setChecked(Setlabel.checkState)

    def chooseClass(self):
        if(self.viewer.currentItem()):
            Setlabel.prevLabelText = self.viewer.currentItem().text()
        self.close()

    # @staticmethod
    # def getClass():
    #     return Setlabel.prevLabelText

    def checkBoxState(self):
        if self.checkBox.isChecked() == True:
            Setlabel.checkState = True
        else:
            Setlabel.checkState = False

    def setClassList(self, classlist):
        self.classlist = classlist
        for i in range(len(self.classlist)):
            item = QListWidgetItem()
            if '\n' in self.classlist[i]:
                self.classlist[i] = self.classlist[i][:-1]
            item.setText(self.classlist[i])
            self.viewer.addItem(item)

    def moveClassSet(self):
        if self.sender().text() == "+":
            dialog = SettingDialog(self)
            dialog.exec_()
        else :
            if len(self.classlist) == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("no element for delete")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            else:
                if self.viewer.currentRow() != -1 :
                    del (self.classlist[self.viewer.currentRow()])
                self.viewer.takeItem(self.viewer.currentRow())
                self.refreshText()

    def refreshText(self) :
        f = open(r".\data\predefined_classes.txt", "w")
        for i in range(len(self.classlist)):
            if i == len(self.classlist)-1:
                f.write(self.classlist[-1])
            else:
                if '\n' in self.classlist[i]:
                    f.write(self.classlist[i])
                else:
                    f.write(self.classlist[i] + '\n')
        f.close()

class SettingDialog(QDialog):
    def __init__(self, dialog):
        super().__init__()
        self.row = 2
        self.dialog = dialog
        self.setWindowTitle("Add Class")
        self.setGeometry(810, 450, 200, 150)

        base_layout = QBoxLayout(QBoxLayout.TopToBottom, self)

        groupBox = QGroupBox()
        groupBox_layout = QBoxLayout(QBoxLayout.TopToBottom)
        groupBox.setLayout(groupBox_layout)

        self.textField = QLineEdit(self)
        self.textField.setReadOnly(False)
        self.textField.setFixedSize(200, 27)

        self.add_button = QPushButton("Add", self)
        self.add_button.clicked.connect(self.buttonClicked)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.buttonClicked)

        groupBox_layout.addWidget(self.textField)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.add_button)
        hlayout.addWidget(self.cancel_button)

        groupBox_layout.addLayout(hlayout)

        base_layout.addWidget(groupBox)

    def setClassList(self, classlist):
        self.classlist = classlist

    def buttonClicked(self):
        if self.sender().text() == "Add":
            self.string = self.textField.text()
            if self.string not in self.dialog.classlist:
                self.dialog.classlist.append(self.string)
                self.dialog.viewer.addItem(self.string)
                self.dialog.refreshText()

                self.close()
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Already exist")
                msg.setWindowTitle("warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

        elif self.sender().text() == "Cancel":
            self.close()

    def addLayers(self, string):
        self.listWidget.addItem(string)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    classlist = []
    dialog = Setlabel()

    f = open(r"..\data\predefined_classes.txt", "r")
    for line in f.readlines():
        classlist.append(line)
    dialog.setClassList(classlist)
    dialog.show()
    sys.exit(app.exec_())
