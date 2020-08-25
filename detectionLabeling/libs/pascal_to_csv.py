import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import *

def xml_to_csv(path):
    classlist = []
    classvalue = []
    if len(glob.glob(path + '/*.xml')) == 0:
        showPopup("There is no xml file", "warning")
        return
    if len(glob.glob(path + '/*.xml')) < 5:
        showPopup("The number of XML files is lower then 5", "warning")
        return

    def convert_to_annotations(start, end):
        xml_list = []
        idxcnt = 0
        filepath = None
        for xml_file in glob.glob(path + '/*.xml')[start:end]:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            if len(root.findall('object')) == 0:
                showPopup("xml with no object", "warning")
            if not filepath:
                    filepath = os.path.dirname(root.find('path').text)
            for member in root.findall('object'):
                bbx = member.find('bndbox')
                xmin = int(bbx.find('xmin').text)
                ymin = int(bbx.find('ymin').text)
                xmax = int(bbx.find('xmax').text)
                ymax = int(bbx.find('ymax').text)
                label = member.find('name').text
                if label not in classlist:
                    classlist.append(label)
                    classvalue.append((label, idxcnt))
                    idxcnt += 1
                value = (root.find('filename').text,
                         xmin,
                         ymin,
                         xmax,
                         ymax,
                         label,
                         )
                xml_list.append(value)
        column_name = ['filename','xmin', 'ymin', 'xmax', 'ymax', 'class']
        return pd.DataFrame(xml_list, columns=column_name), filepath

    end = int(len(glob.glob(path + '/*.xml')) * 4 / 5)
    annotations, filepath = convert_to_annotations(0, end)
    classes = pd.DataFrame(classvalue, columns=['class', 'index'])
    annotations.to_csv(os.path.join(filepath, 'annotations.csv'), header = False, index=False)
    classes.to_csv(os.path.join(filepath, 'classes.csv'), header = False, index=False)

    annotations, filepath = convert_to_annotations(end, len(glob.glob(path + '/*.xml')))
    classes = pd.DataFrame(classvalue, columns=['class', 'index'])
    annotations.to_csv(os.path.join(filepath, 'annotations_val.csv'), header=False, index=False)
    classes.to_csv(os.path.join(filepath, 'classes_val.csv'), header=False, index=False)

    str = "Successfully converted xml to csv.\n (File path : " + filepath + ") \n + annotations.csv, classes.csv + _val"
    showPopup(str)

    return path


def showPopup(str, type = None):
    if type == "warning":
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(str)
        msg.setWindowTitle(type)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return
    msg = QMessageBox()
    msg.setWindowTitle("Success")
    msg.setText(str)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()