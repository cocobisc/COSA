import json
import os.path as osp
import glob
from PyQt5.QtWidgets import *
from .. import PY2

class LabelFileError(Exception):
    pass

def json_to_coco(path):
    jsonlist = []
    namelist = []

    if len(glob.glob(path + '/*.json')) == 0:
        showPopup("There is no json file", "warning")
        return None, None
    for filename in glob.glob(path + '/*.json'):
        fileref = ""
        base64_img_data = ""
        file_attributes = {}
        if osp.basename(filename) == "annotations.json":
            continue

        try:
            with open(filename, 'rb' if PY2 else 'r') as f:
                data = json.load(f)

            # relative path from label file to relative path from cwd
            imagePath = osp.join(osp.abspath(osp.join(osp.dirname(filename), '..')), data['imagePath'])
            imageName = osp.basename(data['imagePath'])
            size = osp.getsize(imagePath)
            imageNameplusSize = imageName + str(size)
            shapes = (
                (
                    s['label'],
                    s['points'],
                )
                for s in data['shapes']
            )
            tempx = []
            tempy = []
            label = []
            for s in shapes:
                tempx.append([i for i in list(map(lambda x: int(x[0]), s[1]))])
                tempy.append([i for i in list(map(lambda x: int(x[1]), s[1]))])
                label.append(s[0])

            namelist.append(imageNameplusSize)
            value = (fileref, size, imageName, base64_img_data,
                     file_attributes, label, tempx, tempy)
            jsonlist.append(value)
        except Exception as e:
            raise LabelFileError(e)
    return namelist, jsonlist
def make_coco(path, namelist, jsonlist):
    data = {}
    for i in namelist:
        data[i] = {}
    for i in range(len(jsonlist)):
        data[namelist[i]]['fileref'] = jsonlist[i][0]
        data[namelist[i]]['size'] = jsonlist[i][1]
        data[namelist[i]]['filename'] = jsonlist[i][2]
        data[namelist[i]]['base64_img_data'] = jsonlist[i][3]
        data[namelist[i]]['file_attributes'] = jsonlist[i][4]
    for i in range(len(jsonlist)):
        data[namelist[i]]['regions'] = {}
        for j in range(len(jsonlist[i][6])):
            data[namelist[i]]['regions'][str(j)] = {'shape_attributes' : {'name' : 'polygon', 'all_points_x' : jsonlist[i][6][j], 'all_points_y' : jsonlist[i][7][j]}, 'region_attributes' : {'name' : jsonlist[i][5][j]}}
    try:
        with open(osp.join(osp.dirname(path),'annotations.json'), 'wb' if PY2 else 'w') as f:
            json.dump(data, f, sort_keys=True, ensure_ascii=False)
        string = "Successfully converted json to coco.\n (File path : " + osp.join(osp.dirname(path),'annotations.json') + ")"
        showPopup(string)
    except Exception as e:
        raise LabelFileError(e)

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

if __name__ == '__main__':
    namelist, jsonlist = json_to_coco(r'C:\Users\user\Desktop\images')
    make_coco(r'C:\Users\user\Desktop\images', namelist, jsonlist)