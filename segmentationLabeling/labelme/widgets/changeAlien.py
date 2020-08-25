import io
import json
import os.path as osp
import PIL.Image
from .._version import __version__
from ..logger import logger
from .. import PY2
from .. import QT4
from .. import utils
import base64


class LabelFileError(Exception):
    pass


def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        logger.error('Failed opening image file: {}'.format(filename))
        return

    # apply orientation to image according to exif
    image_pil = utils.apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(filename)[1].lower()
        if PY2 and QT4:
            format = 'PNG'
        elif ext in ['.jpg', '.jpeg']:
            format = 'JPEG'
        else:
            format = 'PNG'
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()


def changePath(path, mypath):
    with open(path, 'rb' if PY2 else 'r') as f:
        data = json.load(f)
    imagePath = mypath
    imageData = load_image_file(imagePath)
    flags = data.get('flags') or {}
    imageHeight = data.get('imageHeight'),
    imageWidth = data.get('imageWidth'),
    lineColor = data['lineColor']
    fillColor = data['fillColor']
    shapes = []
    for s in data['shapes']:
        shapes.append({
            'label' :  s['label'],
            'line_color'  : s['line_color'],
            'fill_color' : s['fill_color'],
            'points': s['points'],
            'shape_type' : s.get('shape_type', 'polygon'),
            'flags' : s.get('flags', {})
        })

    save(path, imagePath, imageHeight, imageWidth, shapes, imageData, lineColor, fillColor, flags)

def save(
        path,
        imagePath,
        imageHeight,
        imageWidth,
        shapes=None,
        imageData=None,
        lineColor=None,
        fillColor=None,
        flags=None,
):
    imageData = base64.b64encode(imageData).decode('utf-8')
    data = dict(
        version=__version__,
        flags=flags,
        shapes=shapes,
        lineColor=lineColor,
        fillColor=fillColor,
        imagePath=imagePath,
        imageData=imageData,
        imageHeight=imageHeight[0],
        imageWidth=imageWidth[0],
    )
    try:
        with open(path, 'w') as f:
            json.dump(data, f, sort_keys=True, ensure_ascii=False, indent=2)
    except Exception as e:
        raise LabelFileError(e)
