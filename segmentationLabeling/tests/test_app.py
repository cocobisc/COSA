import os.path as osp
import shutil
import tempfile

from ..labelme import app
from ..labelme.config import get_default_config
from ..labelme import testing


here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(here, 'data')


def test_MainWindow_open(qtbot):
    win = app.MainWindow()
    qtbot.addWidget(win)
    win.show()
    win.close()


def test_MainWindow_open_json(qtbot):
    filename = osp.join(data_dir, 'apc2016_obj3.json')
    testing.assert_labelfile_sanity(filename)
    win = app.MainWindow(filename=filename)
    qtbot.addWidget(win)
    win.show()
    win.close()


def test_MainWindow_annotate_jpg(qtbot):
    tmp_dir = tempfile.mkdtemp()
    filename = osp.join(tmp_dir, 'apc2016_obj3.jpg')
    shutil.copy(osp.join(data_dir, 'apc2016_obj3.jpg'),
                filename)
    output_file = osp.join(tmp_dir, 'apc2016_obj3.json')

    config = get_default_config()
    win = app.MainWindow(
        config=config,
        filename=filename,
        output_file=output_file,
    )
    qtbot.addWidget(win)
    win.show()

    def check_imageData():
        assert hasattr(win, 'imageData')
        assert win.imageData is not None

    qtbot.waitUntil(check_imageData)  # wait for loadFile

    label = 'shelf'
    points = [
        (26, 70),
        (176, 730),
        (986, 742),
        (1184, 102),
    ]
    shape = label, points, None, None, 'polygon', {}
    shapes = [shape]
    win.loadLabels(shapes)
    win.saveFile()

    testing.assert_labelfile_sanity(output_file)
