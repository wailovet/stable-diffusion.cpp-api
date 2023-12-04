import tempfile
from krita import Extension, Krita, DockWidgetFactory, DockWidgetFactoryBase, Window, DockWidget
import krita

from PyQt5.QtCore import QByteArray, QUrl, QFile, QIODevice, QObject
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
import json
import os
import sys
import time

from .network import RequestManager

MENU_NAME = 'easydiffusion'
EXTENSION_ID = 'pykrita_easydiffusion'
DOCKER_ID = 'pykrita_easydiffusion'
MENU_ENTRY = 'easydiffusion'

_requests = RequestManager()

from PyQt5.QtCore import QTimer


class Diffusion(Extension):
    def __init__(self, parent):
        super().__init__(parent)

    def setup(self):
        pass

    def createActions(self, window):
        action = window.createAction(EXTENSION_ID, MENU_NAME, MENU_ENTRY)
        action.triggered.connect(self.action_triggered)

    def action_triggered(self):
        # code here.
        pass


from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QApplication, QPushButton, QVBoxLayout)
from PyQt5.QtGui import QPixmap, QImage

port = 21777
tmp_dir = tempfile.gettempdir()
# print("tmp_dir:", tmp_dir)
tmp_file1 = os.path.join(tmp_dir, "krita_diffusion_tmp1.jpg")
tmp_file2 = os.path.join(tmp_dir, "krita_diffusion_tmp2.png")


class ImageDiffusionWidget(DockWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(DOCKER_ID)

        self.timer = QTimer()
        self.timer.timeout.connect(self.task)  # 这个通过调用槽函数来刷新时间
        self.timer.start(500)

        # 新增图片显示区域
        self.label = QLabel(self)
        self.setWidget(self.label)
        self.label.move(0, 120)

        self.generate_button = QPushButton("Generate", self)
        self.generate_button.move(0, 20)
        self.generate_button.clicked.connect(self.generate)

    def task(self):
        print("task run")
        if os.path.exists(tmp_file2):
            img = QImage(tmp_file2)
            img = img.scaled(64 * 4, 64 * 4)
            self.label.setPixmap(QPixmap.fromImage(img))

    def generate(self):
        # code here.
        print("generate")

        doc = Krita.instance().activeDocument()
        pixdata = doc.pixelData(0, 0, doc.width(), doc.height())
        img = QImage(pixdata, doc.width(), doc.height(), QImage.Format_RGB32)
        img = img.scaled(64 * 4, 64 * 4)
        self.label.setPixmap(QPixmap.fromImage(img))

        img.save(tmp_file1)
        try:
            os.remove(tmp_file2)
        except:
            pass

        self.dogenerate()

    def dogenerate(self):
        url = "http://localhost:{}/sdapi/v1/genimg".format(port)
        post_data = {"cfg_scale": "1", "width": "256", "height": "256", "sample_method": "LCM", "sample_steps": "4", "strength": "0.95",
                     "seed": "-1", "output": "", "prompt": "<lora:lcm-lora-sdv1-5:1>1girl", "negative_prompt": "text"}
        post_data["input_path"] = tmp_file1
        post_data["output"] = tmp_file2
        _requests.post(url, post_data)

    def canvasChanged(self, canvas):
        pass


app = Krita.instance()
extension = Diffusion(parent=app)
app.addExtension(extension)
app.addDockWidgetFactory(
    DockWidgetFactory(DOCKER_ID, DockWidgetFactoryBase.DockRight,
                      ImageDiffusionWidget)  # type: ignore
)
