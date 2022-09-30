import time
from imutils.object_detection import non_max_suppression
import numpy as np
import sys
from PyQt5.QtWidgets import QWidget,QFormLayout,QApplication,QVBoxLayout,QPushButton,QFileDialog,QLabel,QLineEdit
from PyQt5.QtGui import QPixmap,QImage
from PIL import Image
import pytesseract
import cv2
import os

class IGUI(QWidget):
    def __init__(self):
        super().__init__()
# GUI
    def ShowImageUi(self):
        self.setWindowTitle("图片文本检测")
# 垂直布局
        qf = QFormLayout(self)
        ql1 = QLabel("路径:", self)
        self.qf1 = QLineEdit(self)
        ql2 = QLabel("识别结果",self)
        self.qf2 = QLineEdit(self)
        pb = QPushButton("图片检测", self)
        pb.clicked.connect(self.ImageShow)
        pb1 = QPushButton("识别文字",self)
        pb1.clicked.connect(self.TextOutput)
        pb0 = QPushButton("打开文件", self)
        pb0.clicked.connect(self.OpenFilePic)


# 添加按钮，充满一行
        self.layout = QVBoxLayout()
        self.layout.addWidget(pb0)
        self.layout.addWidget(pb)
        self.layout.addWidget(pb1)
        qf.addRow(pb0)
        qf.addRow(pb)
        qf.addRow(pb1)
        self.layout.addWidget(self.qf1)
        qf.addRow(ql1, self.qf1)
        self.layout.addWidget(self.qf2)
        qf.addRow(ql2, self.qf2)

# 显示要检测的图片
        self.setLayout(self.layout)
        self.q1 = QLabel(self)
        self.layout.addWidget(self.q1)
        qf.addRow("图片：",self.q1)
        self.resize(800, 1000)
        self.show()

# 打开文件中的图片，并显示出来
    def OpenFilePic(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        self.qf1.setText(self.fname[0]) # 输出文件的路径
        orig = cv2.imread(self.fname[0])
        showImage = QImage(orig.data, orig.shape[1], orig.shape[0], orig.shape[1] * 3, QImage.Format_RGB888)
        self.q1.setPixmap(QPixmap.fromImage(showImage))

# 检测图片文本,先打开文件夹中的图片，选中后程序执行文本检测。框选出来的内容
# 供之后的文本识别做准备。
    def ImageShow(self):
        # 读取文件夹中的图片
        img1 = (self.fname[0])
        # 构建模型
        east = 'frozen_east_text_detection.pb'# 检测要调用的包
        minConfidence = 0.5
        width = 320
        height = 320

# 其作用是复制一份原始的图像,读取宽和高
        image = cv2.imread(img1)
        orig = image.copy()
        (H, W) = image.shape[:2]

        (newW, newH) = (width, height)
        rW = W / float(newW)
        rH = H / float(newH)

        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text

        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

# 加载训练好的EAST检测模型
        print("[提示] 正在加载 EAST 文字识别模块...")
        net = cv2.dnn.readNet(east)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

# 展示模型识别的具体花费时长
        print("[提示] 文字识别共用了 {:.6f} 秒".format(end - start))

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < minConfidence:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        # loop over the bounding boxes

        self.list = []  # 创建空列表
        for (startX, startY, endX, endY) in boxes:
            # 图像坐标
            startX = int(startX * rW-4)
            startY = int(startY * rH-4)
            endX = int(endX * rW+4)
            endY = int(endY * rH+4)
            self.list.append([startX, startY, endX, endY]) # 将图像的坐标点输出到列表中，使其循环
            # 给识别的图像加上边界框，设置颜色和粗细
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
# 展示输出的图片
        b, g, r = cv2.split(orig) # 让图像保持原有的色彩
        orig = cv2.merge([r, g, b])
        showImage = QImage(orig.data, orig.shape[1], orig.shape[0], orig.shape[1] * 3, QImage.Format_RGB888)
        self.q1.setPixmap(QPixmap.fromImage(showImage))
        cv2.waitKey(0) & 0xFF #64位机器

# 输出检测文本中识别出来的文字
    def TextOutput(self):
        # 读取展示的图片文本，输出检测过后的文本
        image = cv2.imread(self.fname[0])
        NewList = []  # 创建新列表

        for i in range(len(self.list)):
            i += 1
            a = self.list[i - 1]  # 从第一位开始读

            img = image[int(a[1]):int(a[3]), int(a[0]):int(a[2])]
            pytesseract.pytesseract.tesseract_cmd = 'tesseract.exe'

            TextOP = pytesseract.image_to_string(img, lang='chi_sim')
            NewList.append(TextOP)

        tx = [i.strip() for i in NewList if i.strip()]
        text = ','.join(str(i) for i in tx)

        print(text)
        self.qf2.setText(text)


if __name__ == '__main__':
    a = QApplication(sys.argv)
    m = IGUI()
    m.ShowImageUi()
    sys.exit(a.exec_())


