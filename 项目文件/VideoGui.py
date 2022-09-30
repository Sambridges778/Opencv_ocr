from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import sys
from PyQt5.QtWidgets import QWidget,QFormLayout,QApplication,QVBoxLayout,QPushButton,QFileDialog,QLabel,QLineEdit
import cv2
import imutils
from PyQt5.QtGui import QPixmap,QImage
class VGUI(QWidget):
    def __init__(self):
        super().__init__()
# GUI
    def ShowVideoUi(self):
        self.setWindowTitle("视频文本检测")
        qf = QFormLayout(self)
        ql1 = QLabel("路径:", self)
        self.qf1 = QLineEdit(self)
        ql2 = QLabel("识别结果", self)
        self.qf2 = QLineEdit(self)
        pb = QPushButton("打开视频", self)
        pb.clicked.connect(self.VideoShow)
        pb1 = QPushButton("打开摄像头", self)
        pb1.clicked.connect(self.CameraShow)


        self.layout = QVBoxLayout()
        self.layout.addWidget(pb1)
        qf.addRow(pb1)
        self.layout.addWidget(pb)
        qf.addRow(pb)
        self.layout.addWidget(self.qf1)
        qf.addRow(ql1, self.qf1)
        self.layout.addWidget(self.qf2)
        qf.addRow(ql2, self.qf2)


        self.setLayout(self.layout)
        self.q1 = QLabel(self)
        self.layout.addWidget(self.q1)
        qf.addRow("视频：",self.q1)
        self.resize(800, 1000)
        self.show()


# 打开视频文件
    def VideoShow(self):
        def decode_predictions(scores, geometry):
            # 从分数卷中获取行和列的数量，然后
            # 初始化我们的一组边界框矩形和相应的置信度得分

            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []

            # 循环行数
            for y in range(0, numRows):
                # 提取分数（概率），然后是
                # 用于得出潜在边界框的几何数据
                # 协调围绕文本
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # 循环列数
                for x in range(0, numCols):
                    # 如果我们的分数没有足够大的概率，
                    # 无视它
                    if scoresData[x] < minConfidence:
                        continue

                    # compute the offset factor as our resulting feature
                    # maps will be 4x smaller than the input image
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)

                    # extract the rotation angle for the prediction and
                    # then compute the sin and cosine
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    # use the geometry volume to derive the width and height
                    # of the bounding box
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]

                    # compute both the starting and ending (x, y)-coordinates
                    # for the text prediction bounding box
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    # add the bounding box coordinates and probability score
                    # to our respective lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

            # return a tuple of the bounding boxes and associated confidences
            return (rects, confidences)

        east = 'frozen_east_text_detection.pb'
        minConfidence = 0.5
        width = 320
        height = 320

        # initialize the original frame dimensions, new frame dimensions,
        # and ratio between the dimensions
        (W, H) = (None, None)
        (newW, newH) = (width, height)
        (rW, rH) = (None, None)

        # 定义EAST检测器模型的两个输出层名称
        # 我们很感兴趣 - 首先是输出概率和
        # 第二可以用于得出文本的边界框坐标
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # 开始从文件夹里读取文件
        net = cv2.dnn.readNet(east)
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        self.qf1.setText(self.fname[0])
        vs = cv2.VideoCapture(self.fname[0])

        # start the FPS throughput estimator
        fps = FPS().start()

        # loop over frames from the video stream
        while True:
            # grab the current frame, then handle if we are using a
            # VideoStream or VideoCapture object
            frame = vs.read()
            frame = frame[1] if vs else frame

            # check to see if we have reached the end of the stream
            if frame is None:
                break

            # resize the frame, maintaining the aspect ratio
            frame = imutils.resize(frame, width=1000)
            orig = frame.copy()

            # if our frame dimensions are None, we still need to compute the
            # ratio of old frame dimensions to new frame dimensions
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                rW = W / float(newW)
                rH = H / float(newH)

            # resize the frame, this time ignoring aspect ratio
            frame = cv2.resize(frame, (newW, newH))

            # construct a blob from the frame and then perform a forward pass
            # of the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                         (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)

            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)

                # draw the bounding box on the frame
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # update the FPS counter
            fps.update()

            # 显示输出帧
            b, g, r = cv2.split(orig)
            orig = cv2.merge([r, g, b]) # 让视频保持原有的色彩
            showImage = QImage(orig.data, orig.shape[1], orig.shape[0], orig.shape[1] * 3, QImage.Format_RGB888)
            self.q1.setPixmap(QPixmap.fromImage(showImage))
            #key = cv2.waitKey(1) & 0xFF # 64位机器
            if cv2.waitKey(1) & 0xff == ord("q"):
                break


            # 如果按下“ f”键，则结束循环
            #if key == ord("f"):
            #    break

        # 停止计时器并显示FPS信息
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # 如果我们使用网络摄像头，请发布指针
        if not vs:
            vs.stop()

        # 否则，发布文件指针
        else:
            vs.release()

        # 关闭所有窗口
        cv2.destroyAllWindows()

# 打开摄像头
    def CameraShow(self):
        def decode_predictions(scores, geometry):
            # grab the number of rows and columns from the scores volume, then
            # initialize our set of bounding box rectangles and corresponding
            # confidence scores
            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []

            # loop over the number of rows
            for y in range(0, numRows):
                # extract the scores (probabilities), followed by the
                # geometrical data used to derive potential bounding box
                # coordinates that surround text
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # loop over the number of columns
                for x in range(0, numCols):
                    # if our score does not have sufficient probability,
                    # ignore it
                    if scoresData[x] < minConfidence:
                        continue

                    # compute the offset factor as our resulting feature
                    # maps will be 4x smaller than the input image
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)

                    # extract the rotation angle for the prediction and
                    # then compute the sin and cosine
                    angle = anglesData[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)

                    # use the geometry volume to derive the width and height
                    # of the bounding box
                    h = xData0[x] + xData2[x]
                    w = xData1[x] + xData3[x]

                    # compute both the starting and ending (x, y)-coordinates
                    # for the text prediction bounding box
                    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    # add the bounding box coordinates and probability score
                    # to our respective lists
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])

            # return a tuple of the bounding boxes and associated confidences
            return (rects, confidences)

        east = 'frozen_east_text_detection.pb'
        minConfidence = 0.5
        width = 320
        height = 320

        # initialize the original frame dimensions, new frame dimensions,
        # and ratio between the dimensions
        (W, H) = (None, None)
        (newW, newH) = (width, height)
        (rW, rH) = (None, None)

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
#从摄像头开始抓取视频
        net = cv2.dnn.readNet(east)
        vs = cv2.VideoCapture(0)

        # start the FPS throughput estimator
        fps = FPS().start()

        # 从视频流中循环
        while True:
            # 抓住当前的框架，然后处理我们正在使用的
            # 视频流或视频关注对象
            frame = vs.read()
            frame = frame[1] if vs else frame

            # check to see if we have reached the end of the stream
            if frame is None:
                break

            # resize the frame, maintaining the aspect ratio
            frame = imutils.resize(frame, width=1000)
            orig = frame.copy()

            # if our frame dimensions are None, we still need to compute the
            # ratio of old frame dimensions to new frame dimensions
            if W is None or H is None:
                (H, W) = frame.shape[:2]
                rW = W / float(newW)
                rH = H / float(newH)

            # resize the frame, this time ignoring aspect ratio
            frame = cv2.resize(frame, (newW, newH))

            # construct a blob from the frame and then perform a forward pass
            # of the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
                                         (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)

            # loop over the bounding boxes
            for (startX, startY, endX, endY) in boxes:
                # scale the bounding box coordinates based on the respective
                # ratios
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)

                # draw the bounding box on the frame
                cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # update the FPS counter
            fps.update()

            # show the output frame
            cv2.imshow("Text Detection", orig)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # if we are using a webcam, release the pointer
        if not vs:
            vs.stop()

        # otherwise, release the file pointer
        else:
            vs.release()

        # close all windows
        cv2.destroyAllWindows()

if __name__ == '__main__':
    a = QApplication(sys.argv)
    m = VGUI()
    m.ShowVideoUi()
    sys.exit(a.exec_())

