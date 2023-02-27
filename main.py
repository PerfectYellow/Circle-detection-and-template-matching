import sys
from PyQt5 import QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel, QPushButton, QComboBox, QLineEdit
from PyQt5.QtGui import QPixmap, QIcon
import cv2
import numpy as np
from HCT import HoughCircleTransform        
from TM import TemplateMatching                      


class Window(QMainWindow, QWidget):
    circleTransform = HoughCircleTransform()
    templateMatching = TemplateMatching()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pattern Recognition")
        self.setGeometry(100, 60, 750, 320)
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.central_widget = QWidget()
        self.validator = QIntValidator()
        self.methods = ["HCT", "TM"]
        self.method = "HCT"
        self.video_flag = False
        self.image_path = None
        self.template_path = None
        self.tm_threshold = 0.8
        self.tm_method = "TM_CCOEFF_NORMED"
        self.tm_mode = "single"

        self.HCT_minDist = 8
        self.HCT_dp = 1.1
        self.HCT_Param1 = 90
        self.HCT_Param2 = 30
        self.HCT_MinR = 10
        self.HCT_MaxR = 90

        self.mainLayout = QVBoxLayout(self) 
        self.mainLayout.setAlignment(Qt.AlignTop)
        self.central_widget.setLayout(self.mainLayout)
        self.mainLayout.setSpacing(10)
        self.setCentralWidget(self.central_widget)
        self.frameLabel = QLabel(self)
        self.frameLabel.setFixedSize(1000, 500)
        self.frameLabel.setScaledContents(True)
        self.mainLayout.addWidget(self.frameLabel)

        #   ------ First Horizontal Layout ------
        #   ------ Hough Circle Transform UI ------

        self.HTC_H_layout = QHBoxLayout(self)
        self.HTC_H_layout.setSpacing(20)

        self.hough_label = QLabel(self)
        self.hough_label.setText("Hough Circle Transform")

        self.HTC_H_layout.addWidget(self.hough_label)

        self.bt_image = QPushButton("Image", self)
        self.bt_image.setFixedSize(QSize(80, 30))
        self.bt_image.clicked.connect(self.load_image_hough)
        self.HTC_H_layout.addWidget(self.bt_image)

        self.bt_video = QPushButton("Video", self)
        self.bt_video.setFixedSize(QSize(80, 30))
        self.bt_video.clicked.connect(self.video_capture)
        self.HTC_H_layout.addWidget(self.bt_video)
        
        self.bt_reset = QPushButton("Reset", self)
        self.bt_reset.setFixedSize(QSize(80, 30))
        self.bt_reset.clicked.connect(self.reset_HCT)
        self.HTC_H_layout.addWidget(self.bt_reset)
        
        self.HCT_num_of_Detected = QLabel(self)
        self.HCT_num_of_Detected.setText(str(0))
        self.HCT_num_of_Detected.setFixedSize(30, 25)
        self.HCT_num_of_Detected.setAlignment(Qt.AlignCenter)
        self.HTC_H_layout.addWidget(self.HCT_num_of_Detected)

        self.minDist = QLineEdit(self)
        self.minDist.resize(100, 20)
        self.minDist.setPlaceholderText("minDist")
        self.minDist.setValidator(self.validator)
        self.HTC_H_layout.addWidget(self.minDist)

        self.dp = QLineEdit(self)
        self.dp.resize(100, 20)
        self.dp.setPlaceholderText("dp")
        self.dp.setValidator(self.validator)
        self.HTC_H_layout.addWidget(self.dp)

        self.param1 = QLineEdit(self)
        self.param1.resize(100, 20)
        self.param1.setPlaceholderText("Param 1")
        self.param1.setValidator(self.validator)
        self.HTC_H_layout.addWidget(self.param1)

        self.param2 = QLineEdit(self)
        self.param2.resize(100, 20)
        self.param2.setPlaceholderText("Param 2")
        self.param2.setValidator(self.validator)
        self.HTC_H_layout.addWidget(self.param2)

        self.Min_R = QLineEdit(self)
        self.Min_R.resize(100, 20)
        self.Min_R.setPlaceholderText("min radious")
        self.Min_R.setValidator(self.validator)
        self.HTC_H_layout.addWidget(self.Min_R)

        self.Max_R = QLineEdit(self)
        self.Max_R.resize(100, 20)
        self.Max_R.setPlaceholderText("max radious")
        self.Max_R.setValidator(self.validator)
        self.HTC_H_layout.addWidget(self.Max_R)

        #   ------ Second Horizontal Layout ------
        #   ------ Template Matching UI ------

        self.TM_H_layout = QHBoxLayout(self)
        self.TM_H_layout.setSpacing(20)

        self.TM_label = QLabel(self)
        self.TM_label.setText("Template Matching")
        self.TM_H_layout.addWidget(self.TM_label)

        self.tm_cb_methods = QComboBox(self)
        self.tm_cb_methods.addItem("TM_CCOEFF")
        self.tm_cb_methods.addItem("TM_CCOEFF_NORMED")
        self.tm_cb_methods.addItem("TM_CCORR")
        self.tm_cb_methods.addItem("TM_CCORR_NORMED")
        self.tm_cb_methods.addItem("TM_SQDIFF")
        self.tm_cb_methods.addItem("TM_SQDIFF_NORMED")
        self.tm_cb_methods.currentIndexChanged.connect(self.change_tm_methods)
        self.TM_H_layout.addWidget(self.tm_cb_methods)

        self.tm_le_threshold = QLineEdit(self)
        self.tm_le_threshold.setFixedSize(70, 20)
        self.tm_le_threshold.setPlaceholderText("Threshold")
        self.tm_le_threshold.setValidator(self.validator)
        self.TM_H_layout.addWidget(self.tm_le_threshold)

        self.TM_num_of_Detected = QLabel(self)
        self.TM_num_of_Detected.setText(str(0))
        self.TM_num_of_Detected.setFixedSize(30, 25)
        self.TM_num_of_Detected.setAlignment(Qt.AlignCenter)
        self.TM_H_layout.addWidget(self.TM_num_of_Detected)

        self.TM_bt_temp = QPushButton("Template", self)
        self.TM_bt_temp.setFixedSize(QSize(80, 30))
        self.TM_bt_temp.clicked.connect(lambda: self.load_image("temp"))
        self.TM_H_layout.addWidget(self.TM_bt_temp)

        self.TM_bt_img = QPushButton("Image", self)
        self.TM_bt_img.setFixedSize(QSize(80, 30))
        self.TM_bt_img.clicked.connect(lambda: self.load_image("img"))
        self.TM_H_layout.addWidget(self.TM_bt_img)

        self.TM_bt_impimg = QPushButton("Imp on Img", self)
        self.TM_bt_impimg.setFixedSize(QSize(80, 30))
        self.TM_bt_impimg.clicked.connect(self.tm_imp_img)
        self.TM_H_layout.addWidget(self.TM_bt_impimg)

        self.TM_bt_video = QPushButton("video", self)
        self.TM_bt_video.setFixedSize(QSize(80, 30))
        self.TM_bt_video.clicked.connect(self.tm_video)
        self.TM_H_layout.addWidget(self.TM_bt_video)

        self.TM_toggle_button = QPushButton("Single mode", self)
        self.TM_toggle_button.setCheckable(True)
        self.TM_toggle_button.clicked.connect(self.change_tm_mode)
        self.TM_toggle_button.setStyleSheet("background-color : lightgrey")
        self.TM_H_layout.addWidget(self.TM_toggle_button)

        self.mainLayout.addLayout(self.HTC_H_layout)
        self.mainLayout.addLayout(self.TM_H_layout)

        self.video_capture()
        self.setLayout(self.mainLayout)
        self.show()


    def load_image(self, flag):
        if flag == "temp":
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            self.template_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp *.gif);;All Files (*)", options=options)
        elif flag == "img":
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp *.gif);;All Files (*)", options=options)


    def video_capture(self):
        if self.video_flag == False:  # self.video.isOpened()
            self.video_flag = True
            self.video = cv2.VideoCapture(0)
        else:
            self.video_flag = False
            self.video.release()


    def tm_imp_img(self):
        if self.video_flag == True:
            self.video_flag = False
            self.video.release()
            result, tm_counter = self.templateMatching.teplate_matching(self.template_path, self.image_path, self.tm_threshold, self.tm_method, self.tm_mode)
            self.set_frame_gray(result)
            self.TM_num_of_Detected.setText(str(tm_counter))
        else:
            result, tm_counter = self.templateMatching.teplate_matching(self.template_path, self.image_path, self.tm_threshold, self.tm_method, self.tm_mode)
            self.set_frame_gray(result)
            self.TM_num_of_Detected.setText(str(tm_counter))


    def tm_video(self):
        self.method = "TM"
        self.video_flag = True


    def tm_threshold_change(self):
        if self.tm_le_threshold.text() != '':
            self.tm_threshold = self.tm_le_threshold.text()
        else:
            self.message_box("Threshold must not be empty")


    def change_tm_methods(self):
        self.tm_method = self.tm_cb_methods.currentText()


    def change_tm_mode(self):
        # if button is checked
        if self.TM_toggle_button.isChecked():
            self.tm_mode = "multi"
            self.TM_toggle_button.setStyleSheet("background-color : lightblue")
            self.TM_toggle_button.setText("Multi mode")
        else:
            self.tm_mode = "single"
            self.TM_toggle_button.setText("Single mode")
            self.TM_toggle_button.setStyleSheet("background-color : lightgrey")


    def load_image_hough(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp *.gif);;All Files (*)", options=options)
        if fileName:
            self.image_path = fileName
            if self.video_flag == True:
                self.video.release()
                self.video_flag = False
                tframe = cv2.imread(self.image_path)
                result = self.circleTransform.liveCameraCircleDitection(tframe, 
                                                                        HCT_minDist = float(self.HCT_minDist), 
                                                                        HCT_dp = float(self.HCT_dp), 
                                                                        HCT_Param1 = float(self.HCT_Param1), 
                                                                        HCT_Param2 = float(self.HCT_Param2), 
                                                                        HCT_MinR = int(self.HCT_MinR), 
                                                                        HCT_MaxR = int(self.HCT_MaxR))
                if result is not None:
                    count, frame = result
                    self.HCT_num_of_Detected.setText(str(count))
                else:
                    count = 0
                    frame = tframe #np.ones((200, 200, 3), np.uint8)
                    self.HCT_num_of_Detected.setText(str(count))
                self.set_frame_colored(frame)
            else:
                self.set_frame_colored(frame)
        else:
            self.message_box("Nothing to Show")


    def set_frame_gray(self, frame):
        # frame = self.scale_to_fit_gray(frame)
        height, width = frame.shape
        qImg = QImage(frame.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qImg)
        self.frameLabel.setPixmap(pixmap)


    def set_frame_colored(self, frame):
        # if self.video_flag == True:
        # frame = self.scale_to_fit_colored(frame)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888) #(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        img_resized = image#.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pixmap = QPixmap.fromImage(img_resized)
        self.frameLabel.setPixmap(pixmap.scaled(self.frameLabel.width(), self.frameLabel.height(), Qt.KeepAspectRatioByExpanding))
        # else:
        #     pixmap = QPixmap(frame)
        #     self.frameLabel.setPixmap(pixmap)


    def update_frame(self):
        if self.method == "HCT":
            if self.video_flag == True:
                ret, oframe = self.video.read()
            else:
                ret, oframe = False, np.ones((200, 200, 3), np.uint8)
            if ret:
                tframe = cv2.cvtColor(oframe, cv2.COLOR_BGR2RGB)
                result = self.circleTransform.liveCameraCircleDitection(tframe, 
                                                                        HCT_minDist = float(self.HCT_minDist), 
                                                                        HCT_dp = float(self.HCT_dp), 
                                                                        HCT_Param1 = float(self.HCT_Param1), 
                                                                        HCT_Param2 = float(self.HCT_Param2), 
                                                                        HCT_MinR = int(self.HCT_MinR), 
                                                                        HCT_MaxR = int(self.HCT_MaxR))
                if result is not None:
                    count, frame = result
                    self.HCT_num_of_Detected.setText(str(count))
                else:
                    count = 0
                    frame = tframe #np.ones((200, 200, 3), np.uint8)
                    self.HCT_num_of_Detected.setText(str(count))
                self.set_frame_colored(frame)
        elif self.method == "TM":
            if self.video_flag == True:
                ret, oframe = self.video.read()
            else:
                ret, oframe = False, np.ones((200, 200, 3), np.uint8)
            if ret:
                tframe = cv2.cvtColor(oframe, cv2.COLOR_BGR2RGB)
                result, tm_counter = self.templateMatching.teplate_matching(self.template_path, oframe, self.tm_threshold, self.tm_method, self.tm_mode)
                # if result is not None:
                #     count, frame = result
                #     self.HCT_num_of_Detected.setText(str(count))
                # else:
                #     count = 0
                #     frame = tframe #np.ones((200, 200, 3), np.uint8)
                #     self.HCT_num_of_Detected.setText(str(count))
                frame = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                self.set_frame_colored(frame)
                self.TM_num_of_Detected.setText(str(tm_counter))


    def reset_HCT(self):
        if self.Max_R.text() != '':
            self.HCT_MaxR = self.Max_R.text()
        if self.Min_R.text() != '':
            self.HCT_MinR = self.Min_R.text()
        if self.param1.text() != '':
            self.HCT_Param1 = self.param1.text()
        if self.param2.text() != '':
            self.HCT_Param2 = self.param2.text()
        if self.minDist.text() != '':
            self.HCT_minDist = self.minDist.text()
        if self.dp.text() != '':
            self.HCT_dp = self.dp.text()
    

        if self.video_flag == False:
            tframe = cv2.imread(self.image_path)
            result = self.circleTransform.liveCameraCircleDitection(tframe, 
                                                                    HCT_minDist = float(self.HCT_minDist), 
                                                                    HCT_dp = float(self.HCT_dp), 
                                                                    HCT_Param1 = float(self.HCT_Param1), 
                                                                    HCT_Param2 = float(self.HCT_Param2), 
                                                                    HCT_MinR = int(self.HCT_MinR), 
                                                                    HCT_MaxR = int(self.HCT_MaxR))
            if result is not None:
                count, frame = result
                self.HCT_num_of_Detected.setText(str(count))
            else:
                count = 0
                frame = tframe #np.ones((200, 200, 3), np.uint8)
                self.HCT_num_of_Detected.setText(str(count))
            self.set_frame_colored(frame)


    def message_box(self, text, type="warning"):
        if type == "warning":
            self.msg = QMessageBox(self)
            self.msg.setIcon(QMessageBox.Warning)
            self.msg.setText(str(text))
            self.msg.setWindowTitle("Warning")
            self.msg.exec_()
        elif type == "info":
            self.msg = QMessageBox(self)
            self.msg.setIcon(QMessageBox.Information)
            self.msg.setText(str(text))
            self.msg.setWindowTitle("Warning")
            self.msg.exec_()
            

    
    def scale_to_fit_colored(self, img):
        label_height, label_width = self.frameLabel.height(), self.frameLabel.width()
        img = cv2.resize(img, (label_width, label_height))
        return img


    def scale_to_fit_gray(self, frame):
        # Get label size
        label_height = frame.height()
        label_width = frame.width()
        # Resize image to fit label size
        img = cv2.resize(frame, (label_width, label_height), interpolation=cv2.INTER_AREA)
        return img







if __name__ == '__main__': 
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Window()
    window.move(350, 90)
    window.resize(900, 750)
    window.show()

    timer = QTimer()
    timer.timeout.connect(window.update_frame)
    timer.start(30)
    sys.exit(app.exec())
