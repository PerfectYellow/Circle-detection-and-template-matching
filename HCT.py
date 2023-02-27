import cv2
import numpy as np


class HoughCircleTransform():

    def liveCameraCircleDitection(self, frame, HCT_minDist, HCT_dp, HCT_Param1, HCT_Param2, HCT_MinR, HCT_MaxR):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # b, g, r = cv2.split(frame)
        # if r.all() == g.all() == b.all():
        #     print("Grayscale image")
        #     gray = frame
        # else:
        #     if r.any():
        #         gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #     else:
        #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, minDist=HCT_minDist, dp=HCT_dp, param1=HCT_Param1, param2=HCT_Param2, minRadius=HCT_MinR, maxRadius=HCT_MaxR)

        # If circles are detected, draw them on the original frame
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # Count the number of circles
            if circles is not None and len(circles) > 0: #and circles.shape[0] != () and frame is not None:
                # count = circles.shape[0]
                num_circles = len(circles)
                return num_circles, frame
            else:
                num_circles = 0
                # frame = np.zeros((200, 200, 3), np.uint8)
                return num_circles, frame

