import cv2
import ast
import numpy as np


class TemplateMatching():
    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    def teplate_matching(self, tmp, frame, threshold, method, mode):
        # parsed_method = ast.literal_eval(method)
        # Load the circle template image
        template = cv2.imread(str(tmp), 0)
        w, h = template.shape[::-1]
        if isinstance(frame, str):
            frame = cv2.imread(str(frame), 0)
        else:
            frame = frame

        if(len(frame.shape)<3):
            gray = frame
        elif len(frame.shape)==3:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the height and width of the template
        h, w = template.shape        
        
        # Perform template matching using the cv2.matchTemplate function
        if method == "TM_CCOEFF":
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
        elif method == "TM_CCOEFF_NORMED":
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        elif method == "TM_CCORR":
            res = cv2.matchTemplate(gray, template, cv2.TM_CCORR)
        elif method == "TM_CCORR_NORMED":
            res = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED)
        elif method == "TM_SQDIFF":
            res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)
        elif method == "TM_SQDIFF_NORMED":
            res = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF_NORMED)
        
        # Get the locations of the matches using the cv2.minMaxLoc function
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        color = (0, 0, 255)  # blue color
        thickness = 2
        threshold = threshold

        if mode == "multi":
            # multiobject detect
            loc = np.where(res >= threshold)
            # Count the number of non-zero elements in the thresholded array
            count = np.count_nonzero(loc)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(frame, pt, (pt[0] + template.shape[0], pt[1] + template.shape[1]), (0, 0, 255), 2)
            return frame, count
        elif mode == 'single':
            # only most similar detected
            # Check if the match is above a certain threshold
            if True: #max_val > threshold:
                # Get the top-left and bottom-right corners of the circle
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                # Draw the circle on the frame using the cv2.rectangle function
                cv2.rectangle(frame, top_left, bottom_right, color, thickness)
                
                # if(len(frame.shape)<3):
                #     print("gray")
                # elif len(frame.shape)==3:
                #     print("color")

                # return the frame
                return frame, 1




# img = cv2.imread('input_image.jpg')
# template = cv2.imread('template.jpg')

# res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.8 # set a threshold for a match
# loc = np.where(res >= threshold)

# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0,0,255), 2)
