import cv2
import imutils
import numpy as np
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
# 
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def image_print_mask(img, mask, mask_applied_img):
        cv2.imshow("image", img)
        cv2.imshow("mask", mask)
        cv2.imshow("image with mask applied", mask_applied_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
		       (x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	RED = (0, 0, 255)

        # # Display original image.
        # image_print(img)
        
        # Convert img from BGR colorspace to HSV colorspace.
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Find range of orange HSV values.
        light_orange_hsv = (1, 190, 153)
        dark_orange_hsv = (28, 255, 255)

        # Apply mask over image, keeping just the orange parts. 
        mask = cv2.inRange(img_hsv, light_orange_hsv, dark_orange_hsv)
        mask_applied_img = cv2.bitwise_and(img, img, mask=mask)
        
        # # Display image with mask applied.
        # image_print_mask(img, mask, mask_applied_img)

        # Use erosion and dilation (opening) to remove outliers.
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask_applied_img, cv2.MORPH_OPEN, kernel)
        # image_print(opening)

        # Grayscale opening image for use as input.
        grayscaled = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)

        # Find the contours in grayscaled image.
        image, contours, hierarchy = cv2.findContours(grayscaled, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find bounding box coordinates.
        biggest_blob = contours[0]
        x1, y2, w, h = cv2.boundingRect(biggest_blob)  # x1 = left-most x, y2 = top_most y
        x2 = x1 + w  # right-most x
        y1 = y2 + h  # bottom-most y

        # Display image with bounding box drawn.
        cv2.rectangle(img, (x1, y2), (x2, y1), RED, 2)
        image_print(img)

        return ((x1, y1), (x2, y2))
