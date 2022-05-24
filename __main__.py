import sys
import cv2 as cv
import numpy as np
import imutils
#TODO search for a blue thingy on the left part of the contour
#TODO treshold for too big objects (rectangles)

# percent values from the area of the half image, where the number plates should be found
maximum_numberplate_area_percent = 2
minimum_numberplate_area_percent = 0.01

def find_candidates(crop_color, crop, max_candidates, draw_rects=False):
    
    image_area = len(crop_color) * len(crop_color[0])
    max_area = maximum_numberplate_area_percent * image_area / 100
    min_area = minimum_numberplate_area_percent * image_area / 100

    #create 13x5 array of ones
    rectKern = cv.getStructuringElement(cv.MORPH_RECT, (13, 5))

    #perform blackhat op (difference between closing and input image)
    blackhat = cv.morphologyEx(crop, cv.MORPH_BLACKHAT, rectKern)
    squareKern = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    light = cv.morphologyEx(crop, cv.MORPH_CLOSE, squareKern)
    light = cv.threshold(light, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    
    grad_x = cv.Sobel(blackhat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
    grad_x = 255 * ((grad_x - min_val) / (max_val - min_val))
    grad_x = grad_x.astype("uint8")

    grad_x = cv.GaussianBlur(grad_x, (5, 5), 0)
    grad_x = cv.morphologyEx(grad_x, cv.MORPH_CLOSE, rectKern)
    thresholded_blurred_x_scharr = cv.threshold(grad_x, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    denoised = cv.erode(thresholded_blurred_x_scharr, None, iterations=2)
    denoised = cv.dilate(denoised, None, iterations=2)

    multiplied_with_light_regions = cv.bitwise_and(denoised, denoised, mask=light)
    denoised = cv.dilate(multiplied_with_light_regions, None, iterations=2)
    denoised = cv.erode(denoised, None, iterations=1)


    cvcnts = cv.findContours(denoised.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cvcnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    
    # Create array of the boxes basing on the contours
    boxes = []
    for c in cnts:
        c_area = cv.contourArea(c)
        if c_area < max_area  and c_area > min_area:
            print("Allowed area: "+str(c_area))
            (x, y, w, h) = cv.boundingRect(c)
            boxes.append([x,y, x+w,y+h])
    boxes = np.asarray(boxes)
    
    boxes = boxes[:max_candidates]
    crop_color_copy = crop_color.copy()

    # draw the boxes
    if(draw_rects):
        for i in range(len(boxes)):
            cv.rectangle(crop_color_copy, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0,255,0), 2 )
    
    return crop_color_copy, boxes

def find_blue_areas(img, max_candidates=5, draw_rectangles=False):

    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    
    # https://de.wikipedia.org/wiki/HSV-Farbraum#Transformation_von_HSV/HSL_und_RGB
    # HSV Color space in openCV interpretation:
    # H - Hue (Farbwinkel)[0-180]
    # S - Saturation (Farbsättigung)[0-255]
    # V - Value (Helligkeit)[0-255]

    lower_blue = np.array([100,192,150])
    upper_blue = np.array([130,255,200])

    mask = cv.inRange(hsv,lower_blue,upper_blue)

    cv.imshow("MASK", mask)

    cnts = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    cnts = cnts[:max_candidates]

    # Create array of the boxes basing on the contours
    boxes = []
    for c in cnts:
        (x, y, w, h) = cv.boundingRect(c)
        boxes.append([x,y, x+w,y+h])
        
    boxes = np.asarray(boxes)

    crop_color_copy = img.copy()

    # draw the boxes
    if(draw_rectangles):
        print ('Drawing the found blue areas')
        for i in range(len(cnts)):  
            if(i>max_candidates):
                break
            cv.rectangle(crop_color_copy, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (100,50,255), 1 )
    
    return crop_color_copy, boxes[:max_candidates]

def on_rectangle_overlap(R1, R2):
    if (R1[0]>=R2[2]) or (R1[2]<=R2[0]) or (R1[3]<=R2[1]) or (R1[1]>=R2[3]):
        return False
    else:
        return True

if __name__ == "__main__":
    default_file = './carplates/car-plate1.jpg'
    filename = sys.argv[1] if len(sys.argv) > 1 else default_file

    # Loads an image
    img_color = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    img_gray = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

    # Check if image is loaded fine
    if img_gray is None:
        print('Error opening image!')
        print('Usage python __main__.py [image_name -- default ' + default_file + '] \n')
    else:


        original_height, original_width = img_gray.shape 

        h=round(len(img_gray) * 0.25)

        crop_gray = img_gray[h:original_height, 0:original_width]
        crop_color = img_color[h:original_height, 0:original_width]

        candidates_result, candidate_boxes = find_candidates(crop_color, crop_gray, 100, True)
        blues_image, blue_boxes = find_blue_areas(crop_color, 5, True)

        crop_color_copy = crop_color.copy()

        for box in candidate_boxes:
            for box2 in blue_boxes:
                if(on_rectangle_overlap(box, box2)):
                    cv.rectangle(crop_color_copy, (box[0], box[1]), (box[2], box[3]), (0,0,255), 1 )

                    
        cv.imshow("Original", img_color)
        cv.imshow("Result", crop_color_copy)  
        cv.imshow("Blues", blues_image)
        cv.imshow("Detected", candidates_result)
        cv.waitKey(0)