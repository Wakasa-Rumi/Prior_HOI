import cv2 as cv
import numpy as np

class_name = '003_cracker_box'

if __name__ == '__main__':
    img = cv.imread("/home/yiyao/HOI/HOI/ho/preprocess/img/yz/003_cracker_box.jpg", 0)
    _, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, 3, 2)
    cnt = contours[0]

    hull = cv.convexHull(cnt)

    image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(image, [hull], True, (0, 0, 255), 2)

    cv.imwrite('hull/003_cracker_box.jpg', image)
