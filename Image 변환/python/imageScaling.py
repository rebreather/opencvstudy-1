import cv2 as cv



img_color = cv.imread('cat.jpg')
cv.imshow("original", img_color)

#fx, fy를 사용하여 이미지 확대 및 축소 비율을 적을 수 있음.
#여기선 2배 확대, 이미지 확대시에는 보간법으로 INTER_CUBIC을 권장함.
img_result = cv.resize(img_color, None, fx=2, fy=2, 
    interpolation = cv.INTER_CUBIC)
cv.imshow("x2 INTER_CUBIC", img_result)

#너비와 높이를 3배 확대
height, width = img_color.shape[:2]
img_result = cv.resize(img_color, (3*width, 3*height), 
    interpolation = cv.INTER_LINEAR )
cv.imshow("x3 INTER_LINEAR", img_result)

#너비와 높이를 0.5배로 줄임
img_result = cv.resize(img_color, None, fx=0.5, fy=0.5, interpolation = cv.INTER_AREA)
cv.imshow("x0.5 INTER_AREA", img_result)


cv.waitKey(0)
cv.destroyAllWindows()