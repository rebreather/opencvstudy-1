import cv2

img_color = cv2.imread('cat.jpg')
cv2.imshow("color", img_color)


height, width = img_color.shape[:2]

#이미지 중앙을 중신으로 반시계 방향 45도 회전 시키는 행렬 생성
M = cv2.getRotationMatrix2D((width/2.0, height/2.0), #g회전 할 때 중심점
45, #회전각도(양수면 반시계, 음수면 시계방향)
1)  #이미지 배율

#회전 행렬 M을 이미지 img_color에 적용함.
img_rotated = cv2.warpAffine(img_color, M, (width, height))

cv2.imshow("rotation", img_rotated)
cv2.waitKey(0)


cv2.destroyAllWindows()