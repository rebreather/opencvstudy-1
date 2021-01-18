import numpy as np
import cv2 as cv

point_list = []

def mouse_callback(event, x, y, flags, param):

 #마우스 왼쪽 버튼 누를 때 마다 좌표를 리스트에 저장
    if event == cv.EVENT_LBUTTONDOWN: 

        print("(%d, %d)" % (x, y))

        point_list.append((x, y))
        cv.circle(img_color, (x, y), 3, (0, 0, 255), -1)

 
cv.namedWindow('source') #콜백 함수 등록
cv.setMouseCallback('source', mouse_callback)

img_color = cv.imread('view.jpg')

#마우스 클릭으로 세 점 지정
while(True): 

    cv.imshow('source', img_color)

    if cv.waitKey(1) == 32: #스페이스 바 누르면 루프에서 빠져나옴
        break


height, weight = img_color.shape[:2]

#오른쪽 상단의 대응점만 Y좌표가 아래로 100만큼 이동하도록.
pts1 = np.float32([point_list[0], point_list[1], point_list[2]])
pts2 = np.float32([point_list[0], point_list[1], point_list[2]])
pts2[1][1] += 100


M = cv.getAffineTransform(pts1,pts2) #아핀 변환 행렬 생성


img_result = cv.warpAffine(img_color, M, (weight,height))

 #결과
cv.imshow("result", img_result)
cv.waitKey(0)
cv.destroyAllWindows()