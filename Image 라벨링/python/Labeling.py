import cv2 as cv


img_color = cv.imread("keyboard.jpeg", cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY) 
cv.imshow("result", img_gray )
cv.waitKey(0)

#엣지를 검출한다. 검출 된 엣지만 흰색이 됨
img_edge = cv.Canny(img_gray, 50, 150)
cv.imshow("result", img_edge )
cv.waitKey(0)

#관심 물체가 흰색이 되어야 하므로 반전시킨다.
img_edge = cv.bitwise_not(img_edge)
cv.imshow("result", img_edge )
cv.waitKey(0) 

#컨투어를 찾아와서 외곽선을 보강한다.
contours = cv.findContours(img_edge.copy(), cv.RETR_LIST, 
            cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img_edge, contours[0], -1, (0, 0, 0), 1)
cv.imshow("result", img_edge )
cv.waitKey(0) 

#흰색 영역에 대해 라벨링함.
nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img_edge)


for i in range(nlabels):

    #배경 제외
    if i < 2:
        continue

    #흰색 영역의 크기, 중심 좌표, 외각 사각형에 대한 정보를 가져온다.
    area = stats[i, cv.CC_STAT_AREA]
    center_x = int(centroids[i, 0])
    center_y = int(centroids[i, 1]) 
    left = stats[i, cv.CC_STAT_LEFT]
    top = stats[i, cv.CC_STAT_TOP]
    width = stats[i, cv.CC_STAT_WIDTH]
    height = stats[i, cv.CC_STAT_HEIGHT]


    #영역의 크기가 50 이상인 경우, 해당 영역에 다음과 같은 정보를 표시.
    if area > 50: 

        #영역 외곽에 사각형을 그림.
        cv.rectangle(img_color, (left, top), (left + width, top + height), (0, 0, 255), 1)
                                                                             ##B,G,R

        #영역 중심 좌표에 원을 그림.
        cv.circle(img_color, (center_x, center_y), 3, (0, 0, 255), 1)

        #라벨 번호를 표시.
        cv.putText(img_color, str(i), (left + 20, top + 20), cv.FONT_HERSHEY_PLAIN, 1, (255,210,0), 1);

        ##############사진, 번호, 위치, 폰트이름, 폰트크기, 폰트색상, 폰트두께

#결과를 화면에 보여줌
cv.imshow("result", img_color)
cv.waitKey(0)