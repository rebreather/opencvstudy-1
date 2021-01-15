#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 참고 사이트 https://ivo-lee.tistory.com/91


# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE 회색
plt.imshow(image, cmap="gray"), plt.axis("off")
plt.show()
type(image) # 데이터 타입을 확인
image # 이미지 데이터를 확인
image.shape # 차원을 확인 (해상도)


# In[3]:


image_bgr = cv2.imread("test.jpg", cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR 컬러
image_bgr[0,0] # 픽셀을 확인
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # RGB로 변환
plt.imshow(image_rgb), plt.axis("off")
plt.show()


# In[4]:


## 이미지 저장 코드 imwrite
image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imwrite("test2.jpg", image) # 이미지를 저장


# In[5]:


## 픽셀 조정 cv2.resize(사진,(픽셀,픽셀))
image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
image_50x50 = cv2.resize(image, (50,50)) # 이미지 크기를 50x50 픽셀로 변경

plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()


# In[6]:


## 이미지 자르기  image[:,:128]
image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
image_cropped = image[0:,:100] # 열의 처음 절반과 모든 행을 선택
## 앞쪽의 숫자가 세로, 뒤쪽의 숫자가 가로

plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()

image_cropped = image[100:,:200] # 열의 처음 절반과 모든 행을 선택
## 앞쪽의 숫자가 세로, 뒤쪽의 숫자가 가로

plt.imshow(image_cropped, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()


# In[7]:


## 이미지 투명도 처리
image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
#각 픽셀 주변의 5X5커널 평균값으로 이미지를 흐리게 합니다.
image_blurry = cv2.blur(image, (5,5))
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()

# 커널 크기의 영향을 강조하기 위해 100X100 커널로 같은 이미지를 흐리게 합니다.
image_very_blurry = cv2.blur(image, (100,100))
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([]) # 이미지를 출력
plt.show()


# In[8]:


## 이미지 투명도 처리 커널 사용
## 커널 PCA와 서포트 벡터 머신이 사용하는 비선형 함수를 커널이라고 합니다.
## Meanshift 알고리즘에서는 샘플의 영향 범위를 커널이라고 합니다.
## 신경망의 가중치를 커널이라고 합니다.
## 커널의 크기는 너비,높이로 지정 합니다.
## 주변 픽셀값의 평균을 계산하는 커널은 이미지를 흐릿하게 만듭니다.
## blur함수는 각 픽셀에 커널 개수의 역수를 곱하며 모두 더합니다.

kernel = np.ones((5,5)) / 25.0 # 커널을 만듭니다.
kernel # 커널을 확인
image_kernel = cv2.filter2D(image, -1, kernel) # 커널을 적용
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()

image_very_blurry = cv2.GaussianBlur(image, (5,5), 0) # 가우시안 블러를 적용
plt.imshow(image_very_blurry, cmap="gray"), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()


# In[9]:


## 이미지 투명도 처리
gaus_vector = cv2.getGaussianKernel(5, 0)
gaus_vector
gaus_kernel = np.outer(gaus_vector, gaus_vector) # 벡터를 외적하여 커널을 만듭니다.
gaus_kernel

# filter2D()로 커널을 이미지에 직접 적용하여 비슷한 흐림 효과를 만들 수 있습니다.
image_kernel = cv2.filter2D(image, -1, gaus_kernel) # 커널을 적용
plt.imshow(image_kernel, cmap="gray"), plt.xticks([]), plt.yticks([]) # 이미지 출력
plt.show()


# In[10]:


##이미지 선명하게 하기
image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]]) # 커널을 만듭니다.

# 이미지를 선명하게 만듭니다.
image_sharp = cv2.filter2D(image, -1, kernel)

plt.imshow(image_sharp, cmap="gray"), plt.axis("off") # 이미지 출력
plt.show()


# In[11]:


## 이미지 대비 높이기
## 히스토그램 평활화는 객체의 형태가 두드러지도록만들어주는 이미지 처리 도구
## Y루마(luma) 또는 밝기이고 U와 V는 컬러를 나타냅니다.
## 흑백 이미지에는 OpenCV의 equalizeHist()를 바로 적용할 수 있습니다.
## 히스토그램 평활화는 픽셀값의 범위가 커지도록 이미지를 변환합니다.
## 히스토그램 평활화는 괌심 대상을 다른 객체나 배경과 잘 구분되도록 만들어줍니다.

image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
image_enhanced = cv2.equalizeHist(image) # 이미지 대비를 향상시킵니다.
plt.imshow(image_enhanced, cmap="gray"), plt.axis("off")
plt.show()

image_bgr = cv2.imread("test.jpg")
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV) # YUV로 변경합니다.
image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0]) # 히스토그램 평활화를 적용
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB) # RGB로 바꿉니다.
plt.imshow(image_rgb), plt.axis("off")
plt.show()


# In[12]:


## 색상 구분
## 이미지에서 한 색상을 구분하려면 색 범위를 정의하고 이미지에 마스크를 적용합니다.
## 이미지를 HSV(색상, 채도, 명도)로 변환 -> 격리시킬 값의 범위를 정의 -> 이미지에 적용할 마스크를 만듭니다.(마스크의 흰색 영역만 유지)
## bitwise_and()는 마스크를 적용하고 원하는 포맷으로 변환

image_bgr = cv2.imread('test.jpg')
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV) # BGR에서 HSV로 변환
lower_blue = np.array([50,100,50]) # HSV에서 파랑 값의 범위를 정의
upper_blue = np.array([130,255,255])
mask = cv2.inRange(image_hsv, lower_blue, upper_blue) # 마스크를 만듭니다.
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask) # 이미지에 마스크를 적용
image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB) # BGR에서 RGB로 변환

plt.imshow(image_rgb), plt.axis("off")
plt.show()

plt.imshow(mask, cmap='gray'), plt.axis("off") # 마스크 출력
plt.show()


# In[13]:


## 이미지 이진화
## 이미지 이진화(임계처리)thresholding은 어떤 값보다 큰 값을 가진 픽셀을 흰색으로 만들고 작은 값을 가진 픽셀은검은색으로 만드는 과정입니다.
## 적응적 이진화(임계처리)adaptive thresholding은 픽셀의 임계값이 주변 픽셀의 강도에 의해 결정됩니다.
## 이진화는 이미지 안의 영역 마다 빛 조건이 달라질 때 도움이 됩니다.
## adaptiveThreshold()의 max_output_value매개변수는 출력 픽셀 강도의 최대값을 결정
## cv2.ADAPTIVE_THRESH_GAUSSIAN_C는 픽셀의 임계값을 주변 픽셀 강도의 가중치 합으로 설정합니다
## cv2.ADAPTIVE_THRESH_MEAN_C는 픽셀의 임계값을 주변 픽셀의 평균으로 설정합니다

image_grey = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
max_output_value = 255
neighborhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey, max_output_value,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         neighborhood_size, subtract_from_mean) # 적응적 임계처리를 적용
plt.imshow(image_binarized, cmap="gray"), plt.axis("off") # 이미지 출력
plt.show()


# In[14]:


## 이미지 이진화
# cv2.ADAPTIVE_THRESH_MEAN_C를 적용합니다.
image_mean_threshold = cv2.adaptiveThreshold(image_grey,
                                             max_output_value,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY,
                                             neighborhood_size,
                                             subtract_from_mean)
plt.imshow(image_mean_threshold, cmap="gray"), plt.axis("off") # 이미지를 출력
plt.show()


# In[15]:


## 배경 제거
## 이미지의 전경만 분리해내려면 원하는 전경 주위에 사각형 박스를 그리고 그랩컷 알고리즘을 실행합니다.
## 그랩컷은 사각형 밖에 있는 모든 것이 배경이라고 가정하고 이 정보를 사용하여 사각형 안에 있는 배경을 찾습니다.
## 검은색 영역은 배경이라고 확실하게 가정한 사각형의 바깥쪽 영역이며, 회색 영역은 그랩컷이 배경이라고 생각하는 영역이고 흰색 영역은 전경입니다.

image_bgr = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # RGB로 변환

rectangle = (0, 56, 256, 150) # 사각형 좌표: 시작점의 x, 시작점의 y, 너비, 높이

mask = np.zeros(image_rgb.shape[:2], np.uint8) # 초기 마스크를 만듭니다.

bgdModel = np.zeros((1, 65), np.float64) # grabCut에 사용할 임시 배열을 만듭니다.
fgdModel = np.zeros((1, 65), np.float64)

# grabCut 실행
cv2.grabCut(image_rgb, # 원본 이미지
            mask, # 마스크
            rectangle, # 사각형
            bgdModel, # 배경을 위한 임시 배열
            fgdModel, # 전경을 위한 임시 배열
            5, # 반복 횟수
            cv2.GC_INIT_WITH_RECT) # 사각형을 사용한 초기화
            
# 배경인 곳은 0, 그외에는 1로 설정한 마스크를 만듭니다.
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# 이미지에 새로운 마스크를 곱해 배경을 제외합니다.
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
plt.imshow(image_rgb_nobg), plt.axis("off") # 이미지 출력
plt.show()

plt.imshow(mask, cmap='gray'), plt.axis("off") # 마스크 출력
plt.show()

plt.imshow(mask_2, cmap='gray'), plt.axis("off") # 마스크 출력
plt.show()


# In[16]:


## 경계선 감지
## 캐니(Canny) 경계선 감지기와 같은 경계선 감지 기술 사용
## 경계선 감지는 컴퓨터 비전의 주요 관심 대상이며 경계선은 많은 정보가 담긴 영역입니다.
## 경계선 감지를 사용하여 정보가 적은 영역을 제거하고 대부분의 정보가 담긴 이미지 영역을 구분할 수 있습니다.
## 캐니 감지기는 그레이디언트 임계값의 저점과 고점을 나타내는 두 매개변수가 필요합니다.
## 낮은 임계값과 높은 임계값 사이의 가능성 있는 경계선 픽셀은 약한 경계선 픽셀로 간주됩니다
## OpenCV의 Canny 함수는 낮은 임곗값과 높은 임곗값이 필수 매개변수입니다.
## Canny를 전체 이미지 모음에 적용하기 전에 몇 개의 이미지를 테스트하여 낮은 임계값과 높은 임곗값의 적절한 쌍을 찾는 것이 좋은 결과를 만듭니다.
## 예제 실습은 낮은 임곗값과 높은 임곗값을 이미지 중간 픽셀 강도의 1표준편차 아래 값과 위 값으로 설정

image_gray = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
median_intensity = np.median(image_gray) # 픽셀 강도의 중간값을 계산

# 중간 픽셀 강도에서 위아래 1 표준 편차 떨어진 값을 임계값으로 지정합니다.
lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

# 캐니 경계선 감지기를 적용합니다.
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

plt.imshow(image_canny, cmap="gray"), plt.axis("off")
plt.show()


# In[17]:


## 모서리 감지
## cornerHarris - 해리스 모서리 감지의 OpenCV 구현
## 해리스 모서리 감지기는 두 개의 경계선이 교차하는 지점을 감지하는 방법으로 사용됩니다.
## 모서리는 정보가 많은 포인트입니다.
## 해리스 모서리 감지기는 윈도(이웃, 패치)안의 픽셀이 작은 움직임에도 크게 변하는 윈도를 찾습니다.
## cornerHarris 매개변수 block_size : 각 픽셀에서 모서리 감지에 사용되는 이웃 픽셀 크기
## cornerHarris 매개변수 aperture : 사용하는 소벨 커널 크기

image_bgr = cv2.imread("test.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

block_size = 2 # 모서리 감지 매개변수를 설정
aperture = 29
free_parameter = 0.04

detector_responses = cv2.cornerHarris(image_gray,
                                      block_size,
                                      aperture,
                                      free_parameter) # 모서리를 감지
detector_responses = cv2.dilate(detector_responses, None) # 모서리 표시를 부각시킵니다.

# 임계값보다 큰 감지 결과만 남기고 흰색으로 표시합니다.
threshold = 0.02
image_bgr[detector_responses >
          threshold *
          detector_responses.max()] = [255,255,255]

image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # 흑백으로 변환

plt.imshow(image_gray, cmap="gray"), plt.axis("off") # 이미지 출력
plt.show()

# 가능성이 높은 모서리를 출력합니다.
plt.imshow(detector_responses, cmap='gray'), plt.axis("off")
plt.show()


# In[18]:


image_bgr = cv2.imread('test.jpg')
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 감지할 모서리 개수
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

corners = cv2.goodFeaturesToTrack(image_gray,
                                  corners_to_detect,
                                  minimum_quality_score,
                                  minimum_distance) # 모서리를 감지
corners = np.float32(corners)

for corner in corners:
    x, y = corner[0]
    cv2.circle(image_bgr, (x,y), 10, (255,255,255), -1) # 모서리마다 흰 원을 그립니다.
    
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) # 흑백 이미지로 변환
plt.imshow(image_rgb, cmap='gray'), plt.axis("off") # 이미지를 출력
plt.show()


# In[ ]:




