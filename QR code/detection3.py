import pyzbar.pyzbar as pyzbar
import cv2

cap = cv2.VideoCapture(0) # 내장캠 연결

i = 0
while(cap.isOpened()):
  ret, img = cap.read() #계속 캡쳐를 한다.

  if not ret:
    continue

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
  decoded = pyzbar.decode(gray) # QR코드 찾기

  for d in decoded: 
    x, y, w, h = d.rect #큐알코드 꼭지점찾기

    barcode_data = d.data.decode("utf-8")
    barcode_type = d.type

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    text = '%s (%s)' % (barcode_data, barcode_type)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    #이미지 안에 텍스트를 넣는다(큐알코드 인식했을 때 나오는 데이터값)

  cv2.imshow('img', img)

  key = cv2.waitKey(1)
  if key == ord('q'): #q키를 누르면 종료
    break
  elif key == ord('s'): #s키를 누르면 캡쳐 
    i += 1
    cv2.imwrite('c_%03d.jpg' % i, img)

cap.release()
cv2.destroyAllWindows()