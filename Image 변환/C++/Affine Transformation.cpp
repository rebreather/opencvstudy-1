#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Point2f src[3];
int idx = 0;
Mat img_color;

void mouse_callback(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN) //마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
	{
		src[idx] = Point2f(x, y);
		idx++;

		cout << "(" << x << ", " << y << ")" << endl;

		circle(img_color, Point(x, y), 3, Scalar(0, 0, 255), -1);
	}
}


int main()
{
	//마우스 콜백함수를 등록
	namedWindow("source");
	setMouseCallback("source", mouse_callback);

	img_color = imread("test.jpg"); //사용할 이미지 불러오기


	while (1) { //반복하면서 마우스 클릭으로 세 점을 지정하도록 함.

		imshow("source", img_color);

		if (waitKey(1) == 32) //스페이스바 누르면 루프에서 빠져나옴
			break;
	}


	int height = img_color.rows;
	int width = img_color.cols;

	//오른쪽 상단의 대응점만 Y좌표가 아래로 100 이동하도록 지정.
	Point2f dst[3];
	dst[0] = src[0];
	dst[1] = Point2f(src[1].x, src[1].y + 100);
	dst[2] = src[2];

	//아핀 변환 생성
	Mat M;
	M = getAffineTransform(src, dst);


	Mat img_result; //아핀변환 적용
	warpAffine(img_color, img_result, M, Size(width, height));


	//결과
	imshow("result", img_result);
	waitKey(0);

	return 0;
}
