#include<opencv2/opencv.hpp>

using namespace cv;

int main()
{
	Mat img_color;
	img_color = imread("test.jpg", IMREAD_COLOR);
	imshow("color", img_color);

	int height = img_color.rows; //행
	int width = img_color.cols; //렬

	//이미지 중앙을 중심으로 반시계 방향으로 45도 회전하는 행렬을 생성.
	Mat M = getRotationMatrix2D(Point(width / 2.0, height / 2.0), //회전 할 때 중심점
		90, // 회전 각도(양수는 반시계 방향, 음수는 시계 방향
		1); //이미지 배율. 1이면 원래 크기

	//회전 행렬 M을 이미지 img_color에 적용한다.
	Mat img_rotated;
	warpAffine(img_color, img_rotated, M, Size(width, height));
	imshow("rotation", img_rotated);
	waitKey(0);

	return 0;
}