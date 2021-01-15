#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img_color = imread("label.jpg", 1);
	Mat img_gray;
	cvtColor(img_color, img_gray, COLOR_BGR2GRAY);
	imshow("result", img_gray);
	waitKey(0);

	// 엣지를 검출 검출된 엣지부분만 흰색
	Mat img_edge;
	Canny(img_gray,img_edge,50, 150);
	imshow("result", img_edge);
	waitKey(0);

	// 관심물체가 흰색이 되어야 하므로 반전
	bitwise_not(img_edge,img_edge);
	imshow("result", img_edge);
	waitKey(0);

	// 컨투어를 찾아서 외곽선을 보강
	vector<vector<Point>> contours;
	findContours(img_edge.clone(), contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
	drawContours(img_edge, contours, -1, Scalar(0, 0, 0), 1);
	imshow("result", img_edge);
	waitKey(0);

	// 흰색 영역에 대해 라벨링
	Mat labels, stats, centroids;
	int nlabels = connectedComponentsWithStats(img_edge, labels, stats, centroids);

	for (int i = 0; i < nlabels; i++)
	{
		// 배경을 제외
		if (i < 2) continue;
		int area = stats.at<int>(i, CC_STAT_AREA);
		int center_x = centroids.at<double>(i, 0);
		int center_y = centroids.at<double>(i, 1);
		int left = stats.at<int>(i, CC_STAT_LEFT);
		int top = stats.at<int>(i, CC_STAT_TOP);
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		int height = stats.at<int>(i, CC_STAT_HEIGHT);

		if (area > 50)
		{
			// 영역 외곽에 사각형을 그립니다.
			rectangle(img_color, Point(left, top), Point(left + width, top + height), Scalar(0, 0, 255), 1);
			// 영역의 중심에 원을 그립니다.
			circle(img_color, Point(center_x, center_y), 5, (255, 0, 0), 1);
			// 라벨 번호 표시
			putText(img_color, std::to_string(i), Point(left + 20, top + 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
		}
	}

	// 결과창
	imshow("result", img_color);
	waitKey(0);
}