#include <opencv2/opencv.hpp>
//#include <iostream>

using namespace cv;
using namespace std;

// 컬러 -> 흑백2 제공 함수 사용하기
//int main(int argc, char** argv)
//{
//	Mat image, image_gray;
//
//	image = imread("test.jpg", 1);
//
//	if (!image.data) { return -1; }
//
//	/// 흑백으로 변환 cvtColor
//	cvtColor(image, image_gray, CV_RGB2GRAY);
//
//	imshow("Image Original", image);
//	imshow("Image Grayed", image_gray);
//
//	waitKey();
//
//	return 0;
//}


// 컬러 -> 흑백으로1
//int main(int argc, char** argv)
//{
//    /// 이미지 불러오기
//    Mat image = imread("test.jpg", 1);
//    Mat image_gray;
//
//    /// 이미지 복사
//    image.copyTo(image_gray);
//
//    int nRows = image.rows;
//    int nCols = image.cols;
//
//    /// 흑백으로 바꾸기
//    float fGray = 0.0f;
//    float chBlue, chGreen, chRed;
//
//    for (int j = 0; j < nRows; j++) {
//
//        for (int i = 0; i < nCols; i++) {
//
//            chBlue = (float)(image.at<Vec3b>(j, i)[0]); // 파랑
//            chGreen = (float)(image.at<Vec3b>(j, i)[1]); // 초록
//            chRed = (float)(image.at<Vec3b>(j, i)[2]); //빨강
//
//            fGray = 0.2126f * chRed + 0.7152f * chGreen + 0.0722f * chBlue;
//
//            if (fGray < 0.0) fGray = 0.0f;
//            if (fGray > 255.0) fGray = 255.0f;
//
//            image_gray.at<Vec3b>(j, i)[0] = (int)fGray;
//            image_gray.at<Vec3b>(j, i)[1] = (int)fGray;
//            image_gray.at<Vec3b>(j, i)[2] = (int)fGray;
//        }
//    }
//
//    imshow("Image Original", image);
//    imshow("Image Grayed", image_gray);
//
//    waitKey();
//
//    return 0;
//}