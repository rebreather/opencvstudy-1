#include <opencv2/opencv.hpp>
//#include <iostream>

using namespace cv;
using namespace std;

// �÷� -> ���2 ���� �Լ� ����ϱ�
//int main(int argc, char** argv)
//{
//	Mat image, image_gray;
//
//	image = imread("test.jpg", 1);
//
//	if (!image.data) { return -1; }
//
//	/// ������� ��ȯ cvtColor
//	cvtColor(image, image_gray, CV_RGB2GRAY);
//
//	imshow("Image Original", image);
//	imshow("Image Grayed", image_gray);
//
//	waitKey();
//
//	return 0;
//}


// �÷� -> �������1
//int main(int argc, char** argv)
//{
//    /// �̹��� �ҷ�����
//    Mat image = imread("test.jpg", 1);
//    Mat image_gray;
//
//    /// �̹��� ����
//    image.copyTo(image_gray);
//
//    int nRows = image.rows;
//    int nCols = image.cols;
//
//    /// ������� �ٲٱ�
//    float fGray = 0.0f;
//    float chBlue, chGreen, chRed;
//
//    for (int j = 0; j < nRows; j++) {
//
//        for (int i = 0; i < nCols; i++) {
//
//            chBlue = (float)(image.at<Vec3b>(j, i)[0]); // �Ķ�
//            chGreen = (float)(image.at<Vec3b>(j, i)[1]); // �ʷ�
//            chRed = (float)(image.at<Vec3b>(j, i)[2]); //����
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