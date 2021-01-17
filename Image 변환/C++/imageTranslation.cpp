#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Mat img_color;
    img_color = imread("cat.jpg");
    imshow("original", img_color);


    int height = img_color.rows; //행
    int width = img_color.cols; //렬

    //이미지를 오른쪽으로 100, 아래로 50 이동시키는 이동행렬을 만듦.
    Mat M(2, 3, CV_64F, Scalar(0.0));

    M.at<double>(0, 0) = 1;
    M.at<double>(1, 1) = 1;
    M.at<double>(0, 2) = 100;
    M.at<double>(1, 2) = 50;

    //이동 행렬을 이미지에 적용함.
    Mat img_translation;
    warpAffine(img_color, img_translation, M, Size(width, height));

    imshow("translation", img_translation);
    waitKey(0);

}