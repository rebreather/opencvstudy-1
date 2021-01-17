#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Mat img_color;

    img_color = imread("cat.jpg"); //이미지 불러오기
    imshow("original", img_color);

    Mat img_result;

    //이미지 확대, 축소 비율을 적을 수 있다.
    //여기에선 가로방향(fx)으로 2배, 세로 방향(fy)으로 2배 확대함
    //이미지 확대시에는 보간법으로 INTER_CUBIC 또는 INTER_LINEAR를 권장함
    resize(img_color, img_result, Size(), 2, 2, INTER_CUBIC);
    imshow("x2 INTER_CUBIC", img_result);

    int width = img_color.cols;
    int height = img_color.rows;

    //확대 및 축소되는 이미지 크기를 너비와 높이로 지정할 수 있음. 여기선 3배 확대
    resize(img_color, img_result, Size(3 * width, 3 * height), INTER_LINEAR);
    imshow("x3 INTER_LINEAR", img_result);

    //이미지의 너비와 높이르 0.5배 줄임
    resize(img_color, img_result, Size(), 0.5, 0.5, INTER_AREA);
    imshow("x0.5 INTER_AREA", img_result);

    waitKey(0);

    return 0;

}