#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    Mat img_color;

    img_color = imread("cat.jpg"); //�̹��� �ҷ�����
    imshow("original", img_color);

    Mat img_result;

    //�̹��� Ȯ��, ��� ������ ���� �� �ִ�.
    //���⿡�� ���ι���(fx)���� 2��, ���� ����(fy)���� 2�� Ȯ����
    //�̹��� Ȯ��ÿ��� ���������� INTER_CUBIC �Ǵ� INTER_LINEAR�� ������
    resize(img_color, img_result, Size(), 2, 2, INTER_CUBIC);
    imshow("x2 INTER_CUBIC", img_result);

    int width = img_color.cols;
    int height = img_color.rows;

    //Ȯ�� �� ��ҵǴ� �̹��� ũ�⸦ �ʺ�� ���̷� ������ �� ����. ���⼱ 3�� Ȯ��
    resize(img_color, img_result, Size(3 * width, 3 * height), INTER_LINEAR);
    imshow("x3 INTER_LINEAR", img_result);

    //�̹����� �ʺ�� ���̸� 0.5�� ����
    resize(img_color, img_result, Size(), 0.5, 0.5, INTER_AREA);
    imshow("x0.5 INTER_AREA", img_result);

    waitKey(0);

    return 0;

}