#include<opencv2/opencv.hpp>

using namespace cv;

int main()
{
	Mat img_color;
	img_color = imread("test.jpg", IMREAD_COLOR);
	imshow("color", img_color);

	int height = img_color.rows; //��
	int width = img_color.cols; //��

	//�̹��� �߾��� �߽����� �ݽð� �������� 45�� ȸ���ϴ� ����� ����.
	Mat M = getRotationMatrix2D(Point(width / 2.0, height / 2.0), //ȸ�� �� �� �߽���
		90, // ȸ�� ����(����� �ݽð� ����, ������ �ð� ����
		1); //�̹��� ����. 1�̸� ���� ũ��

	//ȸ�� ��� M�� �̹��� img_color�� �����Ѵ�.
	Mat img_rotated;
	warpAffine(img_color, img_rotated, M, Size(width, height));
	imshow("rotation", img_rotated);
	waitKey(0);

	return 0;
}