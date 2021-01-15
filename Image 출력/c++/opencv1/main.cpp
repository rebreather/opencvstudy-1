#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int ac, char** av)
{
	Mat img = imread("test.jpg");

	imshow("img", img);
	waitKey(0);

	return 0;
}