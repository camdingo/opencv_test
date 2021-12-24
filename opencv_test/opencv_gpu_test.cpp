// opencv_test.cpp : Defines the entry point for the application.
//

#include <opencv2\core\mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <iostream>

using namespace std;
using namespace cv;

int main()
{

	//std::string filename = "D:/Code/opencv_test/opencv_test/opencv_test/20213570306_GOES16-ABI-CONUS-07-5000x3000.jpg";
	//std::string filename = "D:/Code/opencv_test/opencv_test/opencv_test/20213570501_GOES16-ABI-CONUS-07-2500x1500.jpg";
	//std::string filename = "D:/Code/opencv_test/opencv_test/opencv_test/blackmarble_2016_americas_composite.png";
	//std::string filename = "D:/Code/opencv_test/opencv_test/opencv_test/shapes.png";
	std::string filename = "D:/Code/opencv_test/opencv_test/opencv_test/stars.png";

	Mat img = imread(filename, IMREAD_GRAYSCALE);


	if (img.empty())
	{
		std::cout << "Could not read the image: " << filename << std::endl;
		return -1;
	}

	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	cv::cuda::GpuMat src, edges, circles;
	src.upload(img);

	cv::Ptr <cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(500, 1000);
	cannyDetector->detect(src, edges);

	cv::Ptr < cv::cuda::HoughCirclesDetector> circleDetector = cv::cuda::createHoughCirclesDetector(1, 20, 500, 1, 1, 2);
	circleDetector->detect(src, circles);

	cv::Mat result, edge_result;
	circles.download(result);

	edges.download(edge_result);
	
	vector<Vec3f> circles_final = result;
	//Mat ouput_noImg;
	for (size_t i = 0; i < circles_final.size(); i++)
	{
		Vec3i c = circles_final[i];
		Point center = Point(c[0], c[1]);
		std::cout << center << std::endl;
		// circle center
		circle(img, center, 1, Scalar(0, 100, 100), 3, LINE_AA);
		// circle outline
		int radius = c[2];
		circle(img, center, radius, Scalar(255, 0, 255), 3, LINE_AA);
	}

	cv::imshow("result", img);
	cv::waitKey();

	return 0;
}
