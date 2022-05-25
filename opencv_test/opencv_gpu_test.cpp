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
	//std::string filename = "D:/Code/opencv_test/opencv_test/opencv_test/stars.png";
	std::string filename = "D:/Code/opencv_test/opencv_test/opencv_test/GOES_E_test.png";

	Mat img = imread(filename, IMREAD_GRAYSCALE);


	if (img.empty())
	{
		std::cout << "Could not read the image: " << filename << std::endl;
		return -1;
	}

	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	cv::cuda::GpuMat src, edges, circles;
	src.upload(img);

	cv::Ptr <cv::cuda::CannyEdgeDetector> cannyDetector = cv::cuda::createCannyEdgeDetector(10, 100);
	cannyDetector->detect(src, edges);

	//cv::Ptr < cv::cuda::HoughCirclesDetector> circleDetector = cv::cuda::createHoughCirclesDetector(1, 20, 500, 1, 1, 2);
	//circleDetector->detect(edges, circles);

	cv::Mat result, edge_result;
	//circles.download(result);

	edges.download(edge_result);
	
	//vector<Vec3f> circles_final = edge_result;

	Mat edge_color, img_color;
	//Mat mask;
	
	//note need to mask img not the edges
	//threshold(edge_result, mask, 255, 255, THRESH_BINARY_INV | THRESH_OTSU);
	//edge_result.setTo(Scalar(0, 0, 255), mask);


	applyColorMap(edge_result, edge_color, COLORMAP_PLASMA);
	applyColorMap(img, img_color, COLORMAP_BONE);

	Mat output;
	addWeighted(edge_color,1, img_color,1,0,output);

	cv::imwrite("output.jpg", output);
	cv::imshow("result", output);
	cv::waitKey();


	return 0;
}
