#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "dirent.h"
using namespace cv;
using namespace std;

Rect enlargeROI(Mat frm, Rect boundingBox, int padding) //function to increase or decrease the bounding box 
{ 
    Rect returnRect = Rect(boundingBox.x - padding, boundingBox.y - padding, boundingBox.width + (padding * 2), boundingBox.height + (padding * 2));
    if (returnRect.x < 0)returnRect.x = 0;
    if (returnRect.y < 0)returnRect.y = 0;
    if (returnRect.x+returnRect.width >= frm.cols)returnRect.width = frm.cols-returnRect.x;
    if (returnRect.y+returnRect.height >= frm.rows)returnRect.height = frm.rows-returnRect.y;
    return returnRect;
}
vector<Mat> target(Mat src)
		{
        vector<Mat> retimg;
        Mat __frame;
        src.copyTo(__frame);
      	Mat dst, grey;
      	cv::cvtColor(__frame, dst, CV_BGR2HSV);
      	Scalar avg_color = mean(dst);

      	Mat newimg(src.rows,src.cols, CV_8UC3, avg_color);  // image with the mean colour of the original image
      	Mat final;
        Mat test2;
        cvtColor(newimg,test2,CV_HSV2BGR);
      	cv::subtract(dst, newimg, final);                   // Subtracting the mean colour from the original image 
      	//imwrite("/home/sampledet.jpg",test2);
      	cv::cvtColor(final, final, CV_HSV2BGR);
      	cv::cvtColor(final, final, CV_BGR2GRAY);

      	vector<vector<Point> > contours;
      	vector<Rect> bboxes;
      	Ptr<MSER> mser = MSER::create(5,100,30000,0.1, 0.2, 200, 1.01, 0.002, 5);  //MSER fuction defination
      	mser->detectRegions(final, contours, bboxes);
      	Mat crop2, crop, mask;
        cout<<bboxes.size()<<"\n";
        //int count=0;
      	for(unsigned int i=0;i<bboxes.size();i++)            // To remove overlapping bounding boxes
      			  {
      			   bboxes[i]=enlargeROI(__frame,bboxes[i],30);
      			   if(i>0)
      			   {
      				   if((abs(bboxes[i].width-bboxes[i-1].width)<10)&&(abs(bboxes[i].height-bboxes[i-1].height)<10))
      				   continue;

      			   }

      		
      		__frame(bboxes[i]).clone().copyTo(crop2);
      		Mat img1,test;
      		
      		cv::medianBlur(crop2,crop,5);
      		cv::cvtColor(crop,test, CV_BGR2HSV);
      		std::vector<cv::Mat> channels;
      		cv::split(test, channels);
      		channels[2].copyTo(img1);
      		

          medianBlur(img1,img1,7);
      		Canny(img1, img1, 10, 150,3,true);
      		imwrite("/home/canny.jpg",img1);
      		dilate(img1,img1, Mat(), Point(-1,-1));
      	  erode(img1,img1,Mat(),Point(-1,-1));
      		
      		vector<Vec4i> lines;
      		Mat img2;
      		img1.copyTo(img2);
      		vector<vector<Point> >contours1;
      		
          // Extracting only the target from the background

          findContours(img1,contours1,lines, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE,Point(0,0));
      		Mat mask = Mat::zeros(img1.rows, img1.cols, CV_8UC1);
      		vector<double> areas(contours1.size());
      		for (unsigned int i = 0; i < contours1.size(); i++)
      			areas[i] = contourArea(Mat(contours1[i]));
      		double max;
      		Point maxPosition;
      		cv::minMaxLoc(Mat(areas), 0, &max, 0, &maxPosition);

      		drawContours(mask, contours1, maxPosition.y, Scalar(1), CV_FILLED);
      		Mat crop1(crop.rows, crop.cols, CV_8UC3,Scalar(0));
      		//crop1.setTo(Scalar(0, 0, 0));
      		crop.copyTo(crop1, mask);
      		cv::normalize(mask.clone(), mask, 0.0, 255.0, CV_MINMAX, CV_8UC1);
      		int k= countNonZero(mask);
      		
          // Reject extracted objects that are too small 
          if(k<100)        // number of minimum non black pixels that have to exist for it to be a valid object 
      			continue;
      		double minVal;
      		double maxVal;
      		minMaxLoc(crop1, &minVal, &maxVal);
      		if(maxVal==0)
      			continue;
      		retimg.push_back(crop1);


}
		
      	return retimg;
}
int main()
{   vector<Mat> output;
    Mat src;
    int i;

    //Loop to read in images, If images are numbered 
    for(i=0;i<=1237;i++)
    {
	  src = imread(format("/home/%d.JPG",i));
	  if(!src.data)
		continue;
    output=target(src);
    cout<<output.size()<<"\n";
    if(output.size()!=0)
    {
    for(unsigned int k=0;k<output.size();k++)
    imwrite(format("/home/fff%d.jpg",i*10+k),output[k]);
    }}

 //waitKey(0);
  return 0;
}
