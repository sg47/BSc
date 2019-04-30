#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <std_msgs/Bool.h>


// path to test video
cv::VideoCapture cap("/home/mj/datasets/lprData/20160412-115602.avi");
cv::Mat rgb, gray, scharred_image;


class ImagePubSub{
public:
	ros::Publisher publisher;
	ros::Subscriber sub;
	int i;
	std::vector<cv::Mat> captured;
	void cycleCallback(const std_msgs::BoolConstPtr& cycle_completed){
		if( cap.read(rgb)){
			this->publish(rgb);
		}
        };

	ImagePubSub(ros::NodeHandle nh){
	    	this->publisher = nh.advertise <sensor_msgs::Image> ("/image",1);
		this->i = 0;
    		this->sub = nh.subscribe("/cycle_completed", 1, &ImagePubSub::cycleCallback,this);
	};

	void publish(cv::Mat img){
		cv::Mat rgb;
		cv::resize(img,rgb,cv::Size(600,400));
	  sensor_msgs::ImagePtr rgb_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgb).toImageMsg();
		this->i++;
		this->publisher.publish(rgb_msg);
	};
};

int main(int argc , char **argv){
    ros::init(argc,argv,"camera_node");
    ros::NodeHandle nh;
    ImagePubSub *image_pub_sub = new ImagePubSub(nh);
	ros::spin();
}
