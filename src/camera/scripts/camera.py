#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

cam = cv2.VideoCapture('/home/taher/workspace/BSc/3net/test/20190419_121630.mp4')
bridge = CvBridge()

global image_pub;

def callback(data):
    global image_pub;
    ret_val, img = cam.read()
    print('image sent to estimator')
    cv_image = bridge.cv2_to_imgmsg(img, "bgr8")
    image_pub.publish(cv_image)

def main():
    global image_pub;
    image_pub = rospy.Publisher("/camera/rgb",Image, queue_size=10)
    image_sub = rospy.Subscriber("/cycle_completed",Bool, callback)

if __name__ == '__main__':
    rospy.init_node('image_converter', anonymous=True)
    main()
    rospy.spin()
