#!/usr/bin/env python
import os
import rospy 
from std_msgs.msg import String
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

IMG_PATH = "/media/schoko/BULK/BA/rosbag_extract/biglap3/"
EXTRACT_ORIGINAL = True
EXTRACT_GRAY = False
EXTRACT_MASK = False

os.makedirs(IMG_PATH, exist_ok=True)

def process_image(data):
    rospy.loginfo(rospy.get_caller_id() + " H:%s | W:%s | Seq:%s | Encoding:%s", data.height, data.width, data.header.seq, data.encoding)    

    try:
        height, width = data.height, data.width 
        # ROS Image to OpenCV Image       
        img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")                    

        # save original image
        if (EXTRACT_ORIGINAL):            
            path_orig = IMG_PATH + str(data.header.seq) + "_original.jpg"
            cv2.imwrite(path_orig, img)
            rospy.loginfo("Original image [%s] saved successfully with shape: %s", path_orig, img.shape)
        
        # save gray image
        if (EXTRACT_GRAY):
            path_gray = IMG_PATH + str(data.header.seq) + "_gray.jpg"
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(path_gray, img_gray)
            rospy.loginfo("Gray image [%s] saved successfully with shape: %s", path_gray, img_gray.shape)
        
        if (EXTRACT_MASK):
            mask = np.zeros((height, width), dtype=np.uint8)

            center = (width // 2, height)
            radius = int(width * 0.5)
            cv2.circle(mask, center, radius, 255, -1)

            mask_inverted = cv2.bitwise_not(mask)

            masked_image = img_gray.copy()
            masked_image[mask_inverted == 255] = [0]
                        
            path_masked = IMG_PATH + str(data.header.seq) + "_masked.jpg"
            cv2.imwrite(path_masked, masked_image)
            rospy.loginfo("Masked image saved successfully with shape: %s", path_masked, masked_image.shape)
                      
    except CvBridgeError as e:
        rospy.logerr(f"Failed to convert image: {e}")  
    
def image_processing_loop():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('image_processor', anonymous=True)

    rospy.Subscriber("sensors/hella/image", Image, process_image)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
  
if __name__ == '__main__':
    image_processing_loop()