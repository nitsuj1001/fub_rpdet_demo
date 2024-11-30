#!/usr/bin/env python
import rospy 
from std_msgs.msg import String
from sensor_msgs.msg import Image

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

def process_image(data):
    rospy.loginfo(rospy.get_caller_id() + "H:%s | W:%s | Header:%s | Encoding: %s", data.height, data.width, data.header, data.encoding)
    rospy.loginfo(rospy.get_caller_id() + "Data: step (row length in bytes) - %s", data.step)

    try:
        height, width = data.height, data.width        
        # Konvertiere ROS-Bildnachricht in OpenCV-Bild
        img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        
        # Bild verarbeiten oder speichern
        cv2.imwrite("result.jpg", img)
        rospy.loginfo("Image saved successfully as result.jpg")
        rospy.loginfo("Image Shape: %s", img.shape)
        rospy.loginfo(img[0])

        # Speichere graues Bild
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("result_gray.jpg", img_gray)
        rospy.loginfo(img_gray.shape)
        
        # Schneide Halbkreis aus
        mask = np.zeros((height, width), dtype=np.uint8)

        center = (width // 2, height)
        radius = int(width * 0.5)
        cv2.circle(mask, center, radius, 255, -1)

        mask_inverted = cv2.bitwise_not(mask)

        output_image = img_gray.copy()
        output_image[mask_inverted == 255] = [0]
        
        cv2.imwrite('result_masked.jpg', output_image)
                      
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