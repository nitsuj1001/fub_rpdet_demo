#!/home/schoko/venvs/fub_rpdet/bin/python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose
import tf2_ros
import tf2_geometry_msgs

import sys
#sys.path.append()
print("Python-Pfad:", sys.executable)  # Sollte auf die venv zeigen
print("PYTHONPATH:", sys.path)

from ultralytics import YOLO

class RoadMarkingDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Load camera parameters from file
        self.fx = 1525.34
        self.fy = 1525.34
        self.xc = 913.403
        self.yc = 472.811
        self.kappas = [-1.453264, -3.146470]
        self.lambdas = [5.616413, 3.929874]
        self.model = YOLO("/home/schoko/projects/ba_rp_detection/rp_detector/yolo_roadpictogram_detection/settings_set_3/train5/weights/epoch50.pt")
        
        # Initialize camera matrix with loaded parameters
        self.camera_matrix = np.array([
            [self.fx, 0, self.xc],
            [0, self.fy, self.yc],
            [0, 0, 1]
        ])
        
        rospy.loginfo("Initialized camera matrix with parameters from file")
        
        # Publishers
        self.marker_pub = rospy.Publisher('road_markings', MarkerArray, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('sensors/hella/image', Image, self.image_callback)
        rospy.Subscriber('sensors/hella/camera_info', CameraInfo, self.camera_info_callback)

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.logdebug("Camera matrix initialized: \n%s", self.camera_matrix)
            rospy.logdebug("Distortion coefficients: %s", self.dist_coeffs)

    def project_to_ground(self, bbox, image_height, image_width):
        try:
            # Get bottom center of bounding box in image coordinates
            x_center = (bbox[0] + bbox[2]) / 2
            y_bottom = bbox[3]
            
            # Create homogeneous point
            point_img = np.array([[x_center], [y_bottom], [1.0]])
            
            # Convert to normalized camera coordinates
            point_cam = np.linalg.inv(self.camera_matrix) @ point_img
            
            # Get transform from camera to base_link
            try:
                trans = self.tf_buffer.lookup_transform(
                    'base_link',
                    'hella_camera',
                    rospy.Time(0)
                )
                rospy.logdebug("Got camera transform: %s", trans)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr("TF lookup failed: %s", e)
                return None
            
            # Create ray from camera origin through image point
            ray_direction = np.array([point_cam[0,0], point_cam[1,0], 1.0])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            # Camera position in base_link frame
            cam_pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])
            
            # Ground plane parameters (assuming z=0 is ground)
            ground_normal = np.array([0, 0, 1])
            ground_point = np.array([0, 0, 0])
            
            # Compute intersection with ground plane
            d = np.dot(ground_point - cam_pos, ground_normal) / np.dot(ray_direction, ground_normal)
            intersection_point = cam_pos + d * ray_direction
            
            rospy.logdebug("Projected point: %s", intersection_point)
            return intersection_point.tolist()
            
        except Exception as e:
            rospy.logerr("Error in projection: %s", e)
            return None

    def create_ground_marker(self, position, marker_id, frame_id):
        if position is None:
            return None
            
        marker = Marker()
        marker.header.frame_id = 'base_link'  # Changed to base_link
        marker.header.stamp = rospy.Time.now()
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        
        # Set size based on detection type
        marker.scale.x = 1.0  # width
        marker.scale.y = 1.0  # length
        marker.scale.z = 0.1  # height (thin)
        
        # Set color (yellow for road markings)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        return marker

    def image_callback(self, msg):
        if self.camera_matrix is None:
            rospy.logdebug("Waiting for camera matrix...")
            return

        try:
            # Debug the incoming image encoding
            rospy.logdebug("Incoming image encoding: %s", msg.encoding)
            
            # Convert ROS Image to OpenCV image - changed conversion
            if msg.encoding == '8UC3':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                # Convert from BGR to RGB if needed
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            rospy.logdebug("Successfully converted image with shape: %s", cv_image.shape)
            
            # Run YOLO detection here            
            results = self.model.predict(source=cv_image, imgsz=640, conf=0.1)  # Replace with your YOLO implementation
            # For testing, let's create a dummy detection
            #dummy_detection = [100, 100, 200, 200]  # [x1, y1, x2, y2]
            #detections = [dummy_detection]
            rospy.logdebug("Found %d detections", len(results))
            rospy.logdebug(results)
            
            marker_array = MarkerArray()
            
            result = results[0]

            # Extract boxes from results
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    coords = box.xyxy[0].cpu().numpy()  # Get bbox coordinates
                    bbox = [int(coord) for coord in coords]
                    
                    ground_point = self.project_to_ground(bbox,
                                                        cv_image.shape[0],
                                                        cv_image.shape[1])
                    
                    if ground_point is not None:
                        marker = self.create_ground_marker(ground_point, i, 'base_link')
                        if marker is not None:
                            marker_array.markers.append(marker)
            else:
                rospy.logdebug("No detections found")
                
            rospy.logdebug("Publishing %d markers", len(marker_array.markers))
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

def main():
    # Enable debug logging
    rospy.init_node('road_marking_detector', log_level=rospy.DEBUG)
    rospy.loginfo("Starting Road Marking Detector node")
    detector = RoadMarkingDetector()
    rospy.spin()

if __name__ == '__main__':
    main()
