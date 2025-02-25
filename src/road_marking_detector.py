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

from ultralytics import YOLO

class RoadMarkingDetector:
    def __init__(self, model_path):
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Camera parameters
        self.fx = 1525.34
        self.fy = 1525.34
        self.xc = 913.403
        self.yc = 472.811
        self.kappas = [-1.453264, -3.146470]
        self.lambdas = [5.616413, 3.929874]
        self.model = YOLO(model_path)
        
        # Initialize camera matrix with parameters
        self.camera_matrix = np.array([
            [self.fx, 0, self.xc],
            [0, self.fy, self.yc],
            [0, 0, 1]
        ])
        
        rospy.loginfo("Initialized camera matrix with parameters")
        
        # Publishers
        self.marker_pub = rospy.Publisher('road_markings', MarkerArray, queue_size=10)
        self.image_pub = rospy.Publisher('road_markings/detection_image', Image, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('sensors/hella/image', Image, self.image_callback)
        rospy.Subscriber('sensors/hella/camera_info', CameraInfo, self.camera_info_callback)
        
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            rospy.logdebug("Camera matrix initialized: \n%s", self.camera_matrix)
            rospy.logdebug("Distortion coefficients: %s", self.dist_coeffs)

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        # Extract quaternion components
        x, y, z, w = q.x, q.y, q.z, q.w
        
        # Calculate rotation matrix elements
        r00 = 1 - 2*y*y - 2*z*z
        r01 = 2*x*y - 2*w*z
        r02 = 2*x*z + 2*w*y
        
        r10 = 2*x*y + 2*w*z
        r11 = 1 - 2*x*x - 2*z*z
        r12 = 2*y*z - 2*w*x
        
        r20 = 2*x*z - 2*w*y
        r21 = 2*y*z + 2*w*x
        r22 = 1 - 2*x*x - 2*y*y
        
        return np.array([
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22]
        ])

    def project_point_to_ground(self, x, y):
        """Projects a single pixel coordinate (x, y) onto the ground plane (z=0)."""
        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link',
                'hella_camera',
                rospy.Time(0)
            )

            # Homogeneous coordinate
            point_img = np.array([[x], [y], [1.0]])
            
            # Convert to normalized camera coordinates
            point_cam = np.linalg.inv(self.camera_matrix) @ point_img

            # Camera position in base_link frame
            cam_pos = np.array([
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ])

            # In camera frame, the ray direction should point forward (+z in camera frame)
            ray_direction_cam = np.array([point_cam[0,0], point_cam[1,0], 1.0])
            ray_direction_cam = ray_direction_cam / np.linalg.norm(ray_direction_cam)
            
            # Get rotation matrix
            rotation_matrix = self.quaternion_to_rotation_matrix(trans.transform.rotation)
            
            # Transform ray direction to base_link frame
            # Note: We need to invert the rotation because we're going from camera to base_link
            ray_direction = rotation_matrix.T @ ray_direction_cam

            # Ground plane parameters (z=0)
            ground_normal = np.array([0, 0, 1])
            ground_point = np.array([0, 0, 0])

            # Calculate intersection with ground plane
            denominator = np.dot(ray_direction, ground_normal)
            if abs(denominator) < 1e-6:
                rospy.logwarn("Ray is parallel to ground plane")
                return None

            d = -np.dot(cam_pos, ground_normal) / denominator
            
            if d < 0:
                rospy.logwarn("Ray points away from ground plane")
                return None

            intersection_point = cam_pos + d * ray_direction

            # Debug output
            rospy.logdebug(f"Camera position: {cam_pos}")
            rospy.logdebug(f"Ray direction (camera frame): {ray_direction_cam}")
            rospy.logdebug(f"Ray direction (base frame): {ray_direction}")
            rospy.logdebug(f"Intersection point: {intersection_point}")
            
            return intersection_point.tolist()

        except Exception as e:
            rospy.logerr("Error in project_point_to_ground: %s", e)
            return None

    def create_ground_bbox_marker(self, corners_3d, marker_id, frame_id):
        """Creates a LINE_STRIP marker representing a rectangle on the ground plane."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "road_markings"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Each corner is appended to the marker's points.
        # The first corner is appended at the end again to close the rectangle.
        for corner in corners_3d:
            if corner is not None:
                p = Point()
                p.x = corner[0]
                p.y = corner[1]
                p.z = corner[2]
                marker.points.append(p)
        # Close the loop by repeating the first point if it exists
        if len(marker.points) > 0:
            marker.points.append(marker.points[0])

        # Marker scale and color
        marker.scale.x = 0.05  # line width
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.Duration(0.5)
        return marker

    def image_callback(self, msg):
        if self.camera_matrix is None:
            rospy.logdebug("Waiting for camera matrix...")
            return

        try:
            rospy.logdebug("Incoming image encoding: %s", msg.encoding)

            # Convert image
            if msg.encoding == '8UC3':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            rospy.logdebug("Successfully converted image with shape: %s", cv_image.shape)

            # YOLO detection
            results = self.model.predict(source=cv_image, imgsz=640, conf=0.1, device='cuda')

            # Visualization image (BGR)
            visualization_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            marker_array = MarkerArray()
            result = results[0]
            rospy.logdebug("Found %d detections", len(result))

            # Extract bounding boxes
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    coords = box.xyxy[0].cpu().numpy()
                    bbox = [int(coord) for coord in coords]

                    # Debug bounding box
                    rospy.logdebug(f"Processing bbox: {bbox}")

                    # Draw 2D bounding box on visualization image
                    cv2.rectangle(
                        visualization_image,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),
                        2
                    )

                    # Class name and confidence
                    if hasattr(box, 'cls'):
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        conf = float(box.conf[0])
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(
                            visualization_image,
                            label,
                            (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )

                    # Project all four corners of the bounding box
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    # top-left
                    tl = self.project_point_to_ground(x1, y1)
                    # top-right
                    tr = self.project_point_to_ground(x2, y1)
                    # bottom-right
                    br = self.project_point_to_ground(x2, y2)
                    # bottom-left
                    bl = self.project_point_to_ground(x1, y2)

                    # Create a ground bounding box marker
                    corners_3d = [tl, tr, br, bl]
                    marker = self.create_ground_bbox_marker(corners_3d, i+1, 'base_link')
                    marker_array.markers.append(marker)

            # Publish markers
            rospy.logdebug(f"Publishing MarkerArray with {len(marker_array.markers)} markers")
            self.marker_pub.publish(marker_array)

            # Publish visualization image
            try:
                vis_msg = self.bridge.cv2_to_imgmsg(visualization_image, encoding='bgr8')
                vis_msg.header = msg.header
                self.image_pub.publish(vis_msg)
            except Exception as e:
                rospy.logerr(f"Error publishing visualization image: {e}")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")


def main():
    rospy.init_node('road_marking_detector', log_level=rospy.DEBUG)
    rospy.loginfo("Starting Road Marking Detector node")
    RoadMarkingDetector("/home/schoko/projects/ba_rp_detection/rp_detector/yolo_roadpictogram_detection/settings_set_3/train5/weights/epoch50.pt")
    rospy.spin()

if __name__ == '__main__':
    main()
