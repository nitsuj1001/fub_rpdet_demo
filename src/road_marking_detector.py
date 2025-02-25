#!/home/schoko/venvs/fub_rpdet/bin/python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PointStamped, Vector3Stamped
import tf2_ros
import tf2_geometry_msgs
import math

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
        
        self.yaw_offset = math.radians(-15) 

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
        """Projects a single pixel coordinate (x, y) onto the ground plane (z=0) using tf2."""
        try:
            # Create a point in camera frame
            point_img = np.array([[x], [y], [1.0]])
            
            # Convert to normalized camera coordinates
            point_cam_normalized = np.linalg.inv(self.camera_matrix) @ point_img
            
            # Create a ray direction vector in camera frame
            ray_dir_cam = np.array([point_cam_normalized[0,0], point_cam_normalized[1,0], 1.0])
            ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
            
            # Create a Vector3Stamped for the ray direction in camera frame
            ray_vec_camera = Vector3Stamped()
            ray_vec_camera.header.frame_id = 'hella_camera'
            ray_vec_camera.header.stamp = rospy.Time(0)
            ray_vec_camera.vector.x = ray_dir_cam[0]
            ray_vec_camera.vector.y = ray_dir_cam[1]
            ray_vec_camera.vector.z = ray_dir_cam[2]
            
            # Transform the ray direction vector from camera frame to base_link frame
            try:
                ray_vec_base = self.tf_buffer.transform(ray_vec_camera, 'base_link', rospy.Duration(0.1))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr("TF Error transforming ray vector: %s", e)
                return None
            
            # Extract the transformed direction vector
            ray_dir_base = np.array([
                ray_vec_base.vector.x,
                ray_vec_base.vector.y,
                ray_vec_base.vector.z
            ])
            
            # Get camera position in base_link frame
            try:
                trans = self.tf_buffer.lookup_transform('base_link', 'hella_camera', rospy.Time(0))
                camera_pos = np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z
                ])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr("TF Error looking up camera position: %s", e)
                return None
            
            rospy.logdebug(f"Camera pos: {camera_pos}")
            rospy.logdebug(f"Ray dir (camera): {ray_dir_cam}")
            rospy.logdebug(f"Ray dir (base): {ray_dir_base}")
            
            # Calculate intersection with ground plane (z=0)
            # If ray z-component is very close to 0 or positive, it won't intersect ground
            if abs(ray_dir_base[2]) < 1e-6:
                rospy.logwarn(f"Ray is parallel to ground plane: z={ray_dir_base[2]}")
                return None
                
            if ray_dir_base[2] > 0:
                rospy.logwarn(f"Ray is pointing away from ground plane: z={ray_dir_base[2]}")
                # For debugging, try inverting the z direction to see if that helps
                ray_dir_base[2] = -ray_dir_base[2]
                rospy.logwarn(f"Inverting ray direction to try to get intersection")
            
            # t is the parameter where the ray intersects the ground plane
            t = -camera_pos[2] / ray_dir_base[2]
            
            # Compute the intersection point
            intersection = [
                camera_pos[0] + t * ray_dir_base[0],
                camera_pos[1] + t * ray_dir_base[1],
                0.0  # z=0 by definition
            ]
            
            # Apply yaw rotation
            c = math.cos(self.yaw_offset)
            s = math.sin(self.yaw_offset)
            x_ = intersection[0] * c - intersection[1] * s
            y_ = intersection[0] * s + intersection[1] * c
            intersection[0], intersection[1] = x_, y_

            rospy.logdebug(f"Intersection point: {intersection}, t={t}")
            return intersection
            
        except Exception as e:
            rospy.logerr(f"Error in project_point_to_ground: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
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
