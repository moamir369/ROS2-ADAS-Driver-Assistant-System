
import rclpy
import numpy as np
import cv2

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


class LaneOverlayNode(Node):

    def __init__(self):
        super().__init__('lane_overlay_node')

        self.bridge = CvBridge()


        self.dilation_size = 9
        self.closing_size = 5
        self.overlay_strength = 0.70



        self.image_sub = Subscriber(
            self,
            Image,
            '/front/image_raw'
        )

        self.mask_sub = Subscriber(
            self,
            Image,
            '/lane_detection/mask'
        )


        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.mask_sub],
            queue_size=10,
            slop=0.2
        )

        self.sync.registerCallback(self.callback)

        self.publisher = self.create_publisher(
            Image,
            '/lane_detection/overlay',
            10
        )

        self.get_logger().info(
            'Overlay ready: /front/image_raw + /lane_detection/mask'
        )

        self.get_logger().info(
            'Green lane visualization enabled'
        )

        self.get_logger().info(
            'Publishing to /lane_detection/overlay'
        )



    def callback(self, image_msg, mask_msg):

        stage = 'image conversion'

        try:

            # Convert ROS images to OpenCV
            image = self.bridge.imgmsg_to_cv2(
                image_msg,
                desired_encoding='bgr8'
            )

            mask = self.bridge.imgmsg_to_cv2(
                mask_msg,
                desired_encoding='mono8'
            )

            # Validate image
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    f'Unexpected image shape: {image.shape}'
                )

            if mask.ndim != 2:
                raise ValueError(
                    f'Unexpected mask shape: {mask.shape}'
                )


            stage = 'mask resizing'

            height, width = image.shape[:2]

            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )



            stage = 'mask threshold'

            _, mask = cv2.threshold(
                mask,
                127,
                255,
                cv2.THRESH_BINARY
            )



            stage = 'morphological closing'

            close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.closing_size,
                    self.closing_size
                )
            )

            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                close_kernel
            )



            stage = 'lane dilation'

            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    self.dilation_size,
                    self.dilation_size
                )
            )

            mask = cv2.dilate(
                mask,
                dilation_kernel,
                iterations=1
            )



            stage = 'mask smoothing'

            mask = cv2.GaussianBlur(
                mask,
                (5, 5),
                0
            )


            stage = 'overlay creation'

            alpha = (
                mask.astype(np.float32) / 255.0
            ) * self.overlay_strength


            green = np.zeros_like(image)

            green[:, :, 1] = 255


            result = (
                image.astype(np.float32) *
                (1.0 - alpha[:, :, None])
                +
                green.astype(np.float32) *
                alpha[:, :, None]
            )

            result = np.clip(
                result,
                0,
                255
            ).astype(np.uint8)


            stage = 'message publishing'

            output = Image()

            output.header = image_msg.header

            output.height = int(result.shape[0])
            output.width = int(result.shape[1])

            output.encoding = 'bgr8'
            output.is_bigendian = 0

            output.step = int(result.shape[1] * 3)

            output.data = result.tobytes()

            self.publisher.publish(output)

        except Exception as exc:

            self.get_logger().error(
                f'{stage}: {type(exc).__name__}: {exc}'
            )



def main(args=None):

    rclpy.init(args=args)

    node = LaneOverlayNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

