

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LaneCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 96)

        self.up3 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec3 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 16)
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1)


class LaneCNNNode(Node):
    def __init__(self):
        super().__init__('lane_cnn_node')

        self.declare_parameter('image_topic', '/front/image_raw')
        self.declare_parameter(
            'model_path',
            str(Path.home() / 'adas_training' / 'best_lane_cnn.pth')
        )
        self.declare_parameter('threshold', 0.5)

        image_topic = self.get_parameter('image_topic').value
        model_path = Path(self.get_parameter('model_path').value)
        self.threshold = float(self.get_parameter('threshold').value)

        if not model_path.is_file():
            raise FileNotFoundError(f'Model not found: {model_path}')

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        checkpoint = torch.load(
            str(model_path), map_location=self.device, weights_only=True
        )

        self.model = LaneCNN().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image, '/lane_detection/mask', 10
        )
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        self.get_logger().info(f'Model loaded from: {model_path}')
        self.get_logger().info(f'Inference device: {self.device}')
        self.get_logger().info(f'Subscribed to: {image_topic}')
        self.get_logger().info('Publishing masks to: /lane_detection/mask')

    @torch.inference_mode()
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
            resized = cv2.resize(frame, (640, 360))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0

            tensor = torch.from_numpy(
                rgb.transpose(2, 0, 1).copy()
            ).unsqueeze(0).to(self.device)

            logits = self.model(tensor)
            probabilities = torch.sigmoid(logits)[0, 0]
            mask = (
                (probabilities > self.threshold)
                .to(torch.uint8).mul(255).cpu().numpy()
            )

            output_msg = self.bridge.cv2_to_imgmsg(
                mask, encoding='mono8'
            )
            output_msg.header = msg.header
            self.publisher.publish(output_msg)

        except Exception as exc:
            self.get_logger().error(f'Image processing failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LaneCNNNode()
        rclpy.spin(node)
    except Exception as exc:
        print(f'Lane CNN node failed: {exc}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

