#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MonoCalibrator(Node):
    def __init__(self):
        super().__init__('mono_calibrator')

        # ===== 参数 =====
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('board_cols', 11)          # 内角点列数
        self.declare_parameter('board_rows', 8)           # 内角点行数
        self.declare_parameter('square_size', 0.025)      # 每格边长，单位米
        self.declare_parameter('output_file', 'camera_intrinsics.yaml')
        self.declare_parameter('min_samples', 15)

        self.image_topic = self.get_parameter('image_topic').value
        self.board_cols = int(self.get_parameter('board_cols').value)
        self.board_rows = int(self.get_parameter('board_rows').value)
        self.square_size = float(self.get_parameter('square_size').value)
        self.output_file = self.get_parameter('output_file').value
        self.min_samples = int(self.get_parameter('min_samples').value)

        self.pattern_size = (self.board_cols, self.board_rows)

        self.bridge = CvBridge()
        self.latest_bgr = None
        self.latest_gray = None
        self.image_size = None

        # 世界坐标中的棋盘格角点
        self.objp = np.zeros((self.board_cols * self.board_rows, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_cols, 0:self.board_rows].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.objpoints = []
        self.imgpoints = []

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.timer = self.create_timer(0.03, self.loop)

        self.get_logger().info(f'订阅图像: {self.image_topic}')
        self.get_logger().info('按键说明: s=保存当前样本, c=开始标定, q=退出')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.latest_bgr = frame
            self.latest_gray = gray
            self.image_size = (gray.shape[1], gray.shape[0])
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')

    def find_corners(self, gray):
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if not found:
            return False, None

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )
        return True, corners

    def save_sample(self, corners):
        self.objpoints.append(self.objp.copy())
        self.imgpoints.append(corners.copy())
        self.get_logger().info(f'已保存样本 {len(self.imgpoints)} 张')

    def calibrate(self):
        if len(self.objpoints) < self.min_samples:
            self.get_logger().warn(
                f'样本不足，至少需要 {self.min_samples} 张，当前只有 {len(self.objpoints)} 张'
            )
            return

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.image_size,
            None,
            None
        )

        total_err = 0.0
        for i in range(len(self.objpoints)):
            proj, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(self.imgpoints[i], proj, cv2.NORM_L2) / len(proj)
            total_err += err
        mean_err = total_err / len(self.objpoints)

        self.get_logger().info('========== 标定完成 ==========')
        self.get_logger().info(f'RMS error: {rms}')
        self.get_logger().info(f'Mean reprojection error: {mean_err}')
        self.get_logger().info(f'K =\n{K}')
        self.get_logger().info(f'dist = {dist.ravel()}')

        self.save_yaml(K, dist, mean_err)

    def save_yaml(self, K, dist, mean_err):
        proj = [
            float(K[0, 0]), 0.0, float(K[0, 2]), 0.0,
            0.0, float(K[1, 1]), float(K[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        data = {
            'image_width': int(self.image_size[0]),
            'image_height': int(self.image_size[1]),
            'camera_name': 'orbbec_gemini_330',
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [float(x) for x in K.reshape(-1)]
            },
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {
                'rows': 1,
                'cols': int(dist.size),
                'data': [float(x) for x in dist.reshape(-1)]
            },
            'rectification_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
            },
            'projection_matrix': {
                'rows': 3,
                'cols': 4,
                'data': proj
            },
            'mean_reprojection_error': float(mean_err)
        }

        with open(self.output_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def loop(self):
        if self.latest_bgr is None:
            return

        vis = self.latest_bgr.copy()
        found, corners = self.find_corners(self.latest_gray)

        if found:
            cv2.drawChessboardCorners(vis, self.pattern_size, corners, found)
            text = 'FOUND'
            color = (0, 255, 0)
        else:
            text = 'NOT FOUND'
            color = (0, 0, 255)

        cv2.putText(vis, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis, f'Samples: {len(self.imgpoints)}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.imshow('mono_calibration', vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if found:
                self.save_sample(corners)
            else:
                self.get_logger().warn('当前帧没有检测到角点，不能保存')

        elif key == ord('c'):
            self.calibrate()

        elif key == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main():
    rclpy.init()
    node = MonoCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()