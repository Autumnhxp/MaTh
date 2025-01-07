# subscriber.py
import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image 

import zmq
import json
import time

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

# 假設您的模型模組在此處引入
# from your_model_module import YourModelClass

def is_rotation_exceeding_threshold(R2, threshold_deg=30):

    """
    判断旋转矩阵 R2 与 R1 在 x 轴方向的绝对角度差是否超过阈值。
    """
    R1 = np.array([[0.0, 0.0, 1.0],
                   [0.0, -1.0, 0.0],
                   [1.0, 0.0, 0.0]])
    # 转换阈值为弧度
    threshold_rad = np.deg2rad(threshold_deg)
    
    # 提取 R1 和 R2 的 x 轴方向的向量
    x_axis_R1 = R1[:, 0]  # R1 的第一列，表示 x 轴方向
    x_axis_R2 = R2[:, 0]  # R2 的第一列，表示 x 轴方向

    # 计算 x 轴向量之间的夹角
    dot_product = np.dot(x_axis_R1, x_axis_R2)  # 点积
    cos_theta = np.clip(dot_product, -1.0, 1.0)  # 防止数值误差
    theta = np.arccos(cos_theta)  # 夹角（弧度制）

    # 判断是否超过阈值
    return theta > threshold_rad  # 返回是否超过阈值
    

class VisionModel:
    def __init__(self):
        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()

        # camera data
        self.width = 1280
        self.height = 720
        self.focal_length = 24 # mm
        self.horiz_aperture = 20.955 # mm

        # get camera intrinsics
        self.fx = (self.focal_length/self.horiz_aperture) * self.width
        self.fy = self.fx # should be same in omniverse
        self.cx, self.cy = self.width/2, self.height/2 # width is 1280 and height is 720
        self.scale = 1 # for omniverse camera, since the distance unit of omniverse is also meter
    
        # set workspace to filter output grasps
        self.xmin, self.xmax = -0.2, 0.5
        self.ymin, self.ymax = -0.5, 0.5
        self.zmin, self.zmax = 0.6, 1.0

        self.lims = [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]
    
    def process(self, rgb_path, distance_path):
        # 模擬模型處理邏輯：返回随機的位移和旋轉矩陣
        try:
            colors = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
            depths = np.load(distance_path)
        except Exception as e:
            print(f"Error loading depth file {distance_path}: {e}")

        # get point cloud
        xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depths / self.scale
        points_x = (xmap - self.cx) / self.fx * points_z
        points_y = (ymap - self.cy) / self.fy * points_z

        # set your workspace to crop point cloud
        mask = (points_z > 0) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = colors[mask].astype(np.float32)

        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=self.lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
            return {"translation": None, "rotation": None}

        gg = gg.nms().sort_by_score()

        #rect_gg = gg.to_rect_grasp_group()
        gg_pick = gg[0:20]

        valid_grasp=[]
        for i in gg_pick:
            if not is_rotation_exceeding_threshold(i.rotation_matrix):
                valid_grasp.append({
                    "translation": i.translation.tolist(),
                    "rotation": i.rotation_matrix.tolist()
                })
        if valid_grasp:
            return valid_grasp[0]
        else:
            print(f"No valid grasp after filtering")
            return{"translation": None, "rotation": None}

    
def convert_to_serializable(obj):
    """
    递归将字典中的 numpy.ndarray 转换为列表
        """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # 转换为标准 Python 列表
    else:
        return obj


def main():
    # 初始化模型
    model = VisionModel()

    context = zmq.Context()

    # SUB 套接字：接收來自橋接節點的 ROS 2 資料
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5555")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, '')  # 訂閱所有消息

    # PUB 套接字：將處理後的資料發送回橋接節點
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5556")  # 綁定到本地端口 5556

    print("Subscriber is up and listening for messages...")

    # 等待 PUB 套接字綁定完成
    time.sleep(1)

    take_foto_wp_data = []
    file_paths_data = {}
    simulation_cycle = 0
    grasp_results = {}
    previous_file_paths_nonempty = False  # 記錄上一次 file_paths 是否非空  
    try:
        while True:
            try:
                message = sub_socket.recv_string()
                data = json.loads(message)

                # 根据 topic 区分消息
                if data['topic'] == 'sm_state_wp':
                    print(f"Received from sm_state_wp: {data['data']}")
                elif data['topic'] == 'take_foto_wp':
                    take_foto_wp_data = data['data']
                    print(f"Received from take_foto_wp: {data['data']}")
                elif data['topic'] == 'file_paths':
                    file_paths_data = data['data']
                    print(f"Received from file_paths: {data['data']}")

                    # 檢查是否由非空變為空，標誌新循環開始
                    if previous_file_paths_nonempty and not file_paths_data:
                        simulation_cycle += 1
                        print(f"New simulation cycle detected: {simulation_cycle}")
                        grasp_results = {}  # 重置握取結果

                    previous_file_paths_nonempty = bool(file_paths_data)  # 更新變量

                    # 如果 take_foto_wp 全為 1，且尚未生成握取姿勢，則開始處理環境資料
                    if file_paths_data!={} and grasp_results=={}:
                        env_count = len(take_foto_wp_data)
                        print(f"Detected {env_count} environments.")

                        # 為每個環境生成握取姿勢
                        for env_id, paths in file_paths_data.items():
                            rgb_path = paths.get('rgb')
                            distance_path = paths.get('distance_to_image_plane')
                            if not rgb_path or not distance_path:
                                print(f'Missing data for environment {env_id}. Skipping...')
                                continue
                            # 使用視覺模型生成握取姿勢
                            grasp_result = model.process(rgb_path, distance_path)
                            grasp_results[env_id] = grasp_result

                    # 打印握取結果
                    print(f"Grasp results for cycle {simulation_cycle}: {grasp_results}")

                    # 將握取結果發布回標控節點
                    response = {
                        "cycle": simulation_cycle,
                        "grasp_results": convert_to_serializable(grasp_results)
                    }
                    pub_socket.send_string(json.dumps(response))
                    print(f"Sent grasp results for cycle {simulation_cycle} back to bridge node.")

                else:
                    print(f"Unknown topic: {data['topic']}")
    
            except json.JSONDecodeError:
                print("Error: Received message is not valid JSON.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        print("Shutting down subscriber.")
        
    finally:
        sub_socket.close()
        pub_socket.close()
        context.term()

if __name__ == '__main__':
    main()
    