import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from robotExp.simulator.pseudoSlam import pseudoSlam
import pyastar
import pyastar_contour

WALL_COLOR = 1
UNKNOWN_COLOR = 127
FREE_COLOR = 255
ROBOT_COLOR = 50


def FPS(points, n_samples):
    points = np.array(points)
    points_left = np.arange(len(points))
    sample_inds = np.zeros(n_samples, dtype='int') 
    dists = np.ones_like(points_left) * float('inf')

    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected)

    for i in range(1, n_samples):
        last_added = sample_inds[i-1]
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1)
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left])
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)
    return sample_inds


def find_contour(sparse_k, map, pose): 
    global_size_x = map.shape[0]
    global_size_y = map.shape[1]

    contours_F = []
    contours_W = []
    img = map.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    maze = cv2.erode(img, kernel, 3)

    img[img == UNKNOWN_COLOR] = WALL_COLOR
    grid = cv2.erode(img, kernel, 3)
    grid = grid.astype(np.float32)
    grid[grid == WALL_COLOR] = np.inf
    grid[grid == UNKNOWN_COLOR] = np.inf
    grid[grid == ROBOT_COLOR] = 1
    grid[grid == FREE_COLOR] = 1
    assert grid.min() == 1, "cost of moving must be at least 1"

    start = np.array([int(pose[0]), int(pose[1])])
    end = np.array([int(0.5*global_size_x), int(0.5*global_size_y)])
    path = pyastar_contour.astar_path(grid, start, end, allow_diagonal=True) 
    pointcloud = path

    for point_i in range(pointcloud.shape[0]):
        p_x = pointcloud[point_i][0]
        p_y = pointcloud[point_i][1]
        p_d = pointcloud[point_i][2]
        if is_wall(maze, p_x, p_y, r=4, pixel=WALL_COLOR):
            map_point = np.array([p_x, p_y, p_d, 0])
            contours_W.append(map_point)
        else:
            map_point = np.array([p_x, p_y, p_d, 1])
            contours_F.append(map_point)

    k = sparse_k
    i = 0
    contours_F_sparse = contours_F.copy()
    while len(contours_F_sparse) < k:
        if len(contours_F_sparse) == 0:
            contours_F_sparse = [np.zeros(4)]*k
            break
        else:
            contours_F_sparse.append(contours_F_sparse[i].copy())
            i += 1

    contours_F_sparse = np.array(contours_F_sparse).astype(np.int32)
    if len(contours_F_sparse) > k:
        contour_index = FPS(contours_F_sparse[:, 0:2], k)
        contours_F_sparse = contours_F_sparse[contour_index]

    
    k = sparse_k*3
    i = 0
    contours_W_sparse = contours_W.copy()
    while len(contours_W_sparse) < k:
        if len(contours_W_sparse) == 0:
            contours_W_sparse = [np.zeros(4)]*k
            break
        else:
            contours_W_sparse.append(contours_W_sparse[i].copy())
            i += 1

    contours_W_sparse = np.array(contours_W_sparse).astype(np.int32)
    if len(contours_W_sparse) > k:
        contour_index = FPS(contours_W_sparse[:, 0:2], k)
        contours_W_sparse = contours_W_sparse[contour_index]

    frontiers_num = np.sum(contours_F_sparse[:, -1]).astype(np.int64)

    contours_F = np.array(contours_F)

    contours = np.concatenate((contours_F_sparse, contours_W_sparse))

    return frontiers_num, contours_F, contours[:, 0:4]


def is_wall(maze, p_x, p_y, r, pixel):
    rows, cols = maze.shape
    for i in range(max(0, p_x-r), min(rows, p_x+r+1)):
        for j in range(max(0, p_y-r), min(cols, p_y+r+1)):
            if maze[i, j] == pixel:
                return True
    return False


class RobotExploration_1(gym.Env):
    metadata = {"render_mode": ['human', 'rgb_array'], "render_fps": 4}

    def __init__(self, render_mode= None, save_flag=False, config_path='config.yaml'):
        if config_path.startswith("/"):
            fullpath = config_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "config", config_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.sim = pseudoSlam(fullpath)
        self.sparse_k = 64
        self.render_mode = render_mode
        self.save_flag = save_flag
        self.start_pose = self.sim.get_pose()
        self.start_location = self.sim.get_pose()[0:2]
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.pointcloud = None
        self.total_path = []
        self.info = {}
        self.path = None


    def render(self, mode='human'):
        if mode=='human':
            map = (self.sim.get_state()).astype(np.uint8)
            map = np.stack((map, map, map), axis=2)
            
            map[self.pointcloud[:self.sparse_k, 0], self.pointcloud[:self.sparse_k, 1]] = [0, 201, 34]
            map[self.pointcloud[self.sparse_k:, 0], self.pointcloud[self.sparse_k:, 1]] = [255, 125, 64]
            if not self.path == None:
                for path_step in self.path:
                    map[path_step[0], path_step[1]] = [255, 97, 3]
            map = map.astype(np.uint8)
            plt.figure(2, figsize=(10, 10), dpi = 100)
            plt.clf()

            plt.imshow(map)
            plt.draw()
            plt.pause(0.01)


    def reset(self, seed=None, options=None, order=False):
        super().reset(seed=seed, options=options)
        self.sim.reset(order)
        self.start_pose = self.sim.get_pose()
        self.start_location = self.sim.get_pose()[0:2]
        self.pointcloud = None
        self.total_path = []
        self.info = {}
        self.path = None

        frontiers_num, contour_F, pointcloud = find_contour(self.sparse_k, self.sim.get_state(), self.sim.get_pose())  
        self.info['frontiers_num'] = frontiers_num
        self.pointcloud = pointcloud

        return pointcloud, self.info


    def step(self, action):
        pointcloud = None
        reward_all = 0
        terminated = False
        truncated = False

        reward, truncated = self.step_withAstar(np.array([int(action[0] + 0.5), int(action[1] + 0.5)]))
        reward_all += reward
        
        if not truncated:
            # Next observation
            frontiers_num, contour_F, pointcloud = find_contour(self.sparse_k, self.sim.get_state(), self.sim.get_pose())
            self.info['frontiers_num'] = frontiers_num
            self.pointcloud = pointcloud

            # terminate condition
            if (self.sim.measure_ratio() > 0.999) or (self.sim.measure_ratio() > 0.98 and frontiers_num < 4):
                terminated = True
            else:
                terminated = False
            # draw the experiment results during test
            if terminated:
                if self.save_flag:
                    img = self.sim.world.copy()
                    img_bgr = np.stack((img, img, img), axis=2)
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            if img[i][j] == 0:
                                img_bgr[i][j] = [255, 255, 255]

                    for i in range(len(self.total_path) - 1):
                        start = (int(self.total_path[i][1]), int(self.total_path[i][0]))
                        end = (int(self.total_path[i + 1][1]), int(self.total_path[i + 1][0]))
                        hue = int(120 * (len(self.total_path) - i) / len(self.total_path))  
                        saturation = 255
                        value = 220
                        color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0, 0]
                        color = tuple([int(x) for x in color])
                        cv2.line(img_bgr, start, end, (color[0], color[1], color[2]), lineType=1)
                
                    cv2.circle(img_bgr, (int(self.start_location[1]), int(self.start_location[0])),
                                int(2 * self.sim.robotRadius),
                                (255, 0, 0),
                                thickness=-1)
                    cv2.circle(img_bgr, (int(self.sim.get_pose()[1]), int(self.sim.get_pose()[0])),
                                int(2 * self.sim.robotRadius),
                                (0, 0, 255),
                                thickness=-1)
                    
                    root = os.path.dirname(__file__)
                    result_dir = os.path.join(root, 'result')
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)
                    cv2.imwrite(os.path.join(result_dir, "path_%d.png" % len(self.total_path)), img_bgr)

        return pointcloud, reward_all, terminated, truncated, self.info


    def step_withAstar(self, goal_point):
        truncated = False
        reward_all = 0
        img = self.sim.get_state()
        self.retracking = False

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.erode(img.copy(), kernel, 3)

        grid = dilated.astype(np.float32)
        grid[grid == UNKNOWN_COLOR] = 100000 
        grid[grid == WALL_COLOR] = 100000   
        grid[grid == FREE_COLOR] = 1  
        grid[grid == ROBOT_COLOR] = 1
        start_point = np.array([int(self.sim.get_pose()[0] + 0.5), int(self.sim.get_pose()[1] + 0.5)])
        path = pyastar.astar_path(grid, start_point, goal_point, allow_diagonal=True)

        if len(path) <= 0:
            truncated = True
            reward_all -= 10
        else:
            path_sparse = []
            if len(path) <= 5:
                path_sparse.append(path[-1])
            else:
                for i in range(5, len(path), 5):
                    points = path[i]
                    path_sparse.append(points)
            
            self.path = path_sparse.copy()
            if not self.render_mode == None:
                self.render(self.render_mode)

            # print("------------tracking this path---------------")
            next_location = np.zeros(3)
            map_before = self.sim.get_state().copy()
            map_before[map_before == 50] = 0
            for node_id in range(len(path_sparse)):
                robot_pose = self.sim.get_pose()
                next_location[0:2] = path_sparse[node_id]
                theta_to_next = np.arctan2(-(next_location[0] - robot_pose[0]) , (next_location[1] - robot_pose[1]))
                next_location[2] = theta_to_next
                next_location_step = np.zeros(3)

                if abs(robot_pose[2] - theta_to_next) > np.deg2rad(5):
                    next_location_step[0:2] = robot_pose[0:2]
                    remain_theta = abs((robot_pose[2] - theta_to_next))
                    while remain_theta > np.pi:
                        remain_theta = abs(2*np.pi - remain_theta)
                    while remain_theta > np.deg2rad(5):
                        if robot_pose[2] - theta_to_next > np.pi:
                            act = 1
                        elif robot_pose[2] - theta_to_next > 0:
                            act = 2
                        elif robot_pose[2] - theta_to_next > -np.pi:
                            act = 1
                        else:
                            act = 2
                        if remain_theta > np.deg2rad(15):
                            if act == 1:
                                next_theta = robot_pose[2] + np.deg2rad(15)
                            else:
                                next_theta = robot_pose[2] - np.deg2rad(15)
                        else:
                            next_theta = theta_to_next

                        next_location_step[2] = next_theta
                        reward = self.step_one(next_location_step)
                        reward_all += reward
                        robot_pose = self.sim.get_pose()
                        remain_theta = abs((robot_pose[2] - theta_to_next))
                        while remain_theta > np.pi:
                            remain_theta = abs(2*np.pi - remain_theta)
                    next_location_step[0:2] = path_sparse[node_id]
                    next_location_step[2] = theta_to_next
                    reward = self.step_one(next_location_step)
                    reward_all += reward
                else:
                    next_location[0:2] = path_sparse[node_id]
                    next_location[2] = theta_to_next
                    reward = self.step_one(next_location)
                    reward_all += reward

                current_map = self.sim.get_state()
                current_map[current_map == 50] = 0
                new_area = np.sum(current_map == self.sim.map_color['free'])\
                         - np.sum(map_before == self.sim.map_color['free'])
                if new_area > 200:
                    break
            map_after = self.sim.get_state().copy()
            map_after[map_after == 50] = 0

        return reward_all, truncated


    def step_one(self, targetPose):
        crush_flag = self.sim.moveRobot(targetPose)
        targetPose = targetPose[0:2].astype(np.int64)
        self.total_path.append(targetPose.copy()[0:2])
        reward = self._compute_reward()
        return reward


    def _compute_reward(self):
        reward = -0.01
        return reward


    def _get_action_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.sparse_k, 4), dtype=np.int32)


    def _get_observation_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(self.sparse_k, 4), dtype=np.int32)



if __name__ == '__main__':
    # greedy exploration
    env = RobotExploration_1(render_mode='human', save_flag=False, config_path='config.yaml')
    pointcloud, info = env.reset()
    while 1:
        frontiers_num = info['frontiers_num']
        frontiers = pointcloud[0:frontiers_num]
        if len(frontiers) > 0:
            action_id = np.argmin(frontiers[0:env.sparse_k, 2])
            action = frontiers[action_id][0:2]
        else:
            pointcloud, info = env.reset()
            continue

        pointcloud, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            pointcloud, info = env.reset()
            env.render()
            frontiers_num = info['frontiers_num']





