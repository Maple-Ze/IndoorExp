import numpy as np
import cv2, os
import robotExp.simulator.util as util
# import util
import yaml
from yaml import CLoader as Loader


map_color= {'uncertain':127, 'free':255, 'obstacle':1}

class pseudoSlam():
    def __init__(self, param_file):
        """ pseudoSlam initilization """
        """ define class variable """
        self.root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.m2p = 0
        self.robotRadius= 0
        self.stepLength_linear= 0
        self.stepLength_angular= 0
        self.robotPose_init= np.array([0. ,0. ,0.])
        self.robotResetRandomPose = 0
        self.laser_range=0
        self.laser_fov= 0
        self.laser_resol= 0
        self.world= None
        self.map_id = None
        self.render = 0

        with open(param_file) as stream:
            self.config = yaml.load(stream, Loader)
        
        """ set map_color """
        self.map_color = map_color
        self.map_names = np.loadtxt(os.path.join(self.root, self.config['map_name']))

        """ Initialize user config param """
        self.initialize_param(self.config)

        """ pre calculate radius and angle vector that will be used in building map """
        radius_vect= np.arange(self.laser_range+1)
        self.radius_vect= radius_vect.reshape(1, radius_vect.shape[0]) # generate radius vector of [0,1,2,...,laser_range]

        angles_vect = np.arange(-self.laser_fov*0.5, self.laser_fov*0.5+self.laser_resol, step=self.laser_resol)
        self.angles_vect = angles_vect.reshape(angles_vect.shape[0], 1) # generate angles vector from -laser_angle/2 to laser_angle

        self.robotPose = self.robotPose_init


    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]


    def initialize_param(self, config):
        """ world & robot param """
        self.m2p= config["meter2pixel"] # X pixel= 1 meter
        self.robotRadius= util.meter2pixel(config["robotRadius"], self.m2p) # robot radius in pixel
        self.stepLength_linear= util.meter2pixel(config["stepLength"]["linear"], self.m2p) # robot linear movement in each step in pixel
        self.stepLength_angular= util.deg2rad( config["stepLength"]["angular"] ) # robot angular movement in each step in rad
        
        """ render mode """
        self.render = config["render"]
        
        """ robot starting pose """
        self.robotResetRandomPose= config["resetRandomPose"]

        """ laser param """
        self.laser_range= util.meter2pixel(config["laser"]["range"], self.m2p) # laser range in pixel
        self.laser_fov= util.deg2rad( config["laser"]["fov"] ) # laser field of view in rad
        self.laser_resol= util.deg2rad( config["laser"]["resolution"] ) # laser rotation resolution in rad

        """ unknown mode """
        self.is_exploration = (config["mode"] == 0)
        return


    def reset(self, order=False):
        self.create_world(order)

        if self.robotResetRandomPose==1:
            # randomly generate robot start pose if random is set
            self._randomizeRobotPose()
        else:
            self.robotPose = self.robotPose_init
        while self.robotCrashed(self.robotPose):
            # randomly generate robot start pose where robot is not crashed into obstacle
            self._randomizeRobotPose()
        self.robotCrashed_flag = False
        if self.is_exploration:
            self.slamMap= np.ones(self.world.shape, dtype=np.uint8)*self.map_color["uncertain"]
            self.dslamMap= np.ones(self.world.shape, dtype=np.uint8)*self.map_color["uncertain"]
        else:
            self.slamMap = self.world.copy()
            self.dslamMap = self.world.copy()
        self.build_map(self.robotPose)



    def create_world(self, order=False):
        """ read maps in order if True, else randomly sample"""
        while(True):
            if order:
                map_name = str(int(self.map_names[0]))+'.png'
                self.map_names = np.delete(self.map_names, 0)
            else:
                map_name = str(int(np.random.choice(self.map_names)))+'.png'
            self.map_name = map_name
            self.world = cv2.imread(os.path.join(self.root, self.config['map_dir'], self.map_name), cv2.IMREAD_GRAYSCALE)

            (h,w)= self.world.shape
            break
            # if h < 512 and w < 512 and h > 100 and w > 100:
            #     break
            # else:
            #     pass

        self.robotPose_init[0:2] = int(h*0.5), int(w*0.5)
        self.robotPose_init[2] = 2 * (np.random.random() - 0.5) * np.pi
        return self.world

    
    def _randomizeRobotPose(self):
        # randomly generate robot start pose where robot is not crashed into obstacle
        h, w = self.world.shape
        x_min, x_max = int(0.1 * w), int(0.9 * w)
        y_min, y_max = int(0.1 * h), int(0.9 * h)
        self.robotPose[0] = np.random.randint(y_min, y_max)
        self.robotPose[1] = np.random.randint(x_min, x_max)

        while (self.robotCrashed(self.robotPose)):
            self.robotPose[0] = np.random.randint(y_min, y_max)
            self.robotPose[1] = np.random.randint(x_min, x_max)
        self.robotPose[2] = np.random.rand() * np.pi * 2 - np.pi
        return self.robotPose


    def build_map(self, pose):
        """ build perceived map based on robot position and its simulated laser info
        pose: [y;x;theta] in pixel in img coord | robotPose= pose"""
        """ input pose can be in decimal place, it will be rounded off in _build_map_with_rangeCoordMat """

        self.robotPose = pose.copy()
        """ find the coord matrix that the laser cover """
        angles = self.angles_vect + pose[2]
        # print(angles.shape)
        # print(self.radius_vect.shape)
        y_rangeCoordMat = -np.matmul(np.sin(angles), self.radius_vect) + pose[0]
        x_rangeCoordMat =  np.matmul(np.cos(angles), self.radius_vect) + pose[1]
        
        # need to take a careful look
        self._build_map_with_rangeCoordMat(y_rangeCoordMat, x_rangeCoordMat)
        return self.slamMap


    def _build_map_with_rangeCoordMat(self, y_rangeCoordMat, x_rangeCoordMat):
        # Round y and x coord into int
        y_rangeCoordMat = (np.round(y_rangeCoordMat)).astype(np.int32)
        x_rangeCoordMat = (np.round(x_rangeCoordMat)).astype(np.int32)

        """ Check for index of y_mat and x_mat that are within the world """ \
        '''
            Whether it is essential to check the point within the world or not
            I think just check the obstacle is enough
            In addition, whether setting the point out of the world as the value of boundary point is reasonable
        '''
        inBound_ind= util.within_bound(np.array([y_rangeCoordMat, x_rangeCoordMat]), self.world.shape)

        """ delete coordinate that are not within the bound """
        outside_ind = np.argmax(~inBound_ind, axis=1)
        # print('outside_ind: ', outside_ind, 'outside_ind.shape: ', outside_ind.shape)
        # print(np.where(outside_ind == 0))
        """ np.where return a tuple """ 
        ok_ind = np.where(outside_ind == 0)[0]
        need_amend_ind = np.where(outside_ind != 0)[0]
        outside_ind = np.delete(outside_ind, ok_ind)

        inside_ind = np.copy(outside_ind)
        inside_ind[inside_ind != 0] -= 1
        bound_ele_x = x_rangeCoordMat[need_amend_ind, inside_ind]
        bound_ele_y = y_rangeCoordMat[need_amend_ind, inside_ind]

        count = 0
        for i in need_amend_ind:
            x_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_x[count]
            y_rangeCoordMat[i, ~inBound_ind[i,:]] = bound_ele_y[count]
            count += 1

        """ find obstacle along the laser range """
        obstacle_ind = np.argmax(self.world[y_rangeCoordMat, x_rangeCoordMat] == self.map_color['obstacle'], axis=1)
        obstacle_ind[obstacle_ind == 0] = x_rangeCoordMat.shape[1]


        """ generate a matrix of [[1,2,3,...],[1,2,3...],[1,2,3,...],...] for comparing with the obstacle coord """
        bx = np.arange(x_rangeCoordMat.shape[1]).reshape(1, x_rangeCoordMat.shape[1])
        by = np.ones((x_rangeCoordMat.shape[0], 1))
        b = np.matmul(by, bx)

        """ get the coord that the robot can percieve (ignore pixel beyond obstacle) """
        b = b <= obstacle_ind.reshape(obstacle_ind.shape[0], 1)
        # print('b.shape: ', b.shape)
        y_coord = y_rangeCoordMat[b]  # 1D array
        x_coord = x_rangeCoordMat[b]

        """ no slam error """
        self.slamMap[y_coord, x_coord] = self.world[y_coord, x_coord]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.dslamMap= cv2.dilate(self.slamMap, kernel, iterations=3)
        return self.slamMap


    def moveRobot(self, targetPose):
        """ move robot to targetPose """
        # move by target pose! modified by Xuheng Gao 0627
        # print(f'move robot from {self.robotPose} to {targetPose}')
        # check if robot will crash on obstacle or go out of bound
        if self.robotCrashed(targetPose):
            self.robotCrashed_flag = True
            # print("Robot crash")
            return False
        # build map on the targetPose
        self.build_map( targetPose )

        return True


    def world2state(self):
        # state= cv2.resize(self.slamMap, self.state_size, interpolation=cv2.INTER_LINEAR)
        state = self.slamMap.copy()
        # draw robot position on state
        # self.robotRadius = self.robotRadius * 3
        cv2.circle(state, (int(self.robotPose[1]), int(self.robotPose[0])), self.robotRadius, 50, thickness=-1)

        # draw robot orientation heading on state
        headRadius = np.ceil(self.robotRadius/3.).astype(np.int32)
        headLen = self.robotRadius + headRadius
        ori_theta = self.robotPose[2]
        head_y = self.robotPose[0] - np.sin(ori_theta) * headLen
        head_x = self.robotPose[1] + np.cos(ori_theta) * headLen
        cv2.circle(state, (int(head_x), int(head_y)), headRadius, 50, thickness=-1)

        if not self.is_exploration:
            """Change color for known environment navigation"""
            state[state == self.map_color['free']] = 255
            state[state == self.map_color['obstacle']] = 0
        return state


    def robotCrashed(self, pose):
        # print(pose, self.world.shape, self.robotRadius)
        if ~util.within_bound(pose, self.world.shape, self.robotRadius):
            return True

        py = np.round(pose[0]).astype(int)
        px = np.round(pose[1]).astype(int)
        r  = self.robotRadius

        # make a circle patch around robot location and check if there is obstacle pixel inside the circle
        robotPatch, _ = util.make_circle(r, 1)
        worldPatch = self.world[py-r:py+r+1, px-r:px+r+1]
        if worldPatch.shape[0] != robotPatch.shape[0] or worldPatch.shape[1] != robotPatch.shape[1]:
            return True
        else:
            worldPatch= worldPatch*robotPatch

        return np.sum(worldPatch==self.map_color["obstacle"])!=0 or np.sum(worldPatch==self.map_color["uncertain"])!=0


    def get_state(self):
        return self.world2state().copy()
        # return self.slamMap.copy()


    def get_pose(self):
        return self.robotPose.copy()


    def get_crashed(self):
        return self.robotCrashed_flag


    def measure_ratio(self):
        mapped_pixel= np.sum(self.slamMap==self.map_color['free'])
        world_pixel= np.sum(self.world==self.map_color['free'])

        return 1.*mapped_pixel/world_pixel





if __name__ == '__main__':
    config_path = 'config_train_mini.yaml'
    fullpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", config_path)
    ps = pseudoSlam(fullpath)
    ps.reset()

