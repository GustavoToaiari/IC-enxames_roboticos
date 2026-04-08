import math
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Parameters import *

class Robot:
    def __init__(self, name, base, goal_path, wheel_path, sim):
        self.name = name
        self.base = base
        self.w_left, self.w_right = wheel_path
        self.velocity = np.array([0, 0])
        self.ang_vel = 0
        self.position = np.array([0, 0])
        self.orientation = 0
        self.sim = sim
        self.goal_path = goal_path
        # self.obstacle = obstacle
        self.goal_position = np.array(self.sim.getObjectPosition(self.goal_path, -1)[:2])


    def run(self):
        self.attraction_force()
        self.set_wheel_speeds()

    @staticmethod
    def wrap_to_pi(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    

    def get_pose_2d(self):
        self.position = np.array(self.sim.getObjectPosition(self.base, -1)[:2])
        self.orientation = self.sim.getObjectOrientation(self.base, -1)[2]
    
    @staticmethod
    def get_obstacle_positions(self):
        obstacles = []
        for obs in self.obstacles:
            pos = self.sim.getObjectPosition(obs["handle"], -1)
            obstacles.append((obs["name"], pos[0], pos[1], obs["radius"]))
        return obstacles
    
    def set_wheel_speeds(self):
        angle = np.atan2(self.velocity[1], self.velocity[0]) - self.orientation
        v = self.velocity[0]*np.cos(angle)
        w = angle * K_ROT
        # Colocar função para converter velocidade angular em linear
        wr = (2.0*v + w*AXLE_LENGTH) / (2.0*WHEEL_RADIUS)
        wl = (2.0*v - w*AXLE_LENGTH) / (2.0*WHEEL_RADIUS)

        print(wr)
        
        self.sim.setJointTargetVelocity(self.w_left, float(wl))
        self.sim.setJointTargetVelocity(self.w_right, float(wr))

    def stop_robot(self, robot_key):
        self.sim.setJointTargetVelocity(self.w_left, 0.0)
        self.sim.setJointTargetVelocity(self.w_right, 0.0)

    def attraction_force(self):
        self.velocity = (self.goal_position - self.position) * K_ATT
    

