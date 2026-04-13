import math
import numpy as np
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
        self.get_pose_2d() # Pose do robô é atualizada a cada iteração
        error = self.goal_position - self.position # Erro entre a posição atual do robô e o goal
        rho = np.linalg.norm(error) # Retorna a distância euclidiana até o goal

        # Critério de parada (arrived)
        if rho < GOAL_TOL:
            self.stop_robot()
            return True # Chegou ao goal
        # self.attraction_force()
        self.set_wheel_speeds(error, rho)
        return False # Ainda não chegou ao goal

    @staticmethod
    def wrap_to_pi(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    

    def get_pose_2d(self):
        self.position = np.array(self.sim.getObjectPosition(self.base, -1)[:2])
        self.orientation = self.sim.getObjectOrientation(self.base, -1)[2] + math.pi/2
        self.orientation = self.wrap_to_pi(self.orientation) # Normaliza o ângulo em [-pi, pi]
    
    @staticmethod
    def get_obstacle_positions(self):
        obstacles = []
        for obs in self.obstacles:
            pos = self.sim.getObjectPosition(obs["handle"], -1)
            obstacles.append((obs["name"], pos[0], pos[1], obs["radius"]))
        return obstacles
    
    def set_wheel_speeds(self, error, rho):
        dx, dy = error

        angle_desired = math.atan2(dy, dx) # Para onde o robô deveria estar apontando
        angle = self.wrap_to_pi(angle_desired - self.orientation) # Erro entre a direção desejada e orientação atual
        # angle = np.atan2(self.velocity[1], self.velocity[0]) - self.orientation
        # v = self.velocity[0]*np.cos(angle)
        v = K_V * rho # velocidade linear depende apenas da distância até o goal

        # w = angle * K_ROT
        w = K_W * angle # velocidade angular depende apenas do erro angular

        # conversão de (v, w) para velocidades das rodas
        wr = (2.0*v + w*AXLE_LENGTH) / (2.0*WHEEL_RADIUS)
        wl = (2.0*v - w*AXLE_LENGTH) / (2.0*WHEEL_RADIUS)
        
        self.sim.setJointTargetVelocity(self.w_left, float(wl))
        self.sim.setJointTargetVelocity(self.w_right, float(wr))

    def stop_robot(self):
        self.sim.setJointTargetVelocity(self.w_left, 0.0)
        self.sim.setJointTargetVelocity(self.w_right, 0.0)

    def attraction_force(self):
        self.velocity = (self.goal_position - self.position) * K_ATT
    

