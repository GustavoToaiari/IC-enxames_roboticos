import numpy as np
from Parameters import *

class Robot:
    def __init__(self, name, base, goal_path, wheel_path, sim):
        self.name = name
        self.base = base
        self.w_left, self.w_right = wheel_path
        self.velocity = np.array([0, 0])
        self.force = np.array([0, 0])
        self.ang_vel = 0
        self.position = np.array([0, 0])
        self.orientation = 0
        self.sim = sim
        self.goal_path = goal_path
        # self.obstacle = obstacle
        self.goal_position = np.array(self.sim.getObjectPosition(self.goal_path, -1)[:2])


    def run(self, robots=None): 
        self.get_pose_2d() # Pose do robô é atualizada a cada iteração
        
        if self.arrived():
            self.stop_robot() 
            return True

        self.attraction_force()
        self.velocity = self.force
        self.set_wheel_speeds()
        
        return False # Ainda não chegou ao goal

    @staticmethod
    def wrap_to_pi(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def arrived(self):
        rho = np.linalg.norm(self.goal_position - self.position)
        if rho < GOAL_TOL:
            self.stop_robot()
            return True # Chegou ao goal
    

    def get_pose_2d(self):
        self.position = np.array(self.sim.getObjectPosition(self.base, -1)[:2])

        orientation = self.sim.getObjectOrientation(self.base, -1)[2] + np.pi/2
        self.orientation = self.wrap_to_pi(orientation) # Normaliza o ângulo em [-pi, pi]
    
    @staticmethod
    def get_obstacle_positions(self):
        obstacles = []
        for obs in self.obstacles:
            pos = self.sim.getObjectPosition(obs["handle"], -1)
            obstacles.append((obs["name"], pos[0], pos[1], obs["radius"]))
        return obstacles
    
    def set_wheel_speeds(self):
        rho_v = self.goal_position - self.position # Erro entre a posição atual do robô e o goal
        rho = np.linalg.norm(rho_v)
        dx, dy = rho_v

        angle_desired = np.atan2(dy, dx) # Para onde o robô deveria estar apontando
        angle = self.wrap_to_pi(angle_desired - self.orientation) # Erro entre a direção desejada e orientação atual
        v = K_V * rho # velocidade linear depende apenas da distância até o goal
        v = max(min(v, V_MAX), 0.1*V_MAX) 

        w = K_W * angle # velocidade angular depende apenas do erro angular

        # conversão de (v, w) para velocidades das rodas
        wr = v/WHEEL_RADIUS + (w*AXLE_LENGTH) / (2.0*WHEEL_RADIUS)
        wl = v/WHEEL_RADIUS - (w*AXLE_LENGTH) / (2.0*WHEEL_RADIUS)
        
        # Saturação
        wr = max(min(wr, W_MAX), -W_MAX)
        wl = max(min(wl, W_MAX), -W_MAX)

        self.sim.setJointTargetVelocity(self.w_left, float(wl))
        self.sim.setJointTargetVelocity(self.w_right, float(wr))

    def stop_robot(self):
        self.sim.setJointTargetVelocity(self.w_left, 0.0)
        self.sim.setJointTargetVelocity(self.w_right, 0.0)

    def attraction_force(self):
        self.force += (self.goal_position - self.position) * K_ATT

    
    # função apenas para formação triangulo
    # calcula o vetor resultante de formação triangular
    # def formation_force(self, robots):
    #     force = np.array([0.0, 0.0])
    #     max_error = 0.0

    #     for other in robots:
    #         if other is self: # para o robô não se comparar
    #             continue

    #         # Vetor deste robô até o "outro"
    #         delta = other.position - self.position
    #         dist = np.linalg.norm(delta)

    #         # Evita divisão por zero
    #         if dist < 1e-6:
    #             continue

    #         # Erro de distância:
    #         # > 0  -> longe demais -> aproxima
    #         # < 0  -> perto demais -> afasta
    #         dist_error = dist - DESIRED_DISTANCE

    #         # Guarda o maior erro absoluto, para usar no critério de parada
    #         max_error = max(max_error, abs(dist_error))

    #         # Vetor unitário na direção do outro robô
    #         direction = delta / dist

    #         # Soma contribuição de formação
    #         force += K_FORM * dist_error * direction

    #     return force, max_error
