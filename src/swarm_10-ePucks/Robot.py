import numpy as np
from Parameters import *

class Robot:
    def __init__(self, name, base, goal_path, wheel_path, obstacles, sim):
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
        self.obstacles = obstacles
        self.goal_position = np.array(self.sim.getObjectPosition(self.goal_path, -1)[:2])


    def run(self, robots=None): 
        self.get_pose_2d() # Pose do robô é atualizada a cada iteração
        
        if self.arrived():
            self.stop_robot() 
            return True

        self.attraction_force()
        self.repulsive_force()
        self.repulsive_r2r(robots)
        self.set_wheel_speeds()

        self.force = 0
        
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
    

    def get_obstacle_positions(self):
        obstacles = []
        for i, obs in enumerate(self.obstacles):
            pos = self.sim.getObjectPosition(obs, -1)
            obstacles.append(np.array([pos[0], pos[1]]))
        return obstacles
    
    def set_wheel_speeds(self):

        angle_desired = np.atan2(self.force[1], self.force[0]) # Para onde o robô deveria estar apontando
        angle = self.wrap_to_pi(angle_desired - self.orientation) # Erro entre a direção desejada e orientação atual

        v = K_V * np.linalg.norm(self.force) # velocidade linear depende apenas da distância até o goal
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
        self.force = self.force + (self.goal_position - self.position) * K_ATT

    def repulsive_force(self):
        obstacles = self.get_obstacle_positions()
        obs_rad = 0.25
        F_rep = np.zeros_like(self.force)

        for obs in obstacles:
            dir = self.position - obs
            dist = np.linalg.norm(dir) - obs_rad - ROBOT_RADIUS
            if dist < REP_RANGE:
                F_rep = F_rep + (
                    (K_REP / (dist ** 2)) *
                    ((1 / dist) - (1 / REP_RANGE)) *
                    (dir / REP_RANGE)
                )
            else:
                F_rep = F_rep + np.zeros_like(self.force)

        self.force = self.force + F_rep

    def repulsive_r2r (self, robots):
        REP_RANGE = 3 * ROBOT_RADIUS
        F_rep_robots = np.zeros_like(self.force)

        for robot in robots:
            if robot is self:
                continue

            dir = self.position - robot.position
            norm_dir = np.linalg.norm(dir)

            dist = norm_dir - 2 * ROBOT_RADIUS

            if dist < REP_RANGE:
                F_rep_robots = F_rep_robots + (
                    (K_REP_ROBOTS / (dist ** 2)) *
                    ((1 / dist) - (1 / REP_RANGE)) *
                    (dir / norm_dir)
                )

        self.force = self.force + F_rep_robots


    
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
