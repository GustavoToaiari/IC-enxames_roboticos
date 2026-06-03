import numpy as np
import random
from Parameters import *

class Robot:
    def __init__(self, name, base, goal_path, goal2_path, wheel_path, obstacles, sim):
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
        self.goal2_path = goal2_path
        self.obstacles = obstacles
        self.goal_position = np.array(self.sim.getObjectPosition(self.goal_path, -1)[:2])
        self.goal2_position = np.array(self.sim.getObjectPosition(self.goal2_path, -1)[:2])

        self.leader = None
        self.line_order = None


    def run(self, robots=None, mode="formation"): 
        self.get_pose_2d() # Pose do robô é atualizada a cada iteração
        
        if mode == "formation":
            self.formation_force(robots)
            self.set_wheel_speeds(V_MAX_FORMATION)
            return False
        
        elif mode == "line_formation":
            self.line_formation_force()
            self.set_wheel_speeds(V_MAX_FORMATION)
            return False
        
        elif mode == "go_to_goal":
            if self.arrived():
                return True
            
            self.attraction_force(self.goal_position)
            self.set_wheel_speeds(V_MAX_LEADER_GOAL)

        elif mode == "go_to_goal2":
            if self.arrived2():
                return True
            
            self.attraction_force(self.goal2_position)
            self.set_wheel_speeds(V_MAX_LEADER_GOAL)

        elif mode == "stop":
            self.stop_robot()
            return False

        # if self.arrived():
        #     self.stop_robot() 
        #     return True

        #self.attraction_force()
        #self.repulsive_force()
        #self.repulsive_r2r(robots)
        #self.set_wheel_speeds()

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
        return False
    
    def arrived2(self):
        rho = np.linalg.norm(self.goal2_position - self.position)
        if rho < GOAL_TOL:
            self.stop_robot()
            return True # Chegou ao goal
        return False
    

    def get_pose_2d(self):
        self.position = np.array(self.sim.getObjectPosition(self.base, -1)[:2])

        orientation = self.sim.getObjectOrientation(self.base, -1)[2] + np.pi/2
        self.orientation = self.wrap_to_pi(orientation) # Normaliza o ângulo em [-pi, pi]
    

    def get_obstacle_positions(self):
        obstacles = []
        for obs in self.obstacles:
            pos = self.sim.getObjectPosition(obs, -1)
            obstacles.append(np.array([pos[0], pos[1]]))
        return obstacles
    
    def set_wheel_speeds(self, V_MAX):

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

    def attraction_force(self, target_position):
        self.force = self.force + (target_position - self.position) * K_ATT

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
    def formation_force(self, robots):
        force = np.array([0.0, 0.0])
        max_error = 0

        for other in robots:
            if other is self: # para o robô não se comparar
                continue

            # Vetor deste robô até o "outro"
            delta = other.position - self.position
            dist = np.linalg.norm(delta)

            # Erro de distância:
            # > 0  -> longe demais -> aproxima
            # < 0  -> perto demais -> afasta
            dist_error = dist - DESIRED_DISTANCE

            # Guarda o maior erro absoluto, para usar no critério de parada
            max_error = max(max_error, abs(dist_error))

            # Vetor unitário na direção do outro robô
            direction = delta / dist

            # Soma contribuição de formação
            force += dist_error * direction

        self.force = force
        self.max_error = max_error

    def line_formation_force(self):
        if self.line_order is None or self.leader is None:
            self.force = np.array([0,0])
            self.max_error = 0
            return
        
        index = self.line_order.index(self)

        if index == 0: # Se é o lider
            self.force = np.array([0,0])
            self.max_error = 0
            return
        
        front_robot = self.line_order[index-1]

        self.get_pose_2d()
        front_robot.get_pose_2d()
        self.leader.get_pose_2d()

        # Direção do líde até o Goal2
        direction = self.leader.goal2_position - self.leader.position
        direction = direction / np.linalg.norm(direction)

        # Posição desejada: atrás do robô da frente, considerando a direção do movimento até o Goal2
        desired_position = front_robot.position - LINE_DISTANCE * direction

        error_vector = desired_position - self.position
        self.max_error = np.linalg.norm(error_vector)
        self.force = K_LINE * error_vector

    @staticmethod
    def create_epucks(sim, n_robots):
        created_epucks = []
        epuck_template = sim.getObject('/ePuck1')

        x_min, x_max = -2.0, 2.0  # Limites de X
        y_min, y_max = -2.0, 2.0  # Limites de Y

        for i in range(2, n_robots+1):  # Correção para incluir o último robô
            copied = sim.copyPasteObjects([epuck_template], 1)
            new_model = copied[0]

            sim.setObjectAlias(new_model, f'ePuck{i}')

            # Gerando posições aleatórias dentro dos limites definidos
            x_pos = random.uniform(x_min, x_max)  # Posição aleatória no eixo X
            y_pos = random.uniform(y_min, y_max)  # Posição aleatória no eixo Y
            z_pos = 0.01915

            # Definindo a nova posição do robô
            sim.setObjectPosition(new_model, sim.handle_world, [x_pos, y_pos, z_pos])

            created_epucks.append(new_model)
            

        return created_epucks
    
    @staticmethod
    def remove_created_epucks(sim, created_epucks):
        for epuck in created_epucks:
            sim.removeModel(epuck)