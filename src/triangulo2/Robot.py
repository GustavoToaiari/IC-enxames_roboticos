import numpy as np
import random
from Parameters import *

class Robot:
    def __init__(self, name, base, goal_paths, wheel_path, obstacles, sim):
        self.name = name
        self.base = base
        self.w_left, self.w_right = wheel_path
        self.velocity = np.array([0, 0])
        self.force = np.array([0, 0])
        self.ang_vel = 0
        self.position = np.array([0, 0])
        self.orientation = 0
        self.sim = sim
        
        self.goal_paths = goal_paths
        self.goal_positions = [np.array(self.sim.getObjectPosition(goal, -1)[:2])
                               for goal in self.goal_paths]

        self.obstacles = obstacles

        self.leader = None
        self.line_order = None
        self.line_target_position = None

        self.max_error = 0


    def run(self, robots=None, mode="formation", target_position=None): 
        self.get_pose_2d() # Pose do robô é atualizada a cada iteração
        
        if mode == "formation":
            self.formation_force(robots)
            self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER)
            self.set_wheel_speeds(V_MAX_FORMATION)
            return False
        
        elif mode == "line_formation":
            self.line_formation_force()
            self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER)
            self.set_wheel_speeds(V_MAX_FORMATION)
            return False
        
        elif mode == "formation_with_goal":

            # Líder continua indo para o objetivo
            if self is self.leader:

                self.force = np.array([0.0, 0.0])

                self.attraction_force(target_position)

                self.repulsive_force(
                    repulsion_scale=REP_SCALE_LEADER
                )

            # Seguidores recuperam formação triangular
            else:

                self.formation_force(robots)

                self.repulsive_force(
                    repulsion_scale=REP_SCALE_FOLLOWER
                )


            self.set_wheel_speeds(V_MAX_FORMATION)

            return False
        
        elif mode == "go_to_goal":
            if target_position is None:
                self.stop_robot()
                return False
            
            if self.arrived_target(target_position, stop=False):
                return True
            
            self.force = np.array([0.0,0.0])
            self.attraction_force(target_position)
            self.repulsive_force(repulsion_scale=REP_SCALE_LEADER)
            self.set_wheel_speeds(V_MAX_LEADER_GOAL)

            return False

        elif mode == "stop":
            self.stop_robot()
            return False

        self.force = np.array([0.0, 0.0])
        return False # Ainda não chegou ao goal

    @staticmethod
    def wrap_to_pi(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def arrived_target(self, target_position, stop=True):
        rho = np.linalg.norm(target_position - self.position)

        if rho < GOAL_TOL:
            if stop:
                self.stop_robot()
            return True
        
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
        force_norm = np.linalg.norm(self.force)

        # Se a força for muito pequena, considera que chegou no ponto desejado
        if force_norm < 0.02:
            self.stop_robot()
            return

        angle_desired = np.atan2(self.force[1], self.force[0]) # Para onde o robô deveria estar apontando
        angle = self.wrap_to_pi(angle_desired - self.orientation) # Erro entre a direção desejada e orientação atual

        v = K_V #* np.linalg.norm(self.force) # velocidade linear depende apenas da distância até o goal
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

    def repulsive_force(self, repulsion_scale=1.0):
        F_rep = np.zeros_like(self.force)

        for obs in self.obstacles:
            # Caso 1: obstáculo retangular / parede
            if self.is_wall(obs):
                seg_a, seg_b, thickness = self.get_wall_segment(obs)

                distance_to_centerline, closest_point = Robot.point_to_segment_distance(
                    self.position,
                    seg_a,
                    seg_b
                )
                # distância até a "superficie" da parede
                dist = distance_to_centerline - thickness/2.0 - ROBOT_RADIUS

                direction_vec = self.position - closest_point
                norm_dir = np.linalg.norm(direction_vec)
            
            # Caso 2: obstáculo ciruclar / cilindrico
            else:
                obs_pos = np.array(self.sim.getObjectPosition(obs, -1)[:2])

                direction_vec = self.position - obs_pos
                norm_dir = np.linalg.norm(direction_vec)

                dist = norm_dir - OBSTACLE_RADIUS - ROBOT_RADIUS
            
            # Para evitar erro númerico
            if norm_dir < 1e-6:
                continue

            direction = direction_vec / norm_dir

            # Evita divisão por zero
            dist = max(dist, 0.03)
            
            if dist < REP_RANGE:
                F_rep += (
                    K_REP
                    * ((1.0 / dist) - (1.0 / REP_RANGE))
                    * (1.0 / (dist ** 2))
                    * direction
                )


        F_rep_scaled = repulsion_scale * F_rep

        rep_norm = np.linalg.norm(F_rep_scaled)

        if repulsion_scale < 1.0:
            rep_max = F_REP_MAX_FOLLOWER
        else:
            rep_max = F_REP_MAX_LEADER

        if rep_norm > rep_max:
            F_rep_scaled = (F_rep_scaled / rep_norm) * rep_max

        self.force = self.force + F_rep_scaled


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
        if (
            self.line_order is None
            or self.leader is None
            or self.line_target_position is None
        ):
            self.force = np.array([0.0, 0.0])
            self.max_error = 0.0
            return

        index = self.line_order.index(self)

        # O primeiro robô da lista é o líder
        if index == 0:
            self.force = np.array([0.0, 0.0])
            self.max_error = 0.0
            return

        front_robot = self.line_order[index - 1]

        self.get_pose_2d()
        front_robot.get_pose_2d()
        self.leader.get_pose_2d()

        # Direção do líder até o objetivo atual
        direction = self.line_target_position - self.leader.position
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-6:
            self.force = np.array([0.0, 0.0])
            self.max_error = 0.0
            return

        direction = direction / direction_norm

        # Posição desejada atrás do robô da frente
        desired_position = (
            front_robot.position
            - LINE_DISTANCE * direction
        )

        error_vector = desired_position - self.position

        self.max_error = np.linalg.norm(error_vector)
        self.force = K_LINE * error_vector  

    def detect_narrow_passage(self, target_position):
        target_vector = target_position - self.position
        target_norm = np.linalg.norm(target_vector)

        if target_norm < 1e-6:
            return False

        forward = target_vector / target_norm

        visible_walls = []

        for wall in self.obstacles:
            seg_a, seg_b, thickness = self.get_wall_segment(wall)

            distance_to_wall_centerline, closest_point = Robot.point_to_segment_distance(
                self.position,
                seg_a,
                seg_b
            )

            distance_to_wall_surface = distance_to_wall_centerline - thickness / 2.0

            if distance_to_wall_surface > LEADER_VISION_RADIUS:
                continue

            vec_to_wall = closest_point - self.position
            vec_norm = np.linalg.norm(vec_to_wall)

            if vec_norm < 1e-6:
                continue

            dir_to_wall = vec_to_wall / vec_norm

            dot_value = np.dot(forward, dir_to_wall)
            dot_value = np.clip(dot_value, -1.0, 1.0)

            angle_to_wall = np.arccos(dot_value)

            if angle_to_wall > LEADER_FOV_ANGLE / 2.0:
                continue

            visible_walls.append({
                "a": seg_a,
                "b": seg_b,
                "thickness": thickness
            })

        if len(visible_walls) < 2:
            return False

        for i in range(len(visible_walls)):
            for j in range(i + 1, len(visible_walls)):
                wall_1 = visible_walls[i]
                wall_2 = visible_walls[j]

                centerline_distance = Robot.segment_to_segment_distance(
                    wall_1["a"],
                    wall_1["b"],
                    wall_2["a"],
                    wall_2["b"]
                )

                free_width = (
                    centerline_distance
                    - wall_1["thickness"] / 2.0
                    - wall_2["thickness"] / 2.0
                )

                if 0.0 < free_width < MIN_PASSAGE_WIDTH:
                    return True

        return False
    
    @staticmethod
    def choose_leader(robots, target_position):
        return min(robots, key=lambda robot: np.linalg.norm(target_position - robot.position))
    
    @staticmethod
    def get_last_robot_line(line_order):
        return line_order[-1]

    @staticmethod
    def prepare_line_formation(robots, target_position):
        line_order = sorted(
            robots,
            key=lambda robot: np.linalg.norm(target_position - robot.position)
        )

        leader = line_order[0]

        for robot in robots:
            robot.leader = leader
            robot.line_order = line_order
            robot.line_target_position = target_position.copy()

        return leader, line_order
    
    @staticmethod
    def form_triangle(robots):
        max_errors = []

        for robot in robots:
            robot.run(robots, mode="formation")
            max_errors.append(robot.max_error)

        return max(max_errors) < DIST_TOL
    
    @staticmethod
    def form_line(robots, leader):
        max_errors = []

        for robot in robots:
            if robot is leader:
                robot.stop_robot()
                robot.max_error = 0.0
            else:
                robot.run(robots, mode="line_formation")

            max_errors.append(robot.max_error)

        return max(max_errors) < LINE_TOL
    

    @staticmethod
    def clear_line_data(robots):
        for robot in robots:
            robot.leader = None
            robot.line_order = None
            robot.line_target_position = None


    @staticmethod
    def create_epucks(sim, n_robots):
        created_epucks = []
        epuck_template = sim.getObject('/ePuck1')

        x_min, x_max = 0.5, 1.5  # Limites de X
        y_min, y_max = 1.6, 2.3  # Limites de Y

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

    def is_wall(self, obstacle):
        alias = self.sim.getObjectAlias(obstacle)
        return "Cuboid" in alias

    # Para identificar o obstaculo como parede e não cilindrico
    def get_wall_segment(self, wall):
        min_x = self.sim.getObjectFloatParam(wall, self.sim.objfloatparam_objbbox_min_x)
        max_x = self.sim.getObjectFloatParam(wall, self.sim.objfloatparam_objbbox_max_x)
        min_y = self.sim.getObjectFloatParam(wall, self.sim.objfloatparam_objbbox_min_y)
        max_y = self.sim.getObjectFloatParam(wall, self.sim.objfloatparam_objbbox_max_y)

        size_x = max_x - min_x
        size_y = max_y - min_y

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        if size_x >= size_y:
            p1_local = np.array([min_x, center_y, 0.0])
            p2_local = np.array([max_x, center_y, 0.0])
            thickness = size_y
        else:
            p1_local = np.array([center_x, min_y, 0.0])
            p2_local = np.array([center_x, max_y, 0.0])
            thickness = size_x

        p1_world = self.local_to_world_2d(wall, p1_local)
        p2_world = self.local_to_world_2d(wall, p2_local)

        return p1_world, p2_world, thickness


    def local_to_world_2d(self, obj, point_local):
        matrix = self.sim.getObjectMatrix(obj, -1)

        x = (
            matrix[0] * point_local[0]
            + matrix[1] * point_local[1]
            + matrix[2] * point_local[2]
            + matrix[3]
        )

        y = (
            matrix[4] * point_local[0]
            + matrix[5] * point_local[1]
            + matrix[6] * point_local[2]
            + matrix[7]
        )

        return np.array([x, y])


    @staticmethod
    def point_to_segment_distance(point, seg_a, seg_b):
        ab = seg_b - seg_a
        ab_norm_sq = np.dot(ab, ab)

        if ab_norm_sq < 1e-9:
            return np.linalg.norm(point - seg_a), seg_a

        t = np.dot(point - seg_a, ab) / ab_norm_sq
        t = np.clip(t, 0.0, 1.0)

        closest = seg_a + t * ab
        distance = np.linalg.norm(point - closest)

        return distance, closest


    @staticmethod
    def orientation(a, b, c):
        return (
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )


    @staticmethod
    def on_segment(a, b, c):
        return (
            min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )


    @staticmethod
    def segments_intersect(a, b, c, d):
        o1 = Robot.orientation(a, b, c)
        o2 = Robot.orientation(a, b, d)
        o3 = Robot.orientation(c, d, a)
        o4 = Robot.orientation(c, d, b)

        eps = 1e-9

        if o1 * o2 < 0 and o3 * o4 < 0:
            return True

        if abs(o1) < eps and Robot.on_segment(a, b, c):
            return True

        if abs(o2) < eps and Robot.on_segment(a, b, d):
            return True

        if abs(o3) < eps and Robot.on_segment(c, d, a):
            return True

        if abs(o4) < eps and Robot.on_segment(c, d, b):
            return True

        return False


    @staticmethod
    def segment_to_segment_distance(a, b, c, d):
        if Robot.segments_intersect(a, b, c, d):
            return 0.0

        d1, _ = Robot.point_to_segment_distance(a, c, d)
        d2, _ = Robot.point_to_segment_distance(b, c, d)
        d3, _ = Robot.point_to_segment_distance(c, a, b)
        d4, _ = Robot.point_to_segment_distance(d, a, b)

        return min(d1, d2, d3, d4)