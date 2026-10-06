import numpy as np
import random
import time
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
        self.triangle_transition_target = None

        self.max_error = 0

        self.line_start_time = None

        # Controle de movimento reverso
        self.reverse_mode = False

        # Controle do lado escolhido para contornar obstáculos
        self.avoidance_side = None
        self.avoidance_last_active = None


        

    def start_line_transition(self):
        self.line_start_time = time.time()


    def run(
        self,
        robots=None,
        mode="formation",
        target_position=None,
        formation_context=None
    ):
        self.get_pose_2d()

        if mode == "formation":
            self.formation_force(robots)
            self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER)

            linear_speed = self.calculate_linear_speed(
                self.max_error,
                V_MIN_FORMATION,
                V_MAX_FORMATION,
                FORM_ERROR_MIN,
                FORM_ERROR_MAX
            )

            self.set_wheel_speeds(linear_speed)
            return False

        elif mode == "line_formation":
            if (
                self.line_order is None
                or self not in self.line_order
                or self.line_start_time is None
            ):
                self.stop_robot()
                return False

            index = self.line_order.index(self)
            wait_time = index * LINE_FORMATION_DELAY

            if time.time() - self.line_start_time < wait_time:
                self.stop_robot()
                return False

            self.line_formation_force()
            self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER)

            linear_speed = self.calculate_linear_speed(
                self.max_error,
                V_MIN_FORMATION,
                V_MAX_FORMATION,
                FORM_ERROR_MIN,
                FORM_ERROR_MAX
            )

            self.set_wheel_speeds(linear_speed)
            return False

        elif mode == "formation_with_goal":
            # Mantido por compatibilidade: líder parado e seguidores
            # recuperando a formação triangular ao redor dele.
            if self.leader is self:
                self.stop_robot()
                self.force = np.array([0.0, 0.0])
                self.max_error = 0.0
                return False

            self.formation_force(robots)
            self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER)

            linear_speed = self.calculate_linear_speed(
                self.max_error,
                V_MIN_FORMATION,
                V_MAX_FORMATION,
                FORM_ERROR_MIN,
                FORM_ERROR_MAX
            )

            self.set_wheel_speeds(linear_speed)
            return False

        elif mode == "triangle_transition_leader":
            if target_position is None:
                self.stop_robot()
                self.force = np.array([0.0, 0.0])
                return False

            # Se o líder já chegou ao Goal, ele vira a referência fixa
            # para a reconstrução do triângulo.
            if self.arrived_target(target_position, stop=True):
                self.force = np.array([0.0, 0.0])
                return True

            self.force = np.array([0.0, 0.0])
            self.attraction_force(target_position)
            repulsive = self.repulsive_force(repulsion_scale=REP_SCALE_LEADER)
            self.tangential_avoidance_force(target_position,repulsive)
            self.set_wheel_speeds(V_MAX_LEADER_TRANSITION, allow_reverse=False)
            return False

        elif mode == "triangle_transition":
            self.triangle_transition_force(robots)

            if self.triangle_transition_target is None:
                self.stop_robot()
                return False

            self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER)

            transition_error = np.linalg.norm(
                self.triangle_transition_target - self.position
            )

            linear_speed = self.calculate_linear_speed(
                transition_error,
                V_MIN_FORMATION,
                V_MAX_FORMATION,
                FORM_ERROR_MIN,
                FORM_ERROR_MAX
            )

            self.set_wheel_speeds(linear_speed)
            return False

        elif mode == "go_to_goal":
            if target_position is None:
                self.stop_robot()
                return False

            if self.arrived_target(target_position, stop=True):
                return True

            self.force = np.array([0.0, 0.0])
            self.attraction_force(target_position)
            repulsive = self.repulsive_force(repulsion_scale=REP_SCALE_LEADER)
            self.tangential_avoidance_force(target_position, repulsive)

            rho = np.linalg.norm(target_position - self.position)

            linear_speed = self.calculate_linear_speed(
                rho,
                V_MIN_LEADER,
                V_MAX_LEADER_GOAL,
                GOAL_SPEED_DIST_MIN,
                GOAL_SPEED_DIST_MAX
            )

            cohesion_scale = self.leader_cohesion_scale(
                robots,
                formation_context=formation_context
            )

            linear_speed *= cohesion_scale
            self.set_wheel_speeds(linear_speed, allow_reverse = False)
            return False

        elif mode == "stop":
            self.stop_robot()
            return False

        # Evita que um erro de digitação em mode deixe o robô executando
        # silenciosamente o último comando de roda.
        self.stop_robot()
        raise ValueError(f"Modo de controle desconhecido: {mode}")

    @staticmethod
    def wrap_to_pi(angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    @staticmethod
    def calculate_linear_speed(
        error,
        v_min,
        v_max,
        error_min,
        error_max
    ):
        """
        Calcula uma velocidade linear proporcional ao erro/distância,
        limitada entre v_min e v_max.
        """

        error = abs(float(error))

        # Proteção contra parâmetros inválidos
        if error_max <= error_min:
            return v_max

        # Erro pequeno -> velocidade mínima
        if error <= error_min:
            return v_min

        # Erro grande -> velocidade máxima
        if error >= error_max:
            return v_max

        # Região proporcional
        alpha = (
            (error - error_min)
            / (error_max - error_min)
        )

        return (
            v_min
            + alpha * (v_max - v_min)
        )
    
    def leader_cohesion_scale(self, robots, formation_context="triangle"):
        """
        Reduz a velocidade do líder quando os seguidores ficam para trás.

        formation_context deve ser explicitamente "triangle" ou "line".
        A função não usa line_order para inferir o estado da máquina, pois
        line_order também é mantida temporariamente durante a transição.
        """
        if robots is None or len(robots) <= 1:
            return 1.0

        if formation_context == "line":
            if (
                self.line_order is None
                or self not in self.line_order
                or self.line_order.index(self) != 0
                or len(self.line_order) < 2
            ):
                return 1.0

            first_follower = self.line_order[1]
            distance = np.linalg.norm(first_follower.position - self.position)
            gap = distance - LINE_DISTANCE

        elif formation_context == "triangle":
            gaps = []

            for robot in robots:
                if robot is self:
                    continue

                distance = np.linalg.norm(robot.position - self.position)
                gaps.append(distance - DESIRED_DISTANCE)

            if not gaps:
                return 1.0

            gap = max(gaps)

        else:
            raise ValueError(
                "formation_context deve ser 'triangle' ou 'line'."
            )

        # ============================================================
        # Define a escala mínima conforme a formação
        # ============================================================

        if formation_context == "line":
            # Na linha o líder nunca para completamente.
            # Ele continua avançando devagar enquanto
            # os seguidores recuperam a distância.
            min_scale = LEADER_LINE_MIN_SCALE

        else:
            # No triângulo ainda permitimos parar caso
            # a formação abra demais.
            min_scale = 0.0


        # ============================================================
        # Formação boa
        # ============================================================

        if gap <= LEADER_GAP_SOFT:
            return 1.0


        # ============================================================
        # Formação muito aberta
        # ============================================================

        if gap >= LEADER_GAP_HARD:
            return min_scale


        # ============================================================
        # Região intermediária
        # ============================================================

        alpha = (
            (gap - LEADER_GAP_SOFT)
            /
            (LEADER_GAP_HARD - LEADER_GAP_SOFT)
        )

        scale = (
            1.0
            - alpha * (1.0 - min_scale)
        )

        return float(
            np.clip(
                scale,
                min_scale,
                1.0
            )
        )

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
    
    def set_wheel_speeds(self, linear_speed, allow_reverse = True):

        force_norm = np.linalg.norm(
            self.force
        )

        # Força praticamente nula
        if force_norm < 0.02:
            self.stop_robot()
            return

        angle_desired = np.atan2(
            self.force[1],
            self.force[0]
        )

        angle = self.wrap_to_pi(
            angle_desired
            - self.orientation
        )

        # ========================================
        # Movimento reverso com histerese
        # ========================================

        if allow_reverse:

            if abs(angle) > np.radians(120):
                self.reverse_mode = True

            elif abs(angle) < np.radians(70):
                self.reverse_mode = False

        else:

            self.reverse_mode = False

        # ========================================
        # Velocidade linear
        # ========================================

        v = max(
            0.0,
            float(linear_speed)
        )

        # Quando a ré está desabilitada, reduz a velocidade
        # de translação se o robô ainda não estiver apontando
        # para a direção desejada.
        if not allow_reverse:

            heading_scale = max(
                0.0,
                np.cos(angle)
            )

            v *= heading_scale

        if self.reverse_mode:
            v = -v

        # ========================================
        # Velocidade angular
        # ========================================

        w = K_W * angle

        # ========================================
        # Cinemática diferencial
        # ========================================

        wr = (
            v / WHEEL_RADIUS
            +
            (w * AXLE_LENGTH)
            / (2.0 * WHEEL_RADIUS)
        )

        wl = (
            v / WHEEL_RADIUS
            -
            (w * AXLE_LENGTH)
            / (2.0 * WHEEL_RADIUS)
        )

        # ========================================
        # Saturação das rodas
        # ========================================

        wr = max(
            min(wr, WHEEL_OMEGA_MAX),
            -WHEEL_OMEGA_MAX
        )

        wl = max(
            min(wl, WHEEL_OMEGA_MAX),
            -WHEEL_OMEGA_MAX
        )

        self.sim.setJointTargetVelocity(
            self.w_left,
            float(wl)
        )

        self.sim.setJointTargetVelocity(
            self.w_right,
            float(wr)
        )

    def stop_robot(self):
        self.sim.setJointTargetVelocity(self.w_left, 0.0)
        self.sim.setJointTargetVelocity(self.w_right, 0.0)

    def attraction_force(self, target_position):
        self.force = self.force + (target_position - self.position) * K_ATT

    def repulsive_force(self, repulsion_scale=1.0):
        F_rep = np.zeros_like(self.force)

        for obs in self.obstacles:
            # Caso 1: obstáculo retangular / parede
            if obs["type"] == "wall":

                seg_a = obs["a"]
                seg_b = obs["b"]
                thickness = obs["thickness"]

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
                obs_pos = obs["position"]

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

        return F_rep_scaled
    
    def tangential_avoidance_force(
        self,
        target_position,
        repulsive_force
    ):
        """
        Adiciona uma componente tangencial à força repulsiva.

        Isso evita mínimos locais do campo potencial quando
        atração e repulsão ficam aproximadamente opostas.

        O robô escolhe passar por um dos lados do obstáculo
        e mantém essa decisão temporariamente para evitar
        oscilações esquerda/direita.
        """

        rep_norm = np.linalg.norm(repulsive_force)

        # ========================================================
        # Não existe obstáculo suficientemente próximo
        # ========================================================

        if rep_norm < TANGENTIAL_REP_MIN:

            if self.avoidance_last_active is not None:

                if (
                    time.time()
                    - self.avoidance_last_active
                    > AVOIDANCE_MEMORY_TIME
                ):
                    self.avoidance_side = None
                    self.avoidance_last_active = None

            return

        # Obstáculo está sendo evitado neste instante
        self.avoidance_last_active = time.time()

        # ========================================================
        # Direção radial de repulsão
        # ========================================================

        radial = (
            repulsive_force
            / rep_norm
        )

        # Vetores tangenciais possíveis
        tangent_left = np.array([
            -radial[1],
            radial[0]
        ])

        tangent_right = -tangent_left

        # ========================================================
        # Escolha do lado
        # ========================================================

        if self.avoidance_side is None:

            target_vector = (
                target_position
                - self.position
            )

            target_norm = np.linalg.norm(
                target_vector
            )

            if target_norm > 1e-6:

                target_direction = (
                    target_vector
                    / target_norm
                )

            else:

                target_direction = np.zeros(2)

            # Direção para a qual o robô já está apontando
            heading = np.array([
                np.cos(self.orientation),
                np.sin(self.orientation)
            ])

            # Avalia os dois lados.
            #
            # Queremos:
            # 1 - continuar aproximadamente na direção do Goal
            # 2 - evitar uma mudança brusca em relação à orientação atual

            score_left = (
                np.dot(
                    tangent_left,
                    target_direction
                )
                +
                AVOIDANCE_HEADING_WEIGHT
                * np.dot(
                    tangent_left,
                    heading
                )
            )

            score_right = (
                np.dot(
                    tangent_right,
                    target_direction
                )
                +
                AVOIDANCE_HEADING_WEIGHT
                * np.dot(
                    tangent_right,
                    heading
                )
            )

            if score_left >= score_right:
                self.avoidance_side = 1.0

            else:
                self.avoidance_side = -1.0

        # ========================================================
        # Usa o lado já escolhido
        # ========================================================

        tangent = (
            self.avoidance_side
            * tangent_left
        )

        # Força tangencial proporcional à repulsão
        tangential_magnitude = (
            K_TANGENTIAL
            * rep_norm
        )

        # Saturação
        tangential_magnitude = min(
            tangential_magnitude,
            F_TANGENTIAL_MAX
        )

        F_tangential = (
            tangential_magnitude
            * tangent
        )

        self.force = (
            self.force
            + F_tangential
        )


    def repulsive_r2r(self, robots):
        rep_range = 3 * ROBOT_RADIUS
        F_rep_robots = np.zeros_like(self.force)

        for robot in robots:
            if robot is self:
                continue

            direction_vec = self.position - robot.position
            norm_dir = np.linalg.norm(direction_vec)

            if norm_dir < 1e-6:
                continue

            dist = norm_dir - 2 * ROBOT_RADIUS
            dist = max(dist, 0.01)

            if dist < rep_range:
                F_rep_robots += (
                    (K_REP_ROBOTS / (dist ** 2))
                    * ((1.0 / dist) - (1.0 / rep_range))
                    * (direction_vec / norm_dir)
                )

        self.force = self.force + F_rep_robots

    def triangle_transition_force(self, robots):
        """
        Define o alvo dos dois seguidores durante a transição linha -> triângulo.

        Para um triângulo equilátero de lado D com o líder no vértice frontal:
            deslocamento longitudinal = sqrt(3)/2 * D
            deslocamento lateral     = 1/2 * D
        """
        if (
            self.line_order is None
            or self.line_target_position is None
            or self not in self.line_order
            or len(self.line_order) < 3
        ):
            self.triangle_transition_target = None
            self.force = np.array([0.0, 0.0])
            return

        leader = self.line_order[0]
        index = self.line_order.index(self)

        # Apenas os dois seguidores do triângulo recebem alvos explícitos.
        if index not in (1, 2):
            self.triangle_transition_target = None
            self.force = np.array([0.0, 0.0])
            return

        direction = self.line_target_position - leader.position
        direction_norm = np.linalg.norm(direction)

        # Se o líder estiver exatamente sobre o Goal, usa sua orientação
        # para manter definida a orientação geométrica do triângulo.
        if direction_norm < 1e-6:
            direction = np.array([
                np.cos(leader.orientation),
                np.sin(leader.orientation)
            ])
        else:
            direction = direction / direction_norm

        side = np.array([-direction[1], direction[0]])

        longitudinal_offset = (np.sqrt(3.0) / 2.0) * DESIRED_DISTANCE
        lateral_offset = 0.5 * DESIRED_DISTANCE

        side_sign = 1.0 if index == 1 else -1.0

        target = (
            leader.position
            - direction * longitudinal_offset
            + side_sign * side * lateral_offset
        )

        self.triangle_transition_target = target
        self.force = target - self.position
        self.max_error = np.linalg.norm(self.force)

    @staticmethod
    def triangle_max_error(robots):
        """Maior erro absoluto entre qualquer par e DESIRED_DISTANCE."""
        max_error = 0.0

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                distance = np.linalg.norm(
                    robots[j].position - robots[i].position
                )
                max_error = max(
                    max_error,
                    abs(distance - DESIRED_DISTANCE)
                )

        return max_error

    @staticmethod
    def form_triangle_transition(robots, leader):
        """
        Apenas VERIFICA a transição; não envia comandos novamente.
        """
        if (
            leader is None
            or leader.line_order is None
            or len(leader.line_order) < 3
        ):
            return False

        followers = leader.line_order[1:3]

        if any(robot.triangle_transition_target is None for robot in followers):
            return False

        target_error = max(
            np.linalg.norm(
                robot.triangle_transition_target - robot.position
            )
            for robot in followers
        )

        triangle_error = Robot.triangle_max_error(
            [leader, followers[0], followers[1]]
        )

        return (
            target_error < TRIANGLE_TRANSITION_TOL
            and triangle_error < DIST_TOL
        )

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

            # Evita divisão por zero caso dois centros coincidam numericamente.
            if dist < 1e-6:
                continue

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

            if wall["type"] != "wall":
                continue

            seg_a = wall["a"]
            seg_b = wall["b"]
            thickness = wall["thickness"]

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
    
    def detect_narrow_passage_backward(self):

        # Direção traseira do robô
        backward = np.array([
            -np.cos(self.orientation),
            -np.sin(self.orientation)
        ])


        visible_walls = []


        for wall in self.obstacles:

            if wall["type"] != "wall":
                continue

            seg_a = wall["a"]
            seg_b = wall["b"]
            thickness = wall["thickness"]


            distance_to_wall_centerline, closest_point = Robot.point_to_segment_distance(
                self.position,
                seg_a,
                seg_b
            )


            distance_to_wall_surface = (
                distance_to_wall_centerline
                - thickness / 2
            )


            if distance_to_wall_surface > LEADER_VISION_RADIUS:
                continue


            vec_to_wall = closest_point - self.position
            vec_norm = np.linalg.norm(vec_to_wall)


            if vec_norm < 1e-6:
                continue


            dir_to_wall = vec_to_wall / vec_norm


            dot_value = np.dot(backward, dir_to_wall)
            dot_value = np.clip(dot_value, -1.0, 1.0)


            angle_to_wall = np.arccos(dot_value)


            if angle_to_wall > LEADER_FOV_ANGLE / 2:
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
                    - wall_1["thickness"] / 2
                    - wall_2["thickness"] / 2
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
        for robot in robots:
            robot.run(robots, mode="formation")

        return Robot.triangle_max_error(robots) < DIST_TOL

    @staticmethod
    def form_triangle_around_leader(robots, leader):
        """
        Reagrupa o triângulo mantendo o líder parado no Goal.
        """
        if leader is None:
            return False

        leader.stop_robot()
        leader.force = np.array([0.0, 0.0])

        for robot in robots:
            if robot is leader:
                continue
            robot.run(robots, mode="formation")

        return Robot.triangle_max_error(robots) < DIST_TOL

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
            robot.line_start_time = None
            robot.triangle_transition_target = None


    @staticmethod
    def create_epucks(sim, n_robots):
        created_epucks = []
        epuck_template = sim.getObject('/ePuck1')

        x_min, x_max = 2.0, 4.75  # Limites de X
        y_min, y_max = 3.0, 4.75  # Limites de Y

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
    
    @staticmethod
    def create_obstacle_cache(sim, obstacles):

        cache = []

        for obs in obstacles:

            alias = sim.getObjectAlias(obs)

            # Parede
            if "Cuboid" in alias:

                min_x = sim.getObjectFloatParam(
                    obs,
                    sim.objfloatparam_objbbox_min_x
                )

                max_x = sim.getObjectFloatParam(
                    obs,
                    sim.objfloatparam_objbbox_max_x
                )

                min_y = sim.getObjectFloatParam(
                    obs,
                    sim.objfloatparam_objbbox_min_y
                )

                max_y = sim.getObjectFloatParam(
                    obs,
                    sim.objfloatparam_objbbox_max_y
                )


                size_x = max_x - min_x
                size_y = max_y - min_y


                center_x = (min_x + max_x)/2
                center_y = (min_y + max_y)/2


                if size_x >= size_y:

                    p1_local = np.array(
                        [min_x, center_y, 0]
                    )

                    p2_local = np.array(
                        [max_x, center_y, 0]
                    )

                    thickness = size_y

                else:

                    p1_local = np.array(
                        [center_x, min_y, 0]
                    )

                    p2_local = np.array(
                        [center_x, max_y, 0]
                    )

                    thickness = size_x


                p1_world = Robot.local_to_world_2d_static(
                    sim,
                    obs,
                    p1_local
                )

                p2_world = Robot.local_to_world_2d_static(
                    sim,
                    obs,
                    p2_local
                )


                cache.append(
                    {
                        "type":"wall",
                        "a":p1_world,
                        "b":p2_world,
                        "thickness":thickness
                    }
                )


            # Cilindro
            else:

                pos = sim.getObjectPosition(obs,-1)

                cache.append(
                    {
                        "type":"circle",
                        "position":np.array(
                            [pos[0],pos[1]]
                        )
                    }
                )


        return cache
    
    @staticmethod
    def local_to_world_2d_static(sim,obj,point_local):

        matrix = sim.getObjectMatrix(
            obj,
            -1
        )


        x = (
            matrix[0]*point_local[0]
            +
            matrix[1]*point_local[1]
            +
            matrix[2]*point_local[2]
            +
            matrix[3]
        )


        y = (
            matrix[4]*point_local[0]
            +
            matrix[5]*point_local[1]
            +
            matrix[6]*point_local[2]
            +
            matrix[7]
        )


        return np.array([x,y])