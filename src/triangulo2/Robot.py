"""Robô ePuck: campo potencial, agrupamento circular livre e fila.

O algoritmo de grupo não fixa posições dos seguidores. O círculo engloba
os DEZ robôs, incluindo o líder na parte frontal. Os seguidores equilibram
coesão, separação, contenção no disco e repulsão dos obstáculos; o líder
regula a velocidade conforme o vínculo com o restante do agrupamento. Os métodos geométricos de
obstáculos e os comandos diferenciais vêm da versão original.
"""
import numpy as np
import random
from Parameters import *


class Robot:
    def __init__(self, name, base, goal_paths, wheel_path, obstacles, sim):
        self.name = name
        self.base = base
        self.w_left, self.w_right = wheel_path
        self.velocity = np.zeros(2, dtype=float)
        self.force = np.zeros(2, dtype=float)
        self.position = np.zeros(2, dtype=float)
        self.orientation = 0.0
        self.sim = sim
        self.goal_paths = goal_paths
        self.goal_positions = [np.asarray(sim.getObjectPosition(g, -1)[:2], dtype=float)
                               for g in goal_paths]
        self.obstacles = obstacles
        self.leader = None
        self.line_order = None
        self.line_target_position = None
        self.max_error = 0.0
        # Velocidade linear realmente comandada após saturação das rodas.
        # Permite limitar o avanço dos seguidores em relação ao líder.
        self.commanded_linear_speed = 0.0
        self.reverse_mode = False
        self.avoidance_side = None
        self.avoidance_last_active = None

    @staticmethod
    def unit(vec, fallback=None):
        v = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            return v / norm
        if fallback is not None:
            return Robot.unit(fallback)
        return np.array([1.0, 0.0])

    @staticmethod
    def limit_force(vec, maximum):
        norm = np.linalg.norm(vec)
        return vec * min(1.0, maximum / norm) if norm > 1e-9 else vec

    @staticmethod
    def forward_direction(leader, target_position):
        fallback = np.array([np.cos(leader.orientation), np.sin(leader.orientation)])
        return Robot.unit(np.asarray(target_position) - leader.position, fallback)

    @staticmethod
    def group_center(robots):
        """Centro geométrico de TODOS os robôs, inclusive o líder.

        Assim o disco NÃO fica preso ao líder: o líder participa da dinâmica.
        """
        return np.mean([r.position for r in robots], axis=0)

    @staticmethod
    def leader_front_target(robots, leader, target_position):
        """Ponto de referência flexível na frente dos demais robôs.

        Não é um slot para cada seguidor. Apenas posiciona o líder próximo
        da borda dianteira da nuvem, permitindo aproximar-se dela.
        """
        followers = [r.position for r in robots if r is not leader]
        if not followers:
            return leader.position.copy()
        follower_center = np.mean(followers, axis=0)
        forward = Robot.forward_direction(leader, target_position)
        return follower_center + LEADER_FRONT_OFFSET * forward

    def separation_force(self, robots, desired_distance, gain, max_force):
        """Repulsão suave SOMENTE entre vizinhos próximos.

        Inclui o líder. Um termo adicional age quando há risco de contato.
        O valor de DESIRED_DISTANCE nunca é exigido para pares distantes.
        """
        total = np.zeros(2, dtype=float)
        for other in robots:
            if other is self:
                continue
            delta = self.position - other.position
            distance = np.linalg.norm(delta)
            if distance < 1e-8:
                # Não deixamos duas posições coincidentes anularem a repulsão.
                delta = np.array([1.0, 0.0]) if self.name < other.name else np.array([-1.0, 0.0])
                distance = 1e-8
            if distance >= desired_distance:
                continue
            direction = delta / distance if distance > 1e-6 else Robot.unit(delta)
            intensity = gain * (desired_distance - distance) / desired_distance
            safety_distance = 2 * ROBOT_RADIUS + 0.025
            if distance < safety_distance:
                intensity += gain * 2.0 * (safety_distance - distance) / safety_distance
            total += intensity * direction
        return Robot.limit_force(total, max_force)

    def repulsive_r2r(self, robots):
        """Repulsão de segurança robô-robô, inspirada na função original.

        A repulsão suave entre vizinhos continua em separation_force().
        Esta componente extra só atua quando a distância ENTRE CENTROS
        cai abaixo de (2 * ROBOT_RADIUS + R2R_SAFETY_MARGIN).

        A limitação de magnitude evita a singularidade da versão antiga
        baseada em 1/dist**2 e não combate o LINE_DISTANCE = 0.12 m.
        A força inclui o líder e os seguidores, em todos os modos móveis.
        """
        total = np.zeros(2, dtype=float)
        if robots is None:
            return total

        safe_center_distance = 2.0 * ROBOT_RADIUS + R2R_SAFETY_MARGIN
        margin = max(R2R_SAFETY_MARGIN, 1e-6)

        for other in robots:
            if other is self:
                continue

            delta = self.position - other.position
            distance = np.linalg.norm(delta)
            if distance >= safe_center_distance:
                continue

            if distance < 1e-8:
                # Resolve uma coincidência numérica sem força nula.
                direction = (np.array([1.0, 0.0]) if self.name < other.name
                             else np.array([-1.0, 0.0]))
            else:
                direction = delta / distance

            # Cresce continuamente ao se aproximar de uma colisão.
            # O saturador global limita a soma de vários vizinhos.
            proximity = (safe_center_distance - distance) / margin
            strength = K_REP_ROBOTS * min(proximity, 2.0) ** 2
            total += strength * direction

        total = Robot.limit_force(total, F_REP_ROBOTS_MAX)
        self.force = self.force + total
        return total

    def group_force(self, robots, leader, target_position, moving=False,
                    reforming=False, safety_robots=None):
        """Agrupamento espontâneo: coesão + separação + contenção + obstáculos."""
        center = Robot.group_center(robots)
        offset = center - self.position
        d = np.linalg.norm(offset)

        cohesion = Robot.limit_force(K_GROUP_COHESION * offset, F_GROUP_COHESION_MAX)
        confinement = np.zeros(2, dtype=float)
        active_radius = GROUP_RADIUS - GROUP_BOUNDARY_MARGIN
        if d > active_radius:
            confinement = K_GROUP_CONTAINMENT * (d - active_radius) * Robot.unit(offset)
        separation = self.separation_force(
            robots, DESIRED_DISTANCE, K_GROUP_SEPARATION, F_GROUP_SEPARATION_MAX
        )
        # Barreira longitudinal suave: líder deve ficar à frente dos seguidores.
        forward = Robot.forward_direction(leader, target_position)
        longitudinal = np.dot(self.position - leader.position, forward)
        front_correction = np.zeros(2, dtype=float)
        if longitudinal > -GROUP_FRONT_SLOW_ZONE:
            # Correção gradual já antes de atingir o limite frontal.
            front_correction = (-K_GROUP_FRONT *
                                (longitudinal + GROUP_FRONT_SLOW_ZONE) * forward)
        # O grupo inteiro precisa se deslocar, não apenas ser arrastado pelo líder.
        collective_drive = (K_GROUP_FOLLOW_DRIVE * forward
                            if moving else np.zeros(2, dtype=float))
        # No reagrupamento, o líder é a referência da formação, porém os
        # seguidores continuam livres dentro do disco (sem posições fixas).
        # Seu alvo coletivo é um centro virtual um pouco ATRÁS do líder.
        reform_anchor = np.zeros(2, dtype=float)
        if reforming:
            desired_center = leader.position - LEADER_FRONT_OFFSET * forward
            reform_anchor = Robot.limit_force(
                K_GROUP_REFORM_ANCHOR * (desired_center - self.position),
                F_GROUP_REFORM_ANCHOR_MAX
            )
        self.force = Robot.limit_force(cohesion + confinement + separation
                                       + front_correction + collective_drive
                                       + reform_anchor, F_GROUP_TOTAL_MAX)
        rep = self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER)
        self.tangential_avoidance_force(center, rep)
        self.repulsive_r2r(robots if safety_robots is None else safety_robots)
        self.force = Robot.limit_force(self.force, F_GROUP_TOTAL_MAX)
        # Se atingiu/ultrapassou a faixa frontal, elimina forças que
        # empurrem o seguidor ainda mais à frente. Não interfere na repulsão
        # lateral nem na força que o traz de volta para trás do líder.
        if longitudinal >= -GROUP_FRONT_GAP:
            forward_force = float(np.dot(self.force, forward))
            if forward_force > 0.0:
                self.force = self.force - forward_force * forward
        self.max_error = max(0.0, d - GROUP_RADIUS)

    def leader_group_force(self, robots, target_position, navigating=False):
        """O líder também sente coesão, separação e obstáculos.

        FORM_GROUP: move-se para a frente da nuvem dos seguidores, sem avançar
        sozinho para o Goal.
        MOVE_GROUP: soma atração ao Goal enquanto mantém contato com
        o grupo. Se abrir distância, desacelera sem recuar.
        """
        desired_front = Robot.leader_front_target(robots, self, target_position)
        error = desired_front - self.position
        forward = Robot.forward_direction(self, target_position)

        if navigating:
            # Só bloquear avanço quando o líder estiver à frente demais.
            ahead = max(0.0, np.dot(-error, forward))
            lateral_error = np.linalg.norm(error - np.dot(error, forward) * forward)
            recovery = max(ahead, max(0.0, lateral_error - LEADER_RECOVERY_SOFT))
            goal_scale = float(np.clip(
                (LEADER_RECOVERY_HARD - recovery) /
                (LEADER_RECOVERY_HARD - LEADER_RECOVERY_SOFT), 0.0, 1.0))
            # Corrige desvios laterais e recupera a frente caso o líder
            # esteja atrasado. Se estiver à frente DEMAIS, NÃO aplica
            # componente que o faria retornar para trás: desacelera o
            # avanço para que os seguidores o alcancem.
            longitudinal = np.dot(error, forward)
            lateral = error - longitudinal * forward
            pull = K_LEADER_COHESION * (
                lateral + max(0.0, longitudinal) * forward
            )
            drive = K_LEADER_GOAL * goal_scale * forward
            base = pull + drive
        else:
            base = K_LEADER_FORMATION * error

        # Inclui os seguidores na repulsão física do próprio líder.
        separation = self.separation_force(
            robots, DESIRED_DISTANCE, K_GROUP_SEPARATION, F_GROUP_SEPARATION_MAX
        )
        self.force = Robot.limit_force(base + separation, F_GROUP_TOTAL_MAX)
        # Para não sacrificar a coesão, a repulsão externa permanece saturada.
        rep = self.repulsive_force(
            repulsion_scale=REP_SCALE_LEADER if navigating else REP_SCALE_FOLLOWER
        )
        self.tangential_avoidance_force(
            target_position if navigating else desired_front, rep
        )
        self.repulsive_r2r(robots)
        self.force = Robot.limit_force(self.force, F_GROUP_TOTAL_MAX)
        self.max_error = float(np.linalg.norm(error))
        return self.max_error

    def line_force(self, robots, static=False, line_exit_target=None):
        """Linha fixa atrás do líder durante preparação; em movimento segue o vizinho."""
        if self.line_order is None or self not in self.line_order or self.leader is None:
            self.force = np.zeros(2)
            self.max_error = 0.0
            return
        i = self.line_order.index(self)
        if i == 0:
            self.force = np.zeros(2)
            self.max_error = 0.0
            return
        forward = Robot.forward_direction(self.leader, self.line_target_position)
        if line_exit_target is not None:
            # O vizinho da frente já saiu: manter o alvo na passagem.
            point = np.asarray(line_exit_target, dtype=float)
        elif static:
            point = self.leader.position - i * LINE_DISTANCE * forward
        else:
            front_robot = self.line_order[i - 1]
            point = front_robot.position - LINE_DISTANCE * forward
        error = point - self.position
        self.max_error = np.linalg.norm(error)
        self.force = K_LINE * error
        separation = self.separation_force(robots, LINE_REPULSION_DISTANCE,
                                           K_LINE_SEPARATION, F_LINE_SEPARATION_MAX)
        self.force = Robot.limit_force(self.force + separation, F_GROUP_TOTAL_MAX)
        rep = self.repulsive_force(repulsion_scale=REP_SCALE_FOLLOWER,
                                   rep_range=REP_RANGE_LINE)
        # Na fila também é necessário contornar as paredes, não só repelir.
        self.tangential_avoidance_force(point, rep)
        self.repulsive_r2r(robots)
        self.force = Robot.limit_force(self.force, F_GROUP_TOTAL_MAX)

    def run(self, robots=None, mode="group", target_position=None,
            leader=None, static_line=False, formation_context="group",
            moving=False, reforming=False, safety_robots=None,
            line_exit_target=None, passage_guard=None,
            passage_phase=None, passage_reforming=False):
        # As posições são atualizadas juntas, uma única vez no main.
        if mode == "stop":
            self.stop_robot()
            return False
        if mode == "group_leader_form":
            if target_position is None:
                self.stop_robot()
                return False
            error = self.leader_group_force(robots, target_position, navigating=False)
            speed = self.calculate_linear_speed(
                error, 0.02, LEADER_FORMATION_SPEED_MAX,
                FORM_ERROR_MIN, FORM_ERROR_MAX
            )
            self.set_wheel_speeds(speed, allow_reverse=True)
            return False
        if mode == "group_leader_reform":
            # O líder NÃO recua na reconstrução da formação: segue apontado
            # ao Goal com velocidade baixa, podendo esperar os seguidores.
            if target_position is None:
                self.stop_robot()
                return False
            if self.arrived_target(target_position, stop=True):
                return True
            # Na travessia, depois de liberar a saída, não se afastar
            # indefinidamente enquanto nenhum seguidor terminou de passar.
            if passage_reforming and len(robots) <= 1:
                self.stop_robot()
                return False
            self.force = np.zeros(2, dtype=float)
            self.attraction_force(target_position)
            rep = self.repulsive_force(repulsion_scale=REP_SCALE_LEADER)
            self.tangential_avoidance_force(target_position, rep)
            self.repulsive_r2r(robots if safety_robots is None else safety_robots)
            self.force = Robot.limit_force(self.force, F_GROUP_TOTAL_MAX)
            # Reagrupamento normal permanece INALTERADO. Durante a
            # travessia progressiva, usa-se outra faixa de velocidade;
            # a coesão considera só o conjunto que já saiu do corredor.
            if passage_reforming:
                v_min = V_MIN_LEADER_PASSAGE_REFORM
                v_max = V_MAX_LEADER_PASSAGE_REFORM
            else:
                v_min = V_MIN_LEADER_REFORM
                v_max = V_MAX_LEADER_REFORM
            base_speed = self.calculate_linear_speed(
                np.linalg.norm(target_position - self.position),
                v_min, v_max,
                GOAL_SPEED_DIST_MIN, GOAL_SPEED_DIST_MAX
            )
            speed = base_speed * self.leader_cohesion_scale(
                robots, target_position, formation_context="group"
            )
            # Sem ré, mesmo se uma força repulsiva apontar para trás.
            self.set_wheel_speeds(speed, allow_reverse=False)
            return False
        if mode == "group_leader_move":
            if target_position is None:
                self.stop_robot()
                return False
            if self.arrived_target(target_position, stop=True):
                return True
            error = self.leader_group_force(robots, target_position, navigating=True)
            speed = self.calculate_linear_speed(
                max(error, np.linalg.norm(self.force)),
                V_MIN_LEADER, V_MAX_LEADER_GOAL, FORM_ERROR_MIN, FORM_ERROR_MAX
            )
            # Líder não dá ré para recuperar o grupo: freia/espera.
            self.set_wheel_speeds(speed, allow_reverse=False)
            return False
        if mode == "group":
            if leader is None or target_position is None or self is leader:
                self.stop_robot()
                return False
            self.group_force(robots, leader, target_position,
                             moving=moving, reforming=reforming,
                             safety_robots=safety_robots)
            if passage_guard is not None:
                # Após sair, não permitir que a coesão do grupo puxe o
                # agente de volta para dentro da passagem estreita.
                origin, forward, exit_progress = passage_guard
                progress = float(np.dot(self.position - origin, forward))
                if progress < exit_progress + LEADER_FRONT_OFFSET:
                    backward_force = float(np.dot(self.force, forward))
                    if backward_force < 0.0:
                        self.force = self.force - backward_force * forward
            if reforming:
                # Seguidores mais ágeis para alcançar o líder ainda em avanço.
                linear_speed = self.calculate_linear_speed(
                    np.linalg.norm(self.force), V_MIN_FORMATION,
                    V_MAX_REFORM_FOLLOWER, FORM_ERROR_MIN, REFORM_ERROR_MAX
                )
            else:
                linear_speed = self.calculate_linear_speed(
                    np.linalg.norm(self.force), V_MIN_FORMATION, V_MAX_FORMATION,
                    FORM_ERROR_MIN, FORM_ERROR_MAX
                )
            self.set_wheel_speeds(
                linear_speed, leader=leader, target_position=target_position
            )
            return False
        if mode == "line":
            self.line_force(robots, static=static_line,
                            line_exit_target=line_exit_target)
            linear_speed = self.calculate_linear_speed(
                self.max_error, V_MIN_LINE, V_MAX_LINE,
                LINE_ERROR_MIN, LINE_ERROR_MAX
            )
            self.set_wheel_speeds(linear_speed)
            return False
        if mode == "go_to_goal":
            if target_position is None:
                self.stop_robot()
                return False
            if self.arrived_target(target_position, stop=True):
                return True
            self.force = np.zeros(2)
            self.attraction_force(target_position)
            repulsive = self.repulsive_force(
                repulsion_scale=REP_SCALE_LEADER,
                rep_range=(REP_RANGE_LINE if formation_context == "line"
                           else REP_RANGE)
            )
            self.tangential_avoidance_force(target_position, repulsive)
            self.repulsive_r2r(robots)
            self.force = Robot.limit_force(self.force, F_GROUP_TOTAL_MAX)
            rho = np.linalg.norm(target_position - self.position)
            if passage_phase == 'crossing':
                v_min, v_max = V_MIN_LEADER_CROSSING, V_MAX_LEADER_CROSSING
            elif passage_phase == 'clear_exit':
                v_min, v_max = V_MIN_LEADER_CLEAR_EXIT, V_MAX_LEADER_CLEAR_EXIT
            elif passage_phase is not None:
                raise ValueError(f"Fase de passagem desconhecida: {passage_phase}")
            elif formation_context == "line":
                v_min, v_max = V_MIN_LEADER_LINE, V_MAX_LEADER_LINE
            else:
                v_min, v_max = V_MIN_LEADER, V_MAX_LEADER_GOAL
            v = self.calculate_linear_speed(
                rho, v_min, v_max,
                GOAL_SPEED_DIST_MIN, GOAL_SPEED_DIST_MAX
            )
            # Na passagem, a distância até o primeiro seguidor é proposital
            # (partidas espaçadas em 1 s). Não reduzir a velocidade por ela.
            # As forças de obstáculos e a proteção contra colisões continuam.
            if passage_phase is None:
                v *= self.leader_cohesion_scale(
                    robots, target_position, formation_context)
            self.set_wheel_speeds(v, allow_reverse=False)
            return False
        self.stop_robot()
        raise ValueError(f"Modo desconhecido: {mode}")

    @staticmethod
    def group_metrics(robots, leader, target_position):
        """Coesão do disco dos 10 robôs (líder incluído).

        Além da ocupação/compacidade, mede diretamente a distância entre o
        líder e seu seguidor mais próximo. Isso evita que o líder fique solto
        mesmo quando os nove seguidores formam um aglomerado entre si.
        """
        followers = [r for r in robots if r is not leader]
        if not followers:
            return {"ready": True, "inside": 1.0, "max_radius": 0.0,
                    "centroid_error": 0.0, "min_neighbor": float('inf'),
                    "max_nearest": 0.0, "leader_nearest": 0.0}

        center = Robot.group_center(robots)
        positions = np.stack([r.position for r in robots])
        radii = np.linalg.norm(positions - center, axis=1)
        inside = float(np.mean(radii <= GROUP_RADIUS + DIST_TOL))
        # Diferentemente da versão anterior, este erro NÃO compara o
        # centróide com ele mesmo: mede a integração do líder à nuvem.
        desired_front = Robot.leader_front_target(robots, leader, target_position)
        centroid_error = float(np.linalg.norm(leader.position - desired_front))

        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        closest_distances = distances.min(axis=1)
        leader_index = robots.index(leader)
        leader_nearest = float(closest_distances[leader_index])
        min_neighbor = float(np.min(closest_distances))
        max_nearest = float(np.max(closest_distances))

        ready = (inside == 1.0
                 and centroid_error <= GROUP_CENTER_TOL
                 and min_neighbor >= GROUP_MIN_NEIGHBOR_DISTANCE
                 and max_nearest <= GROUP_MAX_NEAREST_DISTANCE
                 and leader_nearest <= LEADER_NEIGHBOR_HARD_DISTANCE)

        return {"ready": ready, "inside": inside,
                "max_radius": float(np.max(radii)),
                "centroid_error": centroid_error,
                "min_neighbor": min_neighbor, "max_nearest": max_nearest,
                "leader_nearest": leader_nearest}

    @staticmethod
    def line_max_error(robots, leader, target_position):
        forward = Robot.forward_direction(leader, target_position)
        errors = []
        for r in robots:
            if r is leader:
                continue
            i = r.line_order.index(r)
            point = leader.position - i * LINE_DISTANCE * forward
            errors.append(np.linalg.norm(point - r.position))
        return max(errors, default=0.0)

    def leader_cohesion_scale(self, robots, target_position, formation_context="group"):
        if robots is None or len(robots) <= 1:
            return 1.0
        if formation_context == "line":
            if not self.line_order or len(self.line_order) <= 1:
                return 1.0
            first = self.line_order[1]
            gap = np.linalg.norm(first.position - self.position) - LINE_DISTANCE
            minimum = LEADER_LINE_MIN_SCALE
        elif formation_context == "group":
            metrics = Robot.group_metrics(robots, self, target_position)
            overflow = max(0.0, metrics['max_radius'] - GROUP_RADIUS)
            centroid_overflow = max(0.0, metrics['centroid_error']
                                    - LEADER_CENTROID_SOFT_TOL)
            gap = max(overflow, centroid_overflow)
            minimum = LEADER_GROUP_MIN_SCALE
        else:
            raise ValueError(f"Contexto desconhecido: {formation_context}")

        if gap <= LEADER_GAP_SOFT:
            scale = 1.0
        elif gap >= LEADER_GAP_HARD:
            scale = minimum
        else:
            fraction = (gap - LEADER_GAP_SOFT) / (LEADER_GAP_HARD - LEADER_GAP_SOFT)
            scale = 1.0 - fraction * (1.0 - minimum)

        # Só na formação circular: velocidade depende também do elo entre
        # líder e seu seguidor mais próximo. Isso impede uma liderança isolada.
        if formation_context == "group":
            nearest = metrics['leader_nearest']
            if nearest >= LEADER_NEIGHBOR_HARD_DISTANCE:
                proximity_scale = 0.0
            elif nearest <= LEADER_NEIGHBOR_SOFT_DISTANCE:
                proximity_scale = 1.0
            else:
                proximity_scale = ((LEADER_NEIGHBOR_HARD_DISTANCE - nearest) /
                                   (LEADER_NEIGHBOR_HARD_DISTANCE
                                    - LEADER_NEIGHBOR_SOFT_DISTANCE))
            scale = min(scale, proximity_scale)

        return float(np.clip(scale, minimum, 1.0))

    @staticmethod
    def prepare_line_formation(robots, target_position):
        """Ordem ORIGINAL de passagem: distância crescente até o Goal.

        O primeiro é o líder; não há redistribuição por slots atrás dele
        nem pré-alinhamento. A ordem fica congelada durante a travessia.
        """
        if not robots:
            raise ValueError("A fila precisa de pelo menos um robô.")
        target = np.asarray(target_position, dtype=float)
        line_order = sorted(
            robots,
            key=lambda r: np.linalg.norm(r.position - target)
        )
        leader = line_order[0]
        for r in robots:
            r.leader = leader
            r.line_order = line_order
            r.line_target_position = target.copy()
        return leader, line_order

    @staticmethod
    def clear_line_data(robots):
        for r in robots:
            r.line_order = None
            r.line_target_position = None
            r.leader = None

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


    def set_wheel_speeds(self, linear_speed, allow_reverse=True,
                         leader=None, target_position=None):

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
        # Movendo em ré, a orientação alvo do CHASSI é oposta à direção
        # de translação desejada. Corrige o erro angular para esse caso.
        steering_angle = self.wrap_to_pi(angle - np.pi) if self.reverse_mode else angle

        # Evita transladar de lado enquanto ainda está girando. Vale também
        # para os seguidores, que antes se deslocavam mesmo a 90° do alvo.
        v *= max(0.0, np.cos(steering_angle))

        if self.reverse_mode:
            v = -v

        # Segurança geométrica: diminui a translação na direção da parede.
        # A rotação é mantida, para permitir que o robô desvie.
        heading = np.array([np.cos(self.orientation), np.sin(self.orientation)])
        movement_direction = heading if v >= 0 else -heading
        v *= self.obstacle_speed_scale(movement_direction)

        # Barreira de velocidade longitudinal APENAS no agrupamento.
        # Quem persegue a formação pode correr para alcançá-la, porém sua
        # projeção de velocidade rumo ao Goal não pode ultrapassar a do
        # líder ao se aproximar do plano frontal de segurança.
        # Não atua no modo linha, preservando as partidas escalonadas.
        if leader is not None and target_position is not None and leader is not self:
            forward = Robot.forward_direction(leader, target_position)
            distance_behind = float(np.dot(leader.position - self.position, forward))
            if distance_behind < GROUP_FRONT_SLOW_ZONE:
                leader_heading = np.array([
                    np.cos(leader.orientation), np.sin(leader.orientation)
                ])
                leader_forward_speed = max(
                    0.0, float(leader.commanded_linear_speed *
                               np.dot(leader_heading, forward))
                )
                available_gap = max(0.0, distance_behind - GROUP_FRONT_GAP)
                extra_speed = available_gap / max(GROUP_FRONT_RESPONSE_TIME, 1e-6)
                maximum_forward_speed = leader_forward_speed + extra_speed
                current_forward_speed = float(v * np.dot(heading, forward))
                if current_forward_speed > maximum_forward_speed:
                    v *= maximum_forward_speed / current_forward_speed

        # ========================================
        # Velocidade angular
        # ========================================

        w = K_W * steering_angle

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

        self.commanded_linear_speed = 0.5 * (wr + wl) * WHEEL_RADIUS
        self.sim.setJointTargetVelocity(
            self.w_left,
            float(wl)
        )

        self.sim.setJointTargetVelocity(
            self.w_right,
            float(wr)
        )


    def stop_robot(self):
        self.commanded_linear_speed = 0.0
        self.sim.setJointTargetVelocity(self.w_left, 0.0)
        self.sim.setJointTargetVelocity(self.w_right, 0.0)


    def attraction_force(self, target_position):
        """Atração linear perto do Goal e limitada para distâncias maiores.

        Evita que a atração cresça indefinidamente quando os Goals são
        afastados ao ampliar a cena, preservando a direção do objetivo.
        """
        delta = target_position - self.position
        distance = np.linalg.norm(delta)

        if distance < 1e-9:
            return

        direction = delta / distance
        magnitude = min(K_ATT * distance, F_ATT_MAX)
        self.force = self.force + magnitude * direction


    def obstacle_clearance_and_direction(self, point, obs):
        """(distância livre entre superfícies, direção para fora do obstáculo).

        Usa a geometria em planta já existente: parede aproximada por
        segmento com espessura e cilindro aproximado por círculo.
        Uma distância negativa indica interpenetração aproximada.
        """
        p = np.asarray(point, dtype=float)
        if obs['type'] == 'wall':
            distance, closest = Robot.point_to_segment_distance(
                p, obs['a'], obs['b'])
            clearance = distance - obs['thickness'] / 2.0 - ROBOT_RADIUS
            away = p - closest
            if distance < 1e-8:
                segment = np.asarray(obs['b']) - np.asarray(obs['a'])
                away = np.array([-segment[1], segment[0]], dtype=float)
        else:
            away = p - obs['position']
            distance = np.linalg.norm(away)
            clearance = distance - obs.get('radius', OBSTACLE_RADIUS) - ROBOT_RADIUS
        return float(clearance), Robot.unit(away)

    def nearest_obstacle_clearance(self):
        """Menor afastamento robô-obstáculo, para diagnóstico no terminal."""
        return min((self.obstacle_clearance_and_direction(self.position, obs)[0]
                    for obs in self.obstacles), default=float('inf'))

    def obstacle_speed_scale(self, movement_direction):
        """Freia antes da colisão sem bloquear movimento paralelo à parede.

        A força potencial decide a orientação, mas um diferencial não consegue
        alterar sua direção de translação instantaneamente. Este freio limita
        a velocidade ENQUANTO o robô ainda aponta para o obstáculo.
        """
        motion = Robot.unit(movement_direction)
        scale = 1.0
        width = max(OBSTACLE_SLOW_CLEARANCE - OBSTACLE_STOP_CLEARANCE, 1e-6)
        for obs in self.obstacles:
            clearance, away = self.obstacle_clearance_and_direction(self.position, obs)
            if clearance >= OBSTACLE_SLOW_CLEARANCE:
                continue
            # Positivo apenas ao andar na direção da superfície.
            approach = max(0.0, -float(np.dot(motion, away)))
            if approach <= OBSTACLE_APPROACH_THRESHOLD:
                continue
            if clearance <= OBSTACLE_STOP_CLEARANCE:
                return 0.0
            hazard = np.clip((OBSTACLE_SLOW_CLEARANCE - clearance) / width,
                             0.0, 1.0)
            hazard *= approach
            scale = min(scale, 1.0 - hazard)
        return float(np.clip(scale, 0.0, 1.0))

    def repulsive_force(self, repulsion_scale=1.0, rep_range=None):
        """Repulsão contínua que começa cedo e não diverge perto da parede.

        O campo anterior começava só a 8 cm e, nos seguidores, era multiplicado
        por 0,05 e cortado em 0,25: ficava incapaz de vencer a formação.
        A repulsão atual é progressiva ao entrar no alcance de segurança.
        """
        rep = np.zeros(2, dtype=float)
        reach = REP_RANGE if rep_range is None else float(rep_range)
        for obs in self.obstacles:
            clearance, away = self.obstacle_clearance_and_direction(self.position, obs)
            if clearance >= reach:
                continue
            proximity = np.clip(1.0 - clearance / max(reach, 1e-6), 0.0, 1.0)
            rep += K_REP * proximity ** 2 * away

        rep = repulsion_scale * rep
        maximum = (F_REP_MAX_FOLLOWER if repulsion_scale < 1.0
                   else F_REP_MAX_LEADER)
        rep = Robot.limit_force(rep, maximum)
        self.force = self.force + rep
        return rep


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
                    self.sim.getSimulationTime()
                    - self.avoidance_last_active
                    > AVOIDANCE_MEMORY_TIME
                ):
                    self.avoidance_side = None
                    self.avoidance_last_active = None

            return

        # Obstáculo está sendo evitado neste instante
        self.avoidance_last_active = self.sim.getSimulationTime()

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


    def detect_narrow_passage(self, target_position, return_pair=False):
        target_vector = target_position - self.position
        target_norm = np.linalg.norm(target_vector)

        if target_norm < 1e-6:
            return None if return_pair else False

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
            return None if return_pair else False

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
                    return (wall_1, wall_2) if return_pair else True

        return None if return_pair else False


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
    def create_epucks(sim, n_robots, obstacles=None):
        """Copia ePucks sem sobreposição com outros robôs ou obstáculos."""
        created = []
        template = sim.getObject('/ePuck1')
        initial = np.array(sim.getObjectPosition(template, -1)[:2], dtype=float)
        placed = [initial]
        for i in range(2, n_robots + 1):
            new_model = sim.copyPasteObjects([template], 1)[0]
            sim.setObjectAlias(new_model, f'ePuck{i}')
            position = None
            for _ in range(2000):
                candidate = np.array([random.uniform(2.0, 4.75),
                                      random.uniform(3.0, 4.75)])
                if any(np.linalg.norm(candidate - old) < SPAWN_MIN_DISTANCE
                       for old in placed):
                    continue
                safe = True
                for obs in (obstacles or []):
                    if obs['type'] == 'wall':
                        center_distance, _ = Robot.point_to_segment_distance(
                            candidate, obs['a'], obs['b']
                        )
                        surface_distance = (center_distance - obs['thickness'] / 2.0
                                            - ROBOT_RADIUS)
                    else:
                        surface_distance = (np.linalg.norm(candidate - obs['position'])
                                            - obs.get('radius', OBSTACLE_RADIUS) - ROBOT_RADIUS)
                    if surface_distance < SPAWN_OBSTACLE_CLEARANCE:
                        safe = False
                        break
                if safe:
                    position = candidate
                    break
            if position is None:
                # Nunca aceita uma sobreposição forçada em silêncio.
                raise RuntimeError('Não foi possível gerar posições iniciais sem sobreposição.')
            sim.setObjectPosition(new_model, sim.handle_world,
                                  [float(position[0]), float(position[1]), 0.01915])
            placed.append(position)
            created.append(new_model)
        return created

    @staticmethod
    def remove_created_epucks(sim, created_epucks):
        for epuck in created_epucks:
            sim.removeModel(epuck)


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

                # Usa o raio real da geometria em planta, quando acessível.
                try:
                    sx = (sim.getObjectFloatParam(obs, sim.objfloatparam_objbbox_max_x)
                          - sim.getObjectFloatParam(obs, sim.objfloatparam_objbbox_min_x))
                    sy = (sim.getObjectFloatParam(obs, sim.objfloatparam_objbbox_max_y)
                          - sim.getObjectFloatParam(obs, sim.objfloatparam_objbbox_min_y))
                    radius = max(sx, sy) / 2.0
                    if radius <= 0.0:
                        radius = OBSTACLE_RADIUS
                except Exception:
                    radius = OBSTACLE_RADIUS
                cache.append({
                    'type': 'circle',
                    'position': np.array([pos[0], pos[1]]),
                    'radius': float(radius),
                })


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

