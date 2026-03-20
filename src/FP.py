import math
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class RobotFormation:
    # =========================
    # Constantes e parâmetros do robô
    # =========================
    WHEEL_RADIUS = 0.0425 / 2
    AXLE_LENGTH = 0.054
    LEFT_SIGN = 1
    RIGHT_SIGN = 1
    MAX_WHEEL_SPEED = 10.0
    ANG_TOL = 0.08
    K_RHO = 1.5
    K_ALPHA = 3.0
    CONTROL_DT = 0.05
    YAW_OFFSET = math.pi / 2
    DESIRED_DISTANCE = 1.0  # Distância desejada para a formação na fila
    DIST_MIN = 0.8
    DIST_MAX = 1.2
    BEHIND_POINT_TOL = 0.25

    def __init__(self, sim):
        self.sim = sim
        self.leader_elected = False
        self.leader_name = ''
        self.leader_pose = (0, 0, 0)  # Inicializa a pose do líder
        self.time_elapsed = 0  # Contador de tempo para mover o robô 2

        # Handles dos robôs
        self.robot1_base = self.sim.getObject('/base1')
        self.robot1_left = self.sim.getObject('/leftJoint1')
        self.robot1_right = self.sim.getObject('/rightJoint1')

        self.robot2_base = self.sim.getObject('/base2')
        self.robot2_left = self.sim.getObject('/leftJoint2')
        self.robot2_right = self.sim.getObject('/rightJoint2')

        self.robot3_base = self.sim.getObject('/base3')
        self.robot3_left = self.sim.getObject('/leftJoint3')
        self.robot3_right = self.sim.getObject('/rightJoint3')

        print('Handles obtidos com sucesso.')

    # =========================
    # Funções auxiliares
    # =========================
    def wrap_to_pi(self, angle):
        """Limita o ângulo no intervalo de -π a π"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def get_pose_2d(self, base_handle):
        """Retorna a posição 2D (x, y) e o ângulo yaw do robô"""
        pos = self.sim.getObjectPosition(base_handle, -1)
        ori = self.sim.getObjectOrientation(base_handle, -1)

        x = pos[0]
        y = pos[1]
        yaw = self.wrap_to_pi(ori[2] + self.YAW_OFFSET)

        return x, y, yaw

    def set_wheel_speeds(self, left_motor, right_motor, v, w):
        """Define as velocidades das rodas com base nas velocidades linear e angular"""
        wl = (v - (self.AXLE_LENGTH / 2.0) * w) / self.WHEEL_RADIUS
        wr = (v + (self.AXLE_LENGTH / 2.0) * w) / self.WHEEL_RADIUS

        wl *= self.LEFT_SIGN
        wr *= self.RIGHT_SIGN

        wl = max(-self.MAX_WHEEL_SPEED, min(self.MAX_WHEEL_SPEED, wl))
        wr = max(-self.MAX_WHEEL_SPEED, min(self.MAX_WHEEL_SPEED, wr))

        self.sim.setJointTargetVelocity(left_motor, wl)
        self.sim.setJointTargetVelocity(right_motor, wr)

    def stop_robot(self, left_motor, right_motor):
        """Para o robô, definindo velocidade zero nas rodas"""
        self.sim.setJointTargetVelocity(left_motor, 0.0)
        self.sim.setJointTargetVelocity(right_motor, 0.0)

    def controller_to_point(self, x, y, yaw, x_goal, y_goal):
        """Controlador de ponto: calcula a velocidade linear e angular"""
        dx = x_goal - x
        dy = y_goal - y

        rho = math.hypot(dx, dy)
        desired_theta = math.atan2(dy, dx)
        alpha = self.wrap_to_pi(desired_theta - yaw)

        if abs(alpha) > 0.35:
            v = 0.0
            w = self.K_ALPHA * alpha
        else:
            v = self.K_RHO * rho
            w = self.K_ALPHA * alpha

        v = max(-0.20, min(0.20, v))
        w = max(-2.5, min(2.5, w))

        return v, w, rho, alpha

    # =========================
    # Funções de formação
    # =========================
    def formacao_linha(self, x1, y1, yaw1, x2, y2, yaw2, x3, y3, yaw3):
        """Formação linha (fila)"""
        xL, yL, yawL = self.leader_pose
        # Seguidor 1 vai para 1 metro atrás do líder
        xF1_goal = xL + self.DESIRED_DISTANCE  # O seguidor 1 vai 1 metro à frente do líder
        yF1_goal = yL
        # Seguidor 2 vai para 1 metro atrás do seguidor 1
        xF2_goal = xF1_goal + self.DESIRED_DISTANCE  # O seguidor 2 vai 1 metro à frente do seguidor 1
        yF2_goal = yF1_goal

        v1, w1, _, _ = self.controller_to_point(x2, y2, yaw2, xF1_goal, yF1_goal)
        v2, w2, _, _ = self.controller_to_point(x3, y3, yaw3, xF2_goal, yF2_goal)

        self.set_wheel_speeds(self.robot2_left, self.robot2_right, v1, w1)
        self.set_wheel_speeds(self.robot3_left, self.robot3_right, v2, w2)

    def formacao_triangulo(self, x2, y2, yaw2, x3, y3, yaw3):
        """Formação triângulo"""
        xL, yL, yawL = self.leader_pose
        # Seguidor 1 vai para 1 metro atrás do líder
        xF1_goal = xL + self.DESIRED_DISTANCE
        yF1_goal = yL
        # Seguidor 1 vai subir 1 metro no eixo Y
        yF1_goal = yL + self.DESIRED_DISTANCE

        # Seguidor 2 vai para 1 metro atrás do seguidor 1, sem subir no eixo Y
        xF2_goal = xF1_goal + self.DESIRED_DISTANCE
        yF2_goal = yF1_goal

        v1, w1, _, _ = self.controller_to_point(x2, y2, yaw2, xF1_goal, yF1_goal)
        v2, w2, _, _ = self.controller_to_point(x3, y3, yaw3, xF2_goal, yF2_goal)

        self.set_wheel_speeds(self.robot2_left, self.robot2_right, v1, w1)
        self.set_wheel_speeds(self.robot3_left, self.robot3_right, v2, w2)

    # =========================
    # Função principal
    # =========================
    def eleger_lider(self, x1, y1, x2, y2, x3, y3):
        """Escolhe o líder entre os robôs"""
        d1_center = math.hypot(x1, y1)
        d2_center = math.hypot(x2, y2)
        d3_center = math.hypot(x3, y3)

        if d1_center <= d2_center and d1_center <= d3_center:
            self.leader_name = 'ePuck1'
            self.leader_pose = (x1, y1, 0)  # Atualize com a pose real do líder
        elif d2_center <= d1_center and d2_center <= d3_center:
            self.leader_name = 'ePuck2'
            self.leader_pose = (x2, y2, 0)
        else:
            self.leader_name = 'ePuck3'
            self.leader_pose = (x3, y3, 0)

        self.stop_robot(self.robot1_left, self.robot1_right)  # Líder fica parado
        print(f'Líder eleito: {self.leader_name}')

    def controle_formacao(self):
        """Controla o movimento da formação inteira"""
        while True:
            x1, y1, yaw1 = self.get_pose_2d(self.robot1_base)
            x2, y2, yaw2 = self.get_pose_2d(self.robot2_base)
            x3, y3, yaw3 = self.get_pose_2d(self.robot3_base)

            if not self.leader_elected:
                self.eleger_lider(x1, y1, x2, y2, x3, y3)
                self.leader_elected = True

            # Se já elegeu o líder, começa a controlar a formação
            if self.leader_elected:
                # Primeiro, forma a linha (ou triângulo)
                self.formacao_linha(x1, y1, yaw1, x2, y2, yaw2, x3, y3, yaw3)

                # Se for necessário, forma o triângulo
                if self.time_elapsed >= 5:
                    self.formacao_triangulo(x2, y2, yaw2, x3, y3, yaw3)

            # Atualiza o tempo
            self.time_elapsed += self.CONTROL_DT
            time.sleep(self.CONTROL_DT)


def main():
    # =========================
    # Conexão com o CoppeliaSim
    # =========================
    client = RemoteAPIClient()
    try:
        sim = client.getObject('sim')
        print("Conexão com o CoppeliaSim bem-sucedida!")

        # Inicia a simulação
        sim.startSimulation()
        print("Simulação iniciada!")

        robot_formation = RobotFormation(sim)
        robot_formation.controle_formacao()

    except Exception as e:
        print(f"Erro ao conectar ao CoppeliaSim: {e}")
        exit(1)  # Fecha o programa em caso de erro


if __name__ == "__main__":
    main()