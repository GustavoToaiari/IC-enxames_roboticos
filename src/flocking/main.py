import time
import random

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from Robot import Robot
import Parameters


NUM_ROBOTS = 10

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

# Para reproduzir exatamente os sorteios de posicao dos ePucks copiados,
# defina RANDOM_SEED (inteiro) em Parameters.py. None mantem o original.
if Parameters.RANDOM_SEED is not None:
    random.seed(Parameters.RANDOM_SEED)

# ePuck1 ja existe na cena; os demais sao copiados a partir dele.
created_epucks = Robot.create_epucks(sim, NUM_ROBOTS)

goal_paths = [
    sim.getObject('/Goal1'),
    sim.getObject('/Goal2'),
    sim.getObject('/Goal3'),
    sim.getObject('/Goal4')
]

obstacles = [
    sim.getObject('/Cuboid1'), sim.getObject('/Cuboid2'), sim.getObject('/Cuboid3'),
    sim.getObject('/Cuboid4'), sim.getObject('/Cuboid5'), sim.getObject('/Cuboid6'),
    sim.getObject('/Cuboid7'), sim.getObject('/Cuboid8'), sim.getObject('/Cuboid9'),
    sim.getObject('/Cuboid10'), sim.getObject('/Cuboid11'), sim.getObject('/Cuboid12'),
    sim.getObject('/Cylinder1')
]

obstacle_cache = Robot.create_obstacle_cache(sim, obstacles)

robots = []
for i in range(1, NUM_ROBOTS + 1):
    robots.append(
        Robot(
            name=f'ePuck{i}',
            base=sim.getObject(f'/ePuck{i}/base'),
            goal_paths=goal_paths,
            wheel_path=(
                sim.getObject(f'/ePuck{i}/leftJoint'),
                sim.getObject(f'/ePuck{i}/rightJoint')
            ),
            obstacles=obstacle_cache,
            sim=sim
        )
    )

if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()

# A formacao por distancia e mantida APENAS para agregar os robos no inicio.
# Depois disso, o movimento em espaco aberto passa a usar Flocking + APF.
state = 'FORM_GROUP_INITIAL'
leader = None
line_order = None

current_goal_index = 0
target_position = robots[0].goal_positions[current_goal_index]


try:
    while True:
        client.step()

        # Atualizacao sincronizada das poses/velocidades antes dos controles.
        # Isto e importante para a regra de alinhamento do Flocking.
        for robot in robots:
            robot.get_pose_2d()

        # ============================================================
        # AGREGAÇÃO INICIAL
        # ============================================================
        if state == 'FORM_GROUP_INITIAL':
            # Reutiliza o controlador legado apenas para partir de um grupo
            # compacto, aproximando a hipotese do artigo de estado agregado.
            for robot in robots:
                robot.run(robots, mode='formation')

            if Robot.group_radius(robots) < Parameters.INITIAL_GROUP_RADIUS:
                for robot in robots:
                    robot.stop_robot()

                time.sleep(0.5)

                Robot.clear_line_data(robots)
                leader = Robot.choose_leader(robots, target_position)
                state = 'MOVE_FLOCK'

        # ============================================================
        # MOVIMENTO EM GRUPO: APF + FLOCKING
        # ============================================================
        elif state == 'MOVE_FLOCK':
            # Em Flocking nao existe um lider que puxa os seguidores.
            # Mantemos apenas o robo atualmente mais proximo do Goal como
            # referencia para detectar a passagem estreita.
            leader = Robot.choose_leader(robots, target_position)

            if leader.detect_narrow_passage(target_position):
                # Eq. (11) do artigo: ao entrar na estrategia de linha,
                # o Flocking deixa de atuar (alpha = 0 implicitamente).
                leader, line_order = Robot.prepare_line_formation(
                    robots,
                    target_position
                )

                line_start_time = sim.getSimulationTime()
                for robot in robots:
                    robot.line_start_time = line_start_time

                state = 'MOVE_LINE'
                continue

            # Estado group: todos os robos executam APF + Flocking.
            for robot in robots:
                robot.run(
                    robots,
                    mode='flocking',
                    target_position=target_position
                )

            # O Goal so e concluido quando todo o enxame entra na regiao.
            if Robot.flock_goal_reached(robots, target_position):
                for robot in robots:
                    robot.stop_robot()

                time.sleep(0.5)

                if current_goal_index < len(goal_paths) - 1:
                    current_goal_index += 1
                    target_position = robots[0].goal_positions[current_goal_index]
                    leader = Robot.choose_leader(robots, target_position)
                else:
                    time.sleep(1)
                    sim.stopSimulation()
                    break

        # ============================================================
        # MOVIMENTO EM LINHA
        # ============================================================
        elif state == 'MOVE_LINE':
            if line_order is None:
                raise RuntimeError('MOVE_LINE requer line_order valida.')

            last_robot = Robot.get_last_robot_line(line_order)

            # Quando a passagem fica para tras do ultimo robo, a memoria
            # da linha e apagada e o grupo volta diretamente ao Flocking.
            # Nao ha reconstrucao geometrica de triangulo: o Flocking faz
            # o reagrupamento de forma escalavel para 5 ou mais robos.
            if last_robot.detect_narrow_passage_backward():
                Robot.clear_line_data(robots)
                line_order = None
                leader = Robot.choose_leader(robots, target_position)
                state = 'MOVE_FLOCK'
                continue

            for robot in robots:
                if robot is leader:
                    robot.run(
                        robots,
                        mode='go_to_goal',
                        target_position=target_position,
                        formation_context='line'
                    )
                else:
                    robot.run(robots, mode='line_formation')

            # Caso o Goal esteja dentro/proximo da passagem, espera o grupo
            # inteiro chegar a regiao antes de trocar de objetivo.
            if Robot.flock_goal_reached(robots, target_position):
                for robot in robots:
                    robot.stop_robot()

                time.sleep(0.5)
                Robot.clear_line_data(robots)
                line_order = None

                if current_goal_index < len(goal_paths) - 1:
                    current_goal_index += 1
                    target_position = robots[0].goal_positions[current_goal_index]
                    leader = Robot.choose_leader(robots, target_position)
                    state = 'MOVE_FLOCK'
                else:
                    sim.stopSimulation()
                    break

        else:
            raise RuntimeError(f'Estado desconhecido: {state}')

except KeyboardInterrupt:
    client_close = RemoteAPIClient()
    sim = client_close.getObject('sim')
    sim.stopSimulation()


Robot.remove_created_epucks(
    sim,
    created_epucks
)