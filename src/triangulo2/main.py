import time

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from Robot import Robot
import Parameters


NUM_ROBOTS = 3

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

# ePuck1 já existe na cena; os demais são copiados a partir dele.
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

state = 'FORM_TRIANGLE_INITIAL'
leader = None
line_order = None
transition_reason = None  # 'passage' ou 'goal'

current_goal_index = 0
target_position = None


try:
    while True:
        client.step()

        # Atualiza todas as poses uma única vez no início do ciclo.
        for robot in robots:
            robot.get_pose_2d()

        # ============================================================
        # FORMAÇÃO TRIANGULAR INICIAL
        # ============================================================
        if state == 'FORM_TRIANGLE_INITIAL':
            for robot in robots:
                robot.run(robots, mode='formation')

            if Robot.triangle_max_error(robots) < Parameters.DIST_TOL:
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                target_position = robots[0].goal_positions[current_goal_index]

                Robot.clear_line_data(robots)
                leader = Robot.choose_leader(robots, target_position)

                state = 'MOVE_TRIANGLE'

        # ============================================================
        # MOVIMENTO EM TRIÂNGULO
        # ============================================================
        elif state == 'MOVE_TRIANGLE':

            # Se o líder detectou uma passagem estreita, prepara a fila.
            if leader.detect_narrow_passage(target_position):
                leader, line_order = Robot.prepare_line_formation(
                    robots,
                    target_position
                )

                line_start_time = time.time()
                for robot in robots:
                    robot.line_start_time = line_start_time

                state = 'MOVE_LINE'
                continue

            for robot in robots:
                if robot is leader:
                    robot.run(
                        robots,
                        mode='go_to_goal',
                        target_position=target_position,
                        formation_context='triangle'
                    )
                else:
                    robot.run(robots, mode='formation')

            if leader.arrived_target(target_position):
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                if current_goal_index < len(goal_paths) - 1:
                    state = 'FORM_TRIANGLE_AFTER_GOAL'
                else:
                    time.sleep(2)
                    sim.stopSimulation()
                    break

        # ============================================================
        # MOVIMENTO EM LINHA
        # ============================================================
        elif state == 'MOVE_LINE':
            last_robot = Robot.get_last_robot_line(line_order)

            # A passagem estreita apareceu no campo traseiro do último robô:
            # ela já ficou para trás do grupo, então inicia a reconstrução.
            if last_robot.detect_narrow_passage_backward():
                transition_reason = 'passage'
                state = 'FORM_TRIANGLE_TRANSITION'
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

            # Se o Goal foi alcançado ainda em linha, primeiro reconstrói
            # o triângulo em torno desse Goal e só depois avança o objetivo.
            if leader.arrived_target(target_position):
                for robot in robots:
                    robot.stop_robot()

                time.sleep(1)

                if current_goal_index < len(goal_paths) - 1:
                    transition_reason = 'goal'
                    state = 'FORM_TRIANGLE_TRANSITION'
                else:
                    sim.stopSimulation()
                    break

        # ============================================================
        # TRANSIÇÃO LINHA -> TRIÂNGULO
        # ============================================================
        elif state == 'FORM_TRIANGLE_TRANSITION':
            if line_order is None or len(line_order) < 3:
                raise RuntimeError(
                    'FORM_TRIANGLE_TRANSITION requer uma line_order com pelo menos 3 robôs.'
                )

            for robot in robots:
                if robot is leader:
                    # O líder continua devagar. Se já estiver no Goal,
                    # triangle_transition_leader mantém o robô parado.
                    robot.run(
                        robots,
                        mode='triangle_transition_leader',
                        target_position=target_position
                    )
                elif robot in line_order[1:3]:
                    # Os dois seguidores ocupam os dois vértices traseiros
                    # de um triângulo equilátero em torno do líder.
                    robot.run(
                        robots,
                        mode='triangle_transition'
                    )
                else:
                    # Para uma eventual expansão com mais de 3 robôs,
                    # os robôs extras apenas mantêm o controlador de formação.
                    robot.run(robots, mode='formation')

            transition_ready = Robot.form_triangle_transition(
                robots,
                leader
            )

            if transition_ready:
                # line_order deixa de representar o estado atual somente
                # depois que a reconstrução geométrica terminou.
                Robot.clear_line_data(robots)
                line_order = None

                if transition_reason == 'goal':
                    current_goal_index += 1
                    target_position = robots[0].goal_positions[current_goal_index]

                # Após a formação ser reconstruída, volta a escolher o robô
                # mais favorável para liderar até o objetivo atual.
                leader = Robot.choose_leader(robots, target_position)

                transition_reason = None
                state = 'MOVE_TRIANGLE'

        # ============================================================
        # REFORMA TRIÂNGULO NO GOAL
        # ============================================================
        elif state == 'FORM_TRIANGLE_AFTER_GOAL':
            triangle_ready = Robot.form_triangle_around_leader(
                robots,
                leader
            )

            if triangle_ready:
                current_goal_index += 1
                target_position = robots[0].goal_positions[current_goal_index]

                Robot.clear_line_data(robots)
                line_order = None
                leader = Robot.choose_leader(robots, target_position)

                state = 'MOVE_TRIANGLE'

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