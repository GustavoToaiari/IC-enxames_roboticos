"""Navegação de 10 ePucks em agrupamento circular livre.

FORM_GROUP -> MOVE_GROUP -> MOVE_LINE (reagrupamento progressivo) -> REFORM_GROUP -> MOVE_GROUP.
O líder participa da formação e mantém a dianteira na navegação coletiva.
Ao detectar uma passagem estreita, a fila sai SEM pré-alinhamento: líder
imediatamente e demais robôs em ordem de distância ao Goal, a cada 1 s.
Cada robô que sai do corredor passa a se reagrupar somente com quem já saiu.
Tempos calculados em segundos da simulação.
"""
import random
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Robot import Robot
import Parameters as P

NUM_ROBOTS = 10


def main():
    if P.RANDOM_SEED is not None:
        random.seed(P.RANDOM_SEED)

    client = RemoteAPIClient()
    sim = client.getObject('sim')
    client.setStepping(True)
    created_epucks = []
    robots = []

    try:
        goal_paths = [sim.getObject(f'/Goal{i}') for i in range(1, 5)]
        obstacle_paths = ([sim.getObject(f'/Cuboid{i}') for i in range(1, 13)]
                          + [sim.getObject('/Cylinder1')])
        obstacle_cache = Robot.create_obstacle_cache(sim, obstacle_paths)
        created_epucks = Robot.create_epucks(sim, NUM_ROBOTS, obstacle_cache)

        for i in range(1, NUM_ROBOTS + 1):
            robots.append(Robot(
                name=f'ePuck{i}',
                base=sim.getObject(f'/ePuck{i}/base'),
                goal_paths=goal_paths,
                wheel_path=(sim.getObject(f'/ePuck{i}/leftJoint'),
                            sim.getObject(f'/ePuck{i}/rightJoint')),
                obstacles=obstacle_cache, sim=sim
            ))

        if sim.getSimulationState() == sim.simulation_stopped:
            sim.startSimulation()

        for robot in robots:
            robot.get_pose_2d()

        goal_index = 0
        goal = robots[0].goal_positions[goal_index]
        leader = Robot.choose_leader(robots, goal)
        line_order = None
        state = 'FORM_GROUP'
        state_since = sim.getSimulationTime()
        stable_since = None
        line_departure_time = None
        passage_armed = True
        passage_clear_since = None
        # Dados congelados na entrada da passagem. Não mudam a formação original.
        passage_origin = None
        passage_forward = None
        passage_exit_progress = None
        passage_exit_target = None
        crossed_robots = set()

        def change_state(new_state, now):
            nonlocal state, state_since, stable_since
            state = new_state
            state_since = now
            stable_since = None

        def stop_all():
            for robot in robots:
                robot.stop_robot()

        def next_goal(now):
            nonlocal goal_index, goal, leader, line_order
            nonlocal passage_armed, passage_clear_since, crossed_robots
            stop_all()
            goal_index += 1
            if goal_index >= len(goal_paths):
                return False
            goal = robots[0].goal_positions[goal_index]
            Robot.clear_line_data(robots)
            line_order = None
            crossed_robots.clear()
            leader = Robot.choose_leader(robots, goal)
            passage_armed = False
            passage_clear_since = None
            change_state('REFORM_GROUP', now)
            return True

        while True:
            client.step()
            now = sim.getSimulationTime()
            for robot in robots:
                robot.get_pose_2d()

            if not passage_armed and state in ('MOVE_GROUP', 'REFORM_GROUP'):
                if not leader.detect_narrow_passage(goal):
                    if passage_clear_since is None:
                        passage_clear_since = now
                    elif now - passage_clear_since >= P.PASSAGE_REARM_CLEAR_TIME:
                        passage_armed = True
                        passage_clear_since = None
                else:
                    passage_clear_since = None

            if state in ('FORM_GROUP', 'REFORM_GROUP'):
                # Comandar líder PRIMEIRO: o limitador dos seguidores usa
                # a velocidade que ele realmente pediu neste ciclo.
                if state == 'FORM_GROUP':
                    leader.run(robots, mode='group_leader_form',
                               target_position=goal)
                else:
                    leader.run(robots, mode='group_leader_reform',
                               target_position=goal)

                for robot in robots:
                    if robot is not leader:
                        robot.run(robots, mode='group', leader=leader,
                                  target_position=goal,
                                  reforming=(state == 'REFORM_GROUP'))

                metrics = Robot.group_metrics(robots, leader, goal)
                if metrics['ready']:
                    if stable_since is None:
                        stable_since = now
                else:
                    stable_since = None

                stable_enough = (stable_since is not None and
                                 now - stable_since >= P.GROUP_STABLE_TIME)
                timed_out = now - state_since >= P.GROUP_FORM_TIMEOUT
                linked = (metrics['leader_nearest'] <=
                          P.LEADER_NEIGHBOR_SOFT_DISTANCE and
                          metrics['centroid_error'] <= P.GROUP_CENTER_TOL)
                if stable_enough or (timed_out and linked):
                    change_state('MOVE_GROUP', now)
                elif timed_out:
                    # Não libera um líder isolado apenas por timeout.
                    state_since = now
                continue

            if state == 'MOVE_GROUP':
                if leader.arrived_target(goal, stop=False):
                    if not next_goal(now):
                        break
                    continue

                # A detecção continua igual, mas agora também guarda o par de
                # paredes para identificar a saída de CADA agente.
                passage_walls = (leader.detect_narrow_passage(goal, return_pair=True)
                                 if passage_armed else None)
                if passage_walls is not None:
                    passage_origin = leader.position.copy()
                    passage_forward = Robot.forward_direction(leader, goal)
                    lateral = np.array([-passage_forward[1], passage_forward[0]])

                    # A saída fica depois da extremidade mais à frente das
                    # paredes detectadas, com folga para o corpo do robô.
                    passage_exit_progress = max(
                        float(np.dot(endpoint - passage_origin, passage_forward))
                        + 0.5 * wall['thickness']
                        for wall in passage_walls
                        for endpoint in (wall['a'], wall['b'])
                    ) + P.ROBOT_RADIUS + P.PASSAGE_EXIT_MARGIN

                    # Referência fixa no centro da abertura. Um agente que
                    # ainda está na passagem não deve perseguir lateralmente
                    # o vizinho que já saiu e começou a se reagrupar.
                    middle_offsets = []
                    for wall in passage_walls:
                        _, closest = Robot.point_to_segment_distance(
                            passage_origin, wall['a'], wall['b'])
                        middle_offsets.append(float(np.dot(
                            closest - passage_origin, lateral)))
                    passage_exit_target = (
                        passage_origin
                        + (passage_exit_progress + P.LINE_DISTANCE) * passage_forward
                        + (sum(middle_offsets) / 2.0) * lateral
                    )

                    leader, line_order = Robot.prepare_line_formation(robots, goal)
                    crossed_robots.clear()
                    stop_all()
                    line_departure_time = now
                    passage_armed = False
                    passage_clear_since = None
                    change_state('MOVE_LINE', now)
                    # Líder parte em t=0 sem pré-alinhamento. A espera
                    # sequencial não reduz a velocidade dele no corredor.
                    leader.run(robots, mode='go_to_goal', target_position=goal,
                               formation_context='line', passage_phase='crossing')
                    continue

                # Também no movimento, líder PRIMEIRO e seguidores depois.
                leader.run(robots, mode='group_leader_move', target_position=goal)
                for robot in robots:
                    if robot is not leader:
                        robot.run(robots, mode='group', leader=leader,
                                  target_position=goal, moving=True)
                continue

            if state == 'MOVE_LINE':
                elapsed = now - line_departure_time

                # Cada robô deixa a fila quando seu centro ultrapassa o final
                # geométrico das paredes + raio + margem. A saída é permanente.
                for i, robot in enumerate(line_order):
                    if elapsed < i * P.LINE_FORMATION_DELAY:
                        continue
                    progress = float(np.dot(
                        robot.position - passage_origin, passage_forward))
                    if progress >= passage_exit_progress:
                        crossed_robots.add(robot)

                # Somente quando TODOS terminaram a travessia voltamos ao
                # reagrupamento global original, sem mudar seu controlador.
                if len(crossed_robots) == len(line_order):
                    Robot.clear_line_data(robots)
                    line_order = None
                    crossed_robots.clear()
                    passage_armed = False
                    passage_clear_since = None
                    change_state('REFORM_GROUP', now)
                    continue

                # Não muda o Goal enquanto existem robôs na passagem.
                # O líder pode aguardar no Goal, sem abortar a fila.
                if leader.arrived_target(goal, stop=False):
                    leader.stop_robot()
                elif leader in crossed_robots:
                    progress = float(np.dot(
                        leader.position - passage_origin, passage_forward))
                    if progress < (passage_exit_progress
                                   + P.PASSAGE_CLEARANCE_DISTANCE):
                        # Após sair, avança mais um trecho para abrir espaço
                        # para a formação, sem freio pela distância da fila.
                        leader.run(robots, mode='go_to_goal',
                                   target_position=goal, formation_context='line',
                                   passage_phase='clear_exit')
                    else:
                        # Somente depois de liberar espaço, reagrupa com os
                        # agentes já do lado de fora. Se ainda estiver sozinho,
                        # aguarda nesse ponto seguro em vez de se afastar.
                        crossed_group = [r for r in line_order
                                         if r in crossed_robots]
                        leader.run(crossed_group, mode='group_leader_reform',
                                   target_position=goal, safety_robots=robots,
                                   passage_reforming=True)
                else:
                    leader.run(robots, mode='go_to_goal', target_position=goal,
                               formation_context='line', passage_phase='crossing')

                crossed_group = [r for r in line_order if r in crossed_robots]
                for i, robot in enumerate(line_order):
                    if robot is leader:
                        continue
                    if robot in crossed_robots:
                        robot.run(crossed_group, mode='group', leader=leader,
                                  target_position=goal, reforming=True,
                                  safety_robots=robots,
                                  passage_guard=(passage_origin,
                                                 passage_forward,
                                                 passage_exit_progress))
                    elif elapsed < i * P.LINE_FORMATION_DELAY:
                        robot.stop_robot()
                    elif line_order[i - 1] in crossed_robots:
                        # O agente anterior já saiu da fila: segue até a saída
                        # pelo centro da passagem, sem persegui-lo de lado.
                        robot.run(robots, mode='line', static_line=False,
                                  line_exit_target=passage_exit_target)
                    else:
                        robot.run(robots, mode='line', static_line=False)
                continue

            raise RuntimeError(f'Estado inválido: {state}')

    except KeyboardInterrupt:
        pass
    finally:
        for robot in robots:
            try:
                robot.stop_robot()
            except Exception:
                pass
        try:
            if sim.getSimulationState() != sim.simulation_stopped:
                sim.stopSimulation()
        except Exception:
            pass
        for model in reversed(created_epucks):
            try:
                sim.removeModel(model)
            except Exception:
                pass


if __name__ == '__main__':
    main()
