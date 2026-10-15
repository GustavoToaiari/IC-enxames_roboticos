"""Navegação coletiva de N ePucks em formação triangular expansível.

Fluxo: formar em torno do líder -> navegar -> alinhar a fila -> atravessar
com atrasos -> reconstruir formação -> navegar até o próximo Goal.

Requer CoppeliaSim aberto, a cena com ePuck1, Goals e obstáculos
nomeados conforme as listas abaixo e o ZMQ Remote API Client instalado.
"""
import random
import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from Robot import Robot
import Parameters as P


NUM_ROBOTS = 10

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)

if P.RANDOM_SEED is not None:
    random.seed(P.RANDOM_SEED)

# Referências mantidas como na cena original.
goal_paths = [sim.getObject(f'/Goal{i}') for i in range(1, 5)]
obstacle_paths = [
    *[sim.getObject(f'/Cuboid{i}') for i in range(1, 13)],
    sim.getObject('/Cylinder1')
]
obstacle_cache = Robot.create_obstacle_cache(sim, obstacle_paths)

created_epucks = []
robots = []
simulation_started = False

try:
    # Os 9 modelos adicionais são criados sem sobreposição inicial.
    created_epucks = Robot.create_epucks(
        sim, NUM_ROBOTS, obstacle_cache=obstacle_cache
    )

    for i in range(1, NUM_ROBOTS + 1):
        robots.append(Robot(
            name=f'ePuck{i}',
            base=sim.getObject(f'/ePuck{i}/base'),
            goal_paths=goal_paths,
            wheel_path=(
                sim.getObject(f'/ePuck{i}/leftJoint'),
                sim.getObject(f'/ePuck{i}/rightJoint')
            ),
            obstacles=obstacle_cache,
            sim=sim
        ))

    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
    simulation_started = True

    goal_index = 0
    target_position = robots[0].goal_positions[goal_index].copy()
    leader = None
    line_order = None
    state = 'FORM_GROUP'
    ready_cycles = 0
    passage_cooldown_until = 0.0

    def start_goal(index):
        """Escolhe o líder mais próximo do novo Goal e atribui slots fixos."""
        global goal_index, target_position, leader, line_order, state, ready_cycles
        goal_index = index
        target_position = robots[0].goal_positions[goal_index].copy()

        # O método usa as posições medidas no começo do ciclo.
        Robot.clear_line_data(robots)
        line_order = None
        leader = Robot.choose_leader(robots, target_position)
        leader.formation_direction = None  # Novo Goal => nova direção.
        Robot.assign_formation_slots(robots, leader, target_position)
        ready_cycles = 0
        state = 'FORM_GROUP'
        print(f'[Goal {goal_index + 1}] Líder: {leader.name} / robôs: {len(robots)}')

    def reached_stable_condition(ready_now, required_cycles):
        """Evita transições por uma única medição no limite da tolerância."""
        global ready_cycles
        ready_cycles = ready_cycles + 1 if ready_now else 0
        return ready_cycles >= required_cycles

    def leave_line():
        global line_order
        for robot in robots:
            robot.line_order = None
            robot.line_target_position = None
            robot.line_start_time = None
            robot.line_align_target = None
        line_order = None

    # Escolha inicial antes de montar a formação.
    for robot in robots:
        robot.get_pose_2d()
    start_goal(0)

    while True:
        client.step()
        for robot in robots:
            robot.get_pose_2d()

        sim_time = sim.getSimulationTime()

        # Os slots são fixos em relação aos robôs, mas seus alvos acompanham
        # a translação do líder e a direção do Goal.
        if state in ('FORM_GROUP', 'MOVE_GROUP', 'REFORM_GROUP', 'GOAL_SETTLE'):
            Robot.update_formation_targets(robots, leader, target_position)

        if state == 'FORM_GROUP':
            # Líder parado até todos os seguidores estarem posicionados.
            leader.stop_robot()
            for robot in robots:
                if robot is not leader:
                    robot.run(robots, mode='slot_formation')

            ready = Robot.formation_max_slot_error(robots, leader) < P.DIST_TOL
            if reached_stable_condition(ready, P.FORMATION_READY_CYCLES):
                ready_cycles = 0
                state = 'MOVE_GROUP'
                print('[Formação] pronta; navegação liberada')

        elif state == 'MOVE_GROUP':
            # Evita iniciar uma passagem caso o líder já esteja no Goal.
            if leader.arrived_target(target_position):
                ready_cycles = 0
                state = 'GOAL_SETTLE'
                print(f'[Goal {goal_index + 1}] líder chegou; aguardando grupo')
                continue

            # Não detectar novamente a passagem acabada de atravessar.
            if (sim_time >= passage_cooldown_until
                    and leader.detect_narrow_passage(target_position)):
                leader, line_order = Robot.prepare_line_formation(
                    robots, target_position, leader=leader
                )
                ready_cycles = 0
                state = 'FORM_LINE'
                print('[Passagem] detectada; organizando fila antes de entrar')
                continue

            # Calcula todos os destinos antes de o líder checar coesão.
            for robot in robots:
                if robot is not leader:
                    robot.run(robots, mode='slot_formation')
            leader.run(
                robots, mode='go_to_goal',
                target_position=target_position, formation_context='slots'
            )

            if leader.arrived_target(target_position):
                ready_cycles = 0
                state = 'GOAL_SETTLE'
                print(f'[Goal {goal_index + 1}] líder chegou; aguardando grupo')

        elif state == 'FORM_LINE':
            # Não há atrasos aqui: todos se alinham enquanto o líder espera.
            Robot.update_line_align_targets(robots, leader, target_position)
            leader.stop_robot()
            for robot in robots:
                if robot is not leader:
                    robot.run(robots, mode='line_align')

            max_error = Robot.line_max_align_error(robots, leader)
            separation_ok = all(
                float(np.linalg.norm(a.position - b.position))
                >= 2 * P.ROBOT_RADIUS + 0.008
                for i, a in enumerate(robots)
                for b in robots[i + 1:]
            )
            if reached_stable_condition(
                max_error < P.LINE_TOL and separation_ok, P.LINE_READY_CYCLES
            ):
                # A contagem dos atrasos começa SÓ após a fila estar pronta.
                for robot in robots:
                    robot.line_start_time = sim.getSimulationTime()
                ready_cycles = 0
                state = 'MOVE_LINE'
                print('[Fila] pronta; líder parte e seguidores saem com atraso')

        elif state == 'MOVE_LINE':
            last_robot = line_order[-1]
            elapsed = sim_time - line_order[0].line_start_time
            all_started = elapsed >= (len(line_order) - 1) * P.LINE_FORMATION_DELAY

            # Último robô vê a abertura atrás de si: fila atravessou.
            if all_started and last_robot.detect_narrow_passage_backward():
                ready_cycles = 0
                leave_line()
                state = 'REFORM_GROUP'
                print('[Passagem] último robô atravessou; reconstruindo geometria')
                continue

            for robot in robots:
                if robot is not leader:
                    robot.run(robots, mode='line_formation')
            leader.run(
                robots, mode='go_to_goal',
                target_position=target_position, formation_context='line'
            )

            # Se o Goal vier antes da conclusão da fila, esperar os demais
            # se aproximarem. O líder não tenta reconstruir dentro da abertura.
            if leader.arrived_target(target_position):
                errors = [robot.max_error for robot in robots if robot is not leader]
                line_ready = all_started and (max(errors, default=0) < P.LINE_TOL)
                if reached_stable_condition(line_ready, P.LINE_READY_CYCLES):
                    ready_cycles = 0
                    leave_line()
                    state = 'GOAL_SETTLE'
                    print('[Fila] agrupada no Goal; reconstruindo formação')

        elif state == 'REFORM_GROUP':
            # Líder aguarda a geometria inteira; sem assumir apenas 2 seguidores.
            leader.stop_robot()
            for robot in robots:
                if robot is not leader:
                    robot.run(robots, mode='slot_formation')

            ready = Robot.formation_max_slot_error(robots, leader) < P.DIST_TOL
            if reached_stable_condition(ready, P.FORMATION_READY_CYCLES):
                ready_cycles = 0
                passage_cooldown_until = sim_time + 1.5
                if leader.arrived_target(target_position):
                    state = 'GOAL_SETTLE'
                else:
                    state = 'MOVE_GROUP'
                print('[Formação] reconstruída após a passagem')

        elif state == 'GOAL_SETTLE':
            leader.stop_robot()
            for robot in robots:
                if robot is not leader:
                    robot.run(robots, mode='slot_formation')

            ready = Robot.formation_max_slot_error(robots, leader) < P.DIST_TOL
            if reached_stable_condition(ready, P.FORMATION_READY_CYCLES):
                for robot in robots:
                    robot.stop_robot()
                if goal_index + 1 == len(goal_paths):
                    print('[Fim] todos os Goals alcançados; formação concluída')
                    break
                start_goal(goal_index + 1)

        else:
            raise RuntimeError(f'Estado de controle desconhecido: {state}')

except KeyboardInterrupt:
    print('[Interrompido] encerrando simulação')
finally:
    if simulation_started:
        try:
            sim.stopSimulation()
            # A parada do CoppeliaSim pode levar alguns ciclos a ser efetivada.
            # Removemos os modelos copiados apenas depois disso.
            for _ in range(60):
                if sim.getSimulationState() == sim.simulation_stopped:
                    break
                time.sleep(0.05)
        except Exception as exc:
            print(f'Falha ao parar simulação: {exc}')
    if created_epucks:
        try:
            Robot.remove_created_epucks(sim, created_epucks)
        except Exception as exc:
            print(f'Falha ao remover ePucks copiados: {exc}')
