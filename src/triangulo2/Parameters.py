"""Parâmetros do agrupamento circular livre para ePuck no CoppeliaSim.

Distâncias em metros, tempos em segundos de SIMULAÇÃO.
"""
import math

# Robôs e controle diferencial (mantidos da versão original)
WHEEL_RADIUS = 0.0425 / 2
AXLE_LENGTH = 0.054
WHEEL_OMEGA_MAX = 20.0
ROBOT_RADIUS = 0.0724 / 2
OBSTACLE_RADIUS = 0.25
K_W = 1.5
GOAL_TOL = 0.20

# Campo potencial do líder e dos seguidores (original)
K_ATT = 1.0
F_ATT_MAX = 1.0
K_REP = 3.2
REP_RANGE = 0.30  # agrupamento: distância livre da superfície
REP_RANGE_LINE = 0.18  # fila: alcance lateral menor em corredores estreitos
K_TANGENTIAL = 0.65
F_TANGENTIAL_MAX = 1.25
TANGENTIAL_REP_MIN = 0.05
AVOIDANCE_MEMORY_TIME = 0.8
AVOIDANCE_HEADING_WEIGHT = 0.35
REP_SCALE_LEADER = 1.0
REP_SCALE_FOLLOWER = 0.95
F_REP_MAX_FOLLOWER = 3.0
F_REP_MAX_LEADER = 3.0

# Freio preventivo: só corta avanço em direção ao obstáculo.
# Obstáculos laterais não impedem avanço paralelo (importante na passagem).
OBSTACLE_SLOW_CLEARANCE = 0.30
OBSTACLE_STOP_CLEARANCE = 0.065
OBSTACLE_APPROACH_THRESHOLD = 0.20

# Agrupamento livre: os 10 robôs, INCLUSIVE o líder, dentro de um disco.
# DESIRED_DISTANCE vale só para vizinhos muito próximos; NÃO para todos os pares.
DESIRED_DISTANCE = 0.30
DIST_TOL = 0.14   # Margem radial de verificação, não erro entre todos os pares.
GROUP_RADIUS = 0.58
# O círculo é centrado no centróide REAL dos dez robôs, não no líder.
GROUP_FRONT_GAP = 0.12  # distância longitudinal mínima entre os centros
GROUP_FRONT_SLOW_ZONE = 0.26  # inicia frenagem longitudinal progressiva
GROUP_FRONT_RESPONSE_TIME = 0.75  # segundos para consumir a folga restante
K_GROUP_FRONT = 3.5
GROUP_BOUNDARY_MARGIN = 0.10  # inicia contenção antes da borda
K_GROUP_COHESION = 0.75
K_GROUP_CONTAINMENT = 2.2
K_GROUP_FOLLOW_DRIVE = 0.35  # empuxo coletivo rumo ao Goal apenas durante MOVE_GROUP
K_GROUP_SEPARATION = 1.15
F_GROUP_COHESION_MAX = 0.9
F_GROUP_SEPARATION_MAX = 1.45
F_GROUP_TOTAL_MAX = 2.5

# Proteção anticolisão adicional: repulsive_r2r() do Robot.py.
# Ativa apenas abaixo de 0,1024 m entre centros, portanto não impede
# o espaçamento de 0,12 m da fila (nem soma repulsão a longas distâncias).
K_REP_ROBOTS = 0.9
R2R_SAFETY_MARGIN = 0.03
F_REP_ROBOTS_MAX = 0.8

GROUP_MIN_NEIGHBOR_DISTANCE = 0.16  # mínimo para considerar compacto e separado
GROUP_MAX_NEAREST_DISTANCE = 0.47  # não aceita seguidores isolados
GROUP_CENTER_TOL = 0.20  # erro do líder relativo à borda frontal do grupo
GROUP_STABLE_TIME = 0.8
GROUP_FORM_TIMEOUT = 38.0  # libera só quando o líder estiver conectado ao grupo

# Líder participativo: busca ficar um pouco adiante do centróide DOS SEGUIDORES,
# mas pode se mover em direção a eles, inclusive na montagem inicial.
LEADER_FRONT_OFFSET = 0.29
K_LEADER_FORMATION = 1.9
K_LEADER_COHESION = 3.2
K_LEADER_GOAL = 0.8
LEADER_RECOVERY_SOFT = 0.08
LEADER_RECOVERY_HARD = 0.28
LEADER_FORMATION_SPEED_MAX = 0.22

# Reagrupamento após travessia/Goal: o líder segue SEMPRE para frente.
# Os seguidores se reúnem em torno de um centro virtual logo atrás dele.
V_MAX_LEADER_REFORM = 0.065
V_MIN_LEADER_REFORM = 0.025
K_GROUP_REFORM_ANCHOR = 1.15
F_GROUP_REFORM_ANCHOR_MAX = 1.0
V_MAX_REFORM_FOLLOWER = 0.27
REFORM_ERROR_MAX = 0.32

# Velocidades dos seguidores e do líder
V_MIN_FORMATION = 0.025
V_MAX_FORMATION = 0.25
FORM_ERROR_MIN = 0.035
FORM_ERROR_MAX = 0.60
V_MIN_LEADER = 0.06
V_MAX_LEADER_GOAL = 0.22
GOAL_SPEED_DIST_MIN = 0.25
GOAL_SPEED_DIST_MAX = 1.00

# Coesão para o agrupamento: erro de extrapolar o raio do disco
LEADER_GAP_SOFT = 0.04
LEADER_GAP_HARD = 0.18
LEADER_GROUP_MIN_SCALE = 0.0  # líder espera se houver separação excessiva
LEADER_CENTROID_SOFT_TOL = 0.10  # compatibilidade com controle antigo de coesão
# Ligação física do líder ao enxame: distância até o seguidor mais próximo.
# Mesmo com a maior parte do grupo dentro do disco, o líder deve esperar
# se não houver nenhum seguidor suficientemente próximo dele.
LEADER_NEIGHBOR_SOFT_DISTANCE = 0.36
LEADER_NEIGHBOR_HARD_DISTANCE = 0.52

# Passagem estreita: SEM pré-alinhamento; partidas em ordem de proximidade ao Goal
# Velocidade exclusiva para a fila: não altera a navegação circular.
V_MIN_LINE = 0.06
V_MAX_LINE = 0.30
V_MIN_LEADER_LINE = 0.06
V_MAX_LEADER_LINE = 0.22
LINE_ERROR_MIN = 0.02
LINE_ERROR_MAX = 0.22

LINE_DISTANCE = 0.12
LINE_TOL = 0.06
K_LINE = 1.8
LINE_FORMATION_DELAY = 1.0  # líder em t=0; próximo em t=1s; ..., décimo em t=9s
LINE_REPULSION_DISTANCE = 0.145
K_LINE_SEPARATION = 2.3
F_LINE_SEPARATION_MAX = 2.2
LEADER_LINE_MIN_SCALE = 0.30  # líder não fica tão lento durante a fila
LINE_MIN_TRAVEL_TIME = 2.0
PASSAGE_EXIT_MARGIN = 0.12  # espaço após as paredes antes de sair da fila
PASSAGE_REARM_CLEAR_TIME = 1.5

# Velocidades APENAS da travessia progressiva. Não alteram MOVE_GROUP,
# FORM_GROUP, nem o reagrupamento normal entre Goals.
V_MIN_LEADER_CROSSING = 0.15  # líder entre as paredes
V_MAX_LEADER_CROSSING = 0.18
V_MIN_LEADER_CLEAR_EXIT = 0.12  # avançar logo após liberar a passagem
V_MAX_LEADER_CLEAR_EXIT = 0.15
PASSAGE_CLEARANCE_DISTANCE = 0.35  # distância além da saída para abrir espaço
V_MIN_LEADER_PASSAGE_REFORM = 0.035  # após liberar espaço, com seguidores já fora
V_MAX_LEADER_PASSAGE_REFORM = 0.12

# Detecção de passagem (original)
LEADER_VISION_RADIUS = 0.6
MIN_PASSAGE_WIDTH = 0.7
LEADER_FOV_ANGLE = math.radians(120)

# Inicialização e depuração
SPAWN_MIN_DISTANCE = 0.19  # evita ePucks sobrepostos ao copiá-los
SPAWN_OBSTACLE_CLEARANCE = 0.08  # afastamento da superfície dos obstáculos
RANDOM_SEED = 42  # use None para posições iniciais aleatórias
