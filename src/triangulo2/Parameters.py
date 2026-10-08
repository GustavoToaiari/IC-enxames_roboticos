import math

# ============================================================
# Geometria do ePuck
# ============================================================
WHEEL_RADIUS = 0.0425 / 2
AXLE_LENGTH = 0.054
WHEEL_OMEGA_MAX = 20.0

ROBOT_RADIUS = 0.0724 / 2
OBSTACLE_RADIUS = 0.25

# ============================================================
# Objetivo
# ============================================================
GOAL_TOL = 0.20

# ============================================================
# Campo potencial
# ============================================================
# A atração cresce linearmente somente perto do Goal.
# Para distâncias maiores que F_ATT_MAX / K_ATT, sua magnitude
# fica limitada, evitando que o tamanho da cena domine o desvio.
K_ATT = 1.0
F_ATT_MAX = 1.0
K_REP = 0.5
REP_RANGE = 0.08

# ============================================================
# Contorno de obstáculos
# ============================================================

# Ganho da componente tangencial do campo potencial
K_TANGENTIAL = 1.0

# Limite máximo da força tangencial
F_TANGENTIAL_MAX = 3.0

# Só ativa o campo tangencial quando existe
# uma repulsão minimamente relevante
TANGENTIAL_REP_MIN = 0.05

# Tempo em que o robô mantém o lado escolhido
# para contornar o obstáculo
AVOIDANCE_MEMORY_TIME = 0.8

# Peso da orientação atual na escolha esquerda/direita
AVOIDANCE_HEADING_WEIGHT = 0.35

# Controle angular
K_W = 1.5

# Repulsão entre robôs (função disponível para testes futuros)
K_REP_ROBOTS = 0.8

# ============================================================
# Formação triangular
# ============================================================
DESIRED_DISTANCE = 0.30
DIST_TOL = 0.14

# Velocidade variável dos seguidores
V_MIN_FORMATION = 0.05
V_MAX_FORMATION = 0.25

# Faixa de erro usada para mapear erro -> velocidade
FORM_ERROR_MIN = 0.02
FORM_ERROR_MAX = 0.20

# ============================================================
# Movimento do líder até o Goal
# ============================================================
V_MIN_LEADER = 0.06
V_MAX_LEADER_GOAL = 0.20

# Faixa de distância usada para mapear distância ao Goal -> velocidade
GOAL_SPEED_DIST_MIN = 0.25
GOAL_SPEED_DIST_MAX = 1.00

# ============================================================
# Controle de coesão líder-seguidores
# ============================================================
# Até esse erro adicional o líder mantém 100% da velocidade.
LEADER_GAP_SOFT = 0.05

# A partir desse erro adicional o líder deixa de transladar.
LEADER_GAP_HARD = 0.15

# Velocidade mínima relativa do líder enquanto a linha está se formando.
# 0.35 significa que ele mantém no mínimo 35% da velocidade calculada.
LEADER_LINE_MIN_SCALE = 0.35

# ============================================================
# Formação em linha
# ============================================================
LINE_DISTANCE = 0.10
LINE_TOL = 0.05
K_LINE = 1.5
LINE_FORMATION_DELAY = 0.9

# ============================================================
# Transição linha -> triângulo
# ============================================================
V_MAX_LEADER_TRANSITION = 0.05
TRIANGLE_TRANSITION_TOL = 0.03

# ============================================================
# Detecção de passagem estreita
# ============================================================
LEADER_VISION_RADIUS = 0.6
MIN_PASSAGE_WIDTH = 0.7
LEADER_FOV_ANGLE = math.radians(120)

# ============================================================
# Escala / saturação da repulsão
# ============================================================
REP_SCALE_LEADER = 1.0
REP_SCALE_FOLLOWER = 0.05

F_REP_MAX_FOLLOWER = 0.25
F_REP_MAX_LEADER = 4.0

# ============================================================
# Reprodutibilidade das posições iniciais
# ============================================================
# None: mantém as posições iniciais aleatórias como antes.
# Um inteiro (por exemplo, 42): repete as mesmas posições sorteadas
# para ePuck2 e ePuck3 em cada execução, útil ao comparar cenas.
RANDOM_SEED = None