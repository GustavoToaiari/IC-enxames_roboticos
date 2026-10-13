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

# ============================================================
# Repulsão de segurança entre robôs
# ============================================================
# Atua apenas quando dois robôs ficam perigosamente próximos.
# É diferente da separação do Flocking: a separação organiza o grupo,
# enquanto esta parcela funciona como proteção contra colisões.

# Distância entre CENTROS abaixo da qual a repulsão começa [m].
ROBOT_REP_RANGE = 0.16

# Ganho da repulsão de curto alcance.
K_REP_ROBOTS = 0.2

# Saturação da força resultante robô-robô.
F_REP_ROBOTS_MAX = 0.80

# Distância mínima numérica entre centros usada no cálculo [m].
# O diâmetro físico do ePuck é aproximadamente 0.0724 m.
ROBOT_REP_MIN_DIST = 0.075


# ============================================================
# Flocking (Boids)
# ============================================================
# O artigo combina campo potencial e Flocking no estado de grupo:
# F_c = F_pf + alpha * F_f, com F_f = F_coh + F_ali + F_sep.
# Os valores abaixo sao uma sintonia inicial para os ePucks desta cena.

# Vizinhanca local usada por coesao e alinhamento [m]
FLOCK_RADIUS = 0.80

# Campo de visao do Flocking. 360 graus simplifica o primeiro teste e
# evita perder vizinhos durante manobras.
FLOCK_FOV_ANGLE = math.radians(360)

# Vizinhanca menor usada pela regra de separacao [m]
SEPARATION_RADIUS = 0.18

# Ganhos das tres regras de Reynolds
K_COH = 0.40
K_ALI = 0.60
K_SEP = 1.20

# alpha da combinacao F_c = F_pf + alpha * F_f.
# Em grupo: alpha = 1. Na formacao em linha o Flocking nao e aplicado.
FLOCK_ALPHA = 1.0

# Saturacao apenas da parcela de Flocking, para impedir que ela
# sobreponha completamente o campo potencial durante o primeiro teste.
FLOCK_FORCE_MAX = 1.00

# Mapeamento da norma da forca resultante para velocidade linear.
FLOCK_SPEED_GAIN = 0.20
V_MIN_FLOCK = 0.05
V_MAX_FLOCK = 0.20

# Raio maximo do grupo usado apenas para encerrar a agregacao inicial.
# Evita exigir, com 5 robos, que todos os pares tenham a mesma distancia.
INITIAL_GROUP_RADIUS = 0.50

# Considera que o enxame alcancou o Goal quando TODOS os robos
# estiverem dentro deste raio. Deve ser maior que GOAL_TOL porque
# varios robos nao podem ocupar exatamente o mesmo ponto.
FLOCK_GOAL_TOL = 0.55


# ============================================================
# Formação geométrica explícita sobre o Flocking
# ============================================================
# Distância desejada entre robôs vizinhos da formação [m].
# Para 3 robôs, corresponde exatamente ao lado do triângulo equilátero.
# Para N >= 4, define a corda entre vizinhos de um polígono regular
# inscrito em uma circunferência.
FORMATION_SPACING = 0.30

# Ganho da força que leva cada robô ao seu ponto da formação.
K_FORMATION = 2.0

# Saturação da componente de formação. Mantém o desvio de obstáculos
# prioritário quando necessário.
FORMATION_FORCE_MAX = 0.60

# Zona morta ao redor do ponto desejado [m]. Evita pequenas oscilações.
FORMATION_DEADBAND = 0.015

# Peso da componente explícita na força resultante.
FORMATION_ALPHA = 1.0

# A coesão clássica do Flocking puxa todos os robôs para o centro e,
# portanto, tende a encolher uma formação circular. Mantemos apenas uma
# parcela pequena dela; alinhamento e separação continuam integrais.
FORMATION_COHESION_SCALE = 0.15

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
LEADER_LINE_MIN_SCALE = 0.70

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