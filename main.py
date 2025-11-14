import json
import os
import pygame

from env import WumpusEnv, CAVE

# --------- Pygame setup ---------
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hunt the Wumpus - Q-learning Agent")
clock = pygame.time.Clock()

BG = (30, 30, 40)
EDGE = (100, 100, 150)
ROOM_COLOR = (170, 190, 230)
PLAYER_COLOR = (255, 220, 0)
TEXT_COLOR = (240, 240, 240)

font = pygame.font.SysFont("consolas", 20)

# --------- Room positions (layout like your picture) ---------
ROOM_POS = {
    1: (450, 50),     # top
    2: (750, 220),    # top right outer
    3: (640, 550),    # bottom right outer
    4: (260, 550),    # bottom left outer
    5: (150, 220),    # top left outer

    6: (250, 250),    # left mid ring
    7: (330, 170),    # top left mid
    8: (450, 140),    # top center inner
    9: (570, 170),    # top right mid
    10: (650, 250),   # right mid ring

    11: (650, 400),   # lower right mid
    12: (570, 480),   # bottom right mid
    13: (450, 520),   # bottom center inner
    14: (330, 480),   # bottom left mid
    15: (250, 400),   # lower left mid

    16: (350, 360),   # inner pentagon left
    17: (380, 250),   # inner pentagon top-left
    18: (520, 250),   # inner pentagon top-right
    19: (550, 360),   # inner pentagon right
    20: (450, 430),   # inner pentagon bottom
}

# --------- Load Q-table ---------
Q_TABLE_PATH = "q_table.json"
if os.path.exists(Q_TABLE_PATH):
    with open(Q_TABLE_PATH, "r") as f:
        Q_TABLE = json.load(f)
else:
    print("WARNING: q_table.json not found. Agent will behave randomly.")
    Q_TABLE = {}


def get_q(state, action):
    """
    Map state tuple -> JSON key -> Q-value.

    state = (room, arrows, w_alive, smell, rustle, breeze)
    """
    room, arrows, w_alive, smell, rustle, breeze = state
    key_state = f"{room},{arrows},{w_alive},{smell},{rustle},{breeze}"
    key_action = str(action)
    return Q_TABLE.get(key_state, {}).get(key_action, 0.0)


def choose_best_action(state):
    actions = list(range(6))
    qs = [get_q(state, a) for a in actions]
    max_q = max(qs)
    candidates = [a for a, q in zip(actions, qs) if q == max_q]
    # fallback: if all zeros or missing, this is still fine
    import random
    return random.choice(candidates)


# --------- Drawing ---------
def draw_world(env, message=""):
    screen.fill(BG)

    # draw connections
    for room, neighbors in CAVE.items():
        x1, y1 = ROOM_POS[room]
        for n in neighbors:
            x2, y2 = ROOM_POS[n]
            pygame.draw.line(screen, EDGE, (x1, y1), (x2, y2), 2)

    # draw rooms
    for room, (x, y) in ROOM_POS.items():
        color = PLAYER_COLOR if room == env.player_room else ROOM_COLOR
        pygame.draw.circle(screen, color, (int(x), int(y)), 22)
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 22, 2)
        label = font.render(str(room), True, (0, 0, 0))
        screen.blit(label, (x - 10, y - 10))

    # info line
    info1 = f"Room {env.player_room} | Arrows: {env.arrows} | Steps: {env.step_count}"
    text1 = font.render(info1, True, TEXT_COLOR)
    screen.blit(text1, (20, HEIGHT - 80))

    # percepts
    y = HEIGHT - 50
    for p in env.percepts:
        t = font.render(p, True, TEXT_COLOR)
        screen.blit(t, (20, y))
        y -= 24

    # extra message (win/lose, episode)
    if message:
        t_msg = font.render(message, True, (200, 200, 80))
        screen.blit(t_msg, (20, 20))

    pygame.display.flip()


# --------- Autoplay with trained agent ---------
def autoplay(env, episodes=3, delay_ms=200):
    running = True

    for ep in range(1, episodes + 1):
        if not running:
            break

        state = env.reset()
        message = f"Episode {ep}/{episodes}"
        done = False

        while not done and running:
            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            draw_world(env, message)

            if env.game_over:
                done = True
                break

            # Agent chooses action greedily
            action = choose_best_action(state)
            state, reward, done, _info = env.step(action)

            pygame.time.delay(delay_ms)
            clock.tick(60)

        # show result screen for a moment
        if running:
            if env.win:
                message = f"Episode {ep}/{episodes} - WIN!"
            else:
                message = f"Episode {ep}/{episodes} - LOSE!"
            draw_world(env, message)
            pygame.time.delay(1000)


if __name__ == "__main__":
    env = WumpusEnv(CAVE, seed=None)
    autoplay(env, episodes=3, delay_ms=300)
    pygame.quit()
