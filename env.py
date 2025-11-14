import json
import os
import random

# --- Load cave graph from JSON ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAP_PATH = os.path.join(DATA_DIR, "dodecahedron.json")

if not os.path.exists(MAP_PATH):
    raise FileNotFoundError(f"Missing map file: {MAP_PATH}")

with open(MAP_PATH, "r") as f:
    CAVE = {int(k): v for k, v in json.load(f).items()}


class WumpusEnv:
    """
    Hunt the Wumpus environment for Q-learning and visualization.

    Rooms: 1..20, adjacency from CAVE.
    Hazards:
      - 1 Wumpus
      - 2 bats
      - 2 pits

    Actions (0..5):
      0,1,2 -> move to neighbor index 0,1,2  (if exists)
      3,4,5 -> shoot into neighbor index 0,1,2  (if exists)
    """

    def __init__(self, cave, seed=None):
        self.cave = cave
        self.rng = random.Random(seed)
        self.max_steps = 50

        self.player_room = None
        self.arrows = 0
        self.threats = {}
        self.percepts = []
        self.game_over = False
        self.win = False
        self.step_count = 0

        self.reset()

    # ---------- core API ----------

    def reset(self):
        """Randomize world: threats + safe starting room. Returns initial state."""
        self.threats = {}
        rooms = list(self.cave.keys())

        # place threats in empty rooms (no overlap)
        for threat in ["bat", "bat", "pit", "pit", "wumpus"]:
            safe_candidates = [r for r in rooms if r not in self.threats]
            room = self.rng.choice(safe_candidates)
            self.threats[room] = threat

        # player in random safe room (no threats)
        safe_start = [r for r in rooms if r not in self.threats]
        self.player_room = self.rng.choice(safe_start)

        self.arrows = 5
        self.game_over = False
        self.win = False
        self.step_count = 0

        self._update_percepts()
        return self._encode_state()

    def step(self, action):
        """
        Take an action (0..5) and return:
            next_state, reward, done, info

        Reward shaping:
          +5.0  if kill Wumpus (win)
          -5.0  if die (Wumpus/pit/out of arrows)
          -0.05 if shoot and survive
          -0.01 if move and survive
          -0.10 if invalid action
          -2.0  if hit max_steps without finishing
        """
        if self.game_over:
            # Do nothing if already done.
            return self._encode_state(), 0.0, True, {}

        self.step_count += 1
        reward = 0.0

        move_penalty = -0.01
        shoot_penalty = -0.05
        invalid_penalty = -0.10
        death_penalty = -5.0
        win_reward = 5.0

        neighbors = self.cave[self.player_room]

        # ------- apply action -------
        if action in [0, 1, 2]:  # move
            idx = action
            if idx < len(neighbors):
                new_room = neighbors[idx]
                self.player_room = new_room
                reward += move_penalty
                # entering room: check threat
                reward += self._handle_enter_room()
            else:
                reward += invalid_penalty

        elif action in [3, 4, 5]:  # shoot
            idx = action - 3
            if self.arrows <= 0:
                reward += invalid_penalty
            elif idx < len(neighbors):
                target_room = neighbors[idx]
                self.arrows -= 1
                reward += shoot_penalty
                reward += self._handle_shoot(target_room, win_reward, death_penalty)
            else:
                reward += invalid_penalty
        else:
            reward += invalid_penalty

        # out of arrows and Wumpus still alive -> lose
        if (
            not self.game_over
            and self.arrows <= 0
            and self._find_wumpus_room() is not None
        ):
            self.game_over = True
            self.win = False
            reward += death_penalty

        # max steps
        if not self.game_over and self.step_count >= self.max_steps:
            self.game_over = True
            self.win = False
            reward += -2.0

        # override final reward if terminal from win/lose inside handlers
        if self.game_over:
            if self.win:
                reward = max(reward, win_reward)
            else:
                reward = min(reward, -1.0)  # ensure negative

        # update percepts for next state
        if not self.game_over:
            self._update_percepts()
        else:
            self.percepts = []

        return self._encode_state(), reward, self.game_over, {}

    # ---------- helpers ----------

    def get_safe_rooms(self, exclude=None):
        """Rooms with no threats. Optionally exclude some rooms."""
        if exclude is None:
            exclude = []
        return [r for r in self.cave.keys() if r not in self.threats and r not in exclude]

    def _find_wumpus_room(self):
        for r, t in self.threats.items():
            if t == "wumpus":
                return r
        return None

    def _handle_enter_room(self):
        """
        Handle effects of entering current player_room.
        Returns additional reward (usually 0 or death_penalty).
        """
        reward = 0.0
        threat = self.threats.get(self.player_room)

        if threat == "bat":
            # teleport to random empty room (no threats)
            safe_rooms = self.get_safe_rooms(exclude=[self.player_room])
            if safe_rooms:
                self.player_room = self.rng.choice(safe_rooms)
            # no extra reward/penalty; mostly just chaos.
            self._update_percepts()

        elif threat == "pit":
            self.game_over = True
            self.win = False
            reward += -5.0

        elif threat == "wumpus":
            self.game_over = True
            self.win = False
            reward += -5.0

        return reward

    def _handle_shoot(self, target_room, win_reward, death_penalty):
        """
        Handle shooting into target_room, moving Wumpus with 75% chance if missed.
        Returns additional reward.
        """
        reward = 0.0
        w_room = self._find_wumpus_room()
        if w_room is None:
            # No Wumpus alive, nothing happens.
            return reward

        if target_room == w_room:
            # kill Wumpus -> win
            del self.threats[w_room]
            self.game_over = True
            self.win = True
            reward += win_reward
            return reward

        # Missed: Wumpus may move (75% chance)
        if self.rng.random() < 0.75:
            old_room = w_room
            neighbors = self.cave[old_room]
            # Wumpus can move into any neighbor; if it already has threat, skip that room.
            candidates = [r for r in neighbors if self.threats.get(r) is None]
            if not candidates:
                candidates = neighbors[:]  # fallback
            new_room = self.rng.choice(candidates)
            del self.threats[old_room]
            self.threats[new_room] = "wumpus"

            # If it enters player's room -> player dies
            if new_room == self.player_room:
                self.game_over = True
                self.win = False
                reward += death_penalty

        return reward

    def _update_percepts(self):
        """Update percept messages based on adjacent rooms."""
        self.percepts = []
        for nbr in self.cave[self.player_room]:
            t = self.threats.get(nbr)
            if t == "wumpus":
                msg = "You smell something terrible nearby."
                if msg not in self.percepts:
                    self.percepts.append(msg)
            elif t == "bat":
                msg = "You hear a rustling."
                if msg not in self.percepts:
                    self.percepts.append(msg)
            elif t == "pit":
                msg = "You feel a cold wind blowing from a nearby cavern."
                if msg not in self.percepts:
                    self.percepts.append(msg)

    def _percept_flags(self):
        """Return (smell, rustle, breeze) as 0/1 flags."""
        smell = 0
        rustle = 0
        breeze = 0
        for msg in self.percepts:
            if "terrible" in msg:
                smell = 1
            elif "rustling" in msg:
                rustle = 1
            elif "cold wind" in msg:
                breeze = 1
        return smell, rustle, breeze

    def _encode_state(self):
        """
        State for Q-learning:
          (room, arrows, wumpus_alive, smell, rustle, breeze)
        """
        w_alive = 1 if (self._find_wumpus_room() is not None) else 0
        smell, rustle, breeze = self._percept_flags()
        return (self.player_room, self.arrows, w_alive, smell, rustle, breeze)
