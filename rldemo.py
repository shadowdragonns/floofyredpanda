import pygame
import random
import math
import numpy as np
import gym
from gym import spaces
import floofyredpanda as frp  

class Bullet:
    def __init__(self, bid, pos, vel, owner):
        self.id    = bid
        self.pos   = np.array(pos, dtype=np.float32)
        self.vel   = np.array(vel, dtype=np.float32)
        self.owner = owner  # 'player' or 'agent'

    def step(self, dt=1.0):
        self.pos += self.vel * dt

class TwoPlayerShooterEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 width=800, height=600,
                 fps=60, n_bullets_obs=5,
                 training_mode=False):
        super().__init__()
        pygame.init()
        pygame.font.init()

        # toggle for AI-controlled player
        self.training_mode = training_mode

        # throttle FPS for fastest training
        self.fps = 60 if self.training_mode else fps

        # bullet-tracking for miss penalty
        self.next_bullet_id       = 0
        self.active_agent_bullets = set()
        self.MISS_PENALTY         = 1

        # fire-rate cooldown (in frames)
        self.fire_delay      = 10
        self.agent_cooldown  = 0
        self.player_cooldown = 0

        # running score
        self.agent_score = 0.0

        # screen & timing
        self.width  = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Two-Player Shooter')
        self.clock  = pygame.time.Clock()

        # font for score
        self.font = pygame.font.SysFont(None, 24)

        # game constants
        self.MARGIN       = 50
        self.WALL_THICK   = 5
        self.PLAYER_SIZE  = 25
        self.AGENT_SIZE   = 10
        self.BULLET_SPEED = 20.0
        self.SHOOT_REWARD_FACTOR = 0.5
        self.MOVE_SPEED   = 10.0
        self.PLAYER_SPEED = 10.0
        self.N_OBS        = n_bullets_obs
        self.max_dist     = math.hypot(width, height)

        # playable area
        self.min_x = self.MARGIN
        self.max_x = width  - self.MARGIN
        self.min_y = self.MARGIN
        self.max_y = height - self.MARGIN

        # actions: 0 stay, 1–4 move L/R/U/D, 5–12 shoot in 8 dirs
        self.action_space = spaces.Discrete(13)

        # observations: [ax,ay, px,py, angle_to_player] + N×(dist,angle,speed)
        low  = np.array([0,0, 0,0, -1] + [0,-1,0]*self.N_OBS,
                        dtype=np.float32)
        high = np.array([width,height, width,height, 1] + [1,1,1]*self.N_OBS,
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)

        # colours
        self.WHITE = (255,255,255)
        self.BLACK = (  0,  0,  0)
        self.RED   = (255,  0,  0)
        self.BLUE  = (  0,  0,255)

        # dynamic state
        self.player_pos = None
        self.agent_pos  = None
        self.bullets    = []
        self.key_state  = {'w':False,'a':False,'s':False,'d':False}
        self.done       = False
        self.reward     = 0.0

    def reset(self):
        # centre player, random agent
        self.player_pos = np.array(
            [self.width//2, self.height//2],
            dtype=np.float32
        )
        self.agent_pos = np.array([
            random.randint(self.min_x, self.max_x),
            random.randint(self.min_y, self.max_y)
        ], dtype=np.float32)

        # clear bullets, penalties, cooldowns
        self.bullets               = []
        self.next_bullet_id        = 0
        self.active_agent_bullets  = set()
        self.agent_cooldown        = 0
        self.player_cooldown       = 0

        self.done   = False
        self.reward = 0.0
        return self._get_obs()

    def step(self, action):
        if self.done:
            return self._get_obs(), self.reward, True, {}

        # reset this step’s reward
        self.reward = 0.0

        # tick down cooldowns
        if self.agent_cooldown  > 0: self.agent_cooldown  -= 1
        if self.player_cooldown > 0: self.player_cooldown -= 1

        # AGENT: move & (maybe) shoot
        move_delta = np.zeros(2, dtype=np.float32)
        shoot_dir  = None

        if 1 <= action <= 4:
            dirs = {1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
            move_delta = np.array(dirs[action], dtype=np.float32) * self.MOVE_SPEED
        elif action >= 5:
            angle = (action-5) * (math.pi/4)
            shoot_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

        self.agent_pos += move_delta
        self.agent_pos[0] = np.clip(self.agent_pos[0], self.min_x, self.max_x)
        self.agent_pos[1] = np.clip(self.agent_pos[1], self.min_y, self.max_y)

        # spawn agent bullet if cooldown expired
        if shoot_dir is not None and self.agent_cooldown == 0:
            # —— reward‐shape for aiming ——
            rel = self.player_pos - self.agent_pos
            dist = np.linalg.norm(rel)
            if dist > 0:
                unit_rel = rel / dist
                alignment = np.dot(shoot_dir, unit_rel)  # in [-1,1]
                # only reward positive alignments (i.e. roughly pointing at player)
                self.reward += max(0.0, alignment) * self.SHOOT_REWARD_FACTOR

            # now actually fire
            vel = shoot_dir * self.BULLET_SPEED
            bid = self.next_bullet_id
            self.next_bullet_id += 1
            b = Bullet(bid, self.agent_pos.copy(), vel, 'agent')
            self.bullets.append(b)
            self.active_agent_bullets.add(bid)
            self.agent_cooldown = self.fire_delay

        # PLAYER auto-shoot in training mode
        if self.training_mode and self.player_cooldown == 0:
            rel = self.agent_pos - self.player_pos
            d   = np.linalg.norm(rel)
            if d > 0:
                dirv = rel / d
                vel  = dirv * self.BULLET_SPEED
                self.bullets.append(
                    Bullet(None, self.player_pos.copy(), vel, 'player')
                )
                self.player_cooldown = self.fire_delay

        # advance bullets
        for b in self.bullets:
            b.step()

        # check hits and assign rewards
        for b in list(self.bullets):
            if (b.owner=='agent'
                and np.linalg.norm(b.pos-self.player_pos) <= self.PLAYER_SIZE):
                self.done    = True
                self.reward += 100.0
                self.agent_score += 100.0
                self.active_agent_bullets.discard(b.id)
                self.bullets.remove(b)

            elif (b.owner=='player'
                  and np.linalg.norm(b.pos-self.agent_pos) <= self.AGENT_SIZE):
                self.done    = True
                self.reward -= 5.0
                self.agent_score -= 5.0
                self.bullets.remove(b)

        # cull off-screen & penalise agent misses
        for b in list(self.bullets):
            off = not (0 <= b.pos[0] <= self.width and 0 <= b.pos[1] <= self.height)
            if off:
                if b.owner=='agent' and b.id in self.active_agent_bullets:
                    self.reward -= self.MISS_PENALTY
                    self.agent_score -= self.MISS_PENALTY
                    self.active_agent_bullets.discard(b.id)
                self.bullets.remove(b)
        self.reward += 0.05
        self.agent_score += 0.05
        return self._get_obs(), self.reward, self.done, {}

    def render(self, mode='human'):
        # tick down player cooldown
        if self.player_cooldown > 0:
            self.player_cooldown -= 1

        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                self.done = True

        # manual input only if not training
        if not self.training_mode:
            for e in events:
                if e.type == pygame.KEYDOWN:
                    if   e.key==pygame.K_w: self.key_state['w']=True
                    elif e.key==pygame.K_a: self.key_state['a']=True
                    elif e.key==pygame.K_s: self.key_state['s']=True
                    elif e.key==pygame.K_d: self.key_state['d']=True
                elif e.type == pygame.KEYUP:
                    if   e.key==pygame.K_w: self.key_state['w']=False
                    elif e.key==pygame.K_a: self.key_state['a']=False
                    elif e.key==pygame.K_s: self.key_state['s']=False
                    elif e.key==pygame.K_d: self.key_state['d']=False
                elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.player_cooldown==0:
                    mx,my = e.pos
                    dirv  = np.array([mx,my],dtype=np.float32) - self.player_pos
                    d     = np.linalg.norm(dirv)
                    if d>0:
                        dirv /= d
                        vel = dirv * self.BULLET_SPEED
                        self.bullets.append(
                            Bullet(None, self.player_pos.copy(), vel, 'player')
                        )
                        self.player_cooldown = self.fire_delay

            dx = (self.key_state['d'] - self.key_state['a']) * self.PLAYER_SPEED
            dy = (self.key_state['s'] - self.key_state['w']) * self.PLAYER_SPEED
            self.player_pos += np.array([dx,dy], dtype=np.float32)
            self.player_pos[0] = np.clip(self.player_pos[0], self.min_x, self.max_x)
            self.player_pos[1] = np.clip(self.player_pos[1], self.min_y, self.max_y)

        # draw background and walls
        self.screen.fill(self.WHITE)
        pygame.draw.rect(
            self.screen, self.BLACK,
            (self.MARGIN, self.MARGIN,
             self.width-2*self.MARGIN,
             self.height-2*self.MARGIN),
            self.WALL_THICK
        )
        # draw player and agent
        px,py = self.player_pos
        pygame.draw.circle(self.screen, self.BLUE, (int(px),int(py)), self.PLAYER_SIZE)
        ax,ay = self.agent_pos
        pygame.draw.circle(self.screen, self.RED,  (int(ax),int(ay)), self.AGENT_SIZE)
        # draw bullets
        for b in self.bullets:
            pygame.draw.circle(self.screen, self.BLACK,
                               (int(b.pos[0]),int(b.pos[1])), 5)
        # blit the score
        score_surf = self.font.render(f'score: {self.agent_score:.1f}', True, self.BLACK)
        self.screen.blit(score_surf, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()

    def _get_obs(self):
        ax, ay = self.agent_pos
        px, py = self.player_pos

        rel = np.array([px-ax, py-ay], dtype=np.float32)
        angle_to_player = (math.atan2(rel[1], rel[0]) / math.pi
                           if np.linalg.norm(rel)>0 else 0.0)

        obs = [ax, ay, px, py, angle_to_player]

        incoming = [b for b in self.bullets if b.owner=='player']
        incoming.sort(key=lambda b: np.linalg.norm(b.pos-self.agent_pos))
        for b in incoming[:self.N_OBS]:
            relb      = b.pos - self.agent_pos
            dist_norm = np.linalg.norm(relb)/self.max_dist
            ang_norm  = math.atan2(relb[1],relb[0]) / math.pi
            spd_norm  = np.linalg.norm(b.vel)/self.BULLET_SPEED
            obs += [dist_norm, ang_norm, spd_norm]

        while len(obs) < 5 + 3*self.N_OBS:
            obs.append(0.0)

        return np.array(obs, dtype=np.float32)


if __name__ == '__main__':
    training = True  # flip for human vs AI

    env   = TwoPlayerShooterEnv(training_mode=training)
    obs   = env.reset()
    agent = frp.rl(env, 0)

    try:
        while True:
            action = agent.act(obs) if hasattr(agent,'act') else agent(env)
            next_obs, reward, done, _ = env.step(action)
            if hasattr(agent,'learn'):
                agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs if not done else env.reset()
            env.render()
    finally:
        if hasattr(agent,'save'):
            agent.save("agent.pkl")
        env.close()
