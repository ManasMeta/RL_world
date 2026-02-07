"""
HAWKINS HEIST - Complete Runnable Environment
Play with keyboard controls: Arrow keys to move, Shift to sprint
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class PatrolType(Enum):
    LINEAR = "linear"
    ROTATING = "rotating"
    WANDERING = "wandering"


@dataclass
class Guard:
    position: List[float]
    patrol_type: PatrolType
    patrol_points: List[Tuple[int, int]]
    speed: float
    rotation: float
    rotation_speed: float
    vision_radius: int
    vision_angle: int
    current_waypoint: int
    detection_level: float


class HawkinsHeistEnv(gym.Env):
    """
    Hawkins Heist - Stealth Infiltration Environment
    Navigate through an open Soviet laboratory avoiding guard detection
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    GRID_WIDTH = 30
    GRID_HEIGHT = 30
    CELL_SIZE = 24
    START_POS = (1, 1)
    EXIT_POS = (28, 28)
    
    def __init__(self, render_mode: Optional[str] = None, difficulty: str = "medium"):
        super().__init__()
        
        self.difficulty = difficulty
        self.render_mode = render_mode
        
        # Action space: 0=stay, 1-4=move, 5=sprint
        self.action_space = spaces.Discrete(6)
        
        # Observation: agent(2) + 5 guards(15) + detection(1) = 18
        self.observation_space = spaces.Box(
            low=0, high=max(self.GRID_WIDTH, 360),
            shape=(18,), dtype=np.float32
        )
        
        self._create_map_layout()
        
        self.window = None
        self.clock = None
        self.assets = {}
        
        self.agent_pos = list(self.START_POS)
        self.guards: List[Guard] = []
        self.steps = 0
        self.max_steps = 1000
        self.total_detection = 0
        self.detected = False
        self.is_sprinting = False
    
    def _create_map_layout(self):
        """Create open laboratory layout"""
        self.map_layout = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.int32)
        
        # Server racks (obstacles) - 32 positions
        self.obstacle_positions = [
            (5,5), (5,6), (6,5), (8,4), (8,5),
            (4,12), (4,13), (4,14), (7,16), (7,17),
            (12,8), (13,8), (15,12), (16,12), (17,12),
            (14,16), (15,16), (10,20), (11,20),
            (20,7), (21,7), (22,7), (24,11), (24,12),
            (19,15), (20,15), (22,19), (23,19),
            (25,22), (26,22), (23,25),
        ]
        
        # Generators (large cover) - 5 positions
        self.generator_positions = [(9,9), (6,18), (16,6), (13,22), (21,24)]
        
        # Vents (hiding spots) - 5 positions
        self.vent_positions = [(3,8), (11,14), (18,10), (15,20), (24,26)]
        
        self.exit_position = self.EXIT_POS
        
        # Mark on map: 0=empty, 1=obstacle, 2=generator, 3=vent, 4=exit
        for pos in self.obstacle_positions:
            self.map_layout[pos[1], pos[0]] = 1
        for pos in self.generator_positions:
            self.map_layout[pos[1], pos[0]] = 2
        for pos in self.vent_positions:
            self.map_layout[pos[1], pos[0]] = 3
        self.map_layout[self.exit_position[1], self.exit_position[0]] = 4
        
        self._create_guard_patrols()
    
    def _create_guard_patrols(self):
        """Configure guard patrol patterns"""
        configs = {
            "easy": {"num": 3, "radius": 4, "speed": 0.05},
            "medium": {"num": 5, "radius": 5, "speed": 0.08},
            "hard": {"num": 7, "radius": 6, "speed": 0.12},
        }
        cfg = configs[self.difficulty]
        
        all_guards = [
            # Guard 1: Linear top patrol
            {
                "start_pos": [8, 3],
                "patrol_type": PatrolType.LINEAR,
                "waypoints": [(8,3), (22,3), (22,8), (8,8)],
                "speed": cfg["speed"],
                "vision_radius": cfg["radius"],
                "vision_angle": 90,
                "rotation": 90,
            },
            # Guard 2: Rotating center sentinel
            {
                "start_pos": [15, 15],
                "patrol_type": PatrolType.ROTATING,
                "waypoints": [(15,15)],
                "speed": 0,
                "rotation_speed": 1.5,
                "vision_radius": cfg["radius"],
                "vision_angle": 120,
                "rotation": 0,
            },
            # Guard 3: Wandering left
            {
                "start_pos": [5, 12],
                "patrol_type": PatrolType.WANDERING,
                "waypoints": [(5,12), (10,12), (10,18), (5,18)],
                "speed": cfg["speed"] * 0.7,
                "vision_radius": cfg["radius"] - 1,
                "vision_angle": 100,
                "rotation": 180,
            },
            # Guard 4: Linear exit approach
            {
                "start_pos": [18, 18],
                "patrol_type": PatrolType.LINEAR,
                "waypoints": [(18,18), (25,25), (20,25), (20,20)],
                "speed": cfg["speed"],
                "vision_radius": cfg["radius"],
                "vision_angle": 110,
                "rotation": 45,
            },
            # Guard 5: Rotating exit guardian
            {
                "start_pos": [26, 24],
                "patrol_type": PatrolType.ROTATING,
                "waypoints": [(26,24)],
                "speed": 0,
                "rotation_speed": 2.0,
                "vision_radius": cfg["radius"] + 1,
                "vision_angle": 130,
                "rotation": 225,
            },
        ]
        
        self.guard_configs = all_guards[:cfg["num"]]
    
    def _init_guards(self):
        """Initialize guard objects"""
        self.guards = []
        for cfg in self.guard_configs:
            guard = Guard(
                position=cfg["start_pos"].copy(),
                patrol_type=cfg["patrol_type"],
                patrol_points=cfg["waypoints"],
                speed=cfg["speed"],
                rotation=cfg["rotation"],
                rotation_speed=cfg.get("rotation_speed", 0),
                vision_radius=cfg["vision_radius"],
                vision_angle=cfg["vision_angle"],
                current_waypoint=0,
                detection_level=0.0
            )
            self.guards.append(guard)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        
        self.agent_pos = list(self.START_POS)
        self.steps = 0
        self.total_detection = 0
        self.detected = False
        self.is_sprinting = False
        
        self._init_guards()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), self._get_info()
    
    def _get_obs(self):
        """Get observation vector"""
        obs = [
            self.agent_pos[0] / self.GRID_WIDTH,
            self.agent_pos[1] / self.GRID_HEIGHT
        ]
        
        # 5 nearest guards
        guards_sorted = sorted(self.guards, key=lambda g: math.dist(g.position, self.agent_pos))[:5]
        for guard in guards_sorted:
            obs.extend([
                guard.position[0] / self.GRID_WIDTH,
                guard.position[1] / self.GRID_HEIGHT,
                guard.rotation / 360.0,
            ])
        
        # Pad if fewer than 5 guards
        while len(obs) < 17:
            obs.extend([0, 0, 0])
        
        # Max detection level
        max_detection = max([g.detection_level for g in self.guards], default=0)
        obs.append(max_detection)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self):
        return {
            "steps": self.steps,
            "total_detection": self.total_detection,
            "detected": self.detected,
            "distance_to_exit": math.dist(self.agent_pos, self.exit_position)
        }
    
    def _is_valid_move(self, pos: Tuple[int, int]) -> bool:
        """Check if position is walkable"""
        x, y = pos
        if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
            return False
        cell = self.map_layout[y, x]
        return cell not in [1, 2]  # Can't walk through obstacles or generators
    
    def _agent_in_vent(self) -> bool:
        """Check if agent is hiding in vent"""
        return tuple(self.agent_pos) in self.vent_positions
    
    def _update_guards(self):
        """Update all guard positions and rotations"""
        for guard in self.guards:
            if guard.patrol_type == PatrolType.ROTATING:
                guard.rotation = (guard.rotation + guard.rotation_speed) % 360
                
            elif guard.patrol_type == PatrolType.LINEAR:
                target = guard.patrol_points[guard.current_waypoint]
                dx = target[0] - guard.position[0]
                dy = target[1] - guard.position[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 0.2:
                    guard.current_waypoint = (guard.current_waypoint + 1) % len(guard.patrol_points)
                    target = guard.patrol_points[guard.current_waypoint]
                    dx = target[0] - guard.position[0]
                    dy = target[1] - guard.position[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > 0:
                    guard.position[0] += (dx / dist) * guard.speed
                    guard.position[1] += (dy / dist) * guard.speed
                    guard.rotation = math.degrees(math.atan2(dy, dx))
                    
            elif guard.patrol_type == PatrolType.WANDERING:
                if np.random.random() < 0.02:
                    guard.current_waypoint = np.random.randint(0, len(guard.patrol_points))
                
                target = guard.patrol_points[guard.current_waypoint]
                dx = target[0] - guard.position[0]
                dy = target[1] - guard.position[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist > 0.3:
                    guard.position[0] += (dx / dist) * guard.speed
                    guard.position[1] += (dy / dist) * guard.speed
                    guard.rotation = math.degrees(math.atan2(dy, dx))
    
    def _check_line_of_sight(self, from_pos: List[float], to_pos: List[int]) -> bool:
        """Check if line of sight is clear (simplified Bresenham)"""
        x0, y0 = int(from_pos[0]), int(from_pos[1])
        x1, y1 = to_pos[0], to_pos[1]
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        
        if dx > dy:
            error = dx / 2
            while x != x1:
                if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                    if self.map_layout[y, x] in [1, 2]:
                        return False
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2
            while y != y1:
                if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                    if self.map_layout[y, x] in [1, 2]:
                        return False
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc
        
        return True
    
    def _check_detection(self) -> float:
        """Check if agent is detected by any guard"""
        if self._agent_in_vent():
            for guard in self.guards:
                guard.detection_level = max(0, guard.detection_level - 0.1)
            return 0.0
        
        max_detection = 0.0
        
        for guard in self.guards:
            dx = self.agent_pos[0] - guard.position[0]
            dy = self.agent_pos[1] - guard.position[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist <= guard.vision_radius:
                angle_to_agent = math.degrees(math.atan2(dy, dx))
                angle_diff = abs((angle_to_agent - guard.rotation + 180) % 360 - 180)
                
                if angle_diff <= guard.vision_angle / 2:
                    if self._check_line_of_sight(guard.position, self.agent_pos):
                        detection_strength = 1.0 - (dist / guard.vision_radius)
                        if self.is_sprinting:
                            detection_strength *= 1.5
                        
                        guard.detection_level = min(1.0, guard.detection_level + detection_strength * 0.1)
                        max_detection = max(max_detection, guard.detection_level)
                    else:
                        guard.detection_level = max(0, guard.detection_level - 0.05)
                else:
                    guard.detection_level = max(0, guard.detection_level - 0.05)
            else:
                guard.detection_level = max(0, guard.detection_level - 0.05)
        
        return max_detection
    
    def step(self, action):
        """Execute one environment step"""
        self.steps += 1
        
        self.is_sprinting = (action == 5)
        move_action = action if action < 5 else 4
        
        # Move agent
        new_pos = self.agent_pos.copy()
        if move_action == 1:  # up
            new_pos[1] -= 1
        elif move_action == 2:  # right
            new_pos[0] += 1
        elif move_action == 3:  # down
            new_pos[1] += 1
        elif move_action == 4:  # left
            new_pos[0] -= 1
        
        if self._is_valid_move(tuple(new_pos)):
            self.agent_pos = new_pos
        
        # Sprint = move twice
        if self.is_sprinting and move_action != 0:
            sprint_pos = self.agent_pos.copy()
            if move_action == 1:
                sprint_pos[1] -= 1
            elif move_action == 2:
                sprint_pos[0] += 1
            elif move_action == 3:
                sprint_pos[1] += 1
            elif move_action == 4:
                sprint_pos[0] -= 1
            
            if self._is_valid_move(tuple(sprint_pos)):
                self.agent_pos = sprint_pos
        
        # Update guards
        self._update_guards()
        
        # Check detection
        detection_level = self._check_detection()
        self.total_detection += detection_level
        
        # Calculate reward
        reward = -0.01
        terminated = False
        
        if detection_level > 0:
            reward -= detection_level * 0.5
        
        if detection_level > 0.8:
            self.detected = True
            reward = -50
            terminated = True
        
        if tuple(self.agent_pos) == self.exit_position:
            reward = 100 - (self.total_detection * 10)
            terminated = True
        
        if self.steps >= self.max_steps:
            terminated = True
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), reward, terminated, False, self._get_info()
    
    def _load_assets(self):
        """Load PNG assets from assets folder"""
        if self.assets:
            return
        
        asset_files = {
            'clean_floor': 'clean_floor.png',
            'dark_floor': 'dark_floor.png',
            'warning_floor': 'warning_floor.png',
            'exit_tile': 'exit_tile.png',
            'vent_tile': 'vent_tile.png',
            'generator': 'generator_tile.png',
            'server_rack': 'server_rack.png',
            'army': 'army.png',
            'trio': 'trio.png',
        }
        
        assets_dir = 'assets'
        
        for key, filename in asset_files.items():
            filepath = os.path.join(assets_dir, filename)
            if os.path.exists(filepath):
                try:
                    img = pygame.image.load(filepath)
                    self.assets[key] = pygame.transform.scale(img, (self.CELL_SIZE, self.CELL_SIZE))
                except:
                    print(f"Warning: Could not load {filename}")
            else:
                print(f"Warning: Asset not found: {filepath}")
    
    def _render_frame(self):
        """Render the environment"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            window_width = self.GRID_WIDTH * self.CELL_SIZE + 250
            window_height = self.GRID_HEIGHT * self.CELL_SIZE
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Hawkins Heist - Stealth Infiltration")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        self._load_assets()
        
        canvas = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE + 250, self.GRID_HEIGHT * self.CELL_SIZE))
        canvas.fill((20, 25, 30))
        
        # Draw map
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                # Draw floor
                dist_to_exit = math.dist((x, y), self.exit_position)
                if dist_to_exit < 8:
                    color = (60, 55, 50) if 'warning_floor' not in self.assets else None
                    if color:
                        pygame.draw.rect(canvas, color, rect)
                    else:
                        canvas.blit(self.assets.get('warning_floor'), rect)
                elif dist_to_exit < 15:
                    color = (40, 45, 50) if 'dark_floor' not in self.assets else None
                    if color:
                        pygame.draw.rect(canvas, color, rect)
                    else:
                        canvas.blit(self.assets.get('dark_floor'), rect)
                else:
                    color = (50, 55, 60) if 'clean_floor' not in self.assets else None
                    if color:
                        pygame.draw.rect(canvas, color, rect)
                    else:
                        canvas.blit(self.assets.get('clean_floor'), rect)
                
                pygame.draw.rect(canvas, (30, 35, 40), rect, 1)
                
                # Draw objects
                cell = self.map_layout[y, x]
                if cell == 1:  # Obstacle
                    if 'server_rack' in self.assets:
                        canvas.blit(self.assets['server_rack'], rect)
                    else:
                        pygame.draw.rect(canvas, (60, 70, 80), rect)
                elif cell == 2:  # Generator
                    if 'generator' in self.assets:
                        canvas.blit(self.assets['generator'], rect)
                    else:
                        pygame.draw.rect(canvas, (80, 70, 60), rect)
                        pygame.draw.circle(canvas, (255, 200, 0), rect.center, 4)
                elif cell == 3:  # Vent
                    if 'vent_tile' in self.assets:
                        canvas.blit(self.assets['vent_tile'], rect)
                    else:
                        pygame.draw.rect(canvas, (40, 40, 40), rect)
                        pygame.draw.circle(canvas, (80, 80, 80), rect.center, 6)
                elif cell == 4:  # Exit
                    if 'exit_tile' in self.assets:
                        canvas.blit(self.assets['exit_tile'], rect)
                    else:
                        pygame.draw.rect(canvas, (255, 150, 0), rect)
        
        # Draw vision cones
        for guard in self.guards:
            self._draw_vision_cone(canvas, guard)
        
        # Draw guards
        for guard in self.guards:
            gx = int(guard.position[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
            gy = int(guard.position[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            
            if 'army' in self.assets:
                rotated = pygame.transform.rotate(self.assets['army'], -guard.rotation)
                rect = rotated.get_rect(center=(gx, gy))
                canvas.blit(rotated, rect)
            else:
                color = (200, 50, 50) if guard.detection_level > 0.5 else (150, 150, 150)
                pygame.draw.circle(canvas, color, (gx, gy), 10)
                end_x = gx + math.cos(math.radians(guard.rotation)) * 15
                end_y = gy + math.sin(math.radians(guard.rotation)) * 15
                pygame.draw.line(canvas, (255, 255, 255), (gx, gy), (end_x, end_y), 2)
        
        # Draw agent
        ax = int(self.agent_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        ay = int(self.agent_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        if self._agent_in_vent():
            pygame.draw.circle(canvas, (100, 255, 100), (ax, ay), 8)
            pygame.draw.circle(canvas, (50, 200, 50), (ax, ay), 8, 2)
        else:
            if 'trio' in self.assets:
                canvas.blit(self.assets['trio'], pygame.Rect(ax - self.CELL_SIZE//2, ay - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE))
            else:
                pygame.draw.circle(canvas, (0, 255, 100), (ax, ay), 10)
                if self.is_sprinting:
                    pygame.draw.circle(canvas, (255, 255, 0), (ax, ay), 12, 2)
        
        # Draw HUD
        self._draw_hud(canvas)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def _draw_vision_cone(self, canvas, guard):
        """Draw guard vision cone"""
        gx = int(guard.position[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
        gy = int(guard.position[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
        
        radius = guard.vision_radius * self.CELL_SIZE
        start_angle = guard.rotation - guard.vision_angle / 2
        end_angle = guard.rotation + guard.vision_angle / 2
        
        points = [(gx, gy)]
        for angle in range(int(start_angle), int(end_angle) + 1, 5):
            px = gx + math.cos(math.radians(angle)) * radius
            py = gy + math.sin(math.radians(angle)) * radius
            points.append((px, py))
        points.append((gx, gy))
        
        if guard.detection_level > 0.5:
            color = (255, 100, 100, 100)
        elif guard.detection_level > 0.2:
            color = (255, 200, 100, 80)
        else:
            color = (255, 255, 200, 60)
        
        s = pygame.Surface((canvas.get_width(), canvas.get_height()), pygame.SRCALPHA)
        pygame.draw.polygon(s, color, points)
        canvas.blit(s, (0, 0))
        pygame.draw.lines(canvas, (200, 200, 150), False, points, 1)
    
    def _draw_hud(self, canvas):
        """Draw HUD information"""
        hud_x = self.GRID_WIDTH * self.CELL_SIZE + 10
        font_large = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 20)
        
        y = 20
        
        title = font_large.render("HAWKINS HEIST", True, (255, 200, 100))
        canvas.blit(title, (hud_x, y))
        y += 40
        
        status = "INFILTRATING..."
        color = (100, 255, 100)
        if self.detected:
            status = "DETECTED!"
            color = (255, 50, 50)
        elif tuple(self.agent_pos) == self.exit_position:
            status = "ESCAPED!"
            color = (100, 255, 255)
        
        text = font_small.render(status, True, color)
        canvas.blit(text, (hud_x, y))
        y += 30
        
        stats = [
            f"Step: {self.steps}/{self.max_steps}",
            f"Detection: {int(self.total_detection * 100)}",
            f"Distance: {int(math.dist(self.agent_pos, self.exit_position))}",
            "",
            "CONTROLS:",
            "Arrow Keys - Move",
            "Shift - Sprint",
            "Q - Quit",
        ]
        
        for stat in stats:
            text = font_small.render(stat, True, (200, 200, 200))
            canvas.blit(text, (hud_x, y))
            y += 22
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# PLAY MANUALLY
if __name__ == "__main__":
    env = HawkinsHeistEnv(render_mode="human", difficulty="medium")
    obs, info = env.reset()
    
    print("=" * 60)
    print("HAWKINS HEIST - STEALTH INFILTRATION")
    print("=" * 60)
    print("\nMission: Reach exit (28,28) without detection!")
    print("\nControls:")
    print("  Arrow Keys - Move")
    print("  Shift + Arrow - Sprint (faster but louder)")
    print("  Q - Quit")
    print("\nTips:")
    print("  - Hide in vents (grates) for perfect concealment")
    print("  - Use generators and racks as cover")
    print("  - Avoid yellow/red vision cones")
    print("=" * 60)
    
    running = True
    
    while running:
        action = 0
        sprint = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
        
        keys = pygame.key.get_pressed()
        sprint = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        if keys[pygame.K_UP]:
            action = 5 if sprint else 1
        elif keys[pygame.K_RIGHT]:
            action = 5 if sprint else 2
        elif keys[pygame.K_DOWN]:
            action = 5 if sprint else 3
        elif keys[pygame.K_LEFT]:
            action = 5 if sprint else 4
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print("\n" + "=" * 60)
            if info['detected']:
                print("MISSION FAILED - Detected by guards!")
            elif tuple(env.agent_pos) == env.exit_position:
                print("MISSION SUCCESS!")
                print(f"Stealth Score: {100 - int(info['total_detection'] * 10)}/100")
            print(f"Steps: {info['steps']}")
            print("=" * 60)
            print("\nPress R to restart, Q to quit")
            
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            obs, info = env.reset()
                            waiting = False
                        elif event.key == pygame.K_q:
                            running = False
                            waiting = False
    
    env.close()
