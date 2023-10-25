import pygame
import numpy as np
import tcod
import random
from enum import Enum
import heapq

from queue import PriorityQueue

current_hero = (1,2)

restart_flag = 0
i_am_running = 1

LIVES = 3

class Direction(Enum):
    DOWN = -90
    RIGHT = 0
    UP = 90
    LEFT = 180
    NONE = 360


class ScoreType(Enum):
    COOKIE = 10
    POWERUP = 50
    GHOST = 400


class GhostBehaviour(Enum):
    CHASE = 1
    SCATTER = 2


def translate_screen_to_maze(in_coords, in_size=32):
    return int(in_coords[0] / in_size), int(in_coords[1] / in_size)


def translate_maze_to_screen(in_coords, in_size=32):
    return in_coords[0] * in_size, in_coords[1] * in_size


class GameObject:
    def __init__(self, in_surface, x, y,
                 in_size: int, in_color=(255, 0, 0),
                 is_circle: bool = False):
        self._size = in_size
        self._renderer: GameRenderer = in_surface
        self._surface = in_surface._screen
        self.y = y
        self.x = x
        self._color = in_color
        self._circle = is_circle
        self._shape = pygame.Rect(self.x, self.y, in_size, in_size)

    def draw(self):
        if self._circle:
            pygame.draw.circle(self._surface,
                               self._color,
                               (self.x, self.y),
                               self._size)
        else:
            rect_object = pygame.Rect(self.x, self.y, self._size, self._size)
            pygame.draw.rect(self._surface,
                             self._color,
                             rect_object,
                             border_radius=1)

    def tick(self):
        pass

    def get_shape(self):
        return pygame.Rect(self.x, self.y, self._size, self._size)

    def set_position(self, in_x, in_y):
        self.x = in_x
        self.y = in_y

    def get_position(self):
        return (self.x, self.y)


class Wall(GameObject):
    def __init__(self, in_surface, x, y, in_size: int, in_color=(0, 0, 255)): #Blue Color
        super().__init__(in_surface, x * in_size, y * in_size, in_size, in_color)


class GameRenderer:
    def __init__(self, in_width: int, in_height: int):
        global LIVES
        # print(in_width)
        # print(in_height)
        # pygame.init()
        self._width = in_width
        self._height = in_height
        self._screen = pygame.display.set_mode((in_width, in_height))
        pygame.display.set_caption('GHOST_HUNTER')
        self._clock = pygame.time.Clock()
        self._done = False
        self._won = False
        self._game_objects = []
        self._walls = []
        self._cookies = []
        self._powerups = []
        self._ghosts = []
        self._hero: Hero = None
        self._lives = LIVES
        self._score = 0
        self._score_cookie_pickup = 10
        self._score_ghost_eaten = 400
        self._score_powerup_pickup = 50
        self._kokoro_active = False # powerup, special ability
        self._current_mode = GhostBehaviour.SCATTER
        self._mode_switch_event = pygame.USEREVENT + 1  # Custom events defined using unique identifiers.
        self._kokoro_end_event = pygame.USEREVENT + 2
        self._pakupaku_event = pygame.USEREVENT + 3
        self._modes = [ #A list of tuples representing the behavior modes of the ghosts. 
                        #Each tuple contains two values: the number of seconds the mode lasts and the number of seconds the mode should remain active.
            (7, 20),
            (7, 20),
            (5, 20),
            (5, 999999)  # 'infinite' chase seconds
        ]
        self._current_phase = 0

    #Game LOOP
    def tick(self, in_fps: int):
        black = (0, 0, 0)
        global restart_flag
        self.handle_mode_switch()
        pygame.time.set_timer(self._pakupaku_event, 200) # open close mouth
        # self.start_game()
        while not self._done:
            for game_object in self._game_objects:
                game_object.tick()
                game_object.draw()

            self.display_text(f"[Score: {self._score}]  [Lives: {self._lives}]")

            if self._hero is None: 
                self.display_text("YOU DIED", (self._width / 2 - 356, self._height / 2 - 256), 100)
                self.display_text("Press r to RESTART", (self._width / 2 - 356, self._height / 2 - 0), 100)
            if self.get_won(): self.display_text("YOU WON", (self._width / 2 - 256, self._height / 2 - 256), 100)
            pygame.display.flip()
            self._clock.tick(in_fps)
            self._screen.fill(black)
            self._handle_events()
            if restart_flag == 1:
                return

        print("Game over")

    def handle_mode_switch(self):
        current_phase_timings = self._modes[self._current_phase]
        print(f"Current phase: {str(self._current_phase)}, current_phase_timings: {str(current_phase_timings)}")
        scatter_timing = current_phase_timings[0]
        chase_timing = current_phase_timings[1]

        if self._current_mode == GhostBehaviour.CHASE:
            self._current_phase += 1
            self.set_current_mode(GhostBehaviour.SCATTER)
        else:
            self.set_current_mode(GhostBehaviour.CHASE)

        used_timing = scatter_timing if self._current_mode == GhostBehaviour.SCATTER else chase_timing
        pygame.time.set_timer(self._mode_switch_event, used_timing * 1000)

    def start_kokoro_timeout(self):
        pygame.time.set_timer(self._kokoro_end_event, 15000)  # 15s

    def add_game_object(self, obj: GameObject):
        self._game_objects.append(obj)

    def add_cookie(self, obj: GameObject):
        self._game_objects.append(obj)
        self._cookies.append(obj)

    def add_ghost(self, obj: GameObject):
        self._game_objects.append(obj)
        self._ghosts.append(obj)

    def add_powerup(self, obj: GameObject):
        self._game_objects.append(obj)
        self._powerups.append(obj)

    def activate_kokoro(self):
        self._kokoro_active = True
        self.set_current_mode(GhostBehaviour.SCATTER)
        self.start_kokoro_timeout()

    def set_won(self):
        self._won = True

    def get_won(self):
        return self._won

    def add_score(self, in_score: ScoreType):
        self._score += in_score.value

    def get_hero_position(self):
        return self._hero.get_position() if self._hero != None else (0, 0)

    def set_current_mode(self, in_mode: GhostBehaviour):
        self._current_mode = in_mode

    def get_current_mode(self):
        return self._current_mode

    def end_game(self):
        if self._hero in self._game_objects:
            self._game_objects.remove(self._hero)
        self._hero = None

    def kill_pacman(self):
        self._lives -= 1
        self._hero.set_position(32, 32)
        self._hero.set_direction(Direction.NONE)
        if self._lives == 0: self.end_game()

    def display_text(self, text, in_position=(32, 0), in_size=30):
        font = pygame.font.SysFont('Arial', in_size)
        text_surface = font.render(text, False, (255, 255, 255))
        self._screen.blit(text_surface, in_position)

    def is_kokoro_active(self):
        return self._kokoro_active

    def add_wall(self, obj: Wall):
        self.add_game_object(obj)
        self._walls.append(obj)

    def get_walls(self):
        return self._walls

    def get_cookies(self):
        return self._cookies

    def get_ghosts(self):
        return self._ghosts

    def get_powerups(self):
        return self._powerups

    def get_game_objects(self):
        return self._game_objects

    def add_hero(self, in_hero):
        self.add_game_object(in_hero)
        self._hero = in_hero

    def _handle_events(self):
        global restart_flag, i_am_running
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._done = True
                i_am_running = 0

            if event.type == self._mode_switch_event:
                self.handle_mode_switch()

            if event.type == self._kokoro_end_event:
                self._kokoro_active = False

            if event.type == self._pakupaku_event:
                if self._hero is None: break
                self._hero.mouth_open = not self._hero.mouth_open

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_r]:
            restart_flag = 1
        if self._hero is None: return
        if pressed[pygame.K_UP]:
            self._hero.set_direction(Direction.UP)
        elif pressed[pygame.K_LEFT]:
            self._hero.set_direction(Direction.LEFT)
        elif pressed[pygame.K_DOWN]:
            self._hero.set_direction(Direction.DOWN)
        elif pressed[pygame.K_RIGHT]:
            self._hero.set_direction(Direction.RIGHT)


class MovableObject(GameObject):
    def __init__(self, in_surface, x, y, in_size: int, in_color=(255, 0, 0), is_circle: bool = False):
        super().__init__(in_surface, x, y, in_size, in_color, is_circle)
        self.current_direction = Direction.NONE
        self.direction_buffer = Direction.NONE
        self.last_working_direction = Direction.NONE
        self.location_queue = []
        self.next_target = None
        self.image = pygame.image.load('images/ghost.png')

    def get_next_location(self):
        return None if len(self.location_queue) == 0 else self.location_queue.pop(0)

    def set_direction(self, in_direction):
        self.current_direction = in_direction
        self.direction_buffer = in_direction

    def collides_with_wall(self, in_position):
        collision_rect = pygame.Rect(in_position[0], in_position[1], self._size, self._size)
        collides = False
        walls = self._renderer.get_walls()
        for wall in walls:
            collides = collision_rect.colliderect(wall.get_shape())
            if collides: break
        return collides

    def check_collision_in_direction(self, in_direction: Direction):
        desired_position = (0, 0)
        if in_direction == Direction.NONE: return False, desired_position
        if in_direction == Direction.UP:
            desired_position = (self.x, self.y - 1)
        elif in_direction == Direction.DOWN:
            desired_position = (self.x, self.y + 1)
        elif in_direction == Direction.LEFT:
            desired_position = (self.x - 1, self.y)
        elif in_direction == Direction.RIGHT:
            desired_position = (self.x + 1, self.y)

        return self.collides_with_wall(desired_position), desired_position

    def automatic_move(self, in_direction: Direction):
        pass

    def tick(self):
        self.reached_target()
        self.automatic_move(self.current_direction)

    def reached_target(self):
        pass
    
    def draw(self):
        self.image = pygame.transform.scale(self.image, (32, 32))
        self._surface.blit(self.image, self.get_shape())


class Hero(MovableObject):
    def __init__(self, in_surface, x, y, in_size: int):
        super().__init__(in_surface, x, y, in_size, (255, 255, 0), False)
        self.last_non_colliding_position = (0, 0)
        self.open = pygame.image.load("images/paku.png")
        self.closed = pygame.image.load("images/man.png")
        self.image = self.open
        self.mouth_open = True


    def tick(self):
        # TELEPORT
        # global current_hero
        if self.x < 0:
            self.x = self._renderer._width

        if self.x > self._renderer._width:
            self.x = 0

        self.last_non_colliding_position = self.get_position()

        if self.check_collision_in_direction(self.direction_buffer)[0]:
            self.automatic_move(self.current_direction)
        else:
            self.automatic_move(self.direction_buffer)
            self.current_direction = self.direction_buffer

        if self.collides_with_wall((self.x, self.y)):
            self.set_position(self.last_non_colliding_position[0], self.last_non_colliding_position[1])
        # Ghost Real Time coordiante print(self.x,self.y)
        # current_hero = translate_screen_to_maze((self.y,self.x))
        self.handle_cookie_pickup()
        self.handle_ghosts()

    def automatic_move(self, in_direction: Direction):
        collision_result = self.check_collision_in_direction(in_direction)

        desired_position_collides = collision_result[0]
        if not desired_position_collides:
            self.last_working_direction = self.current_direction
            desired_position = collision_result[1]
            self.set_position(desired_position[0], desired_position[1])
        else:
            self.current_direction = self.last_working_direction

    def handle_cookie_pickup(self):
        collision_rect = pygame.Rect(self.x, self.y, self._size, self._size)
        cookies = self._renderer.get_cookies()
        powerups = self._renderer.get_powerups()
        game_objects = self._renderer.get_game_objects()
        cookie_to_remove = None
        for cookie in cookies:
            collides = collision_rect.colliderect(cookie.get_shape())
            if collides and cookie in game_objects:
                game_objects.remove(cookie)
                self._renderer.add_score(ScoreType.COOKIE)
                cookie_to_remove = cookie

        if cookie_to_remove is not None:
            cookies.remove(cookie_to_remove)
        # print("len(self._renderer.get_cookies())   ",len(self._renderer.get_cookies()))
        # if len(self._renderer.get_cookies()) == 200:
        #     self._renderer.set_won()
        if len(self._renderer.get_cookies()) == 0:
            self._renderer.set_won()

        for powerup in powerups:
            collides = collision_rect.colliderect(powerup.get_shape())
            if collides and powerup in game_objects:
                if not self._renderer.is_kokoro_active():
                    game_objects.remove(powerup)
                    self._renderer.add_score(ScoreType.POWERUP)
                    self._renderer.activate_kokoro()

    def handle_ghosts(self):
        collision_rect = pygame.Rect(self.x, self.y, self._size, self._size)
        ghosts = self._renderer.get_ghosts()
        game_objects = self._renderer.get_game_objects()
        for ghost in ghosts:
            collides = collision_rect.colliderect(ghost.get_shape())
            if collides and ghost in game_objects:
                if self._renderer.is_kokoro_active():
                    # game_objects.remove(ghost)
                    self._renderer.add_score(ScoreType.GHOST)
                else:
                    if not self._renderer.get_won():
                        self._renderer.kill_pacman()

    def draw(self):
        half_size = self._size / 2
        self.image = self.open if self.mouth_open else self.closed
        self.image = pygame.transform.rotate(self.image, self.current_direction.value)
        super(Hero, self).draw()


class Ghost(MovableObject):
    def __init__(self, in_surface, x, y, in_size: int, in_game_controller, sprite_path="images/ghost_fright.png"):
        super().__init__(in_surface, x, y, in_size)
        self.game_controller = in_game_controller
        self.sprite_normal = pygame.image.load(sprite_path)
        self.sprite_fright = pygame.image.load("images/ghost_fright.png")

    def reached_target(self):
        if (self.x, self.y) == self.next_target:
            self.next_target = self.get_next_location()
        self.current_direction = self.calculate_direction_to_next_target()

    def set_new_path(self, in_path):
        for item in in_path:
            self.location_queue.append(item)
        self.next_target = self.get_next_location()

    def calculate_direction_to_next_target(self) -> Direction:
        if self.next_target is None:
            if self._renderer.get_current_mode() == GhostBehaviour.CHASE and not self._renderer.is_kokoro_active():
                self.request_path_to_player(self)
            else:
                self.game_controller.request_new_path(self)
            return Direction.NONE

        diff_x = self.next_target[0] - self.x
        diff_y = self.next_target[1] - self.y
        if diff_x == 0:
            return Direction.DOWN if diff_y > 0 else Direction.UP
        if diff_y == 0:
            return Direction.LEFT if diff_x < 0 else Direction.RIGHT

        if self._renderer.get_current_mode() == GhostBehaviour.CHASE and not self._renderer.is_kokoro_active():
            self.request_path_to_player(self)
        else:
            self.game_controller.request_new_path(self)
        return Direction.NONE

    def request_path_to_player(self, in_ghost):
        #Player Postion
        player_position = translate_screen_to_maze(in_ghost._renderer.get_hero_position())
        current_maze_coord = translate_screen_to_maze(in_ghost.get_position())
        path = self.game_controller.p.get_path(current_maze_coord[1], current_maze_coord[0], player_position[1],
                                               player_position[0])

        new_path = [translate_maze_to_screen(item) for item in path]
        in_ghost.set_new_path(new_path)

    def automatic_move(self, in_direction: Direction):
        if in_direction == Direction.UP:
            self.set_position(self.x, self.y - 1)
        elif in_direction == Direction.DOWN:
            self.set_position(self.x, self.y + 1)
        elif in_direction == Direction.LEFT:
            self.set_position(self.x - 1, self.y)
        elif in_direction == Direction.RIGHT:
            self.set_position(self.x + 1, self.y)
    def draw(self):
        self.image = self.sprite_fright if self._renderer.is_kokoro_active() else self.sprite_normal
        super(Ghost, self).draw()

class Cookie(GameObject):
    def __init__(self, in_surface, x, y):
        super().__init__(in_surface, x, y, 4, (255, 255, 0), True)


class Powerup(GameObject):
    def __init__(self, in_surface, x, y):
        super().__init__(in_surface, x, y, 8, (255, 255, 255), True)

# Astar
gROWs = 0
gCOLs = 0

MOVEMENTS = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1)    # Right
]

def is_valid_move(grid, row, col):
    """Check if a given move is valid within the grid boundaries and doesn't contain an obstacle."""
    return 0 <= row < gROWs and 0 <= col < gCOLs and grid[row][col] != 0

def heuristic(a, b):
    """Calculate the Manhattan distance heuristic between two points."""
    return abs(b[0] - a[0]) + abs(b[1] - a[1])

def get_neighbors(grid, row, col):
    """Get valid neighboring cells of a given cell."""
    neighbors = []
    for move in MOVEMENTS:
        new_row = row + move[0]
        new_col = col + move[1]
        if is_valid_move(grid, new_row, new_col):
            neighbors.append((new_row, new_col))
    return neighbors

def reconstruct_path(came_from, current):
    """Reconstruct the path from the start to the goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        if current == '-1':
            break
        path.append(current)
    path.reverse()
    return path


def astar_search(grid, start, goal):
    
    visited = [] # List for visited nodes.

    path = {}
    path[start] = "-1"

    q = PriorityQueue()

    q.put((0+heuristic(start,goal),0,start))

    while not q.empty():
        item = q.get()
        tot = item[0]
        f = item[1]
        now = item[2]
        #print(tot,f,now)
        if now in visited:
            continue
        visited.append(now)
        if now == goal:
            return reconstruct_path(path, now)
        for neighbor in get_neighbors(grid, now[0], now[1]):
            strightlinecost = heuristic(neighbor, goal)
            q.put((strightlinecost+f+1,f+1,neighbor))
            if neighbor not in visited:
                path[neighbor] = now
    

    return None  # No path found
# #########
# MOVEMENTS = [
#     (-1, 0),  # Up
#     (1, 0),   # Down
#     (0, -1),  # Left
#     (0, 1)    # Right
# ]

# def heuristic(a, b):
#     """Calculate the Manhattan distance heuristic between two points."""
#     return abs(b[0] - a[0]) + abs(b[1] - a[1])

# def is_valid_move(grid, row, col):
#     #print(gROWs,gCOLs)
#     """Check if a given move is valid within the grid boundaries and doesn't contain an obstacle."""
#     return 0 <= row < gROWs and 0 <= col < gCOLs and grid[row][col] != 0

# def get_neighbors(grid, row, col):
#     """Get valid neighboring cells of a given cell."""
#     neighbors = []
#     for move in MOVEMENTS:
#         new_row = row + move[0]
#         new_col = col + move[1]
#         if is_valid_move(grid, new_row, new_col):
#             neighbors.append((new_row, new_col))
#     return neighbors

# def reconstruct_path(came_from, current):
#     """Reconstruct the path from the start to the goal."""
#     path = [current]
#     while current in came_from:
#         current = came_from[current]
#         path.append(current)
#     path.reverse()
#     return path

# def astar_search(grid, start, goal):
#     """Find the shortest path from the start to the goal using A* search."""
#     open_set = []
#     closed_set = set()
#     came_from = {}

#     g_score = {start: 0}
#     f_score = {start: heuristic(start, goal)}

#     heapq.heappush(open_set, (f_score[start], start))

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if current == goal:
#             return reconstruct_path(came_from, current)

#         closed_set.add(current)

#         for neighbor in get_neighbors(grid, current[0], current[1]):
#             neighbor_g_score = g_score[current] + 1

#             if neighbor in closed_set and neighbor_g_score >= g_score.get(neighbor, float('inf')):
#                 continue

#             if neighbor_g_score < g_score.get(neighbor, float('inf')):
#                 came_from[neighbor] = current
#                 g_score[neighbor] = neighbor_g_score
#                 f_score[neighbor] = neighbor_g_score + heuristic(neighbor, goal)
#                 if neighbor not in closed_set:
#                     heapq.heappush(open_set, (f_score[neighbor], neighbor))

#     return None  # No path found
# # Astar


class Pathfinder:
    def __init__(self, in_arr): # in_arr = Binary maze
        self.cost= in_arr
        # self.cost = np.array(in_arr, dtype=np.bool_).tolist() 
        # It converts the input array to a NumPy array of 
        # boolean values and then converts it back to a nested list using the tolist() method.
        # This is done to ensure compatibility with the tcod.path.AStar algorithm.
        # self.pf = tcod.path.AStar(cost=self.cost, diagonal=0)
        # The self.pf attribute holds an instance of the tcod.path.AStar class, initialized
        # with the converted grid and a diagonal flag set to 0, indicating that diagonal movement is not allowed.

    def get_path(self, from_x, from_y, to_x, to_y) -> object:
        # res = self.pf.get_path(from_x, from_y, to_x, to_y)
        start = (from_x,from_y)
        if to_x == 0 and to_y == 0:
            to_x = 1
            to_y = 1
        if from_x == 0 and from_y == 0:
            from_x = 1
            from_y = 1
        goal = (to_x,to_y)
        res = astar_search(self.cost,start,goal)
        # print("start   ",start)
        # print("goal   ",goal)
        # print(self.cost[from_x][from_y])
        # print(self.cost[to_x][to_y])
        # print(res)
        #print('Here', res, self.cost)
        tmp = [(sub[1], sub[0]) for sub in res]

        # print("here : ",tmp)
        # print("here : ",res)
        # print(start)
        # print(goal)
        return tmp
        # return res

class PacmanGameController:
    def __init__(self):
        pygame.init()
        in_width = 32*28
        in_height = 32*19
        self._width = in_width
        self._height = in_height
        self._screen = pygame.display.set_mode((in_width, in_height))
        pygame.display.set_caption('GHOST_HUNTER')
        self.ascii_maze = [
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            " P           XX             ",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X XXXXOXXXXX XX XXXXXOXXXX X",
            "X XXXX XXXXX XX XXXXX XXXX X",
            "X                          X",
            "X XXXX XX XXXXXXXX XX XXXX X",
            "X XXXX XX XXXXXXXX XX XXXX X",
            "X      XX    XX    XX      X",
            "XXXXXX XXXXX XX XXXXX XXXXXX",
            "XXXXXX XXXXX XX XXXXX XXXXXX",
            "XXXXXX XX     G    XX XXXXXX",
            "XXXXXX XX XXX  XXX XX XXXXXX",
            "XXXXXX XX X      X XX XXXXXX",
            "          X      X   G      ",
            "XXXXXX XX X      X XX XXXXXX",
            #"XXXXXX XX XXXXXXXX XX XXXXXX",
            #"XXXXXX XX          XX XXXXXX",
            #"XXXXXX XX XXXXXXXX XX XXXXXX",
            #"XXXXXX XX XXXXXXXX XX XXXXXX",
            "X            XX            X",
            #"X XXXX XXXXX XX XXXXX XXXX X",
            #"X XXXX XXXXX XX XXXXX XXXX X",
            "X   XX       G        XX   X",
            # "XXX XX XX XXXXXXXX XX XX XXX",
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
            # "XXX XX XX XXXXXXXX XX XX XXX",
            # "X      XX    XX    XX      X",
            # "X XXXXXXXXXX XX XXXXXXXXXX X",
            # "X XXXXXXXXXX XX XXXXXXXXXX X",
            # "X   O                 O    X",
            # "XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        ]
        self.ascii_maze = [list(row) for row in self.ascii_maze]
        self.numpy_maze = [] #Binary MAZE (0 or 1)
        self.cookie_spaces = []
        self.powerup_spaces = []
        self.reachable_spaces = []
        self.ghost_spawns = []
        self.ghost_colors = [
            "images/ghost.png",
            "images/ghost_pink.png",
            "images/ghost_orange.png",
            "images/ghost_blue.png"
        ]
        self.size = (0, 0)
        # self.convert_maze_to_numpy()
        # print(gROWs,gCOLs)
        self.p = Pathfinder(self.numpy_maze)
        #print(self.numpy_maze)

        self.EMH = -1
        self._difficulty_options = {
            "easy": {
                "fps": 30,
                "lives": 5
            },
            "medium": {
                "fps": 60,
                "lives": 3
            },
            "hard": {
                "fps": 120,
                "lives": 1
            }
        }

    def start_game(self):
        self.display_difficulty_menu()
        # pygame.quit()

    def display_difficulty_menu(self):
        global LIVES
        menu_done = False
        selected_difficulty = None

        while not menu_done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._done = True
                    menu_done = True
                    LIVES = -1
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        selected_difficulty = "easy"
                        self.EMH = 1
                        LIVES = 5
                        menu_done = True
                    elif event.key == pygame.K_2:
                        self.EMH = 2
                        selected_difficulty = "medium"
                        LIVES = 3
                        menu_done = True
                    elif event.key == pygame.K_3:
                        self.EMH = 3
                        selected_difficulty = "hard"
                        LIVES = 2
                        menu_done = True

            self._screen.fill((0, 0, 0))  # Clear the screen
            self.display_text("SELECT DIFFICULTY:", (self._width / 2 - 200, self._height / 2 - 100), 50)
            self.display_text("Press 1. Easy (5 Lives, 2 Ghosts)", (self._width / 2 - 200, self._height / 2))
            self.display_text("Press 2. Medium (3 Lives, 3 Ghots)", (self._width / 2 - 200, self._height / 2 + 50))
            self.display_text("Press 3. Hard (2 Lives, 4 Ghosts)", (self._width / 2 - 200, self._height / 2 + 100))

            pygame.display.flip()  # Update the screen
        if self.EMH == 1: # EASY
            self.ascii_maze[14][21] = ' '
        elif self.EMH == 3: #HARD
            self.ascii_maze[14][3] = 'G'

        if selected_difficulty:
            self.start_game_with_difficulty(selected_difficulty)

    def display_text(self, text, in_position=(32, 0), in_size=30):
        font = pygame.font.SysFont('Arial', in_size)
        text_surface = font.render(text, False, (255, 255, 255))
        self._screen.blit(text_surface, in_position)

    def start_game_with_difficulty(self, difficulty):
        difficulty_options = self._difficulty_options.get(difficulty)
        if difficulty_options:
            fps = difficulty_options.get("fps")
            lives = difficulty_options.get("lives")
            self._lives = lives
        # self.tick(120)

    def call(self):
        global gROWs,gCOLs
        self.convert_maze_to_numpy()
        gROWs = self.size[1]
        gCOLs = self.size[0]
        self.p = Pathfinder(self.numpy_maze)


    def request_new_path(self, in_ghost: Ghost):
        # random_space = (0,0)
        # if in_ghost._renderer.get_current_mode == GhostBehaviour.CHASE:
        #random_space = random.choice(self.reachable_spaces)
        # else:
        if in_ghost._renderer._kokoro_active == True:
            random_space = random.choice(self.reachable_spaces)
        else:
            random_space = translate_screen_to_maze(in_ghost._renderer.get_hero_position())
        # global current_hero
        # random_space = current_hero
        # print(random_space)
        current_maze_coord = translate_screen_to_maze(in_ghost.get_position())
        #print('debug : ',current_maze_coord[1],current_maze_coord[0],random_space[1],random_space[0])
        path = self.p.get_path(current_maze_coord[1], current_maze_coord[0], random_space[1],
                               random_space[0])
        test_path = [translate_maze_to_screen(item) for item in path]
        in_ghost.set_new_path(test_path)

    def convert_maze_to_numpy(self):
        for x, row in enumerate(self.ascii_maze): # x -> row number# and "row" is the full row
            self.size = (len(row), x + 1) # len of column and row number 1 indexed
            binary_row = []
            for y, column in enumerate(row):
                if column == "G":
                    self.ghost_spawns.append((y, x))
                if column == "X":
                    binary_row.append(0) #Blocked cell
                else:
                    binary_row.append(1) 
                    # 1 means we can travel and also 
                    # there're cookies and this cell is reachable
                    self.cookie_spaces.append((y, x))
                    self.reachable_spaces.append((y, x))
                    if column == "O":
                        self.powerup_spaces.append((y, x)) # Power

            self.numpy_maze.append(binary_row)


if __name__ == "__main__":
    unified_size = 32
    while i_am_running:
        restart_flag = 0
        pacman_game = PacmanGameController()
        pacman_game.start_game()
        if LIVES == -1:
            break
        pacman_game.call()
        size = pacman_game.size
        # print(pacman_game.size)
        # print(size)
        game_renderer = GameRenderer(size[0] * unified_size, size[1] * unified_size)

        for y, row in enumerate(pacman_game.numpy_maze):
            for x, column in enumerate(row):
                if column == 0:
                    game_renderer.add_wall(Wall(game_renderer, x, y, unified_size))

        for cookie_space in pacman_game.cookie_spaces:
            translated = translate_maze_to_screen(cookie_space)
            cookie = Cookie(game_renderer, translated[0] + unified_size / 2, translated[1] + unified_size / 2)
            game_renderer.add_cookie(cookie)

        for powerup_space in pacman_game.powerup_spaces:
            translated = translate_maze_to_screen(powerup_space)
            powerup = Powerup(game_renderer, translated[0] + unified_size / 2, translated[1] + unified_size / 2)
            game_renderer.add_powerup(powerup)

        for i, ghost_spawn in enumerate(pacman_game.ghost_spawns):
            translated = translate_maze_to_screen(ghost_spawn)
            ghost = Ghost(game_renderer, translated[0], translated[1], unified_size, pacman_game,
                        pacman_game.ghost_colors[i % 4])
            game_renderer.add_ghost(ghost)

        GHOST_HUNTER = Hero(game_renderer, unified_size, unified_size, unified_size)
        game_renderer.add_hero(GHOST_HUNTER)
        game_renderer.set_current_mode(GhostBehaviour.CHASE)
        # game_renderer.start_game()
        game_renderer.tick(120) #Game Loop

    