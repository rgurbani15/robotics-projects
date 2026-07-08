"""
Project 3: Path Planning Visualizer

Build your own maze by clicking walls, then watch three classic
pathfinding algorithms solve it step by step:

  1 = Dijkstra   (explores evenly outward - guaranteed shortest path)
  2 = A*         (same idea, but smarter - uses distance-to-goal to
                   steer its search toward the target)
  3 = RRT        (Rapidly-exploring Random Tree - grows randomly until
                   it stumbles into the goal; used in real robotics for
                   messy, high-dimensional spaces like robot arms)

Controls:
  Left click / drag : add walls
  Right click / drag: erase walls
  1 / 2 / 3          : run Dijkstra / A* / RRT
  c                  : clear the grid
  SPACE              : re-run the last algorithm
"""

import heapq
import math
import random

import pygame

CELL_SIZE = 20
GRID_W, GRID_H = 40, 30
WINDOW_W, WINDOW_H = GRID_W * CELL_SIZE, GRID_H * CELL_SIZE + 40  # +40 for label bar

COLOR_BG = (20, 20, 30)
COLOR_WALL = (60, 60, 70)
COLOR_GRID = (35, 35, 45)
COLOR_START = (80, 220, 120)
COLOR_END = (220, 90, 90)
COLOR_VISITED = (50, 90, 150)
COLOR_FRONTIER = (230, 200, 60)
COLOR_PATH = (240, 240, 240)
COLOR_TEXT = (220, 220, 220)

START = (2, 2)
END = (GRID_W - 3, GRID_H - 3)

STEPS_PER_FRAME = 8  # higher = faster (less satisfying) animation


def in_bounds(cell):
    x, y = cell
    return 0 <= x < GRID_W and 0 <= y < GRID_H


def neighbors(cell):
    x, y = cell
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        n = (x + dx, y + dy)
        if in_bounds(n):
            yield n


def dijkstra(walls):
    """Yields (visited, frontier, path_or_None) each step."""
    dist = {START: 0}
    prev = {}
    pq = [(0, START)]
    visited = set()

    while pq:
        d, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == END:
            yield visited, set(), reconstruct_path(prev, START, END)
            return

        for n in neighbors(current):
            if n in walls or n in visited:
                continue
            nd = d + 1
            if nd < dist.get(n, float("inf")):
                dist[n] = nd
                prev[n] = current
                heapq.heappush(pq, (nd, n))

        frontier = {c for _, c in pq}
        yield visited, frontier, None

    yield visited, set(), None  # no path found


def a_star(walls):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    g_score = {START: 0}
    prev = {}
    pq = [(heuristic(START, END), START)]
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)

        if current == END:
            yield visited, set(), reconstruct_path(prev, START, END)
            return

        for n in neighbors(current):
            if n in walls or n in visited:
                continue
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(n, float("inf")):
                g_score[n] = tentative_g
                prev[n] = current
                heapq.heappush(pq, (tentative_g + heuristic(n, END), n))

        frontier = {c for _, c in pq}
        yield visited, frontier, None

    yield visited, set(), None


def reconstruct_path(prev, start, end):
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def cell_center(cell):
    x, y = cell
    return (x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2 + 40)


def segment_hits_wall(p1, p2, walls, steps=15):
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        cell = (int(x // CELL_SIZE), int(y // CELL_SIZE))
        if cell in walls or not in_bounds(cell):
            return True
    return False


def rrt(walls, max_iters=3000, step_size=25):
    start_pt = cell_center(START)
    end_pt = cell_center(END)
    nodes = [start_pt]
    parent = {0: None}

    for i in range(max_iters):
        if random.random() < 0.1:
            sample = end_pt
        else:
            sample = (
                random.uniform(0, WINDOW_W),
                random.uniform(40, WINDOW_H),
            )

        nearest_idx = min(range(len(nodes)), key=lambda idx: math.dist(nodes[idx], sample))
        nearest = nodes[nearest_idx]

        dist = math.dist(nearest, sample)
        if dist == 0:
            continue
        ratio = min(step_size / dist, 1.0)
        new_pt = (
            nearest[0] + (sample[0] - nearest[0]) * ratio,
            nearest[1] + (sample[1] - nearest[1]) * ratio,
        )

        if segment_hits_wall(nearest, new_pt, walls):
            continue

        nodes.append(new_pt)
        parent[len(nodes) - 1] = nearest_idx

        if math.dist(new_pt, end_pt) < step_size:
            path_indices = [len(nodes) - 1]
            while parent[path_indices[-1]] is not None:
                # find index of parent point
                p = parent[path_indices[-1]]
                p_idx = nodes.index(p)
                path_indices.append(p_idx)
            path_points = [nodes[idx] for idx in reversed(path_indices)]
            path_points.append(end_pt)
            yield nodes, parent, path_points
            return

        if i % 20 == 0:
            yield nodes, parent, None

    yield nodes, parent, None


def draw_grid(screen, walls, visited, frontier, path, label):
    screen.fill(COLOR_BG)

    font = pygame.font.SysFont("consolas", 20)
    label_surf = font.render(label, True, COLOR_TEXT)
    screen.blit(label_surf, (10, 8))

    for x in range(GRID_W):
        for y in range(GRID_H):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 40, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLOR_GRID, rect, 1)
            cell = (x, y)
            if cell in walls:
                pygame.draw.rect(screen, COLOR_WALL, rect)
            elif cell in visited:
                pygame.draw.rect(screen, COLOR_VISITED, rect)
            elif cell in frontier:
                pygame.draw.rect(screen, COLOR_FRONTIER, rect)

    if path:
        for cell in path:
            rect = pygame.Rect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE + 40, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, COLOR_PATH, rect)

    for cell, color in [(START, COLOR_START), (END, COLOR_END)]:
        rect = pygame.Rect(cell[0] * CELL_SIZE, cell[1] * CELL_SIZE + 40, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, color, rect)

    pygame.display.flip()


def draw_rrt(screen, walls, nodes, parent, path, label):
    screen.fill(COLOR_BG)

    font = pygame.font.SysFont("consolas", 20)
    label_surf = font.render(label, True, COLOR_TEXT)
    screen.blit(label_surf, (10, 8))

    for x in range(GRID_W):
        for y in range(GRID_H):
            cell = (x, y)
            if cell in walls:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 40, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, COLOR_WALL, rect)

    for idx, p in parent.items():
        if p is not None:
            pygame.draw.line(screen, COLOR_VISITED, p, nodes[idx], 2)

    if path:
        for i in range(len(path) - 1):
            pygame.draw.line(screen, COLOR_PATH, path[i], path[i + 1], 3)

    pygame.draw.circle(screen, COLOR_START, cell_center(START), 6)
    pygame.draw.circle(screen, COLOR_END, cell_center(END), 6)

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Path Planning Visualizer - Project 3")
    clock = pygame.time.Clock()

    walls = set()
    active_gen = None
    last_algo = None
    label = "Draw walls, then press 1 (Dijkstra) / 2 (A*) / 3 (RRT)"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    active_gen = dijkstra(walls)
                    last_algo = "1"
                    label = "Running Dijkstra..."
                elif event.key == pygame.K_2:
                    active_gen = a_star(walls)
                    last_algo = "2"
                    label = "Running A*..."
                elif event.key == pygame.K_3:
                    active_gen = rrt(walls)
                    last_algo = "3"
                    label = "Running RRT..."
                elif event.key == pygame.K_c:
                    walls.clear()
                    active_gen = None
                    label = "Cleared. Draw walls, then press 1 / 2 / 3"
                elif event.key == pygame.K_SPACE and last_algo:
                    active_gen = {"1": dijkstra, "2": a_star, "3": rrt}[last_algo](walls)
                    label = "Re-running..."

        # Wall drawing (only when no algorithm is actively animating)
        if active_gen is None:
            buttons = pygame.mouse.get_pressed()
            if buttons[0] or buttons[2]:
                mx, my = pygame.mouse.get_pos()
                if my > 40:
                    cell = (mx // CELL_SIZE, my // CELL_SIZE)
                    if cell != START and cell != END:
                        if buttons[0]:
                            walls.add(cell)
                        elif buttons[2]:
                            walls.discard(cell)

        if active_gen is not None:
            found_path = None
            try:
                for _ in range(STEPS_PER_FRAME):
                    if last_algo == "3":
                        nodes, parent, path = next(active_gen)
                    else:
                        visited, frontier, path = next(active_gen)
                    if path is not None:
                        found_path = path
                        break
            except StopIteration:
                active_gen = None
                if found_path is None:
                    label = "No path found."
                else:
                    label = f"Path found! Length: {len(found_path)}"
            else:
                if found_path is not None:
                    label = f"Path found! Length: {len(found_path)}"
                    active_gen = None

        if last_algo == "3" and active_gen is not None:
            draw_rrt(screen, walls, nodes, parent, None, label)
        elif last_algo == "3":
            draw_rrt(screen, walls, nodes, parent, path, label)
        elif active_gen is not None:
            draw_grid(screen, walls, visited, frontier, None, label)
        elif last_algo:
            draw_grid(screen, walls, visited, set(), path, label)
        else:
            draw_grid(screen, walls, set(), set(), None, label)

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
