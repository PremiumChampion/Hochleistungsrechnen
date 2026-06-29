import random

WALL = "#"
SPACE = " "
START = "+"
FINISH = "-"

def generate_maze(width, height):
    # final output size: (2*height + 1) x (2*width + 1)
    gw = width * 2 + 1
    gh = height * 2 + 1
    grid = [[WALL for _ in range(gw)] for _ in range(gh)]
    visited = [[False] * width for _ in range(height)]

    def carve(x, y):
        visited[y][x] = True
        gx, gy = 2 * x + 1, 2 * y + 1
        # 2-wide corridor
        grid[gy][gx] = SPACE
        if gx + 1 < gw - 1:
            grid[gy][gx + 1] = SPACE
        if gy + 1 < gh - 1:
            grid[gy + 1][gx] = SPACE
        if gx + 1 < gw - 1 and gy + 1 < gh - 1:
            grid[gy + 1][gx + 1] = SPACE

    def carve_between(x1, y1, x2, y2):
        # adjacent cells only
        gx1, gy1 = 2 * x1 + 1, 2 * y1 + 1
        gx2, gy2 = 2 * x2 + 1, 2 * y2 + 1

        if x1 == x2:
            top = min(gy1, gy2)
            for dx in (0, 1):
                grid[top + 1][gx1 + dx] = SPACE
                grid[top + 2][gx1 + dx] = SPACE
        else:
            left = min(gx1, gx2)
            for dy in (0, 1):
                grid[gy1 + dy][left + 1] = SPACE
                grid[gy1 + dy][left + 2] = SPACE

    stack = [(0, 0)]
    carve(0, 0)

    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                neighbors.append((nx, ny))

        if not neighbors:
            stack.pop()
            continue

        nx, ny = random.choice(neighbors)
        carve_between(x, y, nx, ny)
        carve(nx, ny)
        stack.append((nx, ny))

    # Put start/finish on open corridor tiles, not on walls
    grid[1][1] = START
    grid[gh - 2][gw - 2] = FINISH

    return grid

def print_maze(grid):
    print("\n".join("".join(row) for row in grid))

if __name__ == "__main__":
    maze = generate_maze(20, 10)
    print_maze(maze)
