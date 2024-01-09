import argparse
import numpy as np
import imageio
import time
import cv2

import pyastar

from os.path import basename, join


def parse_args():
    parser = argparse.ArgumentParser(
        "An example of using pyastar to find the solution to a maze"
    )
    parser.add_argument(
        "--input", type=str, default="pyastar/mazes/img_0.png",
        help="Path to the black-and-white image to be used as input.",
    )
    parser.add_argument(
        "--output", type=str, help="Path to where the output will be written",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = join("solns", basename(args.input))
        # print(basename(args.input))

    return args


def main():
    args = parse_args()
    maze = imageio.imread(args.input)
    # maze = cv2.imread(args.input)
    print(maze.shape)
    # print(maze[150, 200])
    # maze = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY)
    elements, count = np.unique(maze, return_counts=True)
    print(elements)
    print(count)
    print(maze.shape)
    if maze is None:
        print(f"No file found: {args.input}")
        return
    else:
        print(f"Loaded maze of shape {maze.shape} from {args.input}")

    grid = maze.astype(np.float32)
    # grid[grid == 0] = np.inf
    # grid[grid == 255] = 1
    grid[grid != 0] = np.inf
    grid[grid == 0] = 1


    assert grid.min() == 1, "cost of moving must be at least 1"

    # start is the first white block in the top row
    # start_j, = np.where(grid[0, :] == 1)
    # start = np.array([0, start_j[0]])

    # end is the first white block in the final column
    # end_i, = np.where(grid[:, -1] == 1)
    # end = np.array([end_i[0], grid.shape[0] - 1])
    feasible = np.where(grid == 1)
    print(feasible)
    # print(feasible.shape)
    rand_start = np.random.choice(len(feasible[0]))
    rand_end = np.random.choice(len(feasible[0]))


    start = np.array([feasible[0][rand_start], feasible[1][rand_start]])
    end = np.array([feasible[0][rand_end], feasible[1][rand_end]])

    print(grid[start[0], start[1]])
    print(grid[end[0], end[1]])
    t0 = time.time()
    # set allow_diagonal=True to enable 8-connectivity
    # cv2.imshow('grid', grid)
    path = pyastar.astar_path(grid, start, end, allow_diagonal=True)
    dur = time.time() - t0

    if path.shape[0] > 0:
        print(f"Found path of length {path.shape[0]} in {dur:.6f}s from {start} to {end}")
        print(path)
        maze = np.stack((maze, maze, maze), axis=2)
        maze[path[:, 0], path[:, 1]] = (255, 0, 0)
        cv2.circle(maze, (start[1], start[0]), 2, [0, 0, 255], thickness=-1)
        cv2.circle(maze, (end[1], end[0]), 2, [0, 255, 0], thickness=-1)
        # cv2.imshow('path', maze)
        # cv2.waitKey(0)

        print(f"Plotting path to {args.output}")
        # cv2.imwrite(join('pyastar', args.output), maze)
        imageio.imwrite(join('pyastar', args.output), maze)
    else:
        print("No path found")

    print("Done")


if __name__ == "__main__":
    main()
