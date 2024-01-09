import argparse
import numpy as np
import imageio
import time
from matplotlib import pyplot as plt
import pyastar_contour
import cv2
from os.path import basename, join


def parse_args():
    parser = argparse.ArgumentParser(
        "An example of using pyastar to find the solution to a maze"
    )
    parser.add_argument(
        "--input", type=str, default="pyastar-contour/mazes/map.png",
        help="Path to the black-and-white image to be used as input.",
    )
    parser.add_argument(
        "--output", type=str, help="Path to where the output will be written",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = join("pyastar-contour/solns", basename(args.input))

    return args


def main():
    args = parse_args()
    # maze0 = cv2.imread(args.input)
    # maze = cv2.cvtColor(maze0, cv2.COLOR_BGR2GRAY)
    # print(maze.shape)
    # # cv2.imshow('maze', maze)
    # # cv2.waitKey(0)
    maze = imageio.imread(args.input)

    if maze is None:
        print(f"No file found: {args.input}")
        return
    else:
        print(f"Loaded maze of shape {maze.shape} from {args.input}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    maze = cv2.dilate(maze.copy(), kernel, 10)
    grid = maze.astype(np.float32)
    elements, count = np.unique(grid, return_counts=True)
    print(elements)
    print(count)
    print(grid.shape)
    grid[grid == 200.0] = np.inf
    grid[grid == 1.0] = np.inf
    grid[grid == 101.0] = 1
    assert grid.min() == 1, "cost of moving must be at least 1"
    start = np.array([255, 255])
    end = np.array([257, 257])

    t0 = time.time()
    # set allow_diagonal=True to enable 8-connectivity
    path = pyastar_contour.astar_path(grid, start, end, allow_diagonal=True)
    dur = time.time() - t0
    print(path.shape)
    print(path)
    
    if path.shape[0] > 0:
        print(f"Found contour of length {path.shape[0]} in {dur:.6f}s")

        point_x = []
        point_y = []
        point_z = []
        point_d = []
        pointcloud = path
        print(pointcloud.shape)
        for point_i in range(pointcloud.shape[0]):
            p_x = pointcloud[point_i][0]
            p_y = pointcloud[point_i][1]
            wall_color = 200
            if maze[p_x, p_y] == wall_color or maze[p_x + 2, p_y] == wall_color or \
                    maze[p_x - 2, p_y] == wall_color or maze[p_x, p_y + 2] == wall_color or \
                    maze[p_x, p_y-2] == wall_color or \
                    maze[p_x + 1, p_y-1] == wall_color or maze[p_x - 1, p_y-1] == wall_color or \
                    maze[p_x + 1, p_y+1] == wall_color or maze[p_x - 1, p_y+1] == wall_color:
                point_d.append('r')
            else:
                point_d.append('b')
            # print(pointclouds[point_i])
            point_x.append(pointcloud[point_i][0])
            point_y.append(pointcloud[point_i][1])
            point_z.append(pointcloud[point_i][2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # 将二维转化为三维
        ax.scatter(point_x, point_y,point_z, c=point_d)
        # plt.show()

        maze = np.stack([maze, maze, maze], axis=2)
        maze[path[:, 0], path[:, 1], :] = np.array([255, 0, 0])
        imageio.imwrite(args.output, maze)
        print(f"Plotting path to {args.output}")

        plt.pause(5)

    else:
        print("No path found")

    print("Done")


if __name__ == "__main__":
    main()
