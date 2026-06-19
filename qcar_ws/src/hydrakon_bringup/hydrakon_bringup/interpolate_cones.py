#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

DEFAULT_MAP_YAML    = os.path.join(script_dir, '..', '..', 'hydrakon_description', 'docs', 'new_map1.yaml')
DEFAULT_PATH_CSV    = os.path.join(script_dir, '..', '..', 'hydrakon_description', 'docs', 'new_map1.csv')
DEFAULT_OUTPUT_YAML = os.path.join(script_dir, '..', '..', 'hydrakon_description', 'docs', 'new_map1_augmented.yaml')

OCCUPIED_THRESHOLD = 50    
MIN_CLUSTER_PX     = 2     
MAX_CLUSTER_PX     = 200  
MAX_CONE_GAP_M     = 8.0   
PATH_DOWNSAMPLE    = 20   


def load_map(yaml_path):
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    image_field = meta['image']
    if os.path.isabs(image_field):
        pgm_path = image_field
    else:
        pgm_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image_field)

    grid = np.array(Image.open(pgm_path))
    resolution = float(meta['resolution'])
    origin_x   = float(meta['origin'][0])
    origin_y   = float(meta['origin'][1])

    print(f"Loaded map: {pgm_path}")
    print(f"  Size:       {grid.shape[1]} x {grid.shape[0]} px")
    print(f"  Resolution: {resolution} m/px")
    print(f"  Origin:     ({origin_x}, {origin_y})")
    return grid, resolution, origin_x, origin_y, meta


def save_map(grid, meta, output_yaml_path):
    out_dir     = os.path.dirname(os.path.abspath(output_yaml_path))
    pgm_name    = os.path.splitext(os.path.basename(output_yaml_path))[0] + '.pgm'
    pgm_path    = os.path.join(out_dir, pgm_name)

    Image.fromarray(grid.astype(np.uint8)).save(pgm_path)

    out_meta = dict(meta)
    out_meta['image'] = pgm_name
    with open(output_yaml_path, 'w') as f:
        yaml.dump(out_meta, f, default_flow_style=False)

    print(f"\nSaved: {pgm_path}")
    print(f"Saved: {output_yaml_path}")


def world_to_pixel(wx, wy, origin_x, origin_y, resolution, nrows):
    col = int(round((wx - origin_x) / resolution))
    row = int(round(nrows - 1 - (wy - origin_y) / resolution))
    return row, col


def pixel_to_world(row, col, origin_x, origin_y, resolution, nrows):
    wx = origin_x + col * resolution
    wy = origin_y + (nrows - 1 - row) * resolution
    return wx, wy


def find_cone_centroids(grid, origin_x, origin_y, resolution):
    occupied      = (grid < OCCUPIED_THRESHOLD).astype(np.uint8)
    labeled, n    = ndimage.label(occupied)
    nrows         = grid.shape[0]
    centroids     = []

    for label_id in range(1, n + 1):
        pixels = np.argwhere(labeled == label_id)
        size   = len(pixels)
        if size < MIN_CLUSTER_PX or size > MAX_CLUSTER_PX:
            continue
        row_c = np.mean(pixels[:, 0])
        col_c = np.mean(pixels[:, 1])
        wx, wy = pixel_to_world(row_c, col_c, origin_x, origin_y, resolution, nrows)
        centroids.append((wx, wy))

    print(f"\nCone detection:")
    print(f"  Total clusters found: {n}")
    print(f"  Valid cone candidates ({MIN_CLUSTER_PX}–{MAX_CLUSTER_PX} px): {len(centroids)}")
    return centroids

def load_path(csv_path):
    points = []
    with open(csv_path, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    points.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

    unique = [points[0]]
    for p in points[1:]:
        if math.hypot(p[0] - unique[-1][0], p[1] - unique[-1][1]) > 0.05:
            unique.append(p)

    downsampled = unique[::PATH_DOWNSAMPLE]
    print(f"\nPath:")
    print(f"  Raw points:     {len(points)}")
    print(f"  After dedup:    {len(unique)}")
    print(f"  After downsample (1/{PATH_DOWNSAMPLE}): {len(downsampled)}")
    return downsampled

def classify_cones(centroids, path_points):
    path   = np.array(path_points)
    left   = []
    right  = []
    skipped = 0

    for cx, cy in centroids:
        dists   = np.hypot(path[:, 0] - cx, path[:, 1] - cy)
        idx     = int(np.argmin(dists))

        if dists[idx] > 15.0:
            skipped += 1
            continue

        next_idx = min(idx + 1, len(path) - 1)
        prev_idx = max(idx - 1, 0)

        dx = path[next_idx][0] - path[prev_idx][0]
        dy = path[next_idx][1] - path[prev_idx][1]

        vx = cx - path[idx][0]
        vy = cy - path[idx][1]

        cross_z = dx * vy - dy * vx

        if cross_z > 0:
            left.append((cx, cy))
        else:
            right.append((cx, cy))

    print(f"\nClassification:")
    print(f"  Left boundary:  {len(left)} cones")
    print(f"  Right boundary: {len(right)} cones")
    if skipped:
        print(f"  Skipped (too far from path): {skipped}")
    return left, right


def sort_cones(cones, path_points):
    if not cones:
        return []

    # start from cone nearest to the beginning of path
    start      = np.array(path_points[0])
    start_dists = [math.hypot(c[0] - start[0], c[1] - start[1]) for c in cones]
    first_idx  = int(np.argmin(start_dists))

    remaining   = list(cones)
    ordered     = [remaining.pop(first_idx)]

    while remaining:
        last    = ordered[-1]
        dists   = [math.hypot(c[0] - last[0], c[1] - last[1]) for c in remaining]
        nearest = int(np.argmin(dists))

        if dists[nearest] > MAX_CONE_GAP_M:
            break

        ordered.append(remaining.pop(nearest))

    return ordered


def bresenham(grid, r0, c0, r1, c1):
    nrows, ncols = grid.shape
    dr  = abs(r1 - r0)
    dc  = abs(c1 - c0)
    sr  = 1 if r0 < r1 else -1
    sc  = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0

    while True:
        # Mark a 3x3 area to make walls thicker 
        for dr_off in [-1, 0, 1]:
            for dc_off in [-1, 0, 1]:
                rr, cc = r + dr_off, c + dc_off
                if 0 <= rr < nrows and 0 <= cc < ncols:
                    grid[rr, cc] = 0   # mark as occupied (black)
        
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r   += sr
        if e2 < dr:
            err += dr
            c   += sc


def draw_walls(grid, cones, origin_x, origin_y, resolution):
    nrows        = grid.shape[0]
    walls_drawn  = 0

    for i in range(len(cones) - 1):
        r0, c0 = world_to_pixel(cones[i][0],   cones[i][1],   origin_x, origin_y, resolution, nrows)
        r1, c1 = world_to_pixel(cones[i+1][0], cones[i+1][1], origin_x, origin_y, resolution, nrows)
        bresenham(grid, r0, c0, r1, c1)
        walls_drawn += 1

    return walls_drawn

def check_path_in_map(path_points, origin_x, origin_y, resolution, grid_shape):
    nrows, ncols = grid_shape
    map_x_min = origin_x
    map_x_max = origin_x + ncols * resolution
    map_y_min = origin_y
    map_y_max = origin_y + nrows * resolution

    in_bounds = sum(
        1 for x, y in path_points
        if map_x_min <= x <= map_x_max and map_y_min <= y <= map_y_max
    )
    ratio = in_bounds / len(path_points)
    print(f"\nSanity check: {in_bounds}/{len(path_points)} ({ratio*100:.0f}%) path points inside map bounds")
    if ratio < 0.5:
        print("  WARNING: Most path points are outside the map — possible frame mismatch.")
        print(f"  Map X: [{map_x_min:.1f}, {map_x_max:.1f}]")
        print(f"  Map Y: [{map_y_min:.1f}, {map_y_max:.1f}]")
        print(f"  Path X range: [{min(x for x,y in path_points):.1f}, {max(x for x,y in path_points):.1f}]")
        print(f"  Path Y range: [{min(y for x,y in path_points):.1f}, {max(y for x,y in path_points):.1f}]")

def main():
    map_yaml     = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MAP_YAML
    path_csv     = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PATH_CSV
    output_yaml  = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_YAML

    print("=" * 60)
    print("Cone boundary interpolation")
    print("=" * 60)
    print(f"Map:    {map_yaml}")
    print(f"Path:   {path_csv}")
    print(f"Output: {output_yaml}")

    grid, resolution, origin_x, origin_y, meta = load_map(map_yaml)
    path_points = load_path(path_csv)

    check_path_in_map(path_points, origin_x, origin_y, resolution, grid.shape)

    centroids = find_cone_centroids(grid, origin_x, origin_y, resolution)

    left_cones,  right_cones  = classify_cones(centroids, path_points)

    left_ordered  = sort_cones(left_cones,  path_points)
    right_ordered = sort_cones(right_cones, path_points)

    print(f"\nOrdering:")
    print(f"  Left:  {len(left_ordered)} cones connected")
    print(f"  Right: {len(right_ordered)} cones connected")

    augmented = grid.copy()
    occupied_before = int(np.sum(augmented < OCCUPIED_THRESHOLD))

    left_segs  = draw_walls(augmented, left_ordered,  origin_x, origin_y, resolution)
    right_segs = draw_walls(augmented, right_ordered, origin_x, origin_y, resolution)

    occupied_after = int(np.sum(augmented < OCCUPIED_THRESHOLD))
    print(f"\nWall drawing:")
    print(f"  Left segments:  {left_segs}")
    print(f"  Right segments: {right_segs}")
    print(f"  New wall cells: {occupied_after - occupied_before}")

    save_map(augmented, meta, output_yaml)
    print("\nDone. Point navigation.launch.py at the augmented yaml.")
    print("=" * 60)


if __name__ == '__main__':
    main()