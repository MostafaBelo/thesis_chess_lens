import torch
import numpy as np
from matplotlib import pyplot as plt

import cv2

# autopep8: off
import sys
sys.path.append('../')
from Utils import BoardImageUtils
# autopep8: on


def get_bounds(size: tuple[int, int]) -> torch.Tensor:
    return torch.tensor([
        [0, 0],
        [size[0], 0],
        [size[0], size[1]],
        [0, size[1]],
    ], dtype=torch.float32)


virtual_bounds = get_bounds((256, 256))
YOLO_Image_bounds = get_bounds((640, 640))

# Function to compute intersection of a line with image borders


def get_line_points(rho, theta, width, height):
    """Finds intersection points of a line with image boundaries."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Points where line crosses image borders
    x1, y1, x2, y2 = None, None, None, None

    # Check intersection with vertical borders (x=0 and x=width)
    if sin_t != 0:  # Avoid division by zero
        y_at_x0 = rho / sin_t  # Intersection with x = 0
        y_at_xw = (rho - width * cos_t) / sin_t  # Intersection with x = width

        if 0 <= y_at_x0 <= height:
            x1, y1 = 0, y_at_x0
        if 0 <= y_at_xw <= height:
            x2, y2 = width, y_at_xw

    # Check intersection with horizontal borders (y=0 and y=height)
    if cos_t != 0:
        x_at_y0 = rho / cos_t  # Intersection with y = 0
        # Intersection with y = height
        x_at_yh = (rho - height * sin_t) / cos_t

        if 0 <= x_at_y0 <= width:
            if x1 is None:  # If not set yet
                x1, y1 = x_at_y0, 0
            else:
                x2, y2 = x_at_y0, 0
        if 0 <= x_at_yh <= width:
            if x2 is None:
                x2, y2 = x_at_yh, height
            else:
                x1, y1 = x_at_yh, height

    return (x1, y1, x2, y2)


def intersection_polar(line1, line2) -> torch.Tensor:
    rho1, theta1 = line1[0], line1[1]
    rho2, theta2 = line2[0], line2[1]

    # Construct the coefficient matrix
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])

    # Construct the right-hand side vector
    B = np.array([rho1, rho2])

    # Solve for (x, y)
    if np.linalg.det(A) == 0:  # Check if lines are parallel
        return None

    x, y = np.linalg.solve(A, B)
    return [x, y]


def cross_bin_intersections(bins, height, width):
    res = []

    for b1 in range(len(bins)):
        for b2 in range(b1+1, len(bins)):
            for i in range(len(bins[b1])):
                for j in range(len(bins[b2])):
                    intersection_point = intersection_polar(
                        bins[b1][i], bins[b2][j])
                    if (intersection_point is None):
                        continue

                    if (intersection_point[0] < 0 or intersection_point[0] > width):
                        continue
                    if (intersection_point[1] < 0 or intersection_point[1] > height):
                        continue

                    res.append(intersection_point)

    if (len(res) == 0):
        intersections = None
    elif (len(res) == 1):
        intersections = torch.tensor(res).unsqueeze(dim=0)
    else:
        intersections = torch.tensor(res)

    return intersections


def line_cross_center(H, W, theta):
    x_c, y_c = W / 2, H / 2
    r = x_c * np.cos(theta) + y_c * np.sin(theta)
    return torch.tensor([r, theta])


def cluster(items, deltas):
    grid = {}

    for line in items:
        i1 = line[0]
        i2 = line[1]
        # Compute grid index
        i1_bin = int(torch.Tensor.round(i1 / deltas[0]))
        i2_bin = int(torch.Tensor.round(i2 / deltas[1]))

        if (i1, i2) not in grid:
            grid[(i1_bin, i2_bin)] = (0, torch.zeros(2))
        current = grid[(i1_bin, i2_bin)]
        grid[(i1_bin, i2_bin)] = (current[0] + 1, current[1] + line)

    # Compute mean for each bin
    filtered_items = []
    for idx, (itemCount, vals) in grid.items():
        filtered_items.append((vals[0]/itemCount, vals[1]/itemCount))

    return torch.tensor(filtered_items)


square_size = 32


def getStructBoard() -> torch.tensor:
    board_size = square_size*8
    res = torch.zeros(board_size, board_size)

    for val in range(-2, 3):
        res[np.arange(val, board_size, square_size), :] = 1
        res[:, np.arange(val, board_size, square_size)] = 1

    return res


def calcM(src_pts, dst_pts):
    M = cv2.getPerspectiveTransform(src_pts.numpy(), dst_pts.numpy())
    return torch.tensor(M, dtype=torch.float32)


def Mwarp(M, image, newSize):
    warppedImg = cv2.warpPerspective(image.numpy(), M.numpy(), newSize)
    return warppedImg


def warpPts(src_pts, M):
    tmp_pts = torch.cat([src_pts, torch.ones(src_pts.shape[0], 1)], dim=1)
    dst_pts = (M @ tmp_pts.T).T
    dst_pts_norm = dst_pts / dst_pts[:, 2].unsqueeze(1)
    return dst_pts_norm[:, :2]


def warp(src_pts, dst_pts, image, newSize=(square_size*8, square_size*8)):
    M = calcM(src_pts, dst_pts)
    warppedImg = Mwarp(M, image, newSize)
    return M, warppedImg


def reverseWarp(src_pts, dst_pts, newSize):
    image = getStructBoard()
    return warp(dst_pts, src_pts, image, newSize)


def calc_error(edges, warpped):
    # magnification_factor = 1
    # magnified_edges = magnification_factor*edges
    # magnified_warpped = magnification_factor * warpped
    # return torch.mean((magnified_edges-magnified_warpped)**2)

    return torch.mean((edges * warpped) ** 2)

    # intersection = np.logical_and(edges, warpped).sum()
    # return -(2.0 * intersection) / (edges.sum() + warpped.sum())

    # return ssim(edges.numpy(), warpped, data_range=1)


class BoundedCornerExtractor:
    def __init__(self, EdgeThresholds=(100, 700), LinesThresholds=(1.5, 300), ClusterDeltas=(60, np.pi/180 * 10), PointClusterDeltas=(5, 5), SquareSizeFiltering=100):
        self.EdgeThresholds = EdgeThresholds
        self.LinesThresholds = LinesThresholds
        self.ClusterDeltas = ClusterDeltas
        self.PointClusterDeltas = PointClusterDeltas
        self.SquareSizeFiltering = SquareSizeFiltering

        self.boundary_mask = None

        self.img = None
        self.edges = None
        self.lines = None
        self.clustered_lines = None
        self.line_bins = None
        self.intersections = None
        self.clustered_interesections = None
        self.intercepts = None
        self.square_filtered_bins = None
        self.centrals = None
        self.oriented_centrals = None

    def checkMask(self):
        if (self.boundary_mask is None):
            raise ValueError("Boundary Mask Missing")

    def checkImage(self):
        if (self.img is None):
            raise ValueError("Image Missing")

    def checkEdges(self):
        if (self.edges is None):
            self.detectEdges()

    def checkLines(self):
        if (self.lines is None):
            self.detectLines()

    def checkClustered(self):
        if (self.clustered_lines is None):
            self.compute_clusteredLines()

    def checkLineBins(self):
        if (self.line_bins is None):
            self.compute_line_bins()

    def checkIntersections(self):
        if (self.intersections is None):
            self.compute_intersections()

    def checkClusteredIntersections(self):
        if (self.clustered_interesections is None):
            self.compute_clustered_intersections()

    def checkIntercepts(self):
        if (self.intercepts is None):
            self.compute_intercepts()

    def checkSquareFiltered(self):
        if (self.square_filtered_bins is None):
            self.compute_filtered_bins()

    def checkCentrals(self):
        if (self.centrals is None):
            self.compute_cross_centrals()

    def checkOrientedCentrals(self):
        if (self.oriented_centrals is None):
            self.compute_orient()

    def clear(self):
        self.boundary_mask = None

        self.img = None
        self.edges = None
        self.lines = None
        self.clustered_lines = None
        self.line_bins = None
        self.intersections = None
        self.clustered_interesections = None
        self.intercepts = None
        self.square_filtered_bins = None
        self.centrals = None
        self.oriented_centrals = None

    def setImg(self, x, mask):
        self.clear()
        self.img = x
        self.boundary_mask = mask

    def detectEdges(self):
        self.checkImage()

        self.edges = cv2.Canny(
            self.img, threshold1=self.EdgeThresholds[0], threshold2=self.EdgeThresholds[1])

        return self.edges

    def detectLines(self):
        self.checkEdges()

        self.lines = cv2.HoughLines(
            self.edges, rho=self.LinesThresholds[0], theta=np.pi/180, threshold=self.LinesThresholds[1])

        self.lines = torch.tensor(self.lines).squeeze()
        # self.lines[self.lines[:, 0] < 0, 1] -= np.pi
        # self.lines[self.lines[:, 0] < 0, 0] *= -1

        # self.lines[self.lines[:, 1] < 0, 1] += np.pi

        return self.lines

    def getImageSize(self):
        self.checkImage()

        height, width = self.img.shape[:2]
        return height, width

    def apply(self):
        self.checkImage()
        self.detectEdges()
        self.detectLines()
        self.compute_clusteredLines()
        self.compute_line_bins()
        # self.compute_intersections()
        # self.compute_clustered_intersections()

        self.compute_intercepts()
        self.compute_filtered_bins()
        self.compute_cross_centrals()
        self.compute_orient()

    def interactivePlot(self, lines=None):
        self.checkImage()
        image = self.img

        if (lines is None):
            self.checkLines()
            lines = self.lines

        height, width = self.getImageSize()

        # Prepare figure
        fig, (ax_hough, ax_image) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Hough space (rho vs theta)
        rhos = []
        thetas = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0], line[1]
                rhos.append(rho)
                thetas.append(theta)

        # Scatter plot with picking enabled
        sc = ax_hough.scatter(thetas, rhos, c='blue', picker=True)
        ax_hough.set_xlabel("Theta (radians)")
        ax_hough.set_ylabel("Rho (pixels)")
        ax_hough.set_title("Hough Space: Rho vs Theta")

        # Function to compute intersection of a line with image borders

        def get_line_points(rho, theta, width, height):
            """Finds intersection points of a line with image boundaries."""
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            x1, y1, x2, y2 = None, None, None, None

            if sin_t != 0:
                y_at_x0 = rho / sin_t
                y_at_xw = (rho - width * cos_t) / sin_t

                if 0 <= y_at_x0 <= height:
                    x1, y1 = 0, y_at_x0
                if 0 <= y_at_xw <= height:
                    x2, y2 = width, y_at_xw

            if cos_t != 0:
                x_at_y0 = rho / cos_t
                x_at_yh = (rho - height * sin_t) / cos_t

                if 0 <= x_at_y0 <= width:
                    if x1 is None:
                        x1, y1 = x_at_y0, 0
                    else:
                        x2, y2 = x_at_y0, 0
                if 0 <= x_at_yh <= width:
                    if x2 is None:
                        x2, y2 = x_at_yh, height
                    else:
                        x1, y1 = x_at_yh, height

            return (x1, y1, x2, y2)

        # Plot original image
        ax_image.imshow(image)
        ax_image.set_title("Detected Lines")
        ax_image.axis("off")

        # Plot all lines
        line_objects = []
        for rho, theta in zip(rhos, thetas):
            x1, y1, x2, y2 = get_line_points(rho, theta, width, height)
            if None not in [x1, y1, x2, y2]:
                # Default green, semi-transparent
                line, = ax_image.plot(
                    [x1, x2], [y1, y2], 'g', linewidth=2, alpha=0.5)
                line_objects.append(line)

        # Function to highlight line on click

        def on_pick(event):
            ind = event.ind[0]  # Get the index of the clicked point
            rho, theta = rhos[ind], thetas[ind]

            # Reset all lines
            for line in line_objects:
                line.set_color('g')  # Reset to green
                line.set_linewidth(2)
                line.set_alpha(0.5)  # Reset transparency

            # Highlight selected line
            x1, y1, x2, y2 = get_line_points(rho, theta, width, height)
            line_objects[ind].set_color('r')  # Change to red
            line_objects[ind].set_linewidth(3)
            line_objects[ind].set_alpha(1.0)

            fig.canvas.draw_idle()  # Redraw the updated plot

        # Connect the event handler
        fig.canvas.mpl_connect("pick_event", on_pick)

        # Show the plots
        plt.show()

    def plotLinesGraph(self, lines=None, showGrid=False):
        if (lines is None):
            self.checkLines()
            lines = self.lines

        if (showGrid):
            x_interval = self.ClusterDeltas[1]  # X-axis step
            y_interval = self.ClusterDeltas[0]  # Y-axis step

            # Set X-axis grid every 2 units
            plt.xticks(np.arange(lines[:, 1].min(),
                       lines[:, 1].max(), x_interval))
            # Set Y-axis grid every 0.5 units
            plt.yticks(np.arange(lines[:, 0].min(),
                       lines[:, 0].max(), y_interval))

            # Enable grid with custom styling
            plt.grid(True, linestyle="--", linewidth=0.5, color="gray")

        plt.scatter(lines[:, 1], lines[:, 0])
        print(lines.shape)

    def overlayLines(self, lines=None):
        if (lines is None):
            self.checkLines()
            lines = self.lines

        height, width = self.getImageSize()

        if lines is not None:
            for line in lines:
                rho, theta = line  # Extract rho and theta
                x1, y1, x2, y2 = get_line_points(rho, theta, width, height)
                if None not in [x1, y1, x2, y2]:  # Check if valid points exist
                    # Green infinite line
                    plt.plot([x1, x2], [y1, y2], 'r', linewidth=1)

        plt.imshow(self.img)

    def compute_clusteredLines(self, lines=None, deltas=None):
        """
        Groups and averages lines based on (rho, theta) proximity.
        - lines: Output from cv2.HoughLines()
        - delta_r: Grid size for rho
        - delta_theta: Grid size for theta
        """

        if (lines is None):
            self.checkLines()
            lines = self.lines

        if (deltas is None):
            deltas = self.ClusterDeltas

        self.clustered_lines = cluster(lines, self.ClusterDeltas)
        return self.clustered_lines

    def clustered_interactivePlot(self):
        self.checkClustered()
        self.interactivePlot(self.clustered_lines)

    def clustered_plotLinesGraph(self, showGrid=False):
        self.checkClustered()
        self.plotLinesGraph(self.clustered_lines, showGrid=showGrid)

    def clustered_overlayLines(self):
        self.checkClustered()
        self.overlayLines(self.clustered_lines)

    def compute_line_bins(self, lines=None, theta_threshold=np.pi / 10):
        """
        Clusters Hough line angles (theta values) in 1D using a region-growing approach.

        Args:
            thetas: List of theta values (angles in radians).
            theta_threshold: Maximum angle difference for grouping (default: 5 degrees).

        Returns:
            List of clusters (each cluster is a list of theta values).
        """
        if (lines is None):
            self.checkClustered()
            lines = self.clustered_lines.clone().detach()

        # Sort theta values
        lines = sorted(lines, key=lambda line: line[1])
        lines = torch.stack(lines)

        clusters = []
        current_cluster = [lines[0]]

        for i in range(1, len(lines)):
            if abs(lines[i, 1] - lines[i - 1, 1]) <= theta_threshold:
                # If within threshold, add to current cluster
                current_cluster.append(lines[i])
            else:
                # Otherwise, start a new cluster
                clusters.append(torch.stack(current_cluster))
                current_cluster = [lines[i]]

        # Add the last cluster
        if current_cluster:
            clusters.append(torch.stack(current_cluster))

        if (len(clusters) == 3):  # TODO:
            lines[lines[:, 0] < 0, 1] -= np.pi
            lines[lines[:, 0] < 0, 0] *= -1
            clusters = self.compute_line_bins(lines.clone().detach())

        self.line_bins = clusters

        return self.line_bins

    def preview_bins(self, bins=None):
        if (bins is None):
            self.checkLineBins()
            bins = self.line_bins

        binCount = len(bins)
        plt.figure(figsize=(4*binCount, 4))
        for i in range(1, binCount+1):
            plt.subplot(1, binCount, i)
            self.overlayLines(bins[i-1])

        plt.show()

    def compute_intersections(self, bins=None):
        if (bins is None):
            self.checkLineBins()
            bins = self.line_bins

        height, width = self.getImageSize()

        self.intersections = cross_bin_intersections(bins, height, width)
        return self.intersections

    def overlayIntersections(self, points=None):
        if (points is None):
            self.checkIntersections()
            points = self.intersections

        plt.scatter(points[:, 0], points[:, 1], c='red')
        plt.imshow(self.img)

    def compute_clustered_intersections(self, points=None, deltas=None):
        if (points is None):
            self.checkIntersections()
            points = self.intersections

        if (deltas is None):
            deltas = self.PointClusterDeltas

        self.clustered_interesections = cluster(points, deltas)
        return self.clustered_interesections

    def compute_intercepts(self, bins=None):
        if (bins is None):
            self.checkLineBins()
            bins = self.line_bins

        H, W = self.getImageSize()

        points_bins = []

        for b in bins:
            avg_theta = torch.mean(b[:, 1])
            crossed_line = line_cross_center(
                H, W, avg_theta + np.pi/2)

            res = []
            for line in b:
                res.append(torch.vstack(
                    [torch.tensor(intersection_polar(line, crossed_line)), line]))

            points_bins.append(torch.stack(res))

        self.intercepts = points_bins
        return self.intercepts

    def compute_filtered_bins(self, bins=None, SquareSizeFiltering=None):
        if (bins is None):
            self.checkIntercepts()
            bins = self.intercepts

        if (SquareSizeFiltering is None):
            SquareSizeFiltering = self.SquareSizeFiltering

        res_bins = []

        for points in bins:

            # Compute center of all points
            center = torch.mean(points[:, 0, :], axis=0)

            # Find the point closest to the center
            distances_to_center = np.linalg.norm(
                points[:, 0, :] - center, axis=1)
            center_idx = np.argmin(distances_to_center)
            center_point = points[center_idx]

            # Sort points by distance along the main axis (x-axis if horizontal, y-axis if vertical)
            sorted_points = points[np.argsort(np.linalg.norm(
                # Sort by x-coordinates
                points[:, 0, :] - points[center_idx, 0, :], axis=1))]

            # Start filtering with the center point
            selected = [center_point]

            for p in sorted_points:
                if all(torch.norm(p[0, :] - s[0, :]) >= SquareSizeFiltering for s in selected):
                    selected.append(p)

            res = torch.stack(selected)[:, 1]
            res_bins.append(res)

        self.square_filtered_bins = res_bins
        return self.square_filtered_bins

    def compute_cross_centrals(self, bins=None, grid_size=5):
        if (bins is None):
            self.checkSquareFiltered()
            bins = self.square_filtered_bins

        centeral_lines_binned = []

        for b in range((len(bins))):
            lines = bins[b]
            sorted_lines = lines[lines[:, 0].argsort()]

            # Find the median index
            mid_idx = len(sorted_lines) // 2

            # Get indices for the median 5 (centered around the median index)
            half_n = grid_size // 2
            start_idx = max(0, mid_idx - half_n)
            end_idx = min(len(sorted_lines), start_idx + grid_size)

            # Select the median 5 lines
            selected_lines = sorted_lines[[start_idx, end_idx-1]]

            centeral_lines_binned.append(selected_lines)

        height, width = self.getImageSize()

        self.centrals = cross_bin_intersections(
            centeral_lines_binned, height, width)

        return self.centrals

    def compute_orient(self, centrals=None):
        if (centrals is None):
            self.checkCentrals()
            centrals = self.centrals

        if (centrals.shape[0] != 4 or centrals.shape[1] != 2 or len(centrals.shape) != 2):
            raise ValueError("Invalid Centrals")

        H, W = self.getImageSize()

        # Define the four image corners
        img_corners = torch.tensor([
            [0, 0],      # Top-Left
            [W, 0],      # Top-Right
            [W, H],       # Bottom-Right
            [0, H],      # Bottom-Left
        ], dtype=torch.float32)

        res = []

        for i in range(4):
            idx = torch.norm(centrals - img_corners[i], dim=1).argmin()
            point = centrals[idx].clone().detach()

            centrals = torch.cat((centrals[:idx], centrals[idx+1:]))
            res.append(point)

        self.oriented_centrals = torch.stack(res)
        return self.oriented_centrals

        # # Compute distances between each detected point and each image corner
        # distances = torch.cdist(centrals, img_corners)  # (4,4) matrix

        # # Sort distances and get indices
        # # Sorted point indices for each corner
        # sorted_indices = distances.argsort(dim=1)

        # # Placeholder for ordered points
        # assigned_points = torch.full((4, 2), float('inf'))
        # used_indices = set()  # Track used point indices

        # # Assign each corner its closest available point
        # for corner_idx in range(4):
        #     for i in range(4):  # Iterate over sorted distances
        #         point_idx = sorted_indices[i, corner_idx].item()
        #         if point_idx not in used_indices:  # Ensure unique assignment
        #             assigned_points[corner_idx] = centrals[point_idx]
        #             used_indices.add(point_idx)
        #             break  # Stop once we assign a valid point

        # self.oriented_centrals = assigned_points  # (TL, TR, BL, BR)
        # return self.oriented_centrals

    def fitBoard(self, oriented_centrals=None, grid_size=4):
        if (oriented_centrals is None):
            self.checkOrientedCentrals()
            oriented_centrals = self.oriented_centrals

        self.checkMask()

        H, W = self.getImageSize()

        best_corners = None
        best_error = -1
        best_M = None
        best_mask = None
        best_area = -1

        edges = self.edges
        # dist_transform = torch.tensor(
        #     cv2.distanceTransform(255-edges, cv2.DIST_L2, 5))
        # resized_edges = torch.tensor(cv2.erode(cv2.dilate(cv2.resize(self.edges, (640, 640)), np.ones(
        #     (3, 3)), iterations=2), np.ones((3, 3)), iterations=1)/255, dtype=torch.float32)

        # box_size = 5
        # kernel = np.ones((box_size, box_size), np.uint8)
        # img_dilation = cv2.dilate(edges.numpy(), kernel, iterations=1)
        # edges_dilation = torch.tensor(img_dilation, dtype=torch.float32)

        found_best = False
        configs_count = 0
        for grid_shape_i in range(grid_size, 9):
            if (found_best):
                break

            for grid_shape_j in range(grid_size, 9):
                if (found_best):
                    break

                for i in range(9-grid_shape_i):
                    if (found_best):
                        break

                    for j in range(9-grid_shape_j):
                        if (found_best):
                            break

                        simulated = torch.tensor([
                            [i, j],  # TL
                            [(grid_shape_i + i), j],  # TR
                            [(grid_shape_i+i), (grid_shape_j+j)],  # BR
                            [i, (grid_shape_j+j)],  # BL
                        ], dtype=torch.float32) * square_size

                        # -------------
                        # Ignore out of bounds configurations

                        M = calcM(simulated, oriented_centrals)
                        tmp_corners = warpPts(virtual_bounds, M)

                        image_bounds = get_bounds((W, H))
                        M2 = calcM(image_bounds, YOLO_Image_bounds)
                        scaled_corners = torch.round(
                            warpPts(tmp_corners, M2)).to(int)

                        skip_config = False
                        # if ((scaled_corners < 0).any() or (scaled_corners >= 640).any()):
                        #     skip_config = True
                        # else:
                        #     for corner in scaled_corners:
                        #         if (self.boundary_mask[corner[0], corner[1]] == 0):
                        #             skip_config = True
                        #             break
                        if (skip_config):
                            continue  # skip out of bounds configurations

                        # --------------
                        # IOU

                        mask = BoardImageUtils.get_mask(
                            scaled_corners, (640, 640))
                        iou_error = (1-BoardImageUtils.get_acc_masks(
                            mask, self.boundary_mask))
                        error = iou_error

                        area = grid_shape_i * grid_shape_j
                        # if (iou_error > best_error and best_error != -1 and area > best_area):
                        #     found_best = True
                        #     break

                        # --------------
                        # Distance Transform Error

                        # M, board_mask = reverseWarp(
                        #     oriented_centrals, simulated, (W, H))

                        # board_mask = torch.tensor(cv2.resize(
                        #     board_mask, (640, 640))/255, dtype=torch.float32)
                        # board_mask[board_mask >= .001] = 1
                        # board_mask[board_mask < .001] = 0

                        # error += calc_error(dist_transform, board_mask)

                        # --------------
                        # Adjust Best Config

                        configs_count += 1
                        if ((best_error == -1) or error < best_error):
                            best_corners = simulated.clone().detach()

                            best_error = error
                            best_iou = iou_error

                            best_area = area

            M, board_mask = reverseWarp(
                oriented_centrals, best_corners, (W, H))
            best_M = M
            best_mask = board_mask

            out_corners = warpPts(virtual_bounds, M)

            # print(f"Configurations Count: {configs_count}")
            return best_M, out_corners, best_mask, best_error
