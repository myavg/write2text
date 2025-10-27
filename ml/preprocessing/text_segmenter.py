import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


class TextSegmenter:
    def __init__(self, pad: int = 20):
        self.pad = pad
        self.img = None
        self.img_binary = None

    # ---------------- загрузка и deskew ----------------
    def load_image(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.copyMakeBorder(
            img, self.pad, self.pad, self.pad, self.pad,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        self.img = img
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        self.img_binary = 255 - img_binary

    @staticmethod
    def detect_text_skew_angle(img_binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(cleaned, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return 0
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 20:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 <= angle <= 45:
                    angles.append(angle)
        return np.median(angles) if angles else 0

    @staticmethod
    def deskew_image(image, angle):
        if abs(angle) < 0.5:
            return image.copy()
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w - w) // 2
        M[1, 2] += (new_h - h) // 2
        return cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    # ---------------- utils ----------------
    @staticmethod
    def calculate_rect_distance(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        left1, right1 = x1, x1 + w1
        left2, right2 = x2, x2 + w2
        if right1 < left2:
            return left2 - right1
        elif right2 < left1:
            return left1 - right2
        return 0

    # ---------------- merge_close_contours ----------------
    def merge_close_contours(self, row):
        def change_format(box):
            x, y, w, h = box
            return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        def adaptive_threshold_clustering(distances_list, multiplier=2):
            if np.all(distances_list == distances_list[0]):
                return distances_list[0]
            q25, q75 = np.percentile(distances_list, [25, 75])
            iqr = q75 - q25
            multiplier = 1.8 if iqr > q25 else 1.5
            upper_bound = q75 + multiplier * iqr
            reasonable_upper_limit = max(50, upper_bound)
            filtered = distances_list[distances_list <= reasonable_upper_limit].reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
            labels = kmeans.fit_predict(filtered)
            centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            large_cluster_idx = sorted_indices[1]
            large_cluster_values = filtered[labels == large_cluster_idx]
            return float(np.min(large_cluster_values))

        used = set()
        d = np.array([])
        group_tail = None
        for i in range(len(row)):
            if i in used:
                continue
            if group_tail:
                d = np.append(d, row[i][0] - group_tail)
            group_tail = row[i][0] + row[i][2]
            used.add(i)
            queue = [i]
            while queue:
                current_idx = queue.pop(0)
                for j in range(len(row)):
                    if j in used:
                        continue
                    distance_hor = self.calculate_rect_distance(row[current_idx], row[j])
                    if distance_hor == 0:
                        group_tail = max(group_tail, row[j][0] + row[j][2])
                        used.add(j)
                        queue.append(j)

        hor_dist_thr = adaptive_threshold_clustering(d)
        groups = []
        used = set()
        for i in range(len(row)):
            if i in used:
                continue
            current_group = [i]
            used.add(i)
            queue = [i]
            while queue:
                current_idx = queue.pop(0)
                for j in range(len(row)):
                    if j in used:
                        continue
                    distance_hor = self.calculate_rect_distance(row[current_idx], row[j])
                    if distance_hor < hor_dist_thr:
                        current_group.append(j)
                        used.add(j)
                        queue.append(j)
            groups.append(current_group)

        merged_contours = []
        for group in groups:
            if len(group) == 1:
                merged_contours.append(np.array(change_format(row[group[0]])))
            else:
                merged_points = np.vstack([change_format(row[i]) for i in group])
                merged_contours.append(cv2.convexHull(merged_points))
        return merged_contours

    # ---------------- split/group ----------------
    def split_contours(self, x, y, w, h):
        projection = np.sum(self.img_binary[y:y+h, x:x+w], axis=1)
        threshold = np.max(projection) * 0.25
        lines, in_line = [], False
        for i, val in enumerate(projection):
            if val > threshold and not in_line:
                start = i; in_line = True
            elif val <= threshold and in_line:
                end = i; in_line = False; lines.append((start, end))
        if in_line:
            lines.append((start, i))
        heights = [line[1] - line[0] for line in lines]
        if not heights:
            return []
        mean_height, max_height = np.mean(heights), np.max(heights)
        lines = [line for line in lines if (line[1] - line[0]) > max(mean_height * 0.6, max_height * 0.2)]
        snapshots = [self.img_binary[y + line[0]:y + line[1], x:x + w] for line in lines]
        contours = []
        for i, im in enumerate(snapshots):
            contours_im, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_im:
                x_, y_, w_, h_ = cv2.boundingRect(contour)
                contours.append((x + x_, y + lines[i][0] + y_, w_, h_))
        return contours

    def group_contours_histogram(self, contours, bins=50):
        bboxes = [cv2.boundingRect(cnt) for cnt in contours]
        y_centers = [y + h/2 for _, y, _, h in bboxes]
        hist, bin_edges = np.histogram(y_centers, bins=bins)
        peaks, _ = find_peaks(hist)
        if len(hist) > 1:
            if hist[0] > hist[1]:
                peaks = np.concatenate(([0], peaks))
            if hist[-1] > hist[-2]:
                peaks = np.concatenate((peaks, [len(hist)-1]))
        row_centers = [(bin_edges[p] + bin_edges[p+1]) / 2 for p in peaks]
        rows = [[] for _ in range(len(row_centers))]

        def add_splitted(splitted_contours):
            bbox_centers = [y + h/2 for _, y, _, h in splitted_contours]
            distances = [[abs(c - rc) for rc in row_centers] for c in bbox_centers]
            closest_rows = np.argmin(distances, axis=1)
            for i, row_idx in enumerate(closest_rows):
                rows[row_idx].append(splitted_contours[i])

        for bbox in bboxes:
            x, y, w, h = bbox
            bbox_center = y + h/2
            distances = [abs(bbox_center - rc) for rc in row_centers]
            closest_row = np.argmin(distances)
            checked = False
            if closest_row != 0 and y <= row_centers[closest_row - 1]:
                splitted = self.split_contours(x, y, w, h); add_splitted(splitted); checked = True
            if closest_row != len(distances) - 1 and not checked and y + h >= row_centers[closest_row + 1]:
                splitted = self.split_contours(x, y, w, h); add_splitted(splitted); checked = True
            if not checked:
                rows[closest_row].append(bbox)
        for row in rows:
            row.sort(key=lambda x: x[0])
        return [row for row in rows if row]

    def auto_adjust_bins(self, contours):
        projection = np.sum(self.img_binary, axis=1)
        threshold = np.max(projection) * 0.25
        in_line, estimated_rows = False, 0
        for val in projection:
            if val > threshold and not in_line:
                estimated_rows += 1; in_line = True
            elif val <= threshold and in_line:
                in_line = False
        return estimated_rows * 3

    # ---------------- процесс и сохранение ----------------
    def process_and_save(self, path: str, output_dir: str):
        """Выполняет сегментацию и сохраняет фреймы слов в storage."""
        os.makedirs(output_dir, exist_ok=True)
        self.load_image(path)
        angle = self.detect_text_skew_angle(self.img_binary)
        if abs(angle) > 1.0:
            self.img_binary = self.deskew_image(self.img_binary, angle)
            self.img = self.deskew_image(self.img, angle)

        contours, _ = cv2.findContours(self.img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bins = self.auto_adjust_bins(contours)
        sorted_rows = self.group_contours_histogram(contours, bins=bins)

        frame_count = 0
        for row_idx, row in enumerate(sorted_rows):
            row = [cnt for cnt in np.array(row, dtype=np.int32)]
            merged = self.merge_close_contours(row)
            row_dir = os.path.join(output_dir, f"row_{row_idx}")
            os.makedirs(row_dir, exist_ok=True)
            for contour_idx, contour in enumerate(merged):
                x, y, w, h = cv2.boundingRect(contour)
                crop = self.img[y:y+h, x:x+w]
                out_path = os.path.join(row_dir, f"word_{contour_idx}.png")
                cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                frame_count += 1
        print(f"Saved {frame_count} frames to {output_dir}")
