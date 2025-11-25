import os
import cv2
import json
import numpy as np
import glob
import torch
import time
from tqdm import tqdm
from ultralytics import YOLO  # YOLO trực tiếp
from dotenv import load_dotenv

load_dotenv()


class AeroEyesPredictor:
    def __init__(
        self, model_path, ref_images_dir, conf_threshold=0.001, color_tol=40.0
    ):
        """
        color_tol: Ngưỡng sai lệch màu chấp nhận được (mặc định +-20)
        """
        self.conf_threshold = conf_threshold
        self.color_tol = color_tol  # Ngưỡng sai lệch màu (+-20)

        cuda_available = torch.cuda.is_available()
        device = "cuda:0" if cuda_available else "cpu"

        print(f"🔧 Device thông tin:")
        print(f"   - CUDA available: {cuda_available}")
        print(f"   - Device sử dụng: {device}")
        if cuda_available:
            print(f"   - GPU name: {torch.cuda.get_device_name(0)}")
            print(f"   - CUDA version: {torch.version.cuda}")

        print(f"⏳ Đang load model từ: {model_path}")
        start_time = time.time()

        try:
            self.model = YOLO(model_path)
            self.model.to(device)

            load_time = time.time() - start_time
            print(f"✅ Model loaded thành công!")
            print(f"   - Thời gian load: {load_time:.2f}s")
            print(f"   - Model device: {next(self.model.model.parameters()).device}")

        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            exit(1)

        self.ref_dominant_colors = []  # Lưu list các màu chủ đạo của 3 ảnh ref
        self.load_reference_images(ref_images_dir)

    def get_dominant_color(self, img):
        """
        Tìm màu xuất hiện nhiều nhất trong ảnh dùng K-Means.
        Output: numpy array [B, G, R]
        """
        if img is None or img.size == 0:
            return None

        # Loại bỏ 20% padding các bên để lấy vùng trung tâm
        h, w = img.shape[:2]
        crop_h, crop_w = int(h * 0.6), int(w * 0.6)  # Giữ lại 60% vùng trung tâm
        start_h, start_w = int(h * 0.2), int(w * 0.2)  # Bắt đầu từ 20%
        img_center = img[start_h : start_h + crop_h, start_w : start_w + crop_w]

        # Resize nhỏ lại để chạy K-Means cho nhanh (tốc độ realtime)
        img_small = cv2.resize(img_center, (64, 64), interpolation=cv2.INTER_AREA)

        # Reshape thành danh sách pixel
        data = np.float32(img_small).reshape((-1, 3))

        # K-Means setup
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # Tìm 1 cụm (cluster) đại diện cho màu phổ biến nhất
        try:
            _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)
            return centers[0]  # Trả về [B, G, R]
        except:
            return np.mean(data, axis=0)  # Fallback nếu kmeans lỗi

    def load_reference_images(self, ref_dir):
        if not os.path.exists(ref_dir):
            return

        image_paths = glob.glob(os.path.join(ref_dir, "*.jpg")) + glob.glob(
            os.path.join(ref_dir, "*.png")
        )

        for p in image_paths:
            img = cv2.imread(p)
            if img is not None and img.size > 0:
                # Tính màu chủ đạo cho từng ảnh ref
                dom_color = self.get_dominant_color(img)
                if dom_color is not None:
                    self.ref_dominant_colors.append(dom_color)

        print(f"Loaded {len(self.ref_dominant_colors)} reference colors.")

    def check_color_similarity(self, candidate_crop):
        """
        So sánh màu crop với list màu reference.
        Returns:
           score: khoảng cách màu (càng nhỏ càng tốt),
           passed: True nếu thỏa mãn ngưỡng +-20
        """
        if len(self.ref_dominant_colors) == 0:
            return float("inf"), False

        if candidate_crop is None or candidate_crop.size == 0:
            return float("inf"), False

        cand_color = self.get_dominant_color(candidate_crop)
        if cand_color is None:
            return float("inf"), False

        # Tìm reference khớp nhất (khoảng cách nhỏ nhất)
        best_dist = float("inf")
        is_passed = False

        for ref_color in self.ref_dominant_colors:
            # Tính sai lệch trên từng kênh màu (Abs diff)
            diff = np.abs(ref_color - cand_color)  # [abs(B-B'), abs(G-G'), abs(R-R')]

            # Kiểm tra điều kiện: Không kênh nào lệch quá ngưỡng (ví dụ 20)
            # np.max(diff) <= 20 nghĩa là tất cả các kênh đều <= 20
            if np.max(diff) <= self.color_tol:
                # Nếu pass điều kiện lọc, tính khoảng cách Euclidean để ranking
                dist = np.linalg.norm(ref_color - cand_color)
                if dist < best_dist:
                    best_dist = dist
                    is_passed = True

        return best_dist, is_passed

    def predict_streaming(self, frame_rgb_np, frame_idx):
        try:
            results = self.model.predict(
                frame_rgb_np,
                conf=self.conf_threshold,
                verbose=False,
                iou=0.4,  # NMS iou threshold
            )
        except Exception:
            return None

        result = results[0]
        frame_bgr = cv2.cvtColor(frame_rgb_np, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        if result.boxes is None:
            return None

        valid_candidates = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop = frame_bgr[y1:y2, x1:x2]

            # --- LOGIC MỚI: COLOR FILTERING ---
            dist_score, passed = self.check_color_similarity(crop)

            if passed:
                # Lưu lại bbox và khoảng cách màu để so sánh sau
                valid_candidates.append(
                    {
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "score": dist_score,  # Khoảng cách Euclidean (càng nhỏ càng tốt)
                    }
                )

        if not valid_candidates:
            return None

        # --- LOGIC CHỌN: Lấy bbox có màu gần nhất (score nhỏ nhất) ---
        # Sắp xếp candidates theo score tăng dần
        valid_candidates.sort(key=lambda x: x["score"])

        return valid_candidates[0]["bbox"]


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================


def main():
    data_dir = os.getenv(
        "DATA_DIR", "/data/samples"
    )
    output_path = os.getenv("OUTPUT_PATH", "/result/submission.json")
    model_path = "saved_models/yolov8l_1e.pt"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    final_output = []

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found at {data_dir}")
        return

    subfolders = sorted([f.path for f in os.scandir(data_dir) if f.is_dir()])
    print(f"Found {len(subfolders)} cases.")

    for folder_path in subfolders:
        case_name = os.path.basename(folder_path)
        video_path = os.path.join(folder_path, "drone_video.mp4")
        ref_images_path = os.path.join(folder_path, "object_images")

        if not os.path.exists(video_path):
            continue

        try:
            print(f"\n🚀 Khởi tạo predictor cho case: {case_name}")
            init_start = time.time()

            predictor = AeroEyesPredictor(
                model_path,
                ref_images_path,
                conf_threshold=0.001,
                color_tol=20.0,
            )

            init_time = time.time() - init_start
            print(f"✅ Predictor khởi tạo xong trong: {init_time:.2f}s")

        except Exception as e:
            print(f"❌ Skip {case_name}: {e}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        case_result = {"video_id": case_name, "detections": []}
        has_detections = False

        with tqdm(
            total=total_frames, desc=f"Processing {case_name}", unit="fr", ncols=100
        ) as pbar:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                box = predictor.predict_streaming(frame_rgb, frame_idx)

                if box is not None:
                    if not has_detections:
                        case_result["detections"].append({"bboxes": []})
                        has_detections = True

                    case_result["detections"][0]["bboxes"].append(
                        {
                            "frame": int(frame_idx),
                            "x1": int(box[0]),
                            "y1": int(box[1]),
                            "x2": int(box[2]),
                            "y2": int(box[3]),
                        }
                    )

                    pbar.set_postfix_str("Found: Yes", refresh=False)
                else:
                    pbar.set_postfix_str("Found: No", refresh=False)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        final_output.append(case_result)

    # ---------------------
    # 🔥 FIX JSON ERROR
    # ---------------------
    # Ensure EVERYTHING is JSON serializable by converting recursively
    def convert(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4, default=convert)

    print(f"\nDone! Output saved to {output_path}")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
