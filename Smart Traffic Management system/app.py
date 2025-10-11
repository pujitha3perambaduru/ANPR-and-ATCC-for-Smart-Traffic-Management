import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import pytesseract
import os
import math
import time
from datetime import datetime

# ---- CONFIG ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
VEHICLE_MODEL = "yolov8n.pt"
HELMET_MODEL = "helmet_model.pt"
PLATE_MODEL = "license_plate_detector.pt"

os.makedirs("logs", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---- STREAMLIT UI ----
st.set_page_config(page_title="AI-Powered ANPR & ATCC", layout="wide")
st.title("🚦 ANPR & ATCC SMART TRAFFIC MANAGEMENT SYSTEM")

# Input selection
option = st.selectbox("Select Input Source", ["Upload Video", "Webcam", "Test Image"])
uploaded = st.file_uploader("Upload a video or image", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])
start = st.button("▶ Start Detection")

# Optimization controls (Module 3)
st.sidebar.header("Performance / Optimization")
conf_thres = st.sidebar.slider("Confidence threshold (models)", 0.1, 0.9, 0.35)
frame_skip = st.sidebar.slider("Process every Nth frame (frame skip)", 1, 10, 2)
scale_factor = st.sidebar.slider("Detection scale factor (smaller → faster)", 0.25, 1.0, 0.6)
roi_only = st.sidebar.checkbox("Run helmet & plate detection inside vehicle ROIs only", True)
show_perf = st.sidebar.checkbox("Show performance metrics", True)
debug_overlay = st.sidebar.checkbox("Debug overlays (boxes & ids)", False)

# ---- LOAD MODELS ----
@st.cache_resource
def load_models():
    vehicle_model = YOLO(VEHICLE_MODEL)
    helmet_model = YOLO(HELMET_MODEL)
    plate_model = YOLO(PLATE_MODEL)
    return vehicle_model, helmet_model, plate_model

vehicle_model, helmet_model, plate_model = load_models()

# ---- LOGS & DISPLAY ----
log_cols = ["Timestamp", "Vehicle Type", "Plate Number", "Violation"]
log_df = pd.DataFrame(columns=log_cols)
frame_display = st.image([])
log_placeholder = st.empty()
perf_placeholder = st.empty()

# ---- Helpers: OCR, tracker, helmet check, lane (same logic preserved) ----
def run_ocr(crop):
    try:
        if crop is None or crop.size == 0:
            return ""
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(
            th,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        return "".join([c for c in text if c.isalnum()])[:12]
    except Exception:
        return ""

class SimpleTracker:
    def __init__(self, max_distance=80, max_missed=10):
        self.next_id = 1
        self.objects = {}  # id -> {'centroid':(cx,cy),'box':(x1,y1,x2,y2),'missed':0,'type':name,'counted':bool}
        self.max_distance = max_distance
        self.max_missed = max_missed

    def _centroid(self, box):
        x1,y1,x2,y2 = box
        return ((x1+x2)//2, (y1+y2)//2)

    def update(self, detections):
        mapping = {}
        used = set()
        det_centroids = [self._centroid(d['box']) for d in detections]

        for di, det in enumerate(detections):
            best_id = None
            best_dist = None
            cx,cy = det_centroids[di]
            for oid, info in self.objects.items():
                if oid in used:
                    continue
                ocx, ocy = info['centroid']
                d = math.hypot(cx-ocx, cy-ocy)
                if d <= self.max_distance and (best_dist is None or d < best_dist):
                    best_dist = d; best_id = oid
            if best_id is not None:
                mapping[di] = best_id
                used.add(best_id)
                self.objects[best_id]['centroid'] = det_centroids[di]
                self.objects[best_id]['box'] = detections[di]['box']
                self.objects[best_id]['missed'] = 0
                self.objects[best_id]['type'] = detections[di]['name']

        for di, det in enumerate(detections):
            if di in mapping:
                continue
            oid = self.next_id; self.next_id += 1
            mapping[di] = oid
            self.objects[oid] = {
                'centroid': det_centroids[di],
                'box': det['box'],
                'missed': 0,
                'type': det['name'],
                'counted': False
            }

        for oid in list(self.objects.keys()):
            if oid not in used and oid not in mapping.values():
                self.objects[oid]['missed'] += 1
                if self.objects[oid]['missed'] > self.max_missed:
                    del self.objects[oid]

        return mapping

def helmet_on_head(person_box, helmet_box):
    px1,py1,px2,py2 = person_box
    hx1,hy1,hx2,hy2 = helmet_box
    h_th = py1 + int(0.4 * (py2 - py1))  # top 40% = head
    head_x1, head_y1, head_x2, head_y2 = px1, py1, px2, h_th
    ix1 = max(head_x1, hx1); iy1 = max(head_y1, hy1)
    ix2 = min(head_x2, hx2); iy2 = min(head_y2, hy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter_area = (ix2-ix1)*(iy2-iy1)
    helmet_area = max(1, (hx2-hx1)*(hy2-hy1))
    if inter_area/helmet_area > 0.2 or inter_area > 500:
        return True
    return False

def lane_from_centroid(cx, cy, w, h):
    dx = cx - w/2; dy = cy - h/2
    if abs(dx) >= abs(dy):
        return "East" if dx > 0 else "West"
    else:
        return "South" if dy > 0 else "North"

# ---- Violation logic preserved exactly as before, but used by process_frame ----
def detect_helmet_violations(items, helmets):
    # returns list of (type, box, color) where box is person box for No Helmet or motorbike box for triple
    violations = []
    motorbikes = [it for it in items if it["name"].lower() in ["motorbike", "motorcycle"]]
    persons = [it for it in items if it["name"].lower() == "person"]

    for mb in motorbikes:
        mb_box = mb["box"]
        riders = 0
        for p in persons:
            px1, py1, px2, py2 = p["box"]
            mx1, my1, mx2, my2 = mb_box
            x1 = max(mx1, px1); y1 = max(my1, py1)
            x2 = min(mx2, px2); y2 = min(my2, py2)
            if x2 > x1 and y2 > y1:
                riders += 1
                helmet_found = False
                for h in helmets:
                    if helmet_on_head((px1,py1,px2,py2), h):
                        helmet_found = True
                        break
                if not helmet_found:
                    violations.append(("No Helmet", [px1,py1,px2,py2], (0,0,255)))
        if riders >= 3:
            violations.append(("Triple Riding", mb_box, (0,0,255)))
    return violations

# ---- PROCESS FRAME with optimization (frame skipping & ROI) ----
def process_frame_optimized(frame, vehicle_counter, seen_plates, tracker,
                            last_vehicle_res, last_helmet_res, last_plate_res,
                            frame_idx, skip, scale):
    """
    Returns:
      annotated_frame, lane_vehicles,
      vehicle_res_used, helmet_res_used, plate_res_used
    """
    h_orig, w_orig = frame.shape[:2]

    # scale for detection to speed up
    if scale != 1.0:
        small = cv2.resize(frame, (int(w_orig*scale), int(h_orig*scale)))
    else:
        small = frame.copy()
    h_s, w_s = small.shape[:2]

    # Decide whether to run full detection this frame or reuse last results
    run_detection = (frame_idx % skip == 0)
    vehicle_results = last_vehicle_res
    helmet_results = last_helmet_res
    plate_results = last_plate_res

    if run_detection:
        # vehicle detection on scaled frame
        vehicle_results = vehicle_model(small, conf=conf_thres)[0]

        # For helmet and plate detection we either run on full frame (slower) or only on ROIs
        # We'll run helmet on either full scaled frame or per-vehicle ROI on scaled frame
        if roi_only and vehicle_results.boxes is not None:
            # Build combined ROI list from vehicle boxes on scaled image
            helmet_rois = []
            for vb in vehicle_results.boxes.xyxy.cpu().numpy():
                vx1, vy1, vx2, vy2 = map(int, vb)
                # expand a bit
                pad_x = int(0.1*(vx2-vx1)); pad_y = int(0.2*(vy2-vy1))
                rx1 = max(0, vx1-pad_x); ry1 = max(0, vy1-pad_y)
                rx2 = min(w_s, vx2+pad_x); ry2 = min(h_s, vy2+pad_y)
                helmet_rois.append([rx1, ry1, rx2, ry2])
            # run helmet detection on full scaled frame but we'll later filter boxes by ROI overlap to save from running helmet_model multiple times
            helmet_results = helmet_model(small, conf=conf_thres)[0]
            # plate detection similarly, but run on scaled or full? run plate on small frame and map boxes back
            plate_results = plate_model(small, conf=conf_thres)[0]
        else:
            # Run helmet & plate on full scaled frame
            helmet_results = helmet_model(small, conf=conf_thres)[0]
            plate_results = plate_model(small, conf=conf_thres)[0]

    # Map scaled boxes back to original frame coords when using scaled inference
    def scale_box_to_orig(box_scaled):
        x1,y1,x2,y2 = box_scaled
        if scale != 1.0:
            return [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)]
        else:
            return [int(x1), int(y1), int(x2), int(y2)]

    annotated = frame.copy()
    items = []
    detections_for_tracker = []

    # Use vehicle_results (may be from last run) map to original coords
    if vehicle_results is not None and vehicle_results.boxes is not None:
        for box, cls, conf in zip(
            vehicle_results.boxes.xyxy.cpu().numpy(),
            vehicle_results.boxes.cls.cpu().numpy().astype(int),
            vehicle_results.boxes.conf.cpu().numpy()
        ):
            name = vehicle_model.names[int(cls)]
            bx = scale_box_to_orig(map(int, box)) if scale != 1.0 else list(map(int, box))
            x1,y1,x2,y2 = bx
            items.append({"box": [x1,y1,x2,y2], "name": name, "conf": float(conf)})
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(annotated, name, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            detections_for_tracker.append({'box': (x1,y1,x2,y2), 'name': name})

    # Map helmet boxes
    helmet_boxes = []
    if helmet_results is not None and helmet_results.boxes is not None:
        for box in helmet_results.boxes.xyxy.cpu().numpy():
            bx = scale_box_to_orig(map(int, box))
            helmet_boxes.append(bx)
            # draw helmet on annotated (dark blue)
            cv2.rectangle(annotated, (bx[0],bx[1]), (bx[2],bx[3]), (139,0,0), 2)
            cv2.putText(annotated, "Helmet", (bx[0], bx[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (139,0,0), 2)

    # Map plate boxes and run OCR only inside ROI (original coords)
    plates_in_frame = []
    if plate_results is not None and plate_results.boxes is not None:
        for box in plate_results.boxes.xyxy.cpu().numpy():
            px1,py1,px2,py2 = map(int, box)
            bx = scale_box_to_orig((px1,py1,px2,py2))
            # crop from original frame
            x1c, y1c, x2c, y2c = max(0,bx[0]), max(0,bx[1]), min(w_orig,bx[2]), min(h_orig,bx[3])
            crop = frame[y1c:y2c, x1c:x2c]
            plate_text = run_ocr(crop)
            if plate_text:
                plates_in_frame.append((plate_text, (x1c,y1c,x2c,y2c)))
            # annotate plate detection box (light color)
            cv2.rectangle(annotated, (x1c,y1c), (x2c,y2c), (255,200,0), 1)
            if plate_text:
                cv2.putText(annotated, plate_text, (x1c, y2c+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # Tracker update
    mapping = tracker.update(detections_for_tracker) if detections_for_tracker else {}

    # Collect lane counts (per-frame) and counting logic (plate-first, tracker fallback)
    lane_vehicles = {"North":{}, "South":{}, "East":{}, "West":{}}
    for di, det in enumerate(detections_for_tracker):
        x1,y1,x2,y2 = det['box']
        cx,cy = (x1+x2)//2, (y1+y2)//2
        lane = lane_from_centroid(cx, cy, w_orig, h_orig)
        vtype = det['name'] if det['name'] else "Vehicle"

        # try match plate
        matched_plate = ""
        for ptext, pbox in plates_in_frame:
            px1,py1,px2,py2 = pbox
            x_overlap = max(0, min(x2,px2)-max(x1,px1))
            y_overlap = max(0, min(y2,py2)-max(y1,py1))
            if x_overlap > 0 and y_overlap > 0:
                matched_plate = ptext
                break

        if matched_plate:
            if matched_plate not in seen_plates:
                seen_plates.add(matched_plate)
                vehicle_counter[vtype] = vehicle_counter.get(vtype,0) + 1
            # Log ANPR
            log_df.loc[len(log_df)] = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                vtype,
                matched_plate,
                ""
            ]
        else:
            oid = mapping.get(di, None)
            if oid is not None:
                obj = tracker.objects.get(oid, {})
                if obj and not obj.get('counted', False):
                    vehicle_counter[vtype] = vehicle_counter.get(vtype,0) + 1
                    tracker.objects[oid]['counted'] = True
                    log_df.loc[len(log_df)] = [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        vtype,
                        "",
                        ""
                    ]

        lane_vehicles[lane][vtype] = lane_vehicles[lane].get(vtype,0) + 1

    # Helmet / No-Helmet and triple detection — use items and helmet_boxes
    motorbikes = [it for it in items if it['name'].lower() in ['motorbike','motorcycle']]
    persons = [it for it in items if it['name'].lower() == 'person']

    # Check persons riding motorbike for helmet presence
    for mb in motorbikes:
        mx1,my1,mx2,my2 = mb['box']
        for p in persons:
            px1,py1,px2,py2 = p['box']
            ix = max(0, min(mx2,px2)-max(mx1,px1))
            iy = max(0, min(my2,py2)-max(my1,py1))
            if ix>0 and iy>0:
                helmet_found = False
                for hbox in helmet_boxes:
                    if helmet_on_head((px1,py1,px2,py2), hbox):
                        helmet_found = True
                        break
                if helmet_found:
                    cv2.rectangle(annotated, (px1,py1), (px2,py2), (139,0,0), 2)
                    cv2.putText(annotated, "Helmet", (px1, py1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139,0,0), 2)
                else:
                    cv2.rectangle(annotated, (px1,py1), (px2,py2), (0,0,255), 2)
                    cv2.putText(annotated, "No Helmet", (px1, py1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    log_df.loc[len(log_df)] = [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "motorbike",
                        "",
                        "No Helmet"
                    ]

    # Triple riding detection
    for mb in motorbikes:
        mx1,my1,mx2,my2 = mb['box']
        riders = 0
        for p in persons:
            px1,py1,px2,py2 = p['box']
            ix = max(0, min(mx2,px2)-max(mx1,px1))
            iy = max(0, min(my2,py2)-max(my1,py1))
            if ix>0 and iy>0:
                riders += 1
        if riders >= 3:
            cv2.putText(annotated, "Triple Riding", (mx1, my1-24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.rectangle(annotated, (mx1,my1), (mx2,my2), (0,0,255), 2)
            log_df.loc[len(log_df)] = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "motorbike",
                "",
                "Triple Riding"
            ]

    return annotated, lane_vehicles, vehicle_results, helmet_results, plate_results

# ---- UI: display logs & ATCC summary (unchanged semantics) ----
def compute_signals(lane_vehicles):
    totals = {lane: sum(counts.values()) for lane, counts in lane_vehicles.items()}
    if not totals:
        return {}, {}
    max_total = max(totals.values())
    signals = {}
    for lane, total in totals.items():
        signals[lane] = "🟢 Green" if (total == max_total and max_total>0) else "🔴 Red"
    return signals, totals

def update_log_and_traffic(vehicle_counter, lane_vehicles):
    st.markdown("""
    <style>
    .dataframe th, .dataframe td {
        max-width: 300px;
        word-wrap: break-word;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        if not log_df.empty:
            st.subheader("📋 ANPR Vehicle Log")
            st.dataframe(log_df.tail(200))
    with col2:
        st.subheader("🚗 Vehicle Count (ATCC)")
        if vehicle_counter:
            count_df = pd.DataFrame(list(vehicle_counter.items()), columns=["Vehicle Type", "Count"])
            st.bar_chart(count_df.set_index("Vehicle Type"))
        else:
            st.write("No vehicles counted yet.")
        if lane_vehicles:
            signals, totals = compute_signals(lane_vehicles)
            st.subheader("🚦 Lane counts (current frame) & signals")
            for lane in ["North","South","East","West"]:
                st.markdown(f"**{lane}** — total: {totals.get(lane,0)} — {signals.get(lane,'🔴 Red')}")
                counts = lane_vehicles.get(lane, {})
                if counts:
                    st.caption(", ".join([f"{k}:{v}" for k,v in counts.items()]))

# ---- MAIN ----
if start:
    vehicle_counter = {}
    seen_plates = set()
    tracker = SimpleTracker(max_distance=80, max_missed=20)

    # caching last detections so we can reuse them on skipped frames
    last_vehicle_res = None
    last_helmet_res = None
    last_plate_res = None
    frame_idx = 0

    # performance stats
    last_time = time.time()
    fps_smooth = 0.0

    if option == "Webcam":
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            start_t = time.time()
            annotated, lane_vehicles, last_vehicle_res, last_helmet_res, last_plate_res = process_frame_optimized(
                frame, vehicle_counter, seen_plates, tracker,
                last_vehicle_res, last_helmet_res, last_plate_res,
                frame_idx, frame_skip, scale_factor
            )
            frame_idx += 1

            # compute FPS
            now = time.time()
            dt = now - last_time if last_time else 0.001
            fps = 1.0 / (now - start_t) if (now - start_t)>0 else 0
            fps_smooth = 0.9*fps_smooth + 0.1*fps if fps_smooth else fps
            last_time = now

            # display
            frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            log_placeholder.empty()
            with log_placeholder.container():
                update_log_and_traffic(vehicle_counter, lane_vehicles)

            if show_perf:
                perf_placeholder.markdown(f"**FPS (smoothed):** {fps_smooth:.1f}  \n**Frame index:** {frame_idx}  \n**Frame skip:** {frame_skip}  \n**Scale:** {scale_factor}")

        cap.release()

    elif option == "Upload Video":
        if uploaded:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded.read())
            cap = cv2.VideoCapture("temp_video.mp4")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                start_t = time.time()
                annotated, lane_vehicles, last_vehicle_res, last_helmet_res, last_plate_res = process_frame_optimized(
                    frame, vehicle_counter, seen_plates, tracker,
                    last_vehicle_res, last_helmet_res, last_plate_res,
                    frame_idx, frame_skip, scale_factor
                )
                frame_idx += 1

                now = time.time()
                fps = 1.0 / (now - start_t) if (now - start_t)>0 else 0
                fps_smooth = 0.9*fps_smooth + 0.1*fps if fps_smooth else fps
                last_time = now

                frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                log_placeholder.empty()
                with log_placeholder.container():
                    update_log_and_traffic(vehicle_counter, lane_vehicles)

                if show_perf:
                    perf_placeholder.markdown(f"**FPS (smoothed):** {fps_smooth:.1f}  \n**Frame index:** {frame_idx}  \n**Frame skip:** {frame_skip}  \n**Scale:** {scale_factor}")

            cap.release()
        else:
            st.warning("Please upload a video file first.")

    elif option == "Test Image":
        if uploaded:
            img_path = os.path.join("images", uploaded.name)
            with open(img_path, "wb") as f:
                f.write(uploaded.read())
            frame = cv2.imread(img_path)
            annotated, lane_vehicles, last_vehicle_res, last_helmet_res, last_plate_res = process_frame_optimized(
                frame, vehicle_counter, seen_plates, tracker,
                last_vehicle_res, last_helmet_res, last_plate_res,
                frame_idx, frame_skip, scale_factor
            )
            frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result")
            log_placeholder.empty()
            with log_placeholder.container():
                update_log_and_traffic(vehicle_counter, lane_vehicles)
        else:
            st.warning("Please upload an image for testing.")

# ---- DOWNLOAD LOGS ----
if not log_df.empty:
    st.download_button("Download Logs (CSV)", log_df.to_csv(index=False).encode(), "violations.csv", "text/csv")




