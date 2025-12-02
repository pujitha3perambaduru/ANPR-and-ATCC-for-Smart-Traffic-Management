# 🚦 AI-Powered ANPR & Helmet Detection System

An AI-based smart traffic surveillance system integrating **Automatic Number Plate Recognition (ANPR)** and **Helmet & Triple Riding Detection** using **YOLOv8, OpenCV, and Tesseract OCR**. The system monitors traffic from videos, images, or live webcam feeds and logs traffic violations in real-time.

---

## **Project Features**

- **Vehicle Detection** – Real-time detection of cars, bikes, buses, and trucks with bounding boxes  
- **Helmet Detection** – Identifies Helmet ✅ and No Helmet ❌ violations  
- **Triple Riding Detection** – Highlights triple-riding violations with a red bounding box  
- **License Plate Recognition (ANPR)** – Recognizes license plate numbers (e.g., 3749JPI)  
- **Vehicle Counting (ATCC)** – Counts vehicles and categorizes them with bar charts  
- **Violation Logs** – Maintains structured logs for helmet and triple-riding violations  

---

## **Tech Stack**

- Python  
- YOLOv8 (Vehicle & Violation Detection)  
- OpenCV (Video & Image Processing)  
- Tesseract OCR (License Plate Recognition)  
- Streamlit (Dashboard & Visualization)  
- Pandas, NumPy  

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/your-username/AI-ANPR-Helmet-Detection.git
cd AI-ANPR-Helmet-Detection

2. Create a virtual environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies:

pip install -r requirements.txt


4. Download the pretrained YOLO models:

Helmet Detection: yolov8_helmet.pt

License Plate Recognition: plate_model.pt

5. Run the Streamlit app:

streamlit run app.py


Select Webcam or Upload video

Adjust Confidence Threshold slider

Start detection and monitor real-time outputs

Logs and violation snapshots are saved in the logs/ folder, and you can download the violation CSV file.

6. Folder Structure

AI-ANPR-Helmet-Detection/
│
├─ app.py                  # Main Streamlit app
├─ yolov8_helmet.pt        # Pretrained YOLOv8 Helmet Detection model
├─ plate_model.pt          # Pretrained License Plate Recognition model
├─ logs/                   # Folder to store violation snapshots
├─ temp/                   # (Optional) for storing uploaded videos temporarily
├─ requirements.txt        # Python dependencies
└─ README.md               # Project documentation


7. Key Outcomes

Hands-on experience in real-time computer vision pipelines

Strengthened skills in Python, OpenCV, YOLOv8, Streamlit, and AI deployment

Built a functional prototype suitable for smart city traffic monitoring
