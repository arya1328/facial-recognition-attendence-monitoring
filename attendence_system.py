import cv2
import numpy as np
import os
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk
import mediapipe as mp
from scipy.ndimage import gaussian_filter
import imutils
from sklearn.preprocessing import normalize

class AttendanceSystem:
    def __init__(self):
        # Initialize face recognizer and additional components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.known_faces = []
        self.known_names = []
        self.attendance_list = []
        self.load_known_faces()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Smart Attendance System")
        self.root.geometry("800x600")
        self.root.configure(bg="#2C3E50")
        
        # Style
        style = ttk.Style()
        style.configure("TButton", padding=10, font=('Helvetica', 12))
        
        # Header
        header = tk.Label(self.root, text="OpenCV Attendance System",
                         font=('Helvetica', 24, 'bold'), bg="#2C3E50", fg="white")
        header.pack(pady=20)
        
        # Buttons Frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Buttons with icons
        ttk.Button(button_frame, text="➕ Add New Student", 
                  command=self.add_new_student).pack(pady=10)
        ttk.Button(button_frame, text="📸 Take Attendance", 
                  command=self.take_attendance).pack(pady=10)
        ttk.Button(button_frame, text="📄 Generate Report", 
                  command=self.generate_report).pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(self.root, text="Ready", 
                                   bg="#2C3E50", fg="white", font=('Helvetica', 10))
        self.status_label.pack(pady=10)
        
        self.root.mainloop()
    
    def load_known_faces(self):
        if not os.path.exists('students'):
            os.makedirs('students')
            return
        
        faces = []
        labels = []
        label_id = 0
        
        for file in os.listdir('students'):
            if file.endswith('.jpg'):
                image_path = os.path.join('students', file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    continue
                
                faces_detected = self.face_cascade.detectMultiScale(image, 1.3, 5)
                
                if len(faces_detected) > 0:
                    (x, y, w, h) = faces_detected[0]
                    face_roi = image[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (100, 100))
                    faces.append(face_roi)
                    labels.append(label_id)
                    self.known_names.append(file.split('.')[0])
                    label_id += 1
        
        if faces:
            self.recognizer.train(faces, np.array(labels))
    
    def process_face_image(self, image):
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get facial landmarks
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            height, width = image.shape[:2]
            landmarks = results.multi_face_landmarks[0]
            
            # Get face alignment points
            left_eye = np.mean([(landmarks.landmark[33].x * width,
                                landmarks.landmark[33].y * height),
                               (landmarks.landmark[133].x * width,
                                landmarks.landmark[133].y * height)], axis=0)
            right_eye = np.mean([(landmarks.landmark[362].x * width,
                                 landmarks.landmark[362].y * height),
                                (landmarks.landmark[263].x * width,
                                 landmarks.landmark[263].y * height)], axis=0)
            
            # Calculate angle for alignment
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Rotate image
            rotated = imutils.rotate(image, angle)
            
            # Apply preprocessing
            gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = gaussian_filter(gray, sigma=1)
            
            return gray
        return None

    def add_new_student(self):
        name = simpledialog.askstring("Input", "Enter student name:")
        if name:
            cap = cv2.VideoCapture(0)
            faces_captured = []
            self.status_label.config(text="Capturing... Press SPACE to capture 5 different angles")
            
            while len(faces_captured) < 5:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                # Process frame
                processed_face = self.process_face_image(frame)
                if processed_face is not None:
                    # Draw face mesh
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing.draw_landmarks(
                        frame,
                        self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks[0],
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
                    
                    cv2.putText(frame, f"Captured: {len(faces_captured)}/5", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Capture Face - Press SPACE to capture", frame)
                    
                    if cv2.waitKey(1) & 0xFF == 32:  # SPACE key
                        faces_captured.append(processed_face)
                        self.status_label.config(text=f"Captured {len(faces_captured)}/5 angles")
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            if len(faces_captured) == 5:
                # Save processed faces
                for i, face in enumerate(faces_captured):
                    cv2.imwrite(f"students/{name}_angle{i}.jpg", face)
                
                # Train recognizer
                label_id = len(self.known_names)
                self.known_names.append(name)
                labels = [label_id] * len(faces_captured)
                
                try:
                    self.recognizer.update(faces_captured, np.array(labels))
                except:
                    self.recognizer.train(faces_captured, np.array(labels))
                
                self.status_label.config(text="Student added successfully!")
                messagebox.showinfo("Success", "Student added successfully with multiple angles!")
            else:
                self.status_label.config(text="Registration cancelled")
    
    def take_attendance(self):
        if not self.known_names:
            messagebox.showwarning("Warning", "No students registered yet!")
            return
            
        cap = cv2.VideoCapture(0)
        self.attendance_list = []
        face_counts = {name: 0 for name in self.known_names}
        required_detections = 10
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_face = self.process_face_image(frame)
            if processed_face is not None:
                try:
                    # Get multiple predictions
                    predictions = []
                    for i in range(3):  # Get 3 predictions for each frame
                        label, confidence = self.recognizer.predict(processed_face)
                        predictions.append((label, confidence))
                    
                    # Use majority voting
                    labels, confidences = zip(*predictions)
                    label = max(set(labels), key=labels.count)
                    confidence = np.mean([conf for l, conf in predictions if l == label])
                    
                    if confidence < 60:  # Stricter threshold
                        name = self.known_names[label]
                        face_counts[name] += 1
                        
                        if face_counts[name] >= required_detections and name not in self.attendance_list:
                            self.attendance_list.append(name)
                        
                        # Display recognition info
                        confidence_display = int(100 - confidence)
                        color = (0, 255, 0) if confidence_display > 50 else (0, 255, 255)
                        cv2.putText(frame, f"{name} ({confidence_display}%)", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.9, color, 2)
                        cv2.putText(frame, f"Progress: {face_counts[name]}/{required_detections}",
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, color, 2)
                    
                except Exception as e:
                    print(f"Recognition error: {str(e)}")
            
            cv2.imshow('Attendance (Press Q to quit)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.status_label.config(text=f"Attendance taken for {len(self.attendance_list)} students")

    def generate_report(self):
        if not self.attendance_list:
            messagebox.showwarning("Warning", "No attendance data to generate report!")
            return
        
        # Ask user where to save the PDF
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if not file_path:
            return
            
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Add header
            pdf.set_font("Arial", 'B', 24)
            pdf.cell(190, 20, "Attendance Report", ln=True, align='C')
            
            # Add date and time
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(190, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
            pdf.cell(190, 10, f"Time: {datetime.now().strftime('%H:%M:%S')}", ln=True)
            pdf.ln(10)
            
            # Create table header
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(20, 10, "Sr.No.", border=1)
            pdf.cell(90, 10, "Student Name", border=1)
            pdf.cell(80, 10, "Status", border=1)
            pdf.ln()
            
            # Add student data
            pdf.set_font("Arial", '', 12)
            for i, name in enumerate(self.known_names, 1):
                pdf.cell(20, 10, str(i), border=1)
                pdf.cell(90, 10, name, border=1)
                status = "Present" if name in self.attendance_list else "Absent"
                pdf.cell(80, 10, status, border=1)
                pdf.ln()
            
            # Add summary
            pdf.ln(10)
            pdf.set_font("Arial", 'B', 12)
            present_count = len(self.attendance_list)
            total_count = len(self.known_names)
            pdf.cell(190, 10, f"Summary: {present_count} present out of {total_count} students", ln=True)
            
            # Save PDF
            pdf.output(file_path)
            self.status_label.config(text=f"Report saved: {file_path}")
            messagebox.showinfo("Success", f"Report saved successfully at:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

if __name__ == "__main__":
    AttendanceSystem()