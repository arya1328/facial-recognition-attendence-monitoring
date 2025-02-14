import cv2
import numpy as np
import os
from datetime import datetime
from fpdf import FPDF
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from PIL import Image, ImageTk

class AttendanceSystem:
    def __init__(self):
        # Initialize face recognizer and additional components
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
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
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                cv2.putText(frame, f"Captured: {len(faces_captured)}/5", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Capture Face - Press SPACE to capture", frame)
                
                if cv2.waitKey(1) & 0xFF == 32:  # SPACE key
                    if len(faces) == 1:  # Ensure only one face is detected
                        (x, y, w, h) = faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (100, 100))
                        faces_captured.append(face_roi)
                        self.status_label.config(text=f"Captured {len(faces_captured)}/5 angles")
                    else:
                        messagebox.showwarning("Warning", "Please ensure only one face is visible")
            
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
        recognized_names = set()  # Track currently recognized names
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                
                try:
                    label, confidence = self.recognizer.predict(face_roi)
                    
                    if confidence < 60:  # Stricter threshold
                        name = self.known_names[label]
                        face_counts[name] += 1
                        
                        if face_counts[name] >= required_detections:
                            if name not in self.attendance_list:
                                self.attendance_list.append(name)
                            recognized_names.add(name)
                        
                        # Display only the name without angles
                        if name in recognized_names:
                            cv2.putText(frame, name, 
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"Recognition error: {str(e)}")
            
            # Display total recognized count
            cv2.putText(frame, f"Recognized: {len(self.attendance_list)}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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