import os
import pickle
import traceback
import time
import threading
import warnings
from datetime import datetime
from threading import Thread
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, Student, Teacher, Attendance, Schedule, Bunking
from pathlib import Path

# Suppress Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION
# ==========================================
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "sqlite:///attendance.db"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id): return User.query.get(int(user_id))

if not os.path.exists('embeddings'): os.makedirs('embeddings')
if not os.path.exists('static/bunk_proofs'): os.makedirs('static/bunk_proofs')
PICKLE_DB_PATH = 'embeddings/insight_db.pkl'

# ==========================================
# 2. AI SETUP (HIGH ACCURACY)
# ==========================================
app_face = None
known_faces_db = {}

try:
    from insightface.app import FaceAnalysis
    print("[INFO] Loading InsightFace (buffalo_l) - High Accuracy Mode...")
    # Using Large Model for better detection
    app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app_face.prepare(ctx_id=0, det_size=(640, 640))
    
    if os.path.exists(PICKLE_DB_PATH):
        with open(PICKLE_DB_PATH, 'rb') as f: known_faces_db = pickle.load(f)
    print(f"[INFO] AI Ready. Loaded {len(known_faces_db)} users.")
except Exception as e: print(f"AI Error: {e}")

# ==========================================
# 3. FAST CAMERA STREAM CLASS
# ==========================================
class WebcamStream:
    def __init__(self, src=0):
        import cv2
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self): return self.frame
    def stop(self): self.stopped = True; self.stream.release()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def load_pickle_db():
    if os.path.exists(PICKLE_DB_PATH):
        with open(PICKLE_DB_PATH, 'rb') as f: return pickle.load(f)
    return {}

def save_pickle_db(data):
    with open(PICKLE_DB_PATH, 'wb') as f: pickle.dump(data, f)

def check_pose(face, target_pose):
    try:
        kps = face.kps
        nose_x = kps[2][0]
        eye_center = (kps[0][0] + kps[1][0]) / 2
        eye_dist = kps[1][0] - kps[0][0]
        if eye_dist < 10: return False, 0.0
        
        ratio = (nose_x - eye_center) / eye_dist 
        
        is_valid = False
        # Strict thresholds for High Accuracy Model
        if target_pose == "CENTER" and abs(ratio) < 0.15: is_valid = True
        elif target_pose == "LEFT" and ratio < -0.20: is_valid = True
        elif target_pose == "RIGHT" and ratio > 0.20: is_valid = True
        return is_valid, ratio
    except: return False, 0.0

# --- CAPTURE WINDOW (REGISTRATION) ---
def start_capture_window(roll_no):
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    global app_face
    if app_face is None: 
        return False, "AI Model Not Loaded"

    # UPDATE IP HERE
    ip_url = "http://192.168.1.102:8080/video" 
    cap = cv2.VideoCapture(ip_url)
    
    if not cap.isOpened(): 
        print("[WARN] IP Cam failed, trying Laptop Webcam...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened(): 
        return False, "Cannot Connect to Camera"
    
    samples = []
    tasks = ["CENTER", "LEFT", "RIGHT", "CENTER", "CENTER"]
    current_step = 0
    
    while current_step < len(tasks):
        ret, frame = cap.read()
        if not ret: break
        
        target_pose = tasks[current_step]
        faces = app_face.get(frame)
        msg = f"Step {current_step+1}/5: Look {target_pose}"
        color = (0, 255, 255)
        
        if len(faces) > 0:
            face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            
            orig_h, orig_w = frame.shape[:2]
            disp_w = 800
            disp_scale = disp_w / orig_w
            disp_h = int(orig_h * disp_scale)
            
            valid_pose, ratio = check_pose(face, target_pose)
            
            if valid_pose:
                color = (0, 255, 0)
                emb = face.embedding / np.linalg.norm(face.embedding)
                samples.append(emb)
                current_step += 1
                cv2.waitKey(300)
            else:
                color = (0, 0, 255)
            
            # Landmarks & Info
            kps = face.kps.astype(int)
            for p in kps: cv2.circle(frame, (p[0], p[1]), 3, (0, 255, 255), -1)
            
            frame_disp = cv2.resize(frame, (disp_w, disp_h))
            box = (face.bbox * disp_scale).astype(int)
            cv2.rectangle(frame_disp, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame_disp, f"Acc: {ratio:.2f}", (box[0], box[3] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
             frame_disp = cv2.resize(frame, (800, 600))

        cv2.rectangle(frame_disp, (0, 0), (800, 50), (0,0,0), -1)
        cv2.putText(frame_disp, msg, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow("Face Registration (Server View)", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            cap.release()
            cv2.destroyAllWindows()
            return False, "Registration Cancelled"
            
    cap.release()
    cv2.destroyAllWindows()
    
    if len(samples) >= 5:
        try:
            faces_data = load_pickle_db()
            faces_data[roll_no] = samples # Saving Multi-Template List
            save_pickle_db(faces_data)
            global known_faces_db
            known_faces_db = faces_data
            return True, "Success"
        except Exception as e: 
            return False, f"Save Error: {str(e)}"
            
    return False, "Incomplete Capture"

# --- SURVEILLANCE WORKER (HIGH ACCURACY + MONITOR) ---
surveillance_active = False
current_lecture_info = {}
current_alerts = []
alert_lock = threading.Lock()

def surveillance_worker(camera_url):
    import cv2
    import numpy as np

    global surveillance_active, current_alerts, known_faces_db
    
    try:
        stream = WebcamStream(src=camera_url).start()
    except:
        stream = WebcamStream(src=0).start()

    detection_counts = {}
    
    while surveillance_active:
        frame = stream.read()
        if frame is None: 
            time.sleep(0.01)
            continue
            
        # 1024px for better detection
        frame_large = cv2.resize(frame, (1024, 768)) 
        faces = app_face.get(frame_large)
        
        for face in faces:
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            max_score = 0
            identity = "Unknown"
            color = (0, 0, 255)
            
            for roll, saved_data in known_faces_db.items():
                # Multi-Template Matching
                if isinstance(saved_data, list):
                    scores = [np.dot(emb, saved_emb) for saved_emb in saved_data]
                    best_score = max(scores)
                else:
                    best_score = np.dot(emb, saved_data)
                
                if best_score > max_score:
                    max_score = best_score
                    identity = roll
            
            # Threshold 0.50 for Buffalo_L
            if max_score > 0.50 and identity != "Unknown":
                color = (0, 255, 0)
                detection_counts[identity] = detection_counts.get(identity, 0) + 1
                
                if detection_counts[identity] >= 2: # 2 Consistent Frames
                    detection_counts[identity] = 0
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{identity}_{timestamp}.jpg"
                    upload_dir = Path("/tmp/bunk_proofs")
                    upload_dir.mkdir(parents=True, exist_ok=True)

                    filepath = upload_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    
                    with alert_lock:
                        if not any(a['roll'] == identity for a in current_alerts):
                            current_alerts.append({
                                'roll': identity, 'time': datetime.now().strftime("%H:%M"),
                                'image': f"/static/bunk_proofs/{filename}", 'score': int(max_score*100)
                            })

            # Draw on Monitor
            box = face.bbox.astype(int)
            cv2.rectangle(frame_large, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{identity} {int(max_score*100)}%"
            cv2.putText(frame_large, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display Window
        disp_frame = cv2.resize(frame_large, (800, 600))
        cv2.rectangle(disp_frame, (0, 0), (800, 30), (0,0,0), -1)
        cv2.putText(disp_frame, "CANTEEN SURVEILLANCE (LIVE MONITOR)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Surveillance Monitor", disp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    stream.stop()
    cv2.destroyAllWindows()

# ==========================================
# 5. ROUTES
# ==========================================

@app.route("/")
def home(): 
    return render_template("base.html", title="Home")

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('teacher_dashboard' if current_user.role == 'teacher' else 'student_dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and user.password == request.form.get('password') and user.role == request.form.get('role'):
            login_user(user)
            return redirect(url_for('teacher_dashboard' if user.role == 'teacher' else 'student_dashboard'))
        flash('Invalid Credentials!', 'error')
    return render_template("login.html", title="Login")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- TEACHER DASHBOARD ---
@app.route("/teacher_dashboard")
@login_required
def teacher_dashboard():
    if current_user.role != 'teacher': return redirect(url_for('login'))
    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')
    
    upcoming = Schedule.query.filter(Schedule.teacher_email == current_user.email, Schedule.date >= today).all()
    history = Schedule.query.filter(Schedule.teacher_email == current_user.email, Schedule.date < today).all()
    
    return render_template("teacher_dashboard.html", 
                           user=current_user, 
                           schedules=upcoming, 
                           history=history,
                           current_time=current_time,
                           is_live=surveillance_active)

# --- STUDENT DASHBOARD ---
@app.route("/student_dashboard")
@login_required
def student_dashboard():
    if current_user.role != 'student': return redirect(url_for('login'))
    
    student = Student.query.filter_by(email=current_user.email).first()
    if not student: return "Student record not found. Contact Admin."

    # 1. Fetch All Attendance Records
    attendance_records = Attendance.query.filter_by(roll_no=student.roll_no).order_by(Attendance.date.desc()).all()
    
    # 2. Fetch Bunking Records
    bunks = Bunking.query.filter_by(roll_no=student.roll_no).order_by(Bunking.date.desc()).all()
    
    # 3. Calculate Stats
    total_lectures = len(attendance_records)
    present_count = sum(1 for r in attendance_records if r.status == 'Present')
    
    # Total Absent (Normal Absent + Bunks)
    absent_count = sum(1 for r in attendance_records if r.status == 'Absent') 
    
    # Real Bunk Count (Only Surveillance Caught)
    real_bunk_count = len(bunks)
    
    attendance_percentage = 0
    if total_lectures > 0:
        attendance_percentage = round((present_count / total_lectures) * 100, 1)

    return render_template("student_dashboard.html", 
                           user=current_user, 
                           student=student, 
                           attendance_records=attendance_records,
                           total_lectures=total_lectures,
                           present_count=present_count,
                           absent_count=absent_count,    
                           bunk_count=real_bunk_count,
                           percentage=attendance_percentage,
                           bunks=bunks)

# --- NEW: STUDENT BUNK HISTORY PAGE ---
@app.route("/student_bunk_history")
@login_required
def student_bunk_history():
    if current_user.role != 'student': return redirect(url_for('login'))
    
    student = Student.query.filter_by(email=current_user.email).first()
    bunks = Bunking.query.filter_by(roll_no=student.roll_no).order_by(Bunking.date.desc(), Bunking.time.desc()).all()
    
    return render_template("student_bunk_history.html", bunks=bunks, student=student)

# --- NEW: STUDENT ATTENDANCE HISTORY PAGE ---
@app.route("/student_attendance_history")
@login_required
def student_attendance_history():
    if current_user.role != 'student': return redirect(url_for('login'))
    
    student = Student.query.filter_by(email=current_user.email).first()
    records = Attendance.query.filter_by(roll_no=student.roll_no).order_by(Attendance.date.desc(), Attendance.time.desc()).all()
    
    return render_template("student_attendance_history.html", records=records, student=student)

@app.route("/schedule_lecture")
@login_required
def schedule_lecture():
    edit_id = request.args.get('id')
    data = Schedule.query.get(edit_id) if edit_id else None
    return render_template("schedule_lecture.html", edit_data=data)

@app.route('/save_schedule', methods=['POST'])
@login_required
def save_schedule():
    try:
        data = request.json
        schedule_id = data.get('id')
        if 'id' in data: del data['id']
        
        start = data.get('start_time')
        end = data.get('end_time')
        date = data.get('date')
        
        if start >= end: return jsonify({"status": "error", "msg": "Invalid Time Range!"})

        clashes = Schedule.query.filter_by(date=date).all()
        for lec in clashes:
            if schedule_id and int(schedule_id) == lec.id: continue
            if (start < lec.end_time) and (end > lec.start_time):
                if lec.teacher_email == current_user.email: return jsonify({"status": "error", "msg": "Clash! You are busy."})
                if lec.course == data.get('course'): return jsonify({"status": "error", "msg": f"Clash! {data.get('course')} is busy."})
                if lec.classroom == data.get('classroom'): return jsonify({"status": "error", "msg": f"Clash! Room {data.get('classroom')} occupied."})

        if schedule_id:
            sch = Schedule.query.get(schedule_id)
            sch.course = data.get('course')
            sch.subject = data.get('subject')
            sch.classroom = data.get('classroom')
            sch.date = data.get('date')
            sch.start_time = data.get('start_time')
            sch.end_time = data.get('end_time')
            msg = "Schedule Updated!"
        else:
            new = Schedule(teacher_email=current_user.email, teacher_name=current_user.name, **data)
            db.session.add(new)
            msg = "Lecture Scheduled!"

        db.session.commit()
        return jsonify({"status": "success", "msg": msg})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

@app.route('/delete_schedule/<int:id>', methods=['POST'])
@login_required
def delete_schedule(id):
    Schedule.query.filter_by(id=id).delete()
    db.session.commit()
    return jsonify({"status": "success"})

@app.route('/trigger_capture', methods=['POST'])
def trigger_capture():
    try:
        success, msg = start_capture_window(request.json.get('roll_no'))
        if success: return jsonify({"status": "success", "msg": "Face Saved!"})
        else: return jsonify({"status": "error", "msg": msg})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

@app.route('/submit_student_details', methods=['POST'])
def submit_student_details():
    try:
        data = request.json
        if Student.query.filter_by(roll_no=data['roll_no']).first(): return jsonify({"status": "error", "msg": "Roll No Exists"})
        if User.query.filter_by(email=data['email']).first(): return jsonify({"status": "error", "msg": "Email Exists"})
        new_student = Student(name=data['name'], email=data['email'], roll_no=data['roll_no'], course=data['course'], phone=data.get('phone'))
        db.session.add(new_student)
        new_user = User(name=data['name'], email=data['email'], password=data['password'], role='student')
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "msg": f"Server Error: {str(e)}"})

@app.route('/submit_teacher_details', methods=['POST'])
def submit_teacher_details():
    try:
        data = request.json
        if User.query.filter_by(email=data['email']).first(): return jsonify({"status": "error", "msg": "Email Exists"})
        db.session.add(Teacher(name=data['name'], email=data['email'], phone=data.get('phone')))
        db.session.add(User(name=data['name'], email=data['email'], password=data['password'], role='teacher'))
        db.session.commit()
        return jsonify({"status": "success"})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

@app.route("/take_attendance")
@login_required
def take_attendance():
    course = request.args.get('course')
    students = Student.query.filter_by(course=course).all()
    today_str = datetime.now().strftime('%Y-%m-%d')
    existing_records = Attendance.query.filter_by(course=course, subject=request.args.get('subject'), date=today_str).all()
    attendance_map = {rec.roll_no: rec.status for rec in existing_records}
    
    # Fetch Bunkers to lock UI
    bunks = Bunking.query.filter_by(date=today_str).all()
    bunkers = [b.roll_no for b in bunks]
    
    return render_template("take_attendance.html", 
                           students=students, 
                           course=course, 
                           subject=request.args.get('subject'), 
                           date=today_str, 
                           attendance_map=attendance_map, 
                           bunkers=bunkers)

@app.route('/save_attendance_data', methods=['POST'])
@login_required
def save_attendance_data():
    try:
        data = request.json
        date = data.get('date')
        bunks = Bunking.query.filter_by(date=date).all()
        bunker_rolls = [b.roll_no for b in bunks]
        
        count_updated = 0
        for record in data.get('attendance_data'):
            final_status = record['status']
            if record['roll_no'] in bunker_rolls:
                final_status = 'Absent'
            
            existing = Attendance.query.filter_by(roll_no=record['roll_no'], subject=data.get('subject'), date=date).first()
            if existing:
                existing.status = final_status
                count_updated += 1
            else:
                new_att = Attendance(
                    roll_no=record['roll_no'], name=record['name'], subject=data.get('subject'), 
                    course=data.get('course'), date=date, time=datetime.now().strftime('%H:%M'), status=final_status
                )
                db.session.add(new_att)
        db.session.commit()
        return jsonify({"status": "success", "msg": f"Saved! (Updated: {count_updated})"})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

@app.route('/reset_attendance', methods=['POST'])
@login_required
def reset_attendance():
    try:
        data = request.json
        Attendance.query.filter_by(course=data.get('course'), subject=data.get('subject'), date=data.get('date')).delete()
        db.session.commit()
        return jsonify({"status": "success", "msg": "Reset Complete."})
    except Exception as e: return jsonify({"status": "error", "msg": str(e)})

@app.route('/start_lecture', methods=['POST'])
@login_required
def start_lecture():
    global surveillance_active, current_lecture_info
    if surveillance_active: return jsonify({"status": "error", "msg": "Running!"})
    data = request.json
    current_lecture_info = {"course": data.get('course'), "subject": data.get('subject'), "date": datetime.now().strftime('%Y-%m-%d'), "bunkers": []}
    surveillance_active = True
    cam_url = "http://192.168.1.102:8080/video" # Update IP
    t = threading.Thread(target=surveillance_worker, args=(cam_url,))
    t.daemon = True
    t.start()
    return jsonify({"status": "success"})

@app.route('/stop_lecture', methods=['POST'])
@login_required
def stop_lecture():
    global surveillance_active, current_lecture_info, current_alerts
    surveillance_active = False
    try:
        course = current_lecture_info.get('course')
        subject = current_lecture_info.get('subject')
        date = current_lecture_info.get('date')
        all_students = Student.query.filter_by(course=course).all()
        bunker_rolls = [b.roll_no for b in Bunking.query.filter_by(date=date).all()]
        for student in all_students:
            if not Attendance.query.filter_by(roll_no=student.roll_no, subject=subject, date=date).first():
                status = 'Absent' if student.roll_no in bunker_rolls else 'Present'
                db.session.add(Attendance(roll_no=student.roll_no, name=student.name, course=course, subject=subject, date=date, time=datetime.now().strftime('%H:%M'), status=status))
        db.session.commit()
    except Exception as e: print(f"[ERROR] {e}")
    with alert_lock: current_alerts = []
    return jsonify({"status": "success"})

@app.route('/get_alerts')
@login_required
def get_alerts(): return jsonify(current_alerts)

@app.route('/action_bunking', methods=['POST'])
@login_required
def action_bunking():
    global current_alerts
    data = request.json
    roll = data.get('roll_no')
    with alert_lock:
        if data.get('action') == 'mark':
            alert = next((x for x in current_alerts if x['roll'] == roll), None)
            if alert:
                db.session.add(Bunking(roll_no=roll, name=roll, date=datetime.now().strftime('%Y-%m-%d'), time=alert['time'], proof_image=alert['image']))
                db.session.commit()
        current_alerts = [x for x in current_alerts if x['roll'] != roll]
    return jsonify({"status": "success"})

@app.route("/student_register")
def student_register(): return render_template("student_register.html")
@app.route("/teacher_register")
def teacher_register(): return render_template("teacher_register.html")
@app.route("/demo")
def demo():
    return render_template("demo.html", title="Demo")

if __name__ == "__main__":
    with app.app_context(): db.create_all()
    app.run(debug=False, threaded=True)