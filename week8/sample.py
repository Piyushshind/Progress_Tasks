import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import mediapipe as mp
from deepface  import DeepFace
import logging
from ultralytics import YOLO


from Services.calculation_service import is_Eye_Blinking,is_Head_Moving,is_Mouth_Moving,is_Gaze_Moving,calculate_displacement,is_spoofed
from Services.audio_service import process_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOAD_FOLDER = "./Face_Service/temp_files/"

model = YOLO("./Face_Service/data/Anti-Spoof-Model-4.pt")
model.to('cpu')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()



    
def process_video_stream(image_path, video_path,audio_path,otp):
    image_embedding = extract_face_embedding(image_path)

    cap = cv2.VideoCapture(video_path)
    liveliness_score = 0
    total_valid_frames = 0
    verification_futures = []
    live_count_futures = []
    frames=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        audio_result = executor.submit(process_audio, number = otp, audio_file = audio_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h,w = frame.shape[:2]
            
            scale = min(1280 / w, 720 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            padded = np.zeros((720, 1280, 3), dtype=np.uint8)
            x_offset = (1280 - new_w) // 2
            y_offset = (720 - new_h) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            frame = padded

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)
            if result.multi_face_landmarks :
                if(len(frames)>31):
                    live_count_futures.append(executor.submit(is_spoofed,frame=frames,model=model))
                    frames=[]
                else:
                    frames.append(frame)
                total_valid_frames += 1
                landmarks = result.multi_face_landmarks

                # Check for rapid displacement
                is_moving_fast = check_displacement(landmarks, frame)
                if is_moving_fast:
                    metrics = {"eye_score": 0,"pitch_score": 0,"yaw_score": 0,"roll_score": 0,"mouth_score": 0,"gaze_score": 0}
                else:
                    metrics = process_landmarks(landmarks, frame)

                liveliness_score += (
                    metrics["eye_score"] +
                    metrics["pitch_score"] * 0.25 +
                    metrics["yaw_score"] * 0.25 +
                    metrics["roll_score"] * 0.25 +
                    metrics["gaze_score"] +
                    metrics["mouth_score"] * 0.25
                )

                if total_valid_frames % 60 == 0:
                    # face_points = get_coordinates(facial_landmarks=landmarks,frame=frame,points=[10,152,234,454])

                    detected_face = frame

                    verify_future = executor.submit(verify_face, image_path=image_embedding, detected_face=detected_face)
                    verification_futures.append(verify_future)
        cap.release()
    authenticity_count = get_total_live_count(live_count_futures=live_count_futures)
    authenticity_percentage = round((authenticity_count / total_valid_frames) * 100,2)
    verified_faces,average_match_percentage = get_average_match_percentage(verification_futures=verification_futures)
    liveliness_percentage =round((liveliness_score / total_valid_frames) * 100,2)
    audio_result = audio_result.result()
    otp_verified = audio_result["numberVerified"]

    return {
        "audio_result": audio_result,
        "total_valid_frames":total_valid_frames,
        "authenticity_percentage":authenticity_percentage,
        "liveliness_percentage": liveliness_percentage,
        "match_percentage": round(average_match_percentage,2),
        "verfied": bool(liveliness_percentage>30.0 and verified_faces>=1 and average_match_percentage>50.0 and authenticity_percentage>70.0 and otp_verified)
    }


def process_landmarks(facial_landmarks, frame):
    eye_score = check_Blinking(facial_landmarks, frame)
    pitch_score, yaw_score, roll_score = check_Head_Movement(facial_landmarks, frame)
    mouth_score = check_Mouth_Movement(facial_landmarks, frame)
    gaze_score = check_Gaze_Movement(facial_landmarks, frame)
    return {
        "eye_score": eye_score,
        "pitch_score": pitch_score,
        "yaw_score": yaw_score,
        "roll_score": roll_score,
        "mouth_score": mouth_score,
        "gaze_score": gaze_score
    }

def check_Blinking(facial_landmarks,frame):
    left_eye = get_coordinates(facial_landmarks=facial_landmarks,points=[362,384,385,386,263,374,380,381],frame=frame)
    right_eye = get_coordinates(facial_landmarks=facial_landmarks,points=[33,160,159,158,133,153,145,144],frame=frame)
    return is_Eye_Blinking(left_eye=left_eye,right_eye=right_eye)  

def check_Head_Movement(facial_landmarks,frame):
    head_points = get_coordinates(facial_landmarks=facial_landmarks,frame=frame,points=[234,454,4,152])
    result = is_Head_Moving(head=head_points)
    return result['pitch'],result['yaw'],result['roll']

def check_Mouth_Movement(facial_landmarks,frame):
    mouth_points = get_coordinates(facial_landmarks=facial_landmarks,frame=frame,points=[76,73,11,303,292,    403,15,179])
    return is_Mouth_Moving(mouth=mouth_points)

def check_Gaze_Movement(facial_landmarks,frame):
    left_eye_points = get_coordinates(facial_landmarks=facial_landmarks,frame=frame,points=[263, 249, 390, 373, 374, 380, 381, 382, 362,     466, 388, 387, 386, 385, 384, 398])
    left_eye_points = np.array([(x, y) for x, y, _ in left_eye_points])
    right_eye_points = get_coordinates(facial_landmarks=facial_landmarks,frame=frame,points=[33, 7, 163, 144, 145, 153, 154, 155, 133,     246, 161, 160, 159, 158, 157, 173])
    right_eye_points = np.array([(x, y) for x, y, _ in right_eye_points])

    height, width, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((height,width),np.uint8)
    return is_Gaze_Moving(left_eye_points=left_eye_points,right_eye_points=right_eye_points,mask=mask,gray=gray)

def get_coordinates(facial_landmarks,points,frame):
    frame_height, frame_width,_ = frame.shape
    coordinates =[]
    for facial_landmark in facial_landmarks:
        for idx in points:
            x = int(facial_landmark.landmark[idx].x * frame_width)
            y = int(facial_landmark.landmark[idx].y * frame_height)
            z = facial_landmark.landmark[idx].z
            coordinates.append((x,y,z))
    return coordinates

def check_displacement(facial_landmarks,frame):
    points=get_coordinates(facial_landmarks=facial_landmarks,frame=frame,points=[1])
    return calculate_displacement(points=points)

def verify_face(image_path, detected_face):
    try:
        verification = DeepFace.verify(
            img1_path=image_path,
            img2_path=detected_face, 
            model_name="Facenet512", 
            align= True,
            enforce_detection=False,
            threshold=0.7,
            detector_backend="mtcnn"
        )
        return verification
    except Exception as e:
        logging.info(f"Verification failed: {e}")
        raise e
        return None

def extract_face_embedding(image_path):
    try:
        # Extract faces from the image
        image_numArr = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512", 
            align= True,
            enforce_detection=False,
            detector_backend="mtcnn"
        )
        
        # If no faces are detected, return None
        if not image_numArr:
            logging.info("No faces detected in the image.")
            return None
        
        # If faces are detected, return the first face's embedding
        image_embedding = image_numArr[0]["embedding"]
        return image_embedding
    except Exception as e:
        logging.info(f"Error extracting face embeddings: {e}")
        return None
    
def get_average_match_percentage(verification_futures):
    match_percentages = []
    verified_faces = 0
    for verify_future in as_completed(verification_futures):
        verification_result = verify_future.result() 
        if verification_result:
            if verification_result['verified']:
                verified_faces+=1
            distance = verification_result['distance']
            model_threshold = verification_result['threshold']
            match_percentage = max(0, (1 - (distance / model_threshold)) * 100)
            match_percentages.append(match_percentage)
    return verified_faces,np.mean(match_percentages) if match_percentages else 0

def get_total_live_count(live_count_futures):
    live_count =0
    for live_count_future in as_completed(live_count_futures):
        live_count += live_count_future.result()
    return live_count 