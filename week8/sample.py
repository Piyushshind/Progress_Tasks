import os
import time
import logging
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from Services.video_service import process_video_stream
from Services.audio_service import process_audio

logging.basicConfig(level=logging.INFO)

face_controller = Blueprint('face_controller', __name__)

UPLOAD_FOLDER = ".\Face_Service\\temp_files\\"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@face_controller.route('/receive-video', methods=['POST'])
def receive_video():
    try:
        if 'file' not in request.files or 'image' not in request.files:
            logging.error("Missing file or image part in the request")
            return jsonify({"error": "File and image are required"}), 400
        if 'audio' not in request.files or 'otp' not in request.form:
            logging.error("Missing audio or otp part in the request")
            return jsonify({"error": "audio and otp are required"}), 400

        video_file = request.files['file']
        image_file = request.files['image']
        audio_file = request.files['audio']
        number = request.form.get("otp")
        
        if video_file.filename == '' or image_file.filename == '':
            logging.error("No file or image selected for upload")
            return jsonify({"error": "Both video and image must be selected"}), 400
        if audio_file.filename == '':
            logging.error("No file or image selected for upload")
            return jsonify({"error": "Both audio and otp must be selected"}), 400

        audio_filename = secure_filename(audio_file.filename)  # Ensure filename is valid
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

        video_path = UPLOAD_FOLDER+ 'VideoRecorded.mp4'
        image_path = UPLOAD_FOLDER+ 'actualImage.jpg'
        # audio_path = UPLOAD_FOLDER, secure_filename(audio_file.filename)
        video_file.save(video_path)
        image_file.save(image_path)
        audio_file.save(audio_path)

        start_time = time.time()

        video_result = process_video_stream(video_path=video_path, image_path=image_path,audio_path=audio_path,otp=number)

        end_time = time.time()

        # os.remove(video_path)
        # os.remove(image_path)
        # os.remove(audio_path)
        
        return jsonify({
            "result": video_result,
            "processing_time": end_time - start_time
        }), 200

    except Exception as e:
        logging.exception("Error processing video")
        return jsonify({'error': str(e)}), 500

@face_controller.route('/receive-audio', methods=['POST'])
def receive_audio():
    try:
        if 'audio' not in request.files or 'otp' not in request.form:
            logging.error("Missing audio or otp part in the request")
            return jsonify({"error": "audio and otp are required"}), 400

        audio_file = request.files['audio']

        number = request.form.get("otp")

        if audio_file.filename == '':
            logging.error("No file or image selected for upload")
            return jsonify({"error": "Both audio and otp must be selected"}), 400

        audio_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
        audio_file.save(audio_path)

        start_time = time.time()

        audio_result = process_audio(number=number,audio_path=audio_path)

        end_time = time.time()

        os.remove(audio_path)

        return jsonify({
            "result": audio_result,
            "processing_time": end_time - start_time
        }), 200

    except Exception as e:
        logging.exception("Error processing audio")
        return jsonify({'error': str(e)}), 500
