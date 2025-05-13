import os
import json
import tempfile
import shutil
import numpy as np
import torchaudio
import pymongo
from flask import Flask, request, render_template, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
from speechbrain.pretrained import SpeakerRecognition

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CACHE_FOLDER'] = 'audio_cache'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'your_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

connection_string = "mongodb+srv://szluk133:sncuong2003@cluster0.dggxi.mongodb.net/"
client = pymongo.MongoClient(connection_string)
db = client['voice_embedding_db']
collection = db['voice_samples']

# Tải mô hình ECAPA-TDNN
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

def get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embedding = model.encode_batch(signal).squeeze().tolist()
    return embedding

# Hàm tính toán độ tương đồng cosine
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Hàm tìm file âm thanh tương tự nhất
def find_similar_audio(embedding, count=3):
    # Lấy tất cả các embedding từ cơ sở dữ liệu
    all_documents = collection.find()
    
    # Tính toán độ tương đồng với mỗi embedding
    similarities = []
    for doc in all_documents:
        db_embedding = doc['embedding']
        similarity = cosine_similarity(embedding, db_embedding)
        similarities.append({
            'file_path': doc['file_path'],
            'duration': doc['duration'],
            'similarity': similarity
        })
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:count]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        input_cache_path = os.path.join(app.config['CACHE_FOLDER'], f"input_{filename}")
        shutil.copy2(filepath, input_cache_path)
        session['input_filename'] = f"input_{filename}"

        try:
            embedding = get_embedding(filepath)
            similar_files = find_similar_audio(embedding)

            results = []
            for item in similar_files:
                file_path = item['file_path']
                file_name = os.path.basename(file_path)
                cache_path = os.path.join(app.config['CACHE_FOLDER'], file_name)
                try:
                    shutil.copy2(file_path, cache_path)
                except Exception as e:
                    print(f"Lỗi khi sao chép file: {str(e)}")
                
                results.append({
                    'file_path': file_path,
                    'file_name': file_name,
                    'duration': item['duration'],
                    'similarity': item['similarity'],
                    'audio_url': f"/audio/{file_name}"
                })
            return render_template('index.html', 
                results=results, 
                input_filename=session.get('input_filename'),
                input_audio_url=f"/audio/{session.get('input_filename')}")

        except Exception as e:
            return f"Lỗi khi xử lý file: {str(e)}"
    return redirect(request.url)

@app.route('/audio/<filename>')
def serve_audio(filename):
    cache_path = os.path.join(app.config['CACHE_FOLDER'], filename)
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype='audio/wav')

    if filename.startswith('input_'):
        original_filename = filename[6:]
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        if os.path.exists(upload_path):
            try:
                shutil.copy2(upload_path, cache_path)
                return send_file(cache_path, mimetype='audio/wav')
            except Exception as e:
                return f"Lỗi khi truy cập file: {str(e)}", 500
    result = collection.find_one({'file_path': {'$regex': filename}})

    if result:
        original_path = result['file_path']
        if os.path.exists(original_path):
            try:
                shutil.copy2(original_path, cache_path)
                return send_file(cache_path, mimetype='audio/wav')
            except Exception as e:
                return f"Lỗi khi truy cập file: {str(e)}", 500
        else:
            return "File không tồn tại trên hệ thống", 404
    else:
        return "Không tìm thấy file âm thanh trong cơ sở dữ liệu", 404

if __name__ == '__main__':
    app.run(debug=True)