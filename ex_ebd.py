# import os
# import torchaudio
# import pymongo
# import json
# import torch
# import numpy as np
# from speechbrain.pretrained import SpeakerRecognition

# # Cấu hình kết nối MongoDB Atlas
# connection_string = "mongodb+srv://szluk133:sncuong2003@cluster0.dggxi.mongodb.net/"
# client = pymongo.MongoClient(connection_string)

# # Chọn hoặc tạo database và collection
# db = client['nhom_5']
# collection = db['voice_segments']

# # Load mô hình ECAPA-TDNN
# model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# # Hàm để chia file âm thanh thành các đoạn 1 giây với overlap 50%
# def split_audio_sliding_window(signal, sample_rate, window_size=1.0, overlap=0.5):
#     """
#     Chia tín hiệu âm thanh thành các đoạn với sliding window
    
#     Tham số:
#     - signal: tensor âm thanh đầu vào
#     - sample_rate: tần số lấy mẫu
#     - window_size: kích thước cửa sổ theo giây (mặc định 1 giây)
#     - overlap: độ chồng lấp giữa các đoạn (mặc định 0.5 = 50%)
    
#     Trả về:
#     - danh sách các đoạn âm thanh
#     """
#     # Chuyển window_size và overlap thành số mẫu
#     window_samples = int(window_size * sample_rate)
#     step_samples = int(window_samples * (1 - overlap))
    
#     # Lấy kích thước của signal
#     signal_length = signal.shape[1]
    
#     # Khởi tạo danh sách để lưu các đoạn
#     segments = []
    
#     # Lặp qua signal với kích thước cửa sổ và bước nhảy
#     for start in range(0, signal_length - window_samples + 1, step_samples):
#         end = start + window_samples
#         segment = signal[:, start:end]
#         segments.append(segment)
    
#     return segments

# # Hàm trích xuất embedding cho một đoạn âm thanh
# def get_embedding(audio_segment):
#     embedding = model.encode_batch(audio_segment).squeeze().tolist()
#     return embedding

# # Hàm lưu vào MongoDB Atlas
# def insert_segment_embedding(file_path, segment_index, start_time, embedding):
#     document = {
#         "file_path": file_path,
#         "segment_index": segment_index,
#         "start_time": start_time,
#         "window_size": 1.0,  # 1 giây
#         "overlap": 0.5,      # 50% overlap
#         "embedding": embedding
#     }
#     collection.insert_one(document)

# # Duyệt tất cả file trong thư mục
# def process_directory(folder_path):
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".wav"):  # Chỉ xử lý file WAV
#             file_path = os.path.join(folder_path, filename)
            
#             # Đọc file âm thanh
#             signal, sample_rate = torchaudio.load(file_path)
            
#             # Chia file thành các đoạn 1 giây với overlap 50%
#             segments = split_audio_sliding_window(signal, sample_rate)
            
#             print(f"Đang xử lý {filename}: Chia thành {len(segments)} đoạn")
            
#             # Xử lý từng đoạn
#             for i, segment in enumerate(segments):
#                 # Tính thời gian bắt đầu của đoạn
#                 start_time = i * 0.5  # Mỗi đoạn cách nhau 0.5 giây với overlap 50%
                
#                 # Trích xuất embedding
#                 embedding = get_embedding(segment)
                
#                 # Lưu vào MongoDB
#                 insert_segment_embedding(file_path, i, start_time, embedding)
            
#             print(f"✅ Đã xử lý và lưu {len(segments)} đoạn từ {filename} vào MongoDB Atlas")

# # Chạy chương trình
# if __name__ == "__main__":
#     folder = "E:/NAM_4_KY_2_2025/CSDL_ĐPT/filtered_audio"  # Thư mục chứa file âm thanh
#     process_directory(folder)
#     print("xong")


import os
import torchaudio
import pymongo
import json
from speechbrain.pretrained import SpeakerRecognition

# Cấu hình kết nối MongoDB Atlas
connection_string = "mongodb+srv://szluk133:sncuong2003@cluster0.dggxi.mongodb.net/"
client = pymongo.MongoClient(connection_string)

# Chọn hoặc tạo database và collection
db = client['voice_embedding_db']
collection = db['voice_samples']

# Load mô hình ECAPA-TDNN
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# Hàm trích xuất embedding
def get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embedding = model.encode_batch(signal).squeeze().tolist()
    return embedding

# Hàm lưu vào MongoDB Atlas
def insert_embedding(file_path, duration, embedding):
    document = {
        "file_path": file_path,
        "duration": duration,
        "embedding": embedding
    }
    collection.insert_one(document)

# Duyệt tất cả file trong thư mục
def process_directory(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Chỉ xử lý file WAV
            file_path = os.path.join(folder_path, filename)
            duration = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
            embedding = get_embedding(file_path)
            insert_embedding(file_path, duration, embedding)
            print(f"✅ Đã lưu {filename} vào MongoDB Atlas")

# Chạy chương trình
if __name__ == "__main__":
    folder = "E:/NAM_4_KY_2_2025/CSDL_ĐPT/filtered_audio"  # Thư mục chứa file âm thanh
    process_directory(folder)
    print("🎯 Hoàn thành!")