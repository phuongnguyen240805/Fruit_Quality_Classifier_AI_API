import os
import base64
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use('Agg')  # QUAN TRỌNG: tắt GUI backend trước khi import pyplot
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import my_tf_mod  # Module AI đã được nâng cấp


# ─────────────────────────────────────────────────────
# Khởi tạo Flask App
# ─────────────────────────────────────────────────────
app = Flask(__name__)

# Bật CORS cho toàn bộ API routes
# origins: cho phép request từ Next.js dev server và production
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:3003",
            "http://127.0.0.1:3000",
        ]
    }
})


# ─────────────────────────────────────────────────────
# API MỚI: Kiểm tra sức khỏe server
# GET /api/health
# ─────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    """
    Endpoint đơn giản để Next.js kiểm tra Flask có đang chạy không.
    Next.js gọi endpoint này khi người dùng mở trang để hiển thị trạng thái kết nối.
    """
    return jsonify({
        "status": "ok",
        "message": "Flask AI Backend đang hoạt động",
        "models_loaded": True
    })


# ─────────────────────────────────────────────────────
# API MỚI: Dự đoán chất lượng nông sản + Grad-CAM
# POST /api/predict
# Body: multipart/form-data, key = "file" (ảnh)
# ─────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Endpoint chính để Next.js gửi ảnh và nhận kết quả phân tích AI.

    Luồng xử lý:
      1. Kiểm tra file có được gửi không
      2. Đọc bytes của file → tiền xử lý (resize 100x100, normalize)
      3. Chạy mô hình phân loại quả → fruit_dict {apple, banana, orange}
      4. Chạy mô hình đánh giá chất lượng → [fresh%, rotten%]
      5. Chạy Grad-CAM → heatmap base64 (dùng mô hình quality để visualize)
      6. Chuyển ảnh gốc sang base64 để Frontend preview
      7. Trả về JSON đầy đủ

    Response JSON:
    {
      "success": true,
      "fruit_predictions": {"apple": 95.2, "banana": 3.1, "orange": 1.7},
      "quality_predictions": {"fresh": 98.5, "rotten": 1.5},
      "best_fruit": "apple",
      "quality_status": "fresh",
      "original_image": "data:image/png;base64,...",
      "heatmap_image": "data:image/png;base64,..."
    }
    """
    # ── 1. Kiểm tra file đầu vào ──────────────────────
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "Không tìm thấy file trong request. Key phải là 'file'"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Tên file trống, vui lòng chọn ảnh"}), 400

    try:
        # ── 2. Đọc bytes và tiền xử lý ────────────────
        raw_bytes = file.read()
        org_arr, img_4d, pil_original = my_tf_mod.preprocess_bytes(raw_bytes)

        # Đọc crop_type từ request form (nếu có)
        requested_crop = request.form.get('crop_type', 'auto')

        # ── 3. Phân loại loại quả ─────────────────────
        fruit_dict = my_tf_mod.classify_fruit(
            img_4d=img_4d,
            pil_img=pil_original,
            filename=file.filename,
            requested_crop=requested_crop
        )

        # ── 4. Đánh giá chất lượng ────────────────────
        quality = my_tf_mod.check_rotten(img_4d, pil_img=pil_original)  # [fresh%, rotten%]

        # ── 5. Xác định kết quả cao nhất ──────────────
        best_fruit = max(fruit_dict, key=fruit_dict.get)
        quality_status = "fresh" if quality[0] >= quality[1] else "rotten"

        # ── 6. Sinh Grad-CAM heatmap ───────────────────
        # Dùng quality_model để hiển thị vùng pixel ảnh hưởng đến chất lượng
        # target_class_idx=0 → vùng CNN tập trung khi nhận diện "Fresh"
        heatmap_b64 = my_tf_mod.generate_gradcam(
            img_4d=img_4d,
            model=my_tf_mod.quality_model,
            target_class_idx=0,  # 0 = Fresh class
            alpha=0.45           # 45% heatmap, 55% ảnh gốc
        )

        # ── 7. Chuyển ảnh gốc sang base64 ─────────────
        original_b64 = my_tf_mod.img_to_base64(pil_original, max_size=(400, 400))

        # ── 8. Trả về JSON ─────────────────────────────
        return jsonify({
            "success": True,
            "fruit_predictions": fruit_dict,
            "quality_predictions": {
                "fresh": quality[0],
                "rotten": quality[1]
            },
            "best_fruit": best_fruit,
            "best_fruit_confidence": fruit_dict[best_fruit],
            "quality_status": quality_status,
            "quality_confidence": quality[0] if quality_status == "fresh" else quality[1],
            # Ảnh base64 (Frontend dùng trực tiếp trong <img src="...">)
            "original_image": f"data:image/png;base64,{original_b64}",
            "heatmap_image":  f"data:image/png;base64,{heatmap_b64}"
        })

    except Exception as e:
        # In traceback đầy đủ ra console để debug
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Lỗi xử lý ảnh: {str(e)}"
        }), 500


# ─────────────────────────────────────────────────────
# ROUTE
# ─────────────────────────────────────────────────────
@app.route('/')
def home():
    """Giao diện HTML cũ của Flask (giữ nguyên để tham khảo)"""
    return render_template('home.html')


@app.route('/Prediction', methods=['GET', 'POST'])
def pred():
    """
    Route HTML cũ — Giữ lại để không phá vỡ giao diện Flask cũ.
    THAY ĐỔI: Cập nhật gọi preprocess mới (nhận thêm tham số pil_original)
    """
    if request.method == 'POST':
        file = request.files['file']
        raw_bytes = file.read()
        org_arr, img_4d, pil_original = my_tf_mod.preprocess_bytes(raw_bytes)

        fruit_dict = my_tf_mod.classify_fruit(img_4d)
        rotten = my_tf_mod.check_rotten(img_4d)

        # Encode ảnh gốc sang base64 để hiển thị trong HTML cũ
        img_x = BytesIO()
        plt.imshow(org_arr / 255.0)
        plt.savefig(img_x, format='png')
        plt.close()
        img_x.seek(0)
        plot_url = base64.b64encode(img_x.getvalue()).decode('utf8')

        return render_template('Pred3.html', fruit_dict=fruit_dict, rotten=rotten, plot_url=plot_url)

    return render_template('home.html')


# ─────────────────────────────────────────────────────
# Khởi động server
# ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 Flask AI Backend đang khởi động...")
    print("   - Giao diện cũ:  http://127.0.0.1:5000/")
    print("   - API dự đoán:   POST http://127.0.0.1:5000/api/predict")
    print("   - Health check:  GET  http://127.0.0.1:5000/api/health\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
