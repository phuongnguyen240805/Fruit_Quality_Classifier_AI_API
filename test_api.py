"""
test_api.py — Script kiểm tra Flask AI API (ASCII safe)
======================================================
Chạy: python test_api.py
Yêu cầu: Flask server đang chạy tại http://127.0.0.1:5000
"""

import requests
import json
import os
import sys

BASE_URL = "http://127.0.0.1:5000"

# Màu terminal
GREEN  = ""
RED    = ""
YELLOW = ""
CYAN   = ""
RESET  = ""
BOLD   = ""

def print_header(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def print_ok(msg):
    print(f"  [OK] {msg}")

def print_fail(msg):
    print(f"  [FAIL] {msg}")

def print_info(msg):
    print(f"  [INFO] {msg}")


# ─────────────────────────────────────────────
# TEST 1: Health Check
# ─────────────────────────────────────────────
def test_health():
    print_header("TEST 1: Health Check")
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print_ok(f"Status: {resp.status_code} OK")
            print_ok(f"Message: {data.get('message')}")
            print_ok(f"Models loaded: {data.get('models_loaded')}")
            return True
        else:
            print_fail(f"Status: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_fail("Khong ket noi duoc server! Hay chac chan Flask dang chay.")
        return False


# ─────────────────────────────────────────────
# TEST 2: Gửi ảnh mẫu từ thư mục static
# ─────────────────────────────────────────────
def test_predict_with_file(image_path):
    print_header(f"TEST 2: Predict — {os.path.basename(image_path)}")

    if not os.path.exists(image_path):
        print_fail(f"File khong ton tai: {image_path}")
        return False

    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            resp = requests.post(f"{BASE_URL}/api/predict", files=files, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            if data.get('success'):
                print_ok(f"HTTP Status: {resp.status_code}")
                print_ok(f"Loai qua nhan dien: {data['best_fruit'].upper()} ({data['best_fruit_confidence']:.2f}%)")
                print_ok(f"Chat luong: {data['quality_status'].upper()} ({data['quality_confidence']:.2f}%)")
                print_info("Phan loai chi tiet:")
                for fruit, prob in data['fruit_predictions'].items():
                    bar = '#' * int(prob / 5)
                    print(f"    {fruit:8s}: {bar:<20} {prob:.2f}%")
                print_info("Chat luong chi tiet:")
                fresh  = data['quality_predictions']['fresh']
                rotten = data['quality_predictions']['rotten']
                print(f"    {'fresh':8s}: {'#' * int(fresh/5):<20} {fresh:.2f}%")
                print(f"    {'rotten':8s}: {'#' * int(rotten/5):<20} {rotten:.2f}%")

                # Kiểm tra heatmap có được trả về không
                has_heatmap   = 'heatmap_image' in data and data['heatmap_image'].startswith('data:image')
                has_original  = 'original_image' in data and data['original_image'].startswith('data:image')
                heatmap_len   = len(data.get('heatmap_image', ''))
                print_ok(f"Anh goc base64:    {'CO' if has_original else 'KHONG'}")
                print_ok(f"Heatmap Grad-CAM:  {'CO' if has_heatmap else 'KHONG'} (do dai: {heatmap_len} chars)")
                return True
            else:
                print_fail(f"API tra ve loi: {data.get('error')}")
                return False
        else:
            print_fail(f"HTTP Status: {resp.status_code}")
            print_fail(f"Response: {resp.text[:300]}")
            return False

    except Exception as e:
        print_fail(f"Loi: {e}")
        return False


# ─────────────────────────────────────────────
# TEST 3: Gửi request không có file
# ─────────────────────────────────────────────
def test_no_file():
    print_header("TEST 3: Gui request khong co file (kiem tra xu ly loi)")
    try:
        resp = requests.post(f"{BASE_URL}/api/predict", timeout=5)
        data = resp.json()
        if resp.status_code == 400 and not data.get('success'):
            print_ok(f"Server tra ve loi dung: {data.get('error')}")
            return True
        else:
            print_fail(f"Ky vong status 400, nhan duoc {resp.status_code}")
            return False
    except Exception as e:
        print_fail(f"Loi: {e}")
        return False


# ─────────────────────────────────────────────
# Chạy tất cả tests
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\nBat dau kiem tra Flask AI Backend...")
    print(f"   Server: {BASE_URL}")

    results = []

    # Test 1: Health check
    results.append(test_health())

    if not results[0]:
        print("\nServer chua chay, dung test.")
        print("   Hay kiem tra Docker container.")
        sys.exit(1)

    # Test 2: Gửi ảnh từ static/ nếu có
    static_images = [
        "static/fr1.jpg",
        "static/fr11.jpg",
        "static/bk11.jpg",
    ]
    tested_image = False
    for img_path in static_images:
        if os.path.exists(img_path):
            results.append(test_predict_with_file(img_path))
            tested_image = True
            break  # Test 1 ảnh là đủ

    if not tested_image:
        print_header("TEST 2: Predict")
        print_info("Khong tim thay anh mau trong static/. Bo qua test nay.")

    # Test 3: Request không có file
    results.append(test_no_file())

    # ── Tổng kết ─────────────────────────────
    passed = sum(1 for r in results if r)
    total  = len(results)
    print(f"\n{'='*50}")
    print(f"  KET QUA: {passed}/{total} tests passed")
    if passed == total:
        print("  🎉 TẤT CẢ TESTS ĐỀU THÀNH CÔNG!")
    else:
        print(f"  ⚠️  Co {total - passed} test(s) that bai")
    print(f"{'='*50}\n")
