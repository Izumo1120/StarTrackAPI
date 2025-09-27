# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

# CORS設定: Flutterアプリなどからのアクセスを許可します
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # --- 1. 画像をメモリ上で読み込み & デコード ---
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # カラー画像も必要なら

        if img_gray is None:
            return JSONResponse(status_code=400, content={"error": "画像の読み込みに失敗しました。"})

        # --- 2. ユーザー提供の画像処理ロジック ---
        _, img_binary = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = img_gray.shape[:2]
        
        grid_cols = 4
        max_seconds = 10.0
        offset_seconds = 2.0

        def y_to_seconds_from_bottom(y_pixel):
            return round((1 - y_pixel / height) * max_seconds + offset_seconds, 2)

        detected_stars = []
        min_area = 10

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    
                    col_index = min(int(cX * grid_cols / width), grid_cols - 1)
                    
                    star_data = {
                        "y_pixel_for_sort": cY, # ソート用の一時キー
                        "x": float(col_index),
                        "y": round(cY / height, 2),
                        "soundId": f"button{col_index + 1}",
                        "timing": y_to_seconds_from_bottom(cY)
                    }
                    detected_stars.append(star_data)

        # Y座標が大きい順（画像の下から上へ）にソート
        results_sorted = sorted(detected_stars, key=lambda s: s['y_pixel_for_sort'], reverse=True)

        # 最終的なJSONリストを作成（ソート用キーを削除）
        final_star_list = []
        for star in results_sorted:
            del star['y_pixel_for_sort']
            final_star_list.append(star)

        # --- 3. 指定された形式でJSONを返す ---
        return {"stars": final_star_list}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})