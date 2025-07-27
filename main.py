from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
import uuid
from skimage.metrics import structural_similarity as ssim

app = FastAPI()

# 允许跨域（前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件目录
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 图像比较函数
def compare_images(standard_img, student_img):
    # ✅ 自动 resize 学生图为标准图尺寸
    student_img = cv2.resize(student_img, (standard_img.shape[1], standard_img.shape[0]))

    # 转灰度图
    grayA = cv2.cvtColor(standard_img, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)

    # 结构相似度
    score, diff = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # 找不同区域
    thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    difference_boxes = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w * h > 150:  # 忽略小区域
            difference_boxes.append({"x": x, "y": y, "w": w, "h": h})
            cv2.rectangle(student_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return {
        "similarity": float(score),
        "issues_count": len(difference_boxes),
        "difference_boxes": difference_boxes,
        "marked_image_np": student_img
    }

# 上传接口
@app.post("/compare")
async def compare(standard: UploadFile = File(...), student: UploadFile = File(...)):
    try:
        standard_bytes = await standard.read()
        student_bytes = await student.read()

        # 解码图像
        np_standard = np.frombuffer(standard_bytes, np.uint8)
        np_student = np.frombuffer(student_bytes, np.uint8)
        standard_img = cv2.imdecode(np_standard, cv2.IMREAD_COLOR)
        student_img = cv2.imdecode(np_student, cv2.IMREAD_COLOR)

        if standard_img is None or student_img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image(s)"})

        result = compare_images(standard_img, student_img)

        # 保存有标记的图像
        save_name = f"{uuid.uuid4().hex}.jpg"
        save_path = f"static/{save_name}"
        cv2.imwrite(save_path, result["marked_image_np"])

        return {
            "similarity": result["similarity"],
            "issues_count": result["issues_count"],
            "difference_boxes": result["difference_boxes"],
            "marked_image": f"/static/{save_name}"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
