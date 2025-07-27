from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
from PIL import Image
import io
from skimage.metrics import structural_similarity as compare_ssim
import base64
import os
import traceback

app = FastAPI()


def read_image(uploaded_bytes):
    """从上传的字节数据读取并转换为 OpenCV BGR 图像"""
    try:
        image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise ValueError(f"图像读取失败：{str(e)}")


@app.post("/compare")
async def compare_images(standard: UploadFile = File(...), student: UploadFile = File(...)):
    try:
        # 检查文件扩展名
        if not (standard.filename.lower().endswith(('.png', '.jpg', '.jpeg')) and
                student.filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
            raise ValueError("请上传 PNG / JPG 格式的图像文件")

        std_img = read_image(await standard.read())
        stu_img = read_image(await student.read())

        # 检查图像尺寸
        if std_img.shape[0] < 50 or std_img.shape[1] < 50:
            raise ValueError("标准图像尺寸过小，无法比较")

        # 转换为灰度图
        std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)
        stu_gray = cv2.cvtColor(stu_img, cv2.COLOR_BGR2GRAY)

        # 尺寸不一致则调整
        if std_gray.shape != stu_gray.shape:
            stu_gray = cv2.resize(stu_gray, (std_gray.shape[1], std_gray.shape[0]))
            stu_img = cv2.resize(stu_img, (std_img.shape[1], std_img.shape[0]))

        # SSIM 差异比较
        score, diff = compare_ssim(std_gray, stu_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # 阈值处理 + 轮廓提取
        thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 20 and h >= 20:
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                cv2.rectangle(stu_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 编码为 Base64 图像
        _, img_encoded = cv2.imencode('.png', stu_img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return JSONResponse(content={
            "similarity": float(score),
            "issues_count": len(boxes),
            "difference_boxes": boxes[:50],
            "marked_image": f"data:image/png;base64,{img_base64}"
        })

    except Exception as e:
        print("=== ERROR in /compare ===")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"服务错误：{str(e)}"}
        )


@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>PCB Diff Checker</title></head>
            <body style="font-family: sans-serif; padding: 2rem;">
                <h1>📌 PCB Diff Checker 后端已启动</h1>
                <p>请通过 <code>/compare</code> API 上传标准图和学生图进行比对。</p>
            </body>
        </html>
        """, status_code=200)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
