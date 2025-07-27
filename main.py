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
    return cv2.cvtColor(np.array(Image.open(io.BytesIO(uploaded_bytes))), cv2.COLOR_RGB2BGR)

@app.post("/compare")
async def compare_images(standard: UploadFile = File(...), student: UploadFile = File(...)):
    try:
        std_img = read_image(await standard.read())
        stu_img = read_image(await student.read())

        std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)
        stu_gray = cv2.cvtColor(stu_img, cv2.COLOR_BGR2GRAY)

        # Resize if dimensions mismatch
        if std_gray.shape != stu_gray.shape:
            stu_gray = cv2.resize(stu_gray, (std_gray.shape[1], std_gray.shape[0]))
            stu_img = cv2.resize(stu_img, (std_img.shape[1], std_img.shape[0]))

        # SSIM comparison
        score, diff = compare_ssim(std_gray, stu_gray, full=True)
        diff = (diff * 255).astype("uint8")

        # Threshold and contour
        thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 20 and h >= 20:
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                cv2.rectangle(stu_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

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
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <head><title>PCB Diff Checker</title></head>
            <body>
                <h1>Welcome to PCB Diff Checker!</h1>
                <p>Please upload via /compare API.</p>
            </body>
        </html>
        """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
