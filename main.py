import os
import uuid
import json
import time
import base64
import re
from datetime import datetime, timedelta

import numpy as np
import cv2
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt, JWTError

# =========================
# 환경 변수 로드
# =========================
load_dotenv()

CLOVA_API_URL = os.getenv("CLOVA_API_URL")
CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY")

APP_LOGIN_ID = os.getenv("APP_LOGIN_ID", "hangil")
APP_LOGIN_PASSWORD = os.getenv("APP_LOGIN_PASSWORD", "w010410")

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_TO_SOMETHING_RANDOM")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "43200")
)  # 기본 30일

# =========================
# JWT 관련 유틸
# =========================

security = HTTPBearer()


class LoginRequest(BaseModel):
  username: str
  password: str


class LoginResponse(BaseModel):
  access_token: str
  token_type: str = "bearer"


def create_access_token(subject: str) -> str:
  expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
  to_encode = {"sub": subject, "exp": expire}
  return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(
  credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
  """
  Authorization: Bearer <token> 에서 token을 꺼내서 검증
  """
  token = credentials.credentials
  try:
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    username = payload.get("sub")
    if username is None or username != APP_LOGIN_ID:
      raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다.")
    return username
  except JWTError:
    raise HTTPException(status_code=401, detail="토큰이 유효하지 않습니다.")


# =========================
# 공통 유틸 (기존 코드)
# =========================
def get_left_half_bytes(image_bytes: bytes):
  """
  업로드된 이미지(bytes)의 왼쪽 절반만 추출해서 jpg bytes로 반환
  """
  nparr = np.frombuffer(image_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  if img is None:
    return None

  height, width = img.shape[:2]
  left_half = img[:, : width // 2]
  is_success, buffer = cv2.imencode(".jpg", left_half)
  return buffer.tobytes() if is_success else None


def sanitize_filename(text: str) -> str:
  """
  파일명으로 사용할 수 없는 문자 제거 및 수정
  (프론트에서 써도 되도록 리턴해줄 수 있음)
  """
  text = re.sub(r'[\\/*?:"<>|]', "_", text)
  text = text.replace(" ", "_")
  text = text[:50]
  return text


def call_clova_ocr(left_half_bytes: bytes) -> str:
  """
  왼쪽 절반 이미지 bytes를 CLOVA OCR로 보내고,
  인식된 텍스트를 하나의 문자열로 리턴
  """
  if not CLOVA_API_URL or not CLOVA_SECRET_KEY:
    raise RuntimeError(
      "CLOVA_API_URL 또는 CLOVA_SECRET_KEY 환경변수가 설정되지 않았습니다."
    )

  image_base64 = base64.b64encode(left_half_bytes).decode("utf-8")

  request_json = {
    "images": [
      {
        "format": "jpg",
        "name": "test_image",
        "data": image_base64,
      }
    ],
    "requestId": str(uuid.uuid4()),
    "version": "V2",
    "timestamp": int(round(time.time() * 1000)),
  }

  headers = {
    "X-OCR-SECRET": CLOVA_SECRET_KEY,
    "Content-Type": "application/json",
  }

  response = requests.post(
    CLOVA_API_URL, headers=headers, json=request_json, timeout=30
  )

  if response.status_code != 200:
    raise RuntimeError(f"API 호출 실패 (상태 코드: {response.status_code})")

  result = response.json()
  texts = []
  for image in result.get("images", []):
    for field in image.get("fields", []):
      text = field.get("inferText", "").strip()
      if text:
        texts.append(text)

  if not texts:
    raise RuntimeError("텍스트를 인식하지 못했습니다.")

  recognized_text = " ".join(texts)
  return recognized_text


# =========================
# FastAPI 앱 설정
# =========================
app = FastAPI()

# CORS: Vite 프론트 허용
origins = [
  "http://localhost:5173",  # 개발용
  "https://flower-ocr-frontend.vercel.app"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.get("/")
def root():
  return {"message": "Flower OCR backend running"}


# -------------------------
# 로그인 (JWT 발급)
# -------------------------
@app.post("/login", response_model=LoginResponse)
def login(body: LoginRequest):
  """
  단일 계정 로그인 (아이디/비밀번호는 .env에서 관리)
  """
  if body.username != APP_LOGIN_ID or body.password != APP_LOGIN_PASSWORD:
    raise HTTPException(
      status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다."
    )

  token = create_access_token(APP_LOGIN_ID)
  return LoginResponse(access_token=token)


# -------------------------
# OCR API (로그인 필요)
# -------------------------
@app.post("/ocr")
async def ocr_endpoint(
  file: UploadFile = File(...),
  current_user: str = Depends(get_current_user),
):
  """
  프론트에서 이미지 1장을 업로드하면:
  1) 왼쪽 절반 자르고
  2) CLOVA OCR 호출
  3) 인식된 텍스트와, 파일명용으로 정제된 텍스트를 같이 반환

  ※ JWT 토큰이 있어야 호출 가능
  """

  try:
    image_bytes = await file.read()
    if not image_bytes:
      return {"text": "", "safe_text": "", "error": "empty_file"}

    left_half_bytes = get_left_half_bytes(image_bytes)
    if not left_half_bytes:
      return {"text": "", "safe_text": "", "error": "crop_failed"}

    raw_text = call_clova_ocr(left_half_bytes)
    safe_text = sanitize_filename(raw_text)

    return {
      "text": raw_text,      # 원본 인식 텍스트
      "safe_text": safe_text # 파일명으로 쓰기 좋은 버전
    }

  except Exception as e:
    print("[/ocr 에러]", e)
    return {"text": "", "safe_text": "", "error": str(e)}
