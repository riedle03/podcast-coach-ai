# 🎧 팟캐스트 AI 코치 & 자기 성찰

AI 음성 인식(STT) + 화자 분리 + 발화 분석 + Gemini AI 평가를 통해 팟캐스트, 발표, 토론 등의 대화를 분석하고 리포트를 제공합니다.

## ✨ 주요 기능
- **STT 변환**: Faster-Whisper로 빠르고 정확한 음성 → 텍스트 변환
- **화자 분리**: Hugging Face `pyannote.audio` 기반 화자 분리(화자 수 지정 가능)
- **참여도 분석**: 화자별 발화 횟수, 시간, 비율 산출
- **Gemini AI 평가**: 구성·전달력·내용 충실도·창의성에 대한 AI 피드백
- **시각화**: Plotly 기반 점수 그래프, 발화 타임라인, 화자별 참여도
- **자기 성찰**: 사용자가 직접 점수와 메모를 남길 수 있는 인터페이스

---

## 🛠 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/사용자명/저장소명.git
cd 저장소명
````

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # (Windows) venv\Scripts\activate
```

### 3. 필수 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **권장 버전 고정**
>
> * `pytorch-lightning==2.1.3`
> * `torchmetrics==1.2.0`
> * `pyannote.audio==3.1.1`

### 4. 환경 변수 / secrets 설정

`.streamlit/secrets.toml` 파일 생성:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
HF_TOKEN = "your_huggingface_token"
```

* **Gemini API 키**: [Google AI Studio](https://aistudio.google.com/)에서 발급
* **HF\_TOKEN**: [Hugging Face](https://hf.co/settings/tokens)에서 발급

  * 아래 모델 페이지에서 **Access / Agree** 버튼 클릭 (필수)

    1. [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
    2. [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)

### 5. 앱 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열립니다. (기본: [http://localhost:8501](http://localhost:8501))

---

## 💡 사용 방법

1. **오디오 업로드**

   * 지원 형식: WAV, MP3, M4A
   * 최대 200MB
2. **팟캐스트 주제 입력**

   * 예: `"청소년 노동과 빈곤"`
3. **화자 분리 사용 여부 & 화자 수 지정**

   * HF 토큰이 없으면 단일 화자 처리
4. **AI 분석 시작하기**

   * STT 변환 → 화자 분리 → Gemini 평가 → 결과 시각화
5. **리포트 확인**

   * 점수 그래프, 발화 타임라인, 화자별 참여도, 자기 성찰 탭

---

## 📊 출력 예시

* **Gemini AI 평가**

  * 구성: 4
  * 전달력: 3
  * 내용 충실도: 5
  * 창의성: 4
  * **AI 총평**: “발음이 비교적 명확하고 논리적인 전개가 좋으나, 예시 활용이 부족합니다.”
* **참여도 분석**

  * 화자 A: 60% 발화
  * 화자 B: 40% 발화
* **발화 타임라인**

  * 화자별 색상으로 발화 구간 표시

---

## 🐞 오류 해결 가이드

### 1. 화자 분리 모델 로드 실패

* HF 토큰이 비어있거나 약관 미동의
* 해결: HF 계정 로그인 → [모델 페이지](https://hf.co/pyannote/segmentation-3.0)에서 Access 버튼 클릭

### 2. `RuntimeError: Sizes of tensors must match...`

* 원인: 오디오 세그먼트 길이 불일치
* 해결: 이 앱은 **배치=1 고정**으로 수정되어 대부분 해결됨.
  그래도 발생 시 오디오를 16kHz mono로 재인코딩

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

---

## 📜 라이선스

MIT License

---

© 2025 이대형. All rights reserved.
[https://aicreatorz.netlify.app](https://aicreatorz.netlify.app)