# VideoAudio

## Run locally
**Clone our repository:**

```bash
git clone https://github.com/ZYH-Lightyear/VideoAudio
cd VideoAudio
```

**Create and activate Python environment:**

```bash
python -m venv env
source env/bin/activate
``` 

**Qwen2-VL Weights Download**
```bash
huggingface-cli download --resume-download "Qwen/Qwen2-VL-7B-Instruct" --local-dir Qwen/Qwen2-VL-7B-Instruct
```

**Install ffmpeg**
```bash
sudo apt update
sudo install ffmpeg
```

**Install dependencies and run:**
```bash
pip install -r requirements.txt
python app.py
```










