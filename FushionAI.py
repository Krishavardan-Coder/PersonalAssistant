import subprocess
from datetime import datetime
import requests
import psutil
from gtts import gTTS
import pygame
import os
import speech_recognition as sr
import threading
import time
import ollama
import cv2
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import platform
import datetime as dt



# =========================
# API KEYS & URLs
# =========================
NEWS_API_KEY = "News API Key It's Also Free"
WEATHER_API_KEY = "OpenWeather API Key Don't Worry It's Free"
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
WEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
IP_GEOLOCATION_URL = "https://ipinfo.io/json"

# =========================
# Global variables
# =========================
latest_image_path = "latest_snapshot.jpg"

# =========================
# Helper Functions
# =========================
def get_time_info():
    now = datetime.now()
    return (
        now.strftime("%I:%M %p"),   # Time
        now.strftime("%Y-%m-%d"),   # Date
        now.strftime("%Y"),         # Year
        now.strftime("%B"),         # Month
        now.strftime("%U"),         # Week
        now.strftime("%A")          # Day
    )

def get_weather(location):
    try:
        if location["city"] == "Unknown":
            return "Weather unavailable."
        response = requests.get(WEATHER_BASE_URL, params={
            "q": f"{location['city']},{location['country']}",
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }, timeout=5)
        data = response.json()
        if data.get("weather"):
            desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            return f"{desc}, {temp}°C in {location['city']}."
        return "Weather not found."
    except:
        return "Weather unavailable."

def get_news(location):
    try:
        if not location["country"]:
            return "News unavailable."
        response = requests.get(NEWS_API_URL, params={
            "apiKey": NEWS_API_KEY,
            "country": location["country"].lower(),
            "pageSize": 3
        }, timeout=5)
        data = response.json()
        articles = data.get("articles", [])
        headlines = " | ".join(article["title"] for article in articles)
        return f"Top headlines: {headlines}" if headlines else "No news found."
    except:
        return "News unavailable."

def get_cpu_info():
    per_core = psutil.cpu_percent(percpu=True)
    overall = psutil.cpu_percent()
    core_count = psutil.cpu_count(logical=True)
    core_info = ", ".join([f"{i+1}:{p}%" for i, p in enumerate(per_core)])
    return (
        f"CPU Cores: {core_count}\\n"
        f"CPU Usage: {overall}%\\n"
        f"Per Core Usage: {core_info}"
    )

def get_memory_info():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return (
        f"RAM Usage: {memory.percent}% of {round(memory.total / (1024**3), 2)} GB\\n"
        f"Swap Usage: {swap.percent}% of {round(swap.total / (1024**3), 2)} GB"
    )

def get_disk_info():
    info = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
            info.append(
                f"{part.device} ({part.mountpoint}) — {usage.percent}% of {round(usage.total / (1024**3), 2)} GB"
            )
        except PermissionError:
            continue
    return "Disk Info:\\n" + "\\n".join(info)

def get_disk_io():
    io = psutil.disk_io_counters()
    return (
        f"Disk Read: {round(io.read_bytes / (1024**2), 2)} MB\\n"
        f"Disk Write: {round(io.write_bytes / (1024**2), 2)} MB"
    )

def get_network_info():
    net = psutil.net_io_counters()
    return (
        f"Network Sent: {round(net.bytes_sent / (1024**2), 2)} MB\\n"
        f"Network Received: {round(net.bytes_recv / (1024**2), 2)} MB"
    )

def get_battery_info():
    battery = psutil.sensors_battery()
    if battery:
        plugged = "Charging" if battery.power_plugged else "Not Charging"
        time_left = (
            str(dt.timedelta(seconds=battery.secsleft))
            if battery.secsleft != psutil.POWER_TIME_UNLIMITED
            else "Unlimited"
        )
        return (
            f"Battery: {battery.percent}% - {plugged}\\n"
            f"Time Left: {time_left}"
        )
    return "Battery info not available."

def get_system_info():
    uname = platform.uname()
    boot_time = dt.datetime.fromtimestamp(psutil.boot_time())
    uptime = dt.datetime.now() - boot_time
    return (
        f"System: {uname.system} {uname.release} ({uname.machine})\\n"
        f"Boot Time: {boot_time.strftime('%Y-%m-%d %H:%M:%S')}\\n"
        f"Uptime: {str(uptime).split('.')[0]}"
    )

def get_process_info():
    procs = psutil.pids()
    threads = sum(p.num_threads() for p in map(psutil.Process, procs) if p.is_running())
    return f"Running Processes: {len(procs)}\\nTotal Threads: {threads}"

def full_system_report():
    return "\\n\\n".join([
        get_system_info(),
        get_cpu_info(),
        get_memory_info(),
        get_disk_info(),
        get_disk_io(),
        get_network_info(),
        get_battery_info(),
        get_process_info()
    ])

def speak_text(text, filename="speech.mp3"):
    try:
        print(f"💬 Fusion said: {text}")
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print(f"🔊 Speech error: {e}")

# =========================
# AI Class for Gemma3 (NLP)
# =========================
class ChatAI:
    def __init__(self):
        self.model = "gemma3"
        self.messages = [{
            "role": "system",
            "content": (
                "You are Fusion, an AI Personal Assistant created by Krish, a 13 year old. "
                "Act professional, fun, joke around, max word limit 10-15 words, no emojis."
            )
        }]
        self.location = self.get_location()
        self.weather = get_weather(self.location)
        self.news = get_news(self.location)
        self.add_time_date_info()
        self.add_weather_news_info()

    def get_location(self):
        try:
            response = requests.get(IP_GEOLOCATION_URL, timeout=5)
            data = response.json()
            return {
                "city": data.get("city", "Unknown"),
                "region": data.get("region", ""),
                "country": data.get("country", "")
            }
        except:
            return {"city": "Unknown", "region": "", "country": ""}

    def add_time_date_info(self):
        time_info = get_time_info()
        labels = ["Time", "Date", "Year", "Month", "Week", "Day"]
        for label, value in zip(labels, time_info):
            self.messages.append({"role": "system", "content": f"Current {label}: {value}"})

    def add_weather_news_info(self):
        self.messages.extend([
            {"role": "system", "content": f"Location: {self.location['city']}, {self.location['region']}, {self.location['country']}"},
            {"role": "system", "content": f"Weather Update: {self.weather}"},
            {"role": "system", "content": f"News Update: {self.news}"}
        ])

    def ask(self, user_input: str) -> str:
        system_info = "\n".join([
            get_cpu_info(),
            get_memory_info(),
            get_disk_info(),
            get_battery_info()
        ])

        user_input_full = f"{user_input}\nSystem Info:\n{system_info}"
        self.messages.append({"role": "user", "content": user_input_full})

        response = run_ollama_chat(self.model, self.messages)

        if response:
            self.messages.append({"role": "assistant", "content": response})
            return response
        return "Sorry, I couldn't generate a response."

    # Detection for Image Generation Requests
    def detects_image_generation(self, user_text: str) -> bool:
        prompt = [
            {"role": "system", "content": (
                "You are a helpful assistant that ONLY answers 'yes' or 'no' clearly.\n"
                "Determine if the user wants to GENERATE an image.\n"
                "Examples of image generation requests:\n"
                "- generate an image of a sunset\n"
                "- create a picture of a futuristic city\n"
                "- show me a photo of a cat\n"
                "- draw a dragon breathing fire\n"
                "- paint a landscape with mountains\n"
                "- render an astronaut walking on Mars\n"
                "- make an art of a robot\n"
                "If the input clearly asks for image creation, say 'yes'. Otherwise, say 'no'."
            )},
            {"role": "user", "content": user_text}
        ]
        response = run_ollama_chat(self.model, prompt)
        return response is not None and "yes" in response.lower()

    # Detection for Image Analysis Requests
    def detects_image_analysis(self, user_text: str) -> bool:
        prompt = [
            {"role": "system", "content": (
                "You are a helpful assistant that ONLY answers 'yes' or 'no' clearly.\n"
                "Determine if the user wants IMAGE ANALYSIS or APPEARANCE description.\n"
                "Examples of image analysis requests:\n"
                "- how do I look\n"
                "- describe the image\n"
                "- what do you see\n"
                "- tell me about the image\n"
                "- what's in the picture\n"
                "- explain how I look\n"
                "- what’s on the screen\n"
                "If the input clearly asks to analyze or describe an image or appearance, say 'yes'. Otherwise, say 'no'."
            )},
            {"role": "user", "content": user_text}
        ]
        response = run_ollama_chat(self.model, prompt)
        return response is not None and "yes" in response.lower()

def pull_model(model_name="gemma3"):
    pull_process = subprocess.run(
        ["ollama", "pull", model_name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    if pull_process.returncode != 0:
        print(pull_process.stderr)
        return False
    return True

def run_ollama_chat(model, messages):
    try:
        res = ollama.chat(
            model=model,
            messages=messages
        )
        return res["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Error running Ollama: {e}")
        return None

def capture_webcam_once():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return False
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(latest_image_path, frame)
        cap.release()
        return True
    cap.release()
    return False

def analyze_image_with_llava(image_path):
    messages = [
        {
            "role": "user",
            "content": "Describe this image in 15 words exactly.",
            "images": [image_path]
        }
    ]
    try:
        res = ollama.chat(
            model="llava:7b",
            messages=messages
        )
        return res["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Lava error: {e}")
        return "Couldn't analyze image."

def generate_image(prompt):
    model_id = "stabilityai/stable-diffusion-2-1"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to("cpu")  # use "cuda" if you have GPU

    steps = 20               # Balanced for speed & quality
    guidance_scale = 5.0     # Strong guidance without slowing things down
    height = 448             # Taller than small, but not 512 to save time
    width = 448

    print(f"🎨 Generating HQ image: '{prompt}'")
    result = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width
    )
    image = result.images[0]
    filename = "hq_fast_image.png"
    image.save(filename)
    print(f"✅ Done! Saved as '{filename}'")
    return filename

def listen_loop():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    ai = ChatAI()

    print("🎤 Listening for voice input (press Ctrl+C to stop)...")

    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                user_text = recognizer.recognize_google(audio).lower()
                print(f"🗣️ You said: {user_text}")

                if ai.detects_image_analysis(user_text):
                    print("🤖 Detected image/appearance analysis request.")
                    if capture_webcam_once():
                        explanation = analyze_image_with_llava(latest_image_path)
                        print(f"Lava: {explanation}")

                        gemma_response = ai.ask(f"Explain to user: {explanation}")
                        print(f"Fusion: {gemma_response}")
                        threading.Thread(target=speak_text, args=(gemma_response,)).start()
                    else:
                        print("❌ Failed to capture image.")
                        threading.Thread(target=speak_text, args=("Sorry, couldn't take a picture.",)).start()

                elif ai.detects_image_generation(user_text):
                    print("🤖 Detected image generation request.")
                    # Rephrase user request for better prompt
                    rephrase_prompt = [
                        {"role": "system", "content": (
                            "You are an assistant that rephrases image generation requests into a clear prompt."
                            "Keep it short and descriptive."
                        )},
                        {"role": "user", "content": user_text}
                    ]
                    prompt_for_image = run_ollama_chat(ai.model, rephrase_prompt)
                    if prompt_for_image:
                        filename = generate_image(prompt_for_image)
                        # Show or open the image automatically (platform-dependent)
                        try:
                            if os.name == 'nt':  # Windows
                                os.startfile(filename)
                            else:
                                subprocess.run(["xdg-open", filename])
                        except Exception as e:
                            print(f"Couldn't open the image file: {e}")

                        # Let Gemma know the image was generated
                        gemma_feedback = ai.ask(f"I generated an image with prompt: '{prompt_for_image}'")
                        print(f"Fusion: {gemma_feedback}")
                        threading.Thread(target=speak_text, args=(f"Image generated! How do you like it? {gemma_feedback}",)).start()
                    else:
                        print("❌ Could not rephrase image prompt.")
                        threading.Thread(target=speak_text, args=("Sorry, I couldn't understand what image to generate.",)).start()

                else:
                    # Normal chat response
                    response = ai.ask(user_text)
                    threading.Thread(target=speak_text, args=(response,)).start()

        except sr.UnknownValueError:
            print("🤷 Couldn't understand. Try again.")
        except sr.RequestError as e:
            print(f"❌ API error: {e}")
        except KeyboardInterrupt:
            print("👋 Exiting...")
            break

if __name__ == "__main__":
    if not pull_model("gemma3"):
        exit(1)

    listen_loop()
