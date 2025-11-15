import pyttsx3
import time
import threading
from queue import Queue

class TextToSpeech:
    def __init__(self, cooldown = 1):
        self.engine = pyttsx3.init()
        self.cooldown = cooldown
        self.queue = Queue()
        self.thread = threading.Thread(target = self._process_queue, daemon = True)
        self.thread.start()

    def _process_queue(self):
        last_spoken = 0
        while True:
            text = self.queue.get()
            now = time.time()
            if now - last_spoken < self.cooldown:
                time.sleep(self.cooldown - (now - last_spoken))
            self.engine.say(text)
            self.engine.runAndWait()
            last_spoken = time.time()
            self.queue.task_done()

    def speak(self, text):
        self.queue.put(text)