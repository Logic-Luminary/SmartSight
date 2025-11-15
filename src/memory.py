import time

class DetectionMemory:
    def __init__(self, memory_duration=3.0, cooldown=3.0, distance_tolerance=0.5):
        """
        memory_duration: seconds before forgetting a detection
        cooldown: minimum seconds before the same key can trigger speech again
        distance_tolerance: meters difference required to re-speak sooner
        """
        self.memory_duration = memory_duration
        self.cooldown = cooldown
        self.distance_tolerance = distance_tolerance
        self.last_spoken = {}       # key -> (timestamp, distance)

    def should_speak(self, key: str, current_distance: float) -> bool:
        now = time.time()
        record = self.last_spoken.get(key)

        if record is None:
            # First time seeing this detection
            self.last_spoken[key] = (now, current_distance)
            return True

        last_time, last_distance = record
        time_diff = now - last_time
        dist_diff = abs(current_distance - last_distance)

        # Speak again if enough time has passed or distance changed significantly
        if time_diff > self.cooldown or dist_diff >= self.distance_tolerance:
            self.last_spoken[key] = (now, current_distance)
            return True

        # Forget old entries entirely to prevent memory bloat
        if time_diff > self.memory_duration * 3:
            del self.last_spoken[key]

        return False

    def clear_memory(self):
        # Manually clear all stored detections.
        self.last_spoken.clear()
