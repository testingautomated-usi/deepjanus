from datetime import datetime

from config import RUNTIME


class Timer:
    start = datetime.now()

    @staticmethod
    def get_time():
        return datetime.now()

    @staticmethod
    def get_timestamps():
        now = datetime.now()
        elapsed = now - Timer.start
        return now, elapsed

    @staticmethod
    def get_elapsed_time():
        elapsed_time = datetime.now() - Timer.start
        return elapsed_time

    @staticmethod
    def has_budget():
        elapsed_time = datetime.now() - Timer.start
        if elapsed_time.seconds <= RUNTIME:
            return True
        else:
            return False
