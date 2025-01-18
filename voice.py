import sys
import time
import threading
import queue

import pyaudio

q = queue.Queue()
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 15


def callback_send_to_queue(in_data, frame_count, time_info, status):
    # print(f"in_data: {in_data}")
    # print(f"frame_count: {frame_count}")
    # print(f"time_info: {time_info}")
    # print(f"status: {status}")
    print("inside of callback")
    item = (in_data, frame_count)
    q.put(item=item)
    return (in_data, pyaudio.paContinue)


def record_replay():
    p = pyaudio.PyAudio()

    print("Start of stream")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,  # start recording
                    output=True,  # debug live recording
                    frames_per_buffer=CHUNK,
                    stream_callback=callback_send_to_queue)

    while (stream.is_active()):
        print("Stream is still active, waiting 0.25")
        time.sleep(0.25)

    stream.close()
    print("End of stream")

    p.terminate()


def producer():
    print("Starting worker thread")
    record_replay()


def consumer():
    print("Starting worker thread")
    while True:
        print(f"Queue size: {q.qsize()}")
        if not q.empty():
            (in_data, frame_count) = q.get()
            print(f"Frame count: {frame_count}")
            q.task_done()
        time.sleep(1)


if __name__ == "__main__":
    producer_thread = threading.Thread(target=producer, args=())
    producer_thread.start()
    consumer_thread = threading.Thread(target=consumer, args=())
    consumer_thread.start()

    # replay('output.wav')


