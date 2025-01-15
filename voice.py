import wave
import sys
import time
from typing import Any, Tuple

# import queue

import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 15



def record(filename: str):
    with wave.open(filename, 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True)

        print('Recording...')
        for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
            wf.writeframes(stream.read(CHUNK))
        print('Done')

        stream.close()
        p.terminate()


def callback_send_to_queue(in_data, frame_count, time_info, status):
    # print(f"in_data: {in_data}")
    print(f"frame_count: {frame_count}")
    print(f"time_info: {time_info}")
    print(f"status: {status}")
    return (in_data, pyaudio.paContinue)


def callback_replay_input(in_data, frame_count, time_info, status) -> Tuple[Any, int]:
    """
    Args:
        in_data: Array of byte data. None if stream has input=False
        frame_count: Amount of frames inside buffered input data (in_data)
        time_info:
                - 1 replay timestamp if replay
                - 2 idk
                - 3 idk
        status:
                - 0 no issues(more data incoming)
                - 1 input underflow
                - 2 input overflow
                - 3 output underflow
                - 4 output overflow
                - 16 priming

    Returns:
        tuple (output_data, statuscode)
    """
    # print(f"in_data: {in_data}")
    print(f"frame_count: {frame_count}")
    print(f"time_info: {time_info}")
    print(f"status: {status}")
    return (in_data, pyaudio.paContinue)


# TODO
# - find a way to toggle/ start-end recording
# - add a worker thread for stream processing
# - transfer chunks to worker thread
# - concatenate chunked data
# - start and send data to whisper
# - print continuous results to console
# - use sliding window to transfer word by word
# - add logic to identify recognized words by x repeats in sliding window
# - differentiate between recognized and potential words
# - add result to clipboard

# Optional: When logic stands, add simple frontend to display recognized/potential words
# Optional: When done, find a way to start as a continuous background service
# Optional: Optimise uptime of processing/whisper etc

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


def replay(filename: str):
    with wave.open(filename, 'rb') as wf:
        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        # input=True,
                        output=True,
                        frames_per_buffer=512)

        while len(data := wf.readframes(CHUNK)):
            stream.write(data)

        stream.close()

        p.terminate()


if __name__ == "__main__":
    record_replay()
    # replay('output.wav')




























