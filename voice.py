import time
import threading
import queue

import gc
from faster_whisper import WhisperModel

import numpy as np
import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
q = queue.Queue()
whisper = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4, download_root="./models")


def callback_send_to_queue(in_data, frame_count, time_info, status):
    item = (in_data, frame_count)
    q.put(item=item)
    return (in_data, pyaudio.paContinue)


def producer():
    print("Starting producer thread")
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
        # print("Stream is still active, waiting 0.25")
        time.sleep(0.25)

    stream.close()
    print("End of stream")

    p.terminate()


def consumer():
    print("Starting consumer thread")
    total_data = b''
    window_start_step = 0
    window_stop_step = 5
    available_chunks = 0
    while True:
        if q.qsize() != 0:
            (in_data, frame_count) = q.get()
            total_data += in_data
            q.task_done()

        available_chunks = len(total_data) // CHUNK
        print(f"Available chunks: {available_chunks}, current stop step: {window_stop_step}")
        print(len(total_data))

        if window_stop_step <= available_chunks:
            np_data = get_window(total_data, window_start_step, window_stop_step, CHUNK)
            transcribed = transcribe_window(whisper, np_data, None)
            print(f"Chunk {window_stop_step} transcribed: {transcribed}")
            window_start_step += 1
            window_stop_step += 1
        time.sleep(0.25)


def get_window(total_data, window_start_step: int, window_stop_step: int, chunksize: int) -> np.ndarray:
    start_index = window_start_step * chunksize
    end_index = window_stop_step * chunksize
    return (np.frombuffer(total_data[start_index:end_index], np.int16)
            .flatten()
            .astype(np.float32) / 32768.0
            )


def transcribe_window(model: WhisperModel, np_data: np.ndarray, context: str) -> str:
    segments, _ = model.transcribe(audio=np_data,
                                   language="de",
                                   beam_size=5,
                                   vad_filter=True,
                                   vad_parameters=dict(min_silence_duration_ms=1000))
    segments = [s.text for s in segments]
    transcription = " ".join(segments)
    return transcription

# the following is for OG OpenAi Whisper
# wasn't able to get model.transcribe() working
# so instead using whisper functions directly did work ... somehow
# def transcribe_window(model: whisper.Whisper, np_data: np.ndarray, context: str) -> str:
#     audio = whisper.pad_or_trim(np_data)
#     mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
#     options = whisper.DecodingOptions(language="de", beam_size=5, fp16=False)
#     result = whisper.decode(model, mel, options)
#     return result.text


if __name__ == "__main__":

    # print(whisper.available_models())
    try:
        producer_thread = threading.Thread(target=producer, args=())
        producer_thread.start()
        consumer_thread = threading.Thread(target=consumer, args=())
        consumer_thread.start()
    except Exception as e:
        print(e)
    finally:
        gc.collect()

