import wave
import pyaudio
import gc
import torch
from typing import Tuple, Any

import numpy as np
import whisper

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE * 5
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


def replay(filename: str):
    with wave.open(filename, 'rb') as wf:
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        # input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

        while len(data := wf.readframes(CHUNK)):
            stream.write(data)

        stream.close()

        p.terminate()


def transcribe_file(filename: str):
    model = whisper.load_model("turbo")
    # result = model.transcribe(audio=filename, language="de", verbose=True)
    # print(result["text"])
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Start reading")
    total_data = b""

    while True:
        data = stream.read(CHUNK)
        print("CHUNK read")
        total_data += data
        np_data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        audio = whisper.pad_or_trim(np_data)
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
        options = whisper.DecodingOptions(language="de", beam_size=5, fp16=False)
        result = whisper.decode(model, mel, options)
        print(result)
        # result = model.transcribe(audio=np_data,
        #                           language="de",
        #                           beam_size=5,
        #                           fp16=False,
        #                           verbose=True)
        # print(result["text"])

    stream.close()

    p.terminate()


def transcribe():
    model = whisper.load_model("turbo")
    # result = model.transcribe(audio=filename, language="de", verbose=True)
    # print(result["text"])

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print("Start reading")
    total_data = b""

    while True:
        data = stream.read(CHUNK)
        print("CHUNK read")
        total_data += data
        stream.write(data)
        print("CHUNK replayed")
        np_data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        padded = whisper.pad_or_trim(np_data)
        result = model.transcribe(audio=padded,
                                  language="de",
                                  beam_size=5,
                                  fp16=False,
                                  verbose=True)
        print(result["text"])

    stream.close()

    p.terminate()


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


if __name__ == "__main__":
    try:
        # record("test.wav")
        # replay("test.wav")
        transcribe_file("test.wav")
        # transcribe()
    except Exception as e:
        print(e)
    finally:
        # del model
        torch.cuda.empty_cache()
        gc.collect()
