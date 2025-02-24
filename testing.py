import wave
import pyaudio
import gc
from typing import Tuple, Any

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
    # model = whisper.load_model("turbo")
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
        # np_data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        # audio = whisper.pad_or_trim(np_data)
        # mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
        # options = whisper.DecodingOptions(language="de", beam_size=5, fp16=False)
        # result = whisper.decode(model, mel, options)
        # print(result)
        # result = model.transcribe(audio=np_data,
        #                           language="de",
        #                           beam_size=5,
        #                           fp16=False,
        #                           verbose=True)
        # print(result["text"])

    stream.close()

    p.terminate()


def transcribe():
    # model = whisper.load_model("turbo")
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
        # np_data = np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0
        # padded = whisper.pad_or_trim(np_data)
        # result = model.transcribe(audio=padded,
        #                           language="de",
        #                           beam_size=5,
        #                           fp16=False,
        #                           verbose=True)
        # print(result["text"])

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


def compare_transcriptions():
    sentence_one = "Hallo, ich bin doof"
    words_list_one = sentence_one.split()
    sentence_two = "Hallo, ich bin doof"
    words_list_two = sentence_two.split()

    min_len = min(len(words_list_one), len(words_list_two))
    amount_equal_words = 0
    for i in range(min_len):
        print(f"Word {i}: '{words_list_one[i]}', word {i}: '{words_list_two[i]}'")
        if words_list_one[i] == words_list_two[i]:
            print(f"Found equal word: {words_list_one[i]}")
            amount_equal_words += 1

    print(f"Found a total of {amount_equal_words} equal words: {words_list_one[i]}")


def compare():
    list_of_lists = [
            ['m', 'b', 'c'],
            ['d', 'e', 'f'],
            ['m', 'n', 'o'],
            ['m', 'n', 'd']
            ]
    get_max_dupe(list_of_lists)


def get_max_dupe(list_of_lists: list[list[Any]]) -> list[Any]:
    """
    Finds the maximum number of duplicate words between any two lists in a given list of lists.

    Parameters:
        list_of_lists (list[list[Any]]): A list containing sublists to be compared for duplicates.

    Returns:
        list[Any]: A list where each element is a tuple containing the maximum number of duplicate
                  words and the index of the second sublist that had the most duplicates with the current sublist.
                  If there are no duplicates, it returns 0 and -1 as the index.

    Example:
    >>> get_max_dupe([['a', 'b', 'c'], ['d', 'e', 'f'], ['m', 'n', 'o'], ['m', 'n', 'd']])
    [(0, -1), (0, -1), (2, 3), (0, -1)]
    """
    max_dupes = []
    len_lists = len(list_of_lists)
    for i in range(len_lists):
        current_max_dupe = 0
        compared_to = -1
        for j in range(i + 1, len_lists):
            if j < len_lists:
                amount_dupe, dupe_words = compare_lists(list_of_lists[i], list_of_lists[j])
                compared_to = j if amount_dupe > current_max_dupe else compared_to
                current_max_dupe = amount_dupe if amount_dupe > current_max_dupe else current_max_dupe

        max_dupes.append((current_max_dupe, compared_to))
    print(f"max dupes: {max_dupes}")

    # for i, list_i in enumerate(list_of_lists):
    #     mapped.append( [(j + 1, word) for j, word in enumerate(list_i)] )
    #
    # print(f"Mapped: {flatten(mapped)}")
def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    return [item for sublist in list_of_lists for item in sublist]


def compare_lists(list_a: list[str], list_b: list[str]) -> tuple[int, list[str]]:
    min_len = min(len(list_a), len(list_b))
    duplicate = 0
    words = []
    for i in range(min_len):
        if list_a[i] == list_b[i]:
            duplicate += 1
            words.append(list_a[i])
        else:
            return duplicate, words
    return duplicate, words


# the following is for OG OpenAi Whisper
# wasn't able to get model.transcribe() working
# so instead using whisper functions directly did work ... somehow
# def transcribe_window(model: whisper.Whisper, np_data: np.ndarray, context: str) -> str:
#     audio = whisper.pad_or_trim(np_data)
#     mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
#     options = whisper.DecodingOptions(language="de", beam_size=5, fp16=False)
#     result = whisper.decode(model, mel, options)
#     return result.text


def get_index_dupes(dupes_list: list[int], index_list: list[int]) -> tuple[int, int]:
    min_dupes = min(dupes_list)
    index = len(index_list) - 1
    for current_dupe in reversed(dupes_list):
        if current_dupe == min_dupes:
            break
        index -= 1
    return index, min_dupes


if __name__ == "__main__":
    index_list = [10, 20, 30]
    dupes_list = [5, 15, 5]
    try:
        # record("test.wav")
        # replay("test.wav")
        # transcribe_file("test.wav")
        # transcribe()
        # compare_transcriptions()
        # compare()
        index, min_value = get_index_dupes(dupes_list, index_list)
        print(f"Index: {index}, Minimum Value: {min_value}")
    except Exception as e:
        print(e)
    finally:
        gc.collect()
