# Speech-to-Clipboard
Speech to Clipboard

# Credits for Sounds
- start_recording.wav 
    - Percussive Alert 1_1 by Joao_Janz
    - https://freesound.org/s/504782/
    - License: Creative Commons 0
- end_recording.wav
    - Percussive Alert 1_2 by Joao_Janz
    - https://freesound.org/s/504779/
    - License: Creative Commons 0
- end_processing.wav
    - Percussive Notification 3_1 by Joao_Janz
    - https://freesound.org/s/504789/
    - License: Creative Commons 0

# Done
## Must do
- Start and send data to whisper
- Add a worker thread for stream processing
- Transfer chunks to worker thread
- Concatenate chunked data
- Use sliding window to transscribe word by word
- (print continuous results to terminal)
    - Done in log file, printing to terminal is only done for the last result (confirmed + last transcription) for possible piping
- Add logic to identify recognized words by x repeats in sliding window
- Add result to clipboard
    - Done using Pyperclip
    - Could also be done by Piping, i.e. `python -u voice.py | wl-copy`

## High
- Find a way to toggle/ start-end recording
    - Done using pynput and manual Hotkey management for better control over pressed and released actions.
    - Added a predefined set of keys for start of recording and end of service, and handling of each pressed key
        - Add pressed key to a second set
        - Check if subset of Predefined
            - If so, set recording flag and start recording thread
            - Else unset recording flag
        - Discard any released keys

## Mid
- Differentiate between recognized and potential words
- Add start and stop sound as feedback
    - Make it optional

## Optional - Background Service
- When done, find a way to start as a continuous background service
    - Done by adding pynput keyboard listener. Listener.wait() acts a while True loop until defined set of release keys are pressed (at time of writing ctrl esc)

## Optional - Optimizations
- Optimise uptime of processing/whisper models etc.
    - Currently bound to lifetime of main logic thread
        - Constant Un-/Loading on each use
    - Might be worth it to investigate managing outside of said thread to handle something like configurable uptime after usage for no constant reloading

# To Do
## Optional - Frontend
- When logic stands, add simple frontend to display recognized/potential words
    
## Optional - Optimizations
- Manage model outside of main thread to handle something like configurable uptime after usage for no constant reloading
- Sound files are loaded on each use - add buffer after first read
- Configuration without adjusting code 
    - Config file
    - If frontend available - config editor

## After being done with all
- Add decent description to README.md
    1. Description
    2. Features
    3. Installation
        - Provide requirements.txt
        - Installation steps i.e. how to setup on Linux
    4. Configuration
    5. License
- Think about OSS this little hacked together thingy
    - Contributions?

