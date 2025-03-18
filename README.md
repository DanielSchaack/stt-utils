# Speech-to-Clipboard
Speech to Clipboard

# Requirements
- Developed Linux/Arch, Python 3.11.11, CUDA Version: 12.8
    - Other versions might work

# Setup
```bash
git clone git@github.com:DanielSchaack/speech-to-clipboard.git
cd speech-to-clipboard
echo 'alias speech-to-clipboard="$HOME/Documents/dev/speech-to-clipboard/speech-to-clipboard.sh $@"'
source $HOME/.bashrc
```
- The .sh file checks venv and sets it up if missing, for manual setup:
```bash
python3.11 -m venv .venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# Usage
```bash
source ./venv/bin/activate
python3.11 -u speech-to-clipboard.py &
deactivate
```
or after Setup
```bash
speech-to-clipboard &
```
- Press <ctrl>+<esc> to end the script
- Press and hold <alt>+<r> for transcription
    - Uses default microphone

# Development Tracker
## Done
### Must do
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
    - Could also be done by Piping, i.e. `python3.11 -u voice.py | wl-copy`

### High
- Find a way to toggle/ start-end recording
    - Done using pynput and manual Hotkey management for better control over pressed and released actions.
    - Added a predefined set of keys for start of recording and end of service, and handling of each pressed key
        - Add pressed key to a second set
        - Check if subset of Predefined
            - If so, set recording flag and start recording thread
            - Else unset recording flag
        - Discard any released keys

### Mid
- Differentiate between recognized and potential words
- Add start and stop sound as feedback
    - Make it optional

### Optional - Background Service
- When done, find a way to start as a continuous background service
    - Done by adding pynput keyboard listener. Listener.wait() acts a while True loop until defined set of release keys are pressed (at time of writing ctrl esc)

### Optional - Optimizations
- Optimise uptime of processing/whisper models etc.
    - Currently bound to lifetime of main logic thread
        - Constant Un-/Loading on each use
    - Might be worth it to investigate managing outside of said thread to handle something like configurable uptime after usage for no constant reloading
    - Manage model outside of main thread to handle something like configurable uptime after usage for no constant reloading
    - Added a timer that unloads using a Timer that restarts on use
- Added hint for punctuation, currently for personal main use in german
    - This should result in more output with actual punctuation and capitalisation. These were quite often missing in daily use.
- Configuration without adjusting code 
    - Config file

## To Do
### Optional - Frontend
- When logic stands, add simple frontend to display recognized/potential words
    
### Optional - Optimizations
- Maybe readjust ending condition of consumer thread
    - From Ending with last chunk into processing until done or threshold is met
- Sound files are loaded on each use - add buffer after first read
- Configuration without adjusting code 
    - If frontend available - config editor

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

