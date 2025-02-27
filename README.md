# Speech-to-Clipboard
Speech to Clipboard

# Sounds
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
- start and send data to whisper
- add a worker thread for stream processing
- transfer chunks to worker thread
- concatenate chunked data
- use sliding window to transscribe word by word
- (print continuous results to console)
- add logic to identify recognized words by x repeats in sliding window
- differentiate between recognized and potential words
- Add start and stop sound as feedback
    - Make it optional
    - add result to clipboard

# To Do
## Must do

## High
- find a way to toggle/ start-end recording

## Mid

## Optional - Frontend
- When logic stands, add simple frontend to display recognized/potential words

## Optional - Background Service
- When done, find a way to start as a continuous background service

## Optional - Optimizations
- Optimise uptime of processing/whisper models etc.
- Configuration

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

