# Speech-to-Clipboard
Speech to Clipboard

# Done
- start and send data to whisper
- add a worker thread for stream processing
- transfer chunks to worker thread
- concatenate chunked data

# To Do

## Must do
- use sliding window to transfer word by word
    - print continuous results to console
- add result to clipboard


## High
- add logic to identify recognized words by x repeats in sliding window
- find a way to toggle/ start-end recording

## Mid
- differentiate between recognized and potential words

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

