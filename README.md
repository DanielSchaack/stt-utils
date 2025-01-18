# Speech-to-Clipboard
Speech to Clipboard


# To Do

## Must do
- start and send data to whisper
- add a worker thread for stream processing
- transfer chunks to worker thread
- concatenate chunked data
- use sliding window to transfer word by word
    - print continuous results to console

- add result to clipboard

## High
- find a way to toggle/ start-end recording
- add logic to identify recognized words by x repeats in sliding window

## Mid
- differentiate between recognized and potential words

## Optional - Frontend
When logic stands, add simple frontend to display recognized/potential words
## Optional - Background Service
When done, find a way to start as a continuous background service
## Optional - Optimizations
Optimise uptime of processing/whisper etc.
