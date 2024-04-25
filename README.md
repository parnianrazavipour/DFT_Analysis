# Building a web application to visualize DFT

## Introduction
This web application allows users to visualize the Discrete Fourier Transform (DFT) of signals. It provides tools to upload audio files, create signals by combining sinusoidal waves, and analyze these signals in the frequency domain.

## Getting Started
To run the server locally, navigate to the `/Server` directory and execute the following command:
```bash
docker compose up
````
Once executed, this command sets up the local server, which can then be accessed through the URL http://127.0.0.1:8080 in a web browser.

## Features
- Upload audio files for analysis (.wav and .mp3 files).
- Combine multiple sinusoidal waves to create complex signals.
- Visualize time domain and frequency domain representations of signals.
- Analyze signals using both brute force DFT and Fast Fourier Transform (FFT).

## Usage
1. Start the server using Docker as described above.
2. Navigate to `http://127.0.0.1:8080` in your web browser.
3. Use the "Upload File" section to upload audio files or the "Create Signal" feature to combine sinusoidal waves.
4. Analyze signals and visualize the results using the provided buttons and plots.


