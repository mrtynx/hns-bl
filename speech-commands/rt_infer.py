import sys
import time

import numpy as np
import sounddevice as sd
import torch
from models import CNNLSTMModel

FS = 8000

model = CNNLSTMModel(1, 64, 2, 4)
model.load_state_dict(
    torch.load(r"C:\FEI\ING\S3\HNS\speech-commands\runs\exp1\best_model.pt")
)
model.eval()

label_mapping = ["go", "left", "right", "stop"]


def audio_callback(indata, frames, time, status):
    if status:
        print("Audio callback error\n")
        return

    min_val = np.min(indata)
    max_val = np.max(indata)

    normalized_arr = 2 * (indata - min_val) / (max_val - min_val) - 1

    audio_tensor = torch.from_numpy(normalized_arr).unsqueeze(dim=0)

    with torch.no_grad():
        nn_out = model(audio_tensor).squeeze(dim=0).detach().numpy()
        # nn_out[1] -= 0.5  # Treshold left
        # print(nn_out)
        index = np.argmax(nn_out)

        if nn_out[index] > 0.5:
            print(f"\r{label_mapping[index]}\n", end="", flush=True)


with sd.InputStream(
    callback=audio_callback,
    channels=1,
    samplerate=FS,
    dtype="float32",
    blocksize=FS,
):
    print("Audio stream is active. Listening to voice commands...\n")
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        print("\nAudio stream stopped.")
