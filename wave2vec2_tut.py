import os
import IPython
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
import torchaudio

#Source: https://pytorch.org/tutorials/intermediate/speech_recognition_pipeline_tutorial.html

matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]

torch.random.manual_seed(0)
#Generator for random numbers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPEECH_URL = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
SPEECH_FILE = "_assets/speech.wav"

if not os.path.exists(SPEECH_FILE):
    os.makedirs("_assets", exist_ok=True)
    with open(SPEECH_FILE, "wb") as file:
        file.write(requests.get(SPEECH_URL).content)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
#Wav2Vec2 pipeline, can use pretrained models or fine tune for downstream tasks

print("Sample Rate: ", bundle.sample_rate)
print("Labels: ", bundle.get_labels())

model = bundle.get_model().to(device)
print(model.__class__)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode():
    features, _ = model.extract_features(waveform)

#Feature extraction
fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu())
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis")

plt.tight_layout()
#plt.show()

#Feature classification
with torch.inference_mode():
    emission, _ = model(waveform)
    #Inference mode assumes no interactions with model training, code runs faster

plt.imshow(emission[0].cpu().T)
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
#plt.show()
print("Class labels:", bundle.get_labels())

#Generating Hypothesis (Decoding)
#Decoding takes surrounding observations into consideration, more complex than classification

#Greedy Decoding: Picks best hypothesis at each time step
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels = bundle.get_labels())
transcript = decoder(emission[0])

print(transcript)
