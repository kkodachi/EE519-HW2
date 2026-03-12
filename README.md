# EE519 HW2

Parts 1-4 are in [01_ASR_Robustness.ipynb](./01_ASR_Robustness.ipynb)

Part 5 is in [02_Enhancement_VAD.ipynb](./02_Enhancement_VAD.ipynb)

The relevant graphs can be found at the bottom of both notebooks.

This file contains the required questions and writeup.

## Pre-Experiment Questions

1. conformer_large_960h will likely be the most robust because it is the largest. Model architecture has a huge impact on performance not only in the size but also the type of neural network used. This model features a transformer and convolution which will be useful for spectral patterns. If this network were to fail it would be because the model is too big which might lead to overfitting, but I doubt this will be the case.
2. Background conversation will likely be the most damaging because it is legitimate speech just overlapped and maybe slightly quieter. The model may confuse my recorded words with the people in the background.
3. Enhancement will help more because it improves SNR, reduces spectral masking, and improves phoneme recognition. This would help with the previously mentioned background conversation. VAD helps to remove silence and reduces insertions which wouldn't be useful for that.
4. S (substitutions) will dominate because noise causes phoneme confusion, spectral masking, and misclassification of sounds. This will cause the model to hear speech but map to wrong word.

## Post-Experiment Questions

1. I was very surprised because the smallest models performed the best as they had the smallest WER under real noise. The smallest models also had smaller WER under the synthetic vs real comparisons. They also performed best for the grammatical vs ungrammatical comparisons. All my predictions were wrong because I thought the largest models would perform the best. These charts can be found at the bottom of notebook 1.
2. The white noise revealed the biggest differences between the models as we can see the largest gap between the performance. The smaller models had the lowest WER here.
3. Enhancement reduced WER most in synthetic noise meaning that the stationary noise was effectively suppressed. This isn't true for non statioanary distortions in real world noise.
4. VAD did not help reduce insertions at all as we saw a minimal change in I on the graphs.
5. The first failure mode would be VAD which likely trimmed the phonemes so I propose increasing the padding. The next failure was competing speaker confusion leading to higher WER for which I suggest source separation or diarization.
1. For call centers or hospitals I would pick enhancement and a smaller model because those would perform best to handle background typing, chatter, or keyboard clicks.

## Write Up

The notebooks already contain comments and brief descriptions but I will expand upon them further here.

### Recording and Setup

First I created 3 notebooks to record my speech easily. This allowed me to keep formatting, output, and logistics consistent across all the environments. The code is self explantory and mostly taken from past homeworks. All three notebooks are the same to record [clean](./clean.ipynb), [env1](./env1.ipynb), and [env2](./env2.ipynb). All outputs and results can be found inside the project directory with self explanatory subdirectories.

### [Notebook 1](./01_ASR_Robustness.ipynb)

#### Noise

To evaluate performance under noisy conditions, we generated synthetic noisy versions of the clean audio files by adding controlled noise at specific Signal-to-Noise Ratio (SNR) levels. We defined white noise and traffic noise, and two SNR levels: 20 dB and 10 dB. The system loops over each clean audio file and generates noisy versions for every combination of noise type and SNR. White noise is generated using a standard normal distribution (np.random.randn). The noise is normalized using RMS (root mean square) normalization so that it has unit energy before scaling. If a traffic.wav file is available, it is loaded, repeated or trimmed to match the clean signal length, and RMS-normalized. If the traffic file is not available, the system falls back to generating pink noise, which approximates 1/f noise by shaping the spectrum in the frequency domain using FFT.

#### Models

To evaluate transcription performance under different noise conditions, I used five different Automatic Speech Recognition (ASR) models from both OpenAI Whisper and Hugging Face Transformers. These models vary in size, architecture, and training methodology, allowing me to analyze their performance under different constraints. 

1. Whisper Base
   
   Description: Whisper Base is a medium-sized encoder–decoder Transformer model trained by OpenAI on a large-scale, multilingual, weakly supervised dataset consisting of diverse internet audio. It performs end-to-end speech recognition and directly generates text from audio input.

   Reasoning: The Base model offers a good balance between performance and computational efficiency. It is robust to noise and accents due to its large and diverse training data.
2. Whisper Small 

   Description: Whisper Small is a larger version of the Base model with more parameters, allowing it to capture more complex acoustic and linguistic patterns. Like other Whisper models, it is trained on large-scale multilingual data and uses a sequence-to-sequence Transformer architecture.
   
   Reasoning: This model will offer a good comparison with the base to see performance vs compute tradeoffs.
3. Wav2Vec 2.0
   
   Description: Wav2Vec 2.0 is a self-supervised speech representation learning model developed by Facebook AI. It uses a convolutional feature encoder followed by a Transformer network to learn contextualized speech representations.
   
   Reasoning: The base-960h version is fine-tuned on 960 hours of LibriSpeech labeled data. Unlike Whisper, it is not a sequence-to-sequence model; instead, it uses a Connectionist Temporal Classification (CTC) decoding approach for transcription.

4. HuBERT Large
   
   Description: HuBERT (Hidden-Unit BERT) is another self-supervised speech representation learning model. It improves upon wav2vec-style training by predicting clustered hidden representations instead of raw audio features during pretraining.
   
   Reasoning: The large version has more parameters than the base wav2vec2 model and is fine-tuned on 960 hours of LibriSpeech data. Due to its larger architecture and improved pretraining strategy, it typically achieves better performance, particularly in clean speech conditions.

5. Wav2Vec2 Conformer Large
   
   Description: This model combines Wav2Vec 2.0 with a Conformer architecture. Conformers integrate convolutional layers with Transformer blocks, allowing the model to capture both local acoustic patterns and long-range dependencies effectively.
   
   Reasoning: The large-960h-ft version is fine-tuned on LibriSpeech 960 hours. The Conformer architecture generally improves robustness and accuracy, especially in noisy or complex speech scenarios.

#### Obstacles

After defining the graph models we need to run the models. However, I ran into two major issues. First, the compute_measures function would not work for me. I tried installing multiple versions of jiwer in my virtual environment but was unable to get it working. As a workaround I used an LLM to generate a function with the same capability. The fallback for ASR eval metrics provided: a word-level Levenshtein DP that computes substitutions, insertions, deletions, hits and WER, plus a character-level CER fallback. compute_metrics tokenizes inputs, uses the local DP when external libraries fail, and returns a consistent metrics dictionary while avoiding division-by-zero and backtrace deadlocks. After this, I ran into an issue running the models back to back inside a pipeline and could not resolve this issue. As a result, I was forced to run each in a separate cell which took a good bit longer but still worked. Lastly we output the graphs necessary to analyze results.

### [Notebook 2](./02_Enhancement_VAD.ipynb)

#### Enhancement and VAD

We apply spectral gating–based noise reduction using noisereduce, followed by WebRTC VAD to remove non-speech frames. Audio is converted to 16-bit PCM for VAD processing, segmented into 30 ms frames, and classified as speech/non-speech. A padded VAD version is used to prevent clipping speech boundaries. The enhanced and trimmed audio was then used for ASR evaluation. Next, we run the ASR models using the same code as the previous notebook.

#### Obstacles

Similar to the previous notebook, I was having trouble running the models back to back in a pipeline so I had to rerun the model for each condition. In the future I could fix this by using Google Colab from the very beginning so I don't have to worry about issue on my local machine or virtual environment.

#### Overall Results and Analysis

I was incredibly surprised to see that models 1 and 2 had the best metrics across the board despite being the smallest. This went against my hypothesis and understanding but after looking into the background of the models I will attempt to interpret why these results occured. 

1. Training data mismatch & robustness: Whisper models are trained end-to-end on extremely large, diverse, weakly-supervised datasets. This broad pretraining tends to give Whisper strong robustness to varied background noise and domain shifts. The HF models used (models 3-5) were fine-tuned mainly on clean, read speech (LibriSpeech), which can make them less robust to real-world or synthetic noises.
2. Model Architecture: Whisper is an encoder–decoder model that performs sequence-to-sequence transcription and implicitly models language. wav2vec2 / HuBERT variants typically use CTC decoding. CTC-only models can struggle when audio is noisy or when language context helps disambiguate tokens. Whisper’s decoder helps recover words using context, which reduces substitutions and deletions in ambiguous/noisy frames.
3. Overfitting: This was one of my speculations for why my hypothesis might be wrong in the pre-experiment section. Bigger models can overfit to their fine-tuning data or develop brittle behaviors in new domains. Medium-sized models are sometimes better suited for a specific task enough capacity to model speech and language but not so large that they overfit to fine-tune domain-specific artifacts.