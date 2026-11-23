# 1. IMPORT LIBRARIES
import wave
import struct
import numpy as np
import os
import matplotlib.pyplot as plt

# 2. AUDIO LOADING
def read_audio(file_path):
    with wave.open(file_path, "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        total_frames = audio.getnframes()

        if sample_width != 2:
            raise ValueError("Audio must be 16-bit PCM WAV.")

        raw_data = audio.readframes(total_frames)

    total_samples = len(raw_data) // 2
    samples = struct.unpack("<" + "h" * total_samples, raw_data)

    if channels == 2:
        mono_samples = []

        for i in range(0, len(samples), 2):
            average = (samples[i] + samples[i + 1]) / 2
            mono_samples.append(average)

        samples = mono_samples

    elif channels != 1:
        raise ValueError("Only mono and stereo audio are supported.")

    return np.array(samples, dtype=np.float64), sample_rate

# 3. AUDIO NORMALIZATION
def normalize_audio(samples):
    return samples / 32768.0

# 4. ADD GAUSSIAN NOISE
def add_noise(clean_audio, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(clean_audio))
    noisy_audio = clean_audio + noise

    return noisy_audio

# 5. LINEAR REGRESSION MODEL
def predict(noisy_audio, weights, bias):
    previous = noisy_audio[:-2]
    current = noisy_audio[1:-1]
    following = noisy_audio[2:]

    predictions = (
        weights[0] * previous +
        weights[1] * current +
        weights[2] * following +
        bias
    )

    return predictions

# 6. HALF-MSE
def calculate_loss(predictions, targets):
    errors = predictions - targets
    loss = np.sum(errors ** 2) / (2 * len(targets))

    return loss

# 7. GRADIENT DESCENT
def train_model(noisy_audio, clean_audio, learning_rate=0.5, epochs=1000):

    # Initialize weights and bias
    weights = np.zeros(3)
    bias = 0.0

    # Prepare input samples
    previous = noisy_audio[:-2]
    current = noisy_audio[1:-1]
    following = noisy_audio[2:]

    # Prepare target samples
    targets = clean_audio[1:-1]

    m = len(targets)

    loss_history = []

    # Training process
    for epoch in range(epochs):

        # Make predictions
        predictions = (
            weights[0] * previous +
            weights[1] * current +
            weights[2] * following +
            bias
        )

        # Calculate errors
        errors = predictions - targets

        # Calculate gradients
        dw1 = np.sum(errors * previous) / m
        dw2 = np.sum(errors * current) / m
        dw3 = np.sum(errors * following) / m
        db = np.sum(errors) / m

        # Update weights and bias
        weights[0] -= learning_rate * dw1
        weights[1] -= learning_rate * dw2
        weights[2] -= learning_rate * dw3
        bias -= learning_rate * db

        # Calculate loss
        loss = calculate_loss(predictions, targets)
        loss_history.append(loss)

    return weights, bias, loss_history

# 8. AUDIO DENOISING
def denoise(noisy_audio, weights, bias):
    denoised_audio = np.zeros(len(noisy_audio))

    denoised_audio[1:-1] = predict(
        noisy_audio, weights, bias
    )

    # Preserve the first and last samples
    denoised_audio[0] = noisy_audio[0]
    denoised_audio[-1] = noisy_audio[-1]

    return denoised_audio

# 9. EVALUATE DENOISING
def evaluate(clean_audio, noisy_audio, denoised_audio):
    noisy_mse = np.mean((clean_audio - noisy_audio) ** 2)
    denoised_mse = np.mean((clean_audio - denoised_audio) ** 2)

    print("Noisy Audio MSE:", noisy_mse)
    print("Denoised Audio MSE:", denoised_mse)

# 10. VISUALIZE RESULTS
def plot_results(clean_audio, noisy_audio, denoised_audio, loss_history, sample_rate, result_name):

    # Plot the first 0.05 seconds
    samples_to_plot = int(sample_rate * 0.5)

    time = np.arange(samples_to_plot) / sample_rate

    plt.figure(figsize=(12, 8))

    # Original clean audio
    plt.subplot(4, 1, 1)
    plt.plot(time, clean_audio[:samples_to_plot])
    plt.title("Original Clean Audio")
    plt.ylabel("Amplitude")

    # Noisy audio
    plt.subplot(4, 1, 2)
    plt.plot(time, noisy_audio[:samples_to_plot])
    plt.title("Noisy Audio")
    plt.ylabel("Amplitude")

    # Denoised audio
    plt.subplot(4, 1, 3)
    plt.plot(time, denoised_audio[:samples_to_plot])
    plt.title("Denoised Audio")
    plt.ylabel("Amplitude")

    # Training loss
    plt.subplot(4, 1, 4)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Half-MSE")

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{result_name}_results.png", dpi=300)

    plt.close()

# 11. SAVE DENOISED AUDIO
def save_audio(file_path, audio, sample_rate):
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    with wave.open(file_path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio.tobytes())

def process_audio_pair(clean_path, noisy_path, name):
    clean_audio, sample_rate = read_audio(clean_path)
    clean_audio = normalize_audio(clean_audio)

    noisy_audio, noisy_sample_rate = read_audio(noisy_path)
    noisy_audio = normalize_audio(noisy_audio)

    if sample_rate != noisy_sample_rate:
        raise ValueError("Clean and noisy audio must have the same sample rate.")

    if len(clean_audio) != len(noisy_audio):
        raise ValueError("Clean and noisy audio must have the same length.")

    print("\nProcessing:", name)

    weights, bias, loss_history = train_model(noisy_audio, clean_audio)

    denoised_audio = denoise(noisy_audio, weights, bias)

    save_audio(f"output/{name}_denoised.wav", denoised_audio, sample_rate)

    evaluate(clean_audio, noisy_audio, denoised_audio)

    plot_results(
        clean_audio,
        noisy_audio,
        denoised_audio,
        loss_history,
        sample_rate,
        name
    )


# 12. MAIN PROGRAM
def main():

    os.makedirs("output", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # PROCESS YOUR OWN RECORDING
    clean_audio, sample_rate = read_audio("audio/clean_voice.wav")
    clean_audio = normalize_audio(clean_audio)

    noisy_audio = add_noise(clean_audio)

    save_audio("output/noisy_voice.wav", noisy_audio, sample_rate)

    weights, bias, loss_history = train_model(noisy_audio, clean_audio)

    denoised_audio = denoise(noisy_audio, weights, bias)

    save_audio("output/denoised_voice.wav", denoised_audio, sample_rate)

    evaluate(clean_audio, noisy_audio, denoised_audio)

    plot_results(
        clean_audio,
        noisy_audio,
        denoised_audio,
        loss_history,
        sample_rate,
        "custom"
    )

    # PROCESS DATASET RECORDINGS

    clean_folder = "dataset/clean"
    noisy_folder = "dataset/noisy"

    clean_files = sorted(os.listdir(clean_folder))

    # Start with 5 recordings
    for filename in clean_files[:5]:

        clean_path = os.path.join(clean_folder, filename)
        noisy_path = os.path.join(noisy_folder, filename)

        if not os.path.isfile(noisy_path):
            print("Missing noisy file:", filename)
            continue

        name = os.path.splitext(filename)[0]

        process_audio_pair(clean_path, noisy_path, name)


if __name__ == "__main__":
    main()

    