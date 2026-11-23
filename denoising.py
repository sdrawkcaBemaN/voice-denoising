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
def add_noise(clean_audio, noise_level=0.05):
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
def plot_results(clean_audio, noisy_audio, denoised_audio, loss_history, sample_rate):

    # Plot the first 0.05 seconds
    samples_to_plot = int(sample_rate * 0.05)

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

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/denoising_results.png", dpi=300)

    plt.show()

# 11. MAIN PROGRAM
def main():
    audio, sample_rate = read_audio("audio/clean_voice.wav")

    audio = normalize_audio(audio)

    """print("Normalized first 10 samples:", audio[:10])
    print("Maximum amplitude:", np.max(audio))
    print("Minimum amplitude:", np.min(audio))"""

    noisy_audio = add_noise(audio)

    """print("Noisy first 10 samples:", noisy_audio[:10])
    print("Noisy maximum amplitude:", np.max(noisy_audio))
    print("Noisy minimum amplitude:", np.min(noisy_audio))"""

    weights = np.array([0.0, 1.0, 0.0])
    bias = 0.0

    predictions = predict(noisy_audio, weights, bias)

    print("First 10 predictions:", predictions[:10])
    print("Number of predictions:", len(predictions))

    targets = audio[1:-1]

    loss = calculate_loss(predictions, targets)
    print("Initial Half-MSE:", loss)

    weights, bias, loss_history = train_model(noisy_audio, audio)

    print("Trained weights:", weights)
    print("Trained bias:", bias)
    print("Initial loss:", loss_history[0])
    print("Final loss:", loss_history[-1])

    denoised_audio = denoise(noisy_audio, weights, bias)

    print("Denoised first 10 samples:", denoised_audio[:10])
    print("Denoised maximum amplitude:", np.max(denoised_audio))
    print("Denoised minimum amplitude:", np.min(denoised_audio))
    print("Denoised samples:", len(denoised_audio))

    evaluate(audio, noisy_audio, denoised_audio)
    plot_results(audio, noisy_audio, denoised_audio, loss_history, sample_rate)

    """print("Sample rate:", sample_rate)
    print("Total samples:", len(audio))
    print("Duration:", len(audio) / sample_rate, "seconds")"""


if __name__ == "__main__":
    main()

    