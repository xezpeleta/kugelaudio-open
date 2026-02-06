#!/usr/bin/env python3
"""Basic example of generating speech with KugelAudio.

All generated audio is automatically watermarked for identification.
Voice cloning is not supported. Use pre-encoded voices from the
voices.json registry instead.
"""

import torch

from kugelaudio_open import (
    AudioWatermark,
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)


def main():
    # Configuration
    model_id = "kugelaudio/kugelaudio-0-open"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model {model_id}...")

    # Load model and processor
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    # Strip encoder weights to save VRAM (only decoders needed for inference)
    model.model.strip_encoders()

    processor = KugelAudioProcessor.from_pretrained(model_id)

    # Show available pre-encoded voices
    voices = processor.get_available_voices()
    print(f"Available voices: {voices}")

    # Text to synthesize
    text = "Hello! This is a demonstration of KugelAudio text-to-speech synthesis."

    print(f"Generating speech for: '{text}'")

    # Process input (optionally pass voice="default" to use a pre-encoded voice)
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate speech (watermark is automatically applied)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=3.0,
            max_new_tokens=2048,
        )

    audio = outputs.speech_outputs[0]

    # Verify watermark is present
    print("Verifying watermark...")
    watermark = AudioWatermark()
    result = watermark.detect(audio)
    print(f"Watermark detected: {result.detected}, confidence: {result.confidence:.2%}")

    # Save output
    output_path = "output.wav"
    processor.save_audio(audio, output_path)
    print(f"Audio saved to {output_path}")


if __name__ == "__main__":
    main()
