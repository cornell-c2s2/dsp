# Requires FFMPEG

import os
import argparse
import subprocess


def convert_wav_to_16kHz_ffmpeg(folder):
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root, filename)
                temp_file = file_path + ".tmp.wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        file_path,
                        "-ar",
                        "16000",  # 16 kHz
                        "-ac",
                        "1",  # mono
                        temp_file,
                    ],
                    check=True,
                )

                os.replace(temp_file, file_path)
                print(f"Converted {file_path} to 16kHz")

    print("All WAV files in folder and subfolders have been converted successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all WAV files in folder and subfolders to 16kHz using ffmpeg"
    )
    parser.add_argument(
        "folder", type=str, help="Path to the folder containing WAV files"
    )
    args = parser.parse_args()

    convert_wav_to_16kHz_ffmpeg(args.folder)
