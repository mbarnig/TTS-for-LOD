import os
import pygame
import time

def play_audio_files_in_folder(folder_path):
    # Initialize the mixer module in pygame
    pygame.mixer.init()

    # List all files in the directory
    files = os.listdir(folder_path)

    # Filter the list to include only audio files
    audio_files = [file for file in files if file.endswith(('.mp3', '.wav', '.ogg'))]

    if not audio_files:
        print("No audio files found in the specified folder.")
        return

    # Play each audio file
    for audio_file in audio_files:
        file_path = os.path.join(folder_path, audio_file)
        print(f"Playing: {audio_file}")

        # Load and play the audio file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait until the audio file finishes playing
        while pygame.mixer.music.get_busy():
            time.sleep(1)

    # Quit the mixer module
    pygame.mixer.quit()

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing audio files: ")
    play_audio_files_in_folder(folder_path)