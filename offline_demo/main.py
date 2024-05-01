"""Main file for the Jarvis project"""
import os
from os import PathLike
from time import time
import asyncio
from typing import Union

from dotenv import load_dotenv
import openai
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
import pygame
from pygame import mixer
import elevenlabs

from record import speech_to_text

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))

gpt_client = openai.Client(api_key=OPENAI_API_KEY)
deepgram = DeepgramClient()
mixer.init()

context = "You are Alex, Dheeraj's human assistant. You should answer in 1-2 short sentences."
conversation = {"Conversation": []}
RECORDING_PATH = "audio/recording.wav"


def request_gpt(prompt: str) -> str:
    response = gpt_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content


def transcribe(
    file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]
):
    with open(file_name, "rb") as audio:
        source: FileSource = {"buffer": audio}
        options = PrerecordedOptions(
            smart_format=True,
            summarize="v2",
        )
        response  = deepgram.listen.prerecorded.v("1").transcribe_file(source, options)
        print(f"{response.to_json()}")
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]


def log(log: str):
    print(log)
    with open("status.txt", "w") as f:
        f.write(log)


if __name__ == "__main__":
    while True:
        log("Listening...")
        speech_to_text()
        log("Done listening")

        current_time = time()
        string_words = transcribe(RECORDING_PATH)
        with open("conv.txt", "a") as f:
            f.write(f"{string_words}\n")
        transcription_time = time() - current_time
        log(f"Finished transcribing in {transcription_time:.2f} seconds.")
        current_time = time()
        context += f"\Dheeraj: {string_words}\Alex: "
        response = request_gpt(context)
        context += response
        gpt_time = time() - current_time
        log(f"Finished generating response in {gpt_time:.2f} seconds.")

        current_time = time()
        audio = elevenlabs.generate(
            text=response, voice="Adam", model="eleven_monolingual_v1"
        )
        elevenlabs.save(audio, "audio/response.wav")
        audio_time = time() - current_time
        log(f"Finished generating audio in {audio_time:.2f} seconds.")

        log("Speaking...")
        sound = mixer.Sound("audio/response.wav")
        with open("conv.txt", "a") as f:
            f.write(f"{response}\n")
        sound.play()
        pygame.time.wait(int(sound.get_length() * 1000))
        print(f"\n --- Dheeraj: {string_words}\n --- Alex: {response}\n")
