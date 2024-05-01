import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import av
import threading
import numpy as np
import speech_recognition as sr


from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from speech_recognition import AudioSource
from dotenv import load_dotenv
import openai

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
gpt_client = openai.Client(api_key=OPENAI_API_KEY)

class AudioTransformationTrackO(MediaStreamTrack):

    kind = "audio"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        print(frame)
        return frame
    
class WebRTCSource(AudioSource):
    def __init__(self, sample_rate=None, chunk_size=1024, sample_width=4):
        # Those are the only 4 properties required by the recognizer.listen method
        self.stream = WebRTCSource.MicrophoneStream()
        self.SAMPLE_RATE = sample_rate  # sampling rate in Hertz
        self.CHUNK = chunk_size  # number of frames stored in each buffer
        self.SAMPLE_WIDTH = sample_width  # size of each sample

    class MicrophoneStream(object):
        def __init__(self):
            self.stream = av.AudioFifo()
            self.event = threading.Event()

        def write(self, frame: av.AudioFrame):
            assert type(frame) is av.AudioFrame, "Tried to write something that is not AudioFrame"
            self.stream.write(frame=frame)
            self.event.set()

        def read(self, size) -> bytes:
            frames: av.AudioFrame = self.stream.read(size)

            # while no frame, wait until some is written using an event
            while frames is None:
                self.event.wait()
                self.event.clear()
                frames = self.stream.read(size)

            # convert the frame to bytes
            data: np.ndarray = frames.to_ndarray()
            return data.tobytes()

    
class AudioTransformTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        rate = 16_000  # Whisper has a sample rate of 16000
        audio_format = 's16p'
        sample_width = av.AudioFormat(audio_format).bytes
        self.resampler = av.AudioResampler(format=audio_format, layout='mono', rate=rate)
        self.source = WebRTCSource(sample_rate=rate, sample_width=sample_width)

    async def recv(self):
        out_frame: av.AudioFrame = await self.track.recv()
        print(out_frame)

        out_frames = self.resampler.resample(out_frame)

        for frame in out_frames:
            self.source.stream.write(frame)

        return out_frame
    
def listen(source: AudioSource):
    recognizer = sr.Recognizer()

    while True:
        audio = recognizer.listen(source)
        # do something with the audio...
        print("Sphinx thinks you said " + recognizer.recognize_sphinx(audio))

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

async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    context = "You are Helper, Dheeraj's human assistant. You are witty and full of personality. Your answers should be limited to 1-2 short sentences."


    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str):
                nonlocal context
                context += f"\Dheeaj: {message}\nHelper: "
                response = request_gpt(context)
                context += response
                channel.send(response)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "audio":
            # t = AudioTransformTrack(relay.subscribe(track))
            # thread = threading.Thread(target=listen, args=(t.source,))
            # thread.start()

            pc.addTrack(AudioTransformationTrackO(relay.subscribe(track)))
            recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
