"""Hermes MQTT server for Rhasspy ASR using Pocketsphinx"""
import io
import json
import logging
import subprocess
import typing
import wave
from collections import defaultdict

import attr

from rhasspyhermes.base import Message
from rhasspyhermes.asr import (
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOn,
    AsrToggleOff,
)
from rhasspyhermes.audioserver import AudioFrame
from rhasspyasr import Transcriber
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger(__name__)


class AsrHermesMqtt:
    """Hermes MQTT server for Rhasspy ASR using Pocketsphinx."""

    def __init__(
        self,
        client,
        transcriber: Transcriber,
        siteIds: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
    ):
        self.client = client
        self.transcriber = transcriber
        self.siteIds = siteIds or []
        self.enabled = enabled

        # Required audio format
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        # No timeout
        self.make_recorder = make_recorder or (
            lambda: WebRtcVadRecorder(max_seconds=None)
        )

        # WAV buffers for each session
        self.session_recorders = defaultdict(VoiceCommandRecorder)

        # Topic to listen for WAV chunks on
        self.audioframe_topics: typing.List[str] = []
        for siteId in self.siteIds:
            self.audioframe_topics.append(AudioFrame.topic(siteId=siteId))

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    def start_listening(self, message: AsrStartListening):
        """Start recording audio data for a session."""
        if message.sessionId not in self.session_recorders:
            self.session_recorders[message.sessionId] = self.make_recorder()

        # Start session
        self.session_recorders[message.sessionId].start()
        _LOGGER.debug("Starting listening (sessionId=%s)", message.sessionId)
        self.first_audio = True

    def stop_listening(self, message: AsrStopListening):
        """Stop recording audio data for a session."""
        if message.sessionId in self.session_recorders:
            # Stop session
            self.session_recorders[message.sessionId].stop()
            self.session_recorders.pop(message.sessionId)

        _LOGGER.debug("Stopping listening (sessionId=%s)", message.sessionId)

    def transcribe(
        self, audio_data: bytes, siteId: str = "default", sessionId: str = ""
    ):
        """Transcribe audio data and publish captured text."""
        try:
            with io.BytesIO() as wav_buffer:
                wav_file: wave.Wave_write = wave.open(wav_buffer, mode="wb")
                with wav_file:
                    wav_file.setframerate(self.sample_rate)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setnchannels(self.channels)
                    wav_file.writeframesraw(audio_data)

                transcription = self.transcriber.transcribe_wav(wav_buffer.getvalue())
                if transcription:
                    # Actual transcription
                    self.publish(
                        AsrTextCaptured(
                            text=transcription.text,
                            likelihood=transcription.likelihood,
                            seconds=transcription.transcribe_seconds,
                            siteId=siteId,
                            sessionId=sessionId,
                        )
                    )
                else:
                    _LOGGER.warning("Received empty transcription")

                    # Empty transcription
                    self.publish(
                        AsrTextCaptured(
                            text="",
                            likelihood=0,
                            seconds=0,
                            siteId=siteId,
                            sessionId=sessionId,
                        )
                    )
        except Exception:
            _LOGGER.exception("transcribe")

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [
                AsrToggleOn.topic(),
                AsrToggleOff.topic(),
                AsrStartListening.topic(),
                AsrStopListening.topic(),
            ]

            if self.audioframe_topics:
                # Specific siteIds
                topics.extend(self.audioframe_topics)
            else:
                # All siteIds
                topics.append(AudioFrame.topic(siteId="#"))

            for topic in topics:
                self.client.subscribe(topic)
                _LOGGER.debug("Subscribed to %s", topic)
        except Exception:
            _LOGGER.exception("on_connect")

    def on_message(self, client, userdata, msg):
        """Received message from MQTT broker."""
        try:
            if not msg.topic.endswith("/audioFrame"):
                _LOGGER.debug("Received %s byte(s) on %s", len(msg.payload), msg.topic)

            # Check enable/disable messages
            if msg.topic == AsrToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = True
                    _LOGGER.debug("Enabled")
            elif msg.topic == AsrToggleOn.topic():
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    self.enabled = False
                    _LOGGER.debug("Disabled")

            if not self.enabled:
                # Disabled
                return

            if (not self.audioframe_topics) or (msg.topic in self.audioframe_topics):
                # Add to all active sessions
                if self.first_audio:
                    _LOGGER.debug("Receiving audio")
                    self.first_audio = False

                # Extract audio data.
                audio_data = self.maybe_convert_wav(msg.payload)

                # Add to every open session
                siteId = AudioFrame.get_siteId(msg.topic)
                for sessionId, recorder in self.session_recorders.items():
                    command = recorder.process_chunk(audio_data)
                    if command and (command.result == VoiceCommandResult.SUCCESS):
                        _LOGGER.debug(
                            "Voice command recorded for session %s (%s byte(s))",
                            sessionId,
                            len(command.audio_data),
                        )
                        self.transcribe(
                            command.audio_data, siteId=siteId, sessionId=sessionId
                        )

                        # Reset session (but keep open)
                        recorder.stop()
                        recorder.start()

            elif msg.topic == AsrStartListening.topic():
                # hermes/asr/startListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.start_listening(AsrStartListening(**json_payload))
            elif msg.topic == AsrStopListening.topic():
                # hermes/asr/stopListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.stop_listening(AsrStopListening(**json_payload))
        except Exception:
            _LOGGER.exception("on_message")

    def publish(self, message: Message, **topic_args):
        """Publish a Hermes message to MQTT."""
        try:
            _LOGGER.debug("-> %s", message)
            topic = message.topic(**topic_args)
            payload = json.dumps(attr.asdict(message))
            _LOGGER.debug("Publishing %s char(s) to %s", len(payload), topic)
            self.client.publish(topic, payload)
        except Exception:
            _LOGGER.exception("on_message")

    # -------------------------------------------------------------------------

    def _check_siteId(self, json_payload: typing.Dict[str, typing.Any]) -> bool:
        if self.siteIds:
            return json_payload.get("siteId", "default") in self.siteIds

        # All sites
        return True

    # -------------------------------------------------------------------------

    def _convert_wav(self, wav_data: bytes) -> bytes:
        """Converts WAV data to required format with sox. Return raw audio."""
        return subprocess.run(
            [
                "sox",
                "-t",
                "wav",
                "-",
                "-r",
                str(self.sample_rate),
                "-e",
                "signed-integer",
                "-b",
                str(self.sample_width * 8),
                "-c",
                str(self.channels),
                "-t",
                "raw",
                "-",
            ],
            check=True,
            stdout=subprocess.PIPE,
            input=wav_data,
        ).stdout

    def maybe_convert_wav(self, wav_bytes: bytes) -> bytes:
        """Converts WAV data to required format if necessary. Returns raw audio."""
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io, "rb") as wav_file:
                if (
                    (wav_file.getframerate() != self.sample_rate)
                    or (wav_file.getsampwidth() != self.sample_width)
                    or (wav_file.getnchannels() != self.channels)
                ):
                    # Return converted wav
                    return self._convert_wav(wav_bytes)

                # Return original audio
                return wav_file.readframes(wav_file.getnframes())
