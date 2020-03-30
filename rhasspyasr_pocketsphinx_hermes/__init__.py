"""Hermes MQTT server for Rhasspy ASR using Pocketsphinx"""
import gzip
import logging
import os
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import rhasspyasr_pocketsphinx
import rhasspynlu
from rhasspyasr import Transcriber
from rhasspyhermes.asr import (
    AsrAudioCaptured,
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame, AudioSessionFrame
from rhasspyhermes.base import Message
from rhasspyhermes.client import GeneratorType, HermesClient, TopicArgs
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce, G2pPronunciation
from rhasspynlu.g2p import PronunciationsType
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger("rhasspyasr_pocketsphinx_hermes")

# -----------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """Information about an open session."""

    sessionId: str
    start_listening: AsrStartListening
    recorder: typing.Optional[VoiceCommandRecorder] = None
    transcription_sent: bool = False
    num_wav_bytes: int = 0
    audio_buffer: typing.Optional[bytes] = None


@dataclass
class PronunciationDictionary:
    """Details of a phonetic dictionary."""

    path: Path
    pronunciations: PronunciationsType = field(default_factory=dict)
    mtime_ns: typing.Optional[int] = None


# -----------------------------------------------------------------------------


class AsrHermesMqtt(HermesClient):
    """Hermes MQTT server for Rhasspy ASR using Pocketsphinx."""

    def __init__(
        self,
        client,
        transcriber_factory: typing.Callable[[], Transcriber],
        dictionary: Path,
        language_model: Path,
        base_dictionaries: typing.Optional[typing.List[Path]] = None,
        dictionary_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        g2p_model: typing.Optional[Path] = None,
        g2p_word_transform: typing.Optional[typing.Callable[[str], str]] = None,
        unknown_words: typing.Optional[Path] = None,
        no_overwrite_train: bool = False,
        siteIds: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
    ):
        super().__init__(
            "rhasspyasr_pocketsphinx_hermes",
            client,
            siteIds=siteIds,
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        )

        self.subscribe(
            AsrToggleOn,
            AsrToggleOff,
            AsrStartListening,
            AsrStopListening,
            G2pPronounce,
            AudioFrame,
            AudioSessionFrame,
            AsrTrain,
        )

        self.make_transcriber = transcriber_factory
        self.transcriber: typing.Optional[Transcriber] = None

        # Files to write during training
        self.dictionary = dictionary
        self.language_model = language_model

        # Pronunciation dictionaries and word transform function
        base_dictionaries = base_dictionaries or []
        self.base_dictionaries = [
            PronunciationDictionary(path=path) for path in base_dictionaries
        ]
        self.dictionary_word_transform = dictionary_word_transform

        # Grapheme-to-phonme model (Phonetisaurus FST) and word transform
        # function.
        self.g2p_model = g2p_model
        self.g2p_word_transform = g2p_word_transform

        # Path to write missing words and guessed pronunciations
        self.unknown_words = unknown_words

        # If True, dictionary and language model won't be overwritten during training
        self.no_overwrite_train = no_overwrite_train

        # True if ASR system is enabled
        self.enabled = enabled
        self.disabled_reasons: typing.Set[str] = set()

        # No timeout
        def default_recorder():
            return WebRtcVadRecorder(max_seconds=None)

        self.make_recorder = make_recorder or default_recorder

        # WAV buffers for each session
        self.sessions: typing.Dict[str, SessionInfo] = {}

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    async def start_listening(self, message: AsrStartListening) -> None:
        """Start recording audio data for a session."""
        session = self.sessions.get(message.sessionId)
        if not session:
            session = SessionInfo(sessionId=message.sessionId, start_listening=message)

            if message.stopOnSilence:
                # Use voice command recorder
                session.recorder = self.make_recorder()
            else:
                # Use buffer
                session.audio_buffer = bytes()

            self.sessions[message.sessionId] = session

        # Start session
        assert session

        if session.recorder:
            session.recorder.start()

        _LOGGER.debug("Starting listening (sessionId=%s)", message.sessionId)
        self.first_audio = True

    async def stop_listening(
        self, message: AsrStopListening
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Stop recording audio data for a session."""
        try:
            session = self.sessions.pop(message.sessionId, None)
            if session:
                # Stop session
                if session.recorder:
                    audio_data = session.recorder.stop()
                else:
                    assert session.audio_buffer is not None
                    audio_data = session.audio_buffer

                wav_bytes = self.to_wav_bytes(audio_data)

                _LOGGER.debug(
                    "Received a total of %s byte(s) for WAV data for session %s",
                    session.num_wav_bytes,
                    message.sessionId,
                )

                if not session.transcription_sent:
                    # Send transcription
                    session.transcription_sent = True

                    yield (
                        await self.transcribe(
                            wav_bytes,
                            siteId=message.siteId,
                            sessionId=message.sessionId,
                        )
                    )

                    if session.start_listening.sendAudioCaptured:
                        # Send audio data
                        yield (
                            AsrAudioCaptured(wav_bytes=wav_bytes),
                            {"siteId": message.siteId, "sessionId": message.sessionId},
                        )

            _LOGGER.debug("Stopping listening (sessionId=%s)", message.sessionId)
        except Exception as e:
            _LOGGER.exception("stop_listening")
            yield AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=message.siteId,
                sessionId=message.sessionId,
            )

    async def handle_audio_frame(
        self,
        frame_wav_bytes: bytes,
        siteId: str = "default",
        sessionId: typing.Optional[str] = None,
    ) -> typing.AsyncIterable[
        typing.Union[
            AsrTextCaptured,
            AsrError,
            typing.Tuple[AsrAudioCaptured, typing.Dict[str, typing.Any]],
        ]
    ]:
        """Process single frame of WAV audio"""
        audio_data = self.maybe_convert_wav(frame_wav_bytes)

        if sessionId is None:
            # Add to every open session
            target_sessions = list(self.sessions.items())
        else:
            # Add to single session
            target_sessions = [(sessionId, self.sessions[sessionId])]

        # Add audio to session(s)
        for target_id, session in target_sessions:
            try:
                # Skip if siteId doesn't match
                if session.start_listening.siteId != siteId:
                    continue

                session.num_wav_bytes += len(frame_wav_bytes)
                if session.recorder:
                    # Check for end of voice command
                    command = session.recorder.process_chunk(audio_data)
                    if command and (command.result == VoiceCommandResult.SUCCESS):
                        assert command.audio_data is not None
                        _LOGGER.debug(
                            "Voice command recorded for session %s (%s byte(s))",
                            target_id,
                            len(command.audio_data),
                        )

                        session.transcription_sent = True
                        wav_bytes = self.to_wav_bytes(command.audio_data)

                        yield (
                            await self.transcribe(
                                wav_bytes, siteId=siteId, sessionId=target_id
                            )
                        )

                        if session.start_listening.sendAudioCaptured:
                            # Send audio data
                            yield (
                                AsrAudioCaptured(wav_bytes=wav_bytes),
                                {"siteId": siteId, "sessionId": target_id},
                            )

                        # Reset session (but keep open)
                        session.recorder.stop()
                        session.recorder.start()
                else:
                    # Add to buffer
                    assert session.audio_buffer is not None
                    session.audio_buffer += audio_data
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(self.transcriber),
                    siteId=siteId,
                    sessionId=target_id,
                )

    async def transcribe(
        self, wav_bytes: bytes, siteId: str = "default", sessionId: str = ""
    ) -> AsrTextCaptured:
        """Transcribe audio data and publish captured text."""
        try:
            if not self.transcriber:
                self.transcriber = self.make_transcriber()

            _LOGGER.debug("Transcribing %s byte(s) of audio data", len(wav_bytes))
            transcription = self.transcriber.transcribe_wav(wav_bytes)
            if transcription:
                # Actual transcription
                return AsrTextCaptured(
                    text=transcription.text,
                    likelihood=transcription.likelihood,
                    seconds=transcription.transcribe_seconds,
                    siteId=siteId,
                    sessionId=sessionId,
                )

            _LOGGER.warning("Received empty transcription")

        except Exception:
            _LOGGER.exception("transcribe")

        # Empty transcription
        return AsrTextCaptured(
            text="", likelihood=0, seconds=0, siteId=siteId, sessionId=sessionId
        )

    async def handle_train(
        self, train: AsrTrain, siteId: str = "default"
    ) -> typing.AsyncIterable[
        typing.Union[typing.Tuple[AsrTrainSuccess, TopicArgs], AsrError]
    ]:
        """Re-trains ASR system"""
        try:
            if not self.base_dictionaries:
                _LOGGER.warning(
                    "No base dictionaries provided. Training will likely fail."
                )

            # Load base dictionaries
            pronunciations: PronunciationsType = defaultdict(list)
            for base_dict in self.base_dictionaries:
                if not os.path.exists(base_dict.path):
                    _LOGGER.warning(
                        "Base dictionary does not exist: %s", base_dict.path
                    )
                    continue

                # Re-load dictionary if modification time has changed
                dict_mtime_ns = os.stat(base_dict.path).st_mtime_ns
                if (base_dict.mtime_ns is None) or (
                    base_dict.mtime_ns != dict_mtime_ns
                ):
                    base_dict.mtime_ns = dict_mtime_ns
                    _LOGGER.debug("Loading base dictionary from %s", base_dict.path)
                    with open(base_dict.path, "r") as base_dict_file:
                        rhasspynlu.g2p.read_pronunciations(
                            base_dict_file, word_dict=base_dict.pronunciations
                        )

                for word in base_dict.pronunciations:
                    pronunciations[word].extend(base_dict.pronunciations[word])

            if not self.no_overwrite_train:
                _LOGGER.debug("Loading %s", train.graph_path)
                with gzip.GzipFile(train.graph_path, mode="rb") as graph_gzip:
                    graph = nx.readwrite.gpickle.read_gpickle(graph_gzip)

                # Generate dictionary/language model
                _LOGGER.debug("Starting training")
                rhasspyasr_pocketsphinx.train(
                    graph,
                    self.dictionary,
                    self.language_model,
                    pronunciations,
                    dictionary_word_transform=self.dictionary_word_transform,
                    g2p_model=self.g2p_model,
                    g2p_word_transform=self.g2p_word_transform,
                    missing_words_path=self.unknown_words,
                )
            else:
                _LOGGER.warning("Not overwriting dictionary/language model")

            _LOGGER.debug("Re-loading transcriber")
            self.transcriber = self.make_transcriber()

            yield (AsrTrainSuccess(id=train.id), {"siteId": siteId})
        except Exception as e:
            _LOGGER.exception("handle_train")
            yield AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=siteId,
                sessionId=train.id,
            )

    async def handle_pronounce(
        self, pronounce: G2pPronounce
    ) -> typing.AsyncIterable[typing.Union[G2pPhonemes, G2pError]]:
        """Looks up or guesses word pronunciation(s)."""
        try:
            result = G2pPhonemes(
                id=pronounce.id, siteId=pronounce.siteId, sessionId=pronounce.sessionId
            )

            # Load base dictionaries
            pronunciations: typing.Dict[str, typing.List[typing.List[str]]] = {}

            for base_dict in self.base_dictionaries:
                if base_dict.path.is_file():
                    _LOGGER.debug("Loading base dictionary from %s", base_dict.path)
                    with open(base_dict.path, "r") as base_dict_file:
                        rhasspynlu.g2p.read_pronunciations(
                            base_dict_file, word_dict=pronunciations
                        )

            # Try to look up in dictionary first
            missing_words: typing.Set[str] = set()
            if pronunciations:
                for word in pronounce.words:
                    # Handle case transformation
                    if self.dictionary_word_transform:
                        word = self.dictionary_word_transform(word)

                    word_prons = pronunciations.get(word)
                    if word_prons:
                        # Use dictionary pronunciations
                        result.wordPhonemes[word] = [
                            G2pPronunciation(phonemes=p, guessed=False)
                            for p in word_prons
                        ]
                    else:
                        # Will have to guess later
                        missing_words.add(word)
            else:
                # All words must be guessed
                missing_words.update(pronounce.words)

            if missing_words:
                if self.g2p_model:
                    _LOGGER.debug("Guessing pronunciations of %s", missing_words)
                    guesses = rhasspynlu.g2p.guess_pronunciations(
                        missing_words,
                        self.g2p_model,
                        g2p_word_transform=self.g2p_word_transform,
                        num_guesses=pronounce.numGuesses,
                    )

                    # Add guesses to result
                    for guess_word, guess_phonemes in guesses:
                        result_phonemes = result.wordPhonemes.get(guess_word) or []
                        result_phonemes.append(
                            G2pPronunciation(phonemes=guess_phonemes, guessed=True)
                        )
                        result.wordPhonemes[guess_word] = result_phonemes
                else:
                    _LOGGER.warning("No g2p model. Cannot guess pronunciations.")

            yield result
        except Exception as e:
            _LOGGER.exception("handle_pronounce")
            yield G2pError(
                error=str(e),
                context=repr(self.transcriber),
                id=pronounce.id,
                siteId=pronounce.siteId,
                sessionId=pronounce.sessionId,
            )

    # -------------------------------------------------------------------------

    async def on_message(
        self,
        message: Message,
        siteId: typing.Optional[str] = None,
        sessionId: typing.Optional[str] = None,
        topic: typing.Optional[str] = None,
    ) -> GeneratorType:
        """Received message from MQTT broker."""
        # Check enable/disable messages
        if isinstance(message, AsrToggleOn):
            self.disabled_reasons.discard(message.reason)
            if self.disabled_reasons:
                _LOGGER.debug("Still disabled: %s", self.disabled_reasons)
            else:
                self.enabled = True
                self.first_audio = True
                _LOGGER.debug("Enabled")
        elif isinstance(message, AsrToggleOff):
            self.enabled = False
            self.disabled_reasons.add(message.reason)
            _LOGGER.debug("Disabled")
        elif isinstance(message, AudioFrame):
            if self.enabled:
                assert siteId, "Missing siteId"
                if self.first_audio:
                    _LOGGER.debug("Receiving audio")
                    self.first_audio = False

                # Add to all active sessions
                async for frame_result in self.handle_audio_frame(
                    message.wav_bytes, siteId=siteId
                ):
                    yield frame_result
        elif isinstance(message, AudioSessionFrame):
            if self.enabled:
                assert siteId and sessionId, "Missing siteId or sessionId"
                if sessionId in self.sessions:
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    # Add to specific session only
                    async for session_frame_result in self.handle_audio_frame(
                        message.wav_bytes, siteId=siteId, sessionId=sessionId
                    ):
                        yield session_frame_result
        elif isinstance(message, AsrStartListening):
            # hermes/asr/startListening
            await self.start_listening(message)
        elif isinstance(message, AsrStopListening):
            # hermes/asr/stopListening
            async for stop_result in self.stop_listening(message):
                yield stop_result
        elif isinstance(message, AsrTrain):
            # rhasspy/asr/<siteId>/train
            assert siteId, "Missing siteId"
            async for train_result in self.handle_train(message, siteId=siteId):
                yield train_result
        elif isinstance(message, G2pPronounce):
            # rhasspy/g2p/pronounce
            async for pronounce_result in self.handle_pronounce(message):
                yield pronounce_result
        else:
            _LOGGER.warning("Unexpected message: %s", message)
