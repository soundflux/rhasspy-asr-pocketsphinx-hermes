"""Hermes MQTT server for Rhasspy ASR using Pocketsphinx"""
import io
import json
import logging
import subprocess
import typing
import wave
from pathlib import Path

import attr
import rhasspyasr_pocketsphinx
from rhasspyasr import Transcriber
from rhasspyhermes.asr import (
    AsrError,
    AsrStartListening,
    AsrStopListening,
    AsrTextCaptured,
    AsrToggleOff,
    AsrToggleOn,
    AsrTrain,
    AsrTrainSuccess,
)
from rhasspyhermes.audioserver import AudioFrame
from rhasspyhermes.base import Message
from rhasspyhermes.g2p import G2pError, G2pPhonemes, G2pPronounce, G2pPronunciation
from rhasspysilence import VoiceCommandRecorder, VoiceCommandResult, WebRtcVadRecorder

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


@attr.s(auto_attribs=True, slots=True)
class SessionInfo:
    """Information about an open session."""

    sessionId: str
    recorder: VoiceCommandRecorder
    transcription_sent: bool = False


# -----------------------------------------------------------------------------


class AsrHermesMqtt:
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
        siteIds: typing.Optional[typing.List[str]] = None,
        enabled: bool = True,
        sample_rate: int = 16000,
        sample_width: int = 2,
        channels: int = 1,
        make_recorder: typing.Callable[[], VoiceCommandRecorder] = None,
    ):
        self.client = client
        self.make_transcriber = transcriber_factory
        self.transcriber: typing.Optional[Transcriber] = None
        self.dictionary = dictionary
        self.language_model = language_model
        self.base_dictionaries = base_dictionaries or []
        self.dictionary_word_transform = dictionary_word_transform
        self.g2p_model = g2p_model
        self.g2p_word_transform = g2p_word_transform
        self.siteIds = siteIds or []
        self.enabled = enabled

        # Required audio format
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

        # No timeout
        def default_recorder():
            return WebRtcVadRecorder(max_seconds=None)

        self.make_recorder = make_recorder or default_recorder

        # WAV buffers for each session
        self.sessions: typing.Dict[str, SessionInfo] = {}

        # Topic to listen for WAV chunks on
        self.audioframe_topics: typing.List[str] = [
            AudioFrame.topic(siteId=siteId) for siteId in self.siteIds
        ]

        self.first_audio: bool = True

    # -------------------------------------------------------------------------

    def start_listening(self, message: AsrStartListening):
        """Start recording audio data for a session."""
        session = self.sessions.get(message.sessionId)
        if not session:
            session = SessionInfo(
                sessionId=message.sessionId, recorder=self.make_recorder()
            )
            self.sessions[message.sessionId] = session

        # Start session
        assert session
        session.recorder.start()
        _LOGGER.debug("Starting listening (sessionId=%s)", message.sessionId)
        self.first_audio = True

    def stop_listening(
        self, message: AsrStopListening
    ) -> typing.Iterable[typing.Union[AsrTextCaptured, AsrError]]:
        """Stop recording audio data for a session."""
        try:
            session = self.sessions.pop(message.sessionId, None)
            if session:
                # Stop session
                audio_data = session.recorder.stop()
                if not session.transcription_sent and audio_data:
                    # Send transcription
                    session.transcription_sent = True

                    yield self.transcribe(
                        audio_data, siteId=message.siteId, sessionId=message.sessionId
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

    def handle_audio_frame(
        self, wav_bytes: bytes, siteId: str = "default"
    ) -> typing.Iterable[typing.Union[AsrTextCaptured, AsrError]]:
        """Process single frame of WAV audio"""
        audio_data = self.maybe_convert_wav(wav_bytes)

        # Add to every open session
        for sessionId, session in self.sessions.items():
            try:
                command = session.recorder.process_chunk(audio_data)
                if command and (command.result == VoiceCommandResult.SUCCESS):
                    assert command.audio_data is not None
                    _LOGGER.debug(
                        "Voice command recorded for session %s (%s byte(s))",
                        sessionId,
                        len(command.audio_data),
                    )

                    session.transcription_sent = True

                    yield self.transcribe(
                        command.audio_data, siteId=siteId, sessionId=sessionId
                    )

                    # Reset session (but keep open)
                    session.recorder.stop()
                    session.recorder.start()
            except Exception as e:
                _LOGGER.exception("handle_audio_frame")
                yield AsrError(
                    error=str(e),
                    context=repr(self.transcriber),
                    siteId=siteId,
                    sessionId=sessionId,
                )

    def transcribe(
        self, audio_data: bytes, siteId: str = "default", sessionId: str = ""
    ) -> typing.Union[AsrTextCaptured, AsrError]:
        """Transcribe audio data and publish captured text."""
        try:
            if not self.transcriber:
                self.transcriber = self.make_transcriber()

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
                    return AsrTextCaptured(
                        text=transcription.text,
                        likelihood=transcription.likelihood,
                        seconds=transcription.transcribe_seconds,
                        siteId=siteId,
                        sessionId=sessionId,
                    )

                _LOGGER.warning("Received empty transcription")

                # Empty transcription
                return AsrTextCaptured(
                    text="", likelihood=0, seconds=0, siteId=siteId, sessionId=sessionId
                )
        except Exception as e:
            _LOGGER.exception("transcribe")
            return AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=siteId,
                sessionId=sessionId,
            )

    def handle_train(
        self, train: AsrTrain, siteId: str = "default"
    ) -> typing.Union[AsrTrainSuccess, AsrError]:
        """Re-trains ASR system"""
        try:
            if not self.base_dictionaries:
                _LOGGER.warning(
                    "No base dictionaries provided. Training will likely fail."
                )

            graph = rhasspynlu.json_to_graph(train.graph_dict)

            # Generate counts
            intent_counts = rhasspynlu.get_intent_ngram_counts(graph)

            # pylint: disable=W0511
            # TODO: Balance counts

            # Use mitlm to create language model
            vocabulary: typing.Set[str] = set()

            with tempfile.NamedTemporaryFile(mode="w") as lm_file:

                # Create ngram counts
                with tempfile.NamedTemporaryFile(mode="w") as count_file:
                    for ngram_count in intent_counts.values():
                        for ngram, count in ngram_count.items():
                            # word [word] ... <TAB> count
                            print(*ngram, "\t", count, file=count_file)

                    count_file.seek(0)
                    with tempfile.NamedTemporaryFile(mode="w+") as vocab_file:
                        ngram_command = [
                            "estimate-ngram",
                            "-order",
                            "3",
                            "-counts",
                            count_file.name,
                            "-write-lm",
                            lm_file.name,
                            "-write-vocab",
                            vocab_file.name,
                        ]

                        _LOGGER.debug(ngram_command)
                        subprocess.check_call(ngram_command)

                        # Extract vocabulary
                        vocab_file.seek(0)
                        for line in vocab_file:
                            line = line.strip()
                            if not line.startswith("<"):
                                vocabulary.add(line)

                # Write dictionary
                if vocabulary:

                    # Load base dictionaries
                    pronunciations: typing.Dict[str, typing.List[str]] = {}

                    for base_dict_path in self.base_dictionaries:
                        _LOGGER.debug("Loading base dictionary from %s", base_dict_path)
                        with open(base_dict_path, "r") as base_dict_file:
                            rhasspyasr.utils.read_dict(
                                base_dict_file, word_dict=pronunciations
                            )

                    with tempfile.NamedTemporaryFile(mode="w") as dict_file:
                        missing_words: typing.Set[str] = set()

                        # Look up each word
                        for word in vocabulary:
                            word_phonemes = pronunciations.get(word)
                            if not word_phonemes:
                                # Add to missing word list
                                _LOGGER.warning("Missing word '%s'", word)
                                missing_words.add(word)
                                continue

                            # Write CMU format
                            for i, phonemes in enumerate(word_phonemes):
                                if i == 0:
                                    print(word, phonemes, file=dict_file)
                                else:
                                    print(f"{word}({i+1})", phonemes, file=dict_file)

                        if missing_words:
                            # Fail if no g2p model is available
                            if not self.g2p_model:
                                raise MissingWordPronunciationsException(
                                    list(missing_words)
                                )

                            # Guess word pronunciations
                            _LOGGER.debug(
                                "Guessing pronunciations for %s", missing_words
                            )
                            with tempfile.NamedTemporaryFile(mode="w") as wordlist_file:
                                # pylint: disable=W0511
                                # TODO: Handle casing
                                print(*missing_words, file=wordlist_file, sep="\n")

                                wordlist_file.seek(0)
                                g2p_command = [
                                    "phonetisaurus-apply",
                                    "--model",
                                    str(self.g2p_model),
                                    "--word_list",
                                    wordlist_file.name,
                                    "--nbest",
                                    "1",
                                ]

                                _LOGGER.debug(g2p_command)
                                g2p_lines = subprocess.check_output(
                                    g2p_command, universal_newlines=True
                                ).splitlines()
                                for line in g2p_lines:
                                    line = line.strip()
                                    if line:
                                        word, *parts = line.split()
                                        phonemes = " ".join(parts).strip()
                                        print(word.strip(), phonemes, file=dict_file)

                        # -----------------------------------------------------

                        # Copy dictionary
                        dict_file.seek(0)
                        shutil.copy(dict_file.name, self.transcriber.dictionary)
                        _LOGGER.debug(
                            "Wrote dictionary to %s", str(self.transcriber.dictionary)
                        )

                # -------------------------------------------------------------

                # Copy language model
                lm_file.seek(0)
                shutil.copy(lm_file.name, self.transcriber.language_model)
                _LOGGER.debug(
                    "Wrote language model to %s", str(self.transcriber.language_model)
                )

            # Generate dictionary/language model
            rhasspyasr_pocketsphinx.train(
                train.graph_dict,
                self.dictionary,
                self.language_model,
                self.base_dictionaries,
                dictionary_word_transform=self.dictionary_word_transform,
                g2p_model=self.g2p_model,
                g2p_word_transform=self.g2p_word_transform,
            )

            _LOGGER.debug("Re-loading transcriber")
            self.transcriber = self.make_transcriber()

            return AsrTrainSuccess(id=train.id)
        except Exception as e:
            _LOGGER.exception("handle_train")
            return AsrError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=siteId,
                sessionId=train.id,
            )

    def handle_pronounce(
        self, pronounce: G2pPronounce
    ) -> typing.Union[G2pPhonemes, G2pError]:
        """Looks up or guesses word pronunciation(s)."""
        try:
            result = G2pPhonemes(
                id=pronounce.id, siteId=pronounce.siteId, sessionId=pronounce.sessionId
            )

            # Load base dictionaries
            pronunciations: typing.Dict[str, typing.List[typing.List[str]]] = {}

            for base_dict_path in self.base_dictionaries:
                _LOGGER.debug("Loading base dictionary from %s", base_dict_path)
                with open(base_dict_path, "r") as base_dict_file:
                    rhasspyasr_pocketsphinx.read_dict(
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
                    guesses = rhasspyasr_pocketsphinx.guess_pronunciations(
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

            return result
        except Exception as e:
            _LOGGER.exception("handle_pronounce")
            return G2pError(
                error=str(e),
                context=repr(self.transcriber),
                siteId=pronounce.siteId,
                sessionId=pronounce.id,
            )

    # -------------------------------------------------------------------------

    def on_connect(self, client, userdata, flags, rc):
        """Connected to MQTT broker."""
        try:
            topics = [
                AsrToggleOn.topic(),
                AsrToggleOff.topic(),
                AsrStartListening.topic(),
                AsrStopListening.topic(),
                G2pPronounce.topic(),
            ]

            if self.audioframe_topics:
                # Specific siteIds
                topics.extend(self.audioframe_topics)
            else:
                # All siteIds
                topics.append(AudioFrame.topic(siteId="+"))

            if self.siteIds:
                # Specific siteIds
                topics.extend(
                    [AsrTrain.topic(siteId=siteId) for siteId in self.siteIds]
                )
            else:
                # All siteIds
                topics.append(AsrTrain.topic(siteId="+"))

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

            if AudioFrame.is_topic(msg.topic):
                # Check siteId
                if (not self.audioframe_topics) or (
                    msg.topic in self.audioframe_topics
                ):
                    # Add to all active sessions
                    if self.first_audio:
                        _LOGGER.debug("Receiving audio")
                        self.first_audio = False

                    siteId = AudioFrame.get_siteId(msg.topic)
                    for result in self.handle_audio_frame(msg.payload, siteId=siteId):
                        self.publish(result)

            elif msg.topic == AsrStartListening.topic():
                # hermes/asr/startListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    self.start_listening(AsrStartListening(**json_payload))
            elif msg.topic == AsrStopListening.topic():
                # hermes/asr/stopListening
                json_payload = json.loads(msg.payload)
                if self._check_siteId(json_payload):
                    for result in self.stop_listening(AsrStopListening(**json_payload)):
                        self.publish(result)
            elif AsrTrain.is_topic(msg.topic):
                # rhasspy/asr/<siteId>/train
                siteId = AsrTrain.get_siteId(msg.topic)
                if (not self.siteIds) or (siteId in self.siteIds):
                    json_payload = json.loads(msg.payload)
                    result = self.handle_train(AsrTrain(**json_payload), siteId=siteId)
                    self.publish(result)
            elif msg.topic == G2pPronounce.topic():
                # rhasspy/g2p/pronounce
                json_payload = json.loads(msg.payload or "{}")
                if self._check_siteId(json_payload):
                    result = self.handle_pronounce(G2pPronounce(**json_payload))
                    self.publish(result)
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
