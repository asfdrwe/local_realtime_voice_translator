# SPDX-License-Identifier: CC0-1.0
#
# This work is marked with CC0 1.0 Universal. 
# To view a copy of this license, visit http://creativecommons.org

import queue
import base64
import sys
import os
import json
import math
import time
import wave
import tempfile
import numpy as np
import requests
import sounddevice as sd

from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QSettings
from PySide6.QtGui import QIcon, QAction, QFont, QPixmap, QColor, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox,
    QDoubleSpinBox, QTextEdit, QGroupBox, QSystemTrayIcon, QMenu,
    QMessageBox, QSplitter
)

DEFAULT_VAD_THRESHOLD = 0.3
DEFAULT_INPUT_GAIN = 1.0
DEFAULT_MIN_DYNAMIC_REF_LEN = 0.5
MAX_VOICE_REF_BYTES = 5 * 1024 * 1024

DEFAULT_SYSTEM_PROMPT = (
    "You are a real-time speech translator. Translate the given text between {lang_a} and {lang_b}. "
    "If the input text is in {lang_a}, translate it to {lang_b}. "
    "If it is in {lang_b}, translate it to {lang_a}. "
    "Output ONLY the raw translated text without any explanation, markdown formatting, or notes."
)

def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio_data
    duration = len(audio_data) / orig_sr
    target_length = int(len(audio_data) * target_sr / orig_sr)
    orig_time = np.linspace(0, duration, len(audio_data), endpoint=False)
    target_time = np.linspace(0, duration, target_length, endpoint=False)
    return np.interp(target_time, orig_time, audio_data).astype(np.int16)

def extract_asr_confidence(asr_result: dict) -> float | None:
    """ASRレスポンスから信頼度スコア (0.0 - 1.0) を抽出"""
    if not isinstance(asr_result, dict):
        return None
        
    for key in ["confidence", "score", "prob", "probability", "avg_prob"]:
        if key in asr_result and isinstance(asr_result[key], (int, float)):
            val = float(asr_result[key])
            if val > 1.0:
                val /= 100.0
            return val
            
    segments = asr_result.get("segments")
    if isinstance(segments, list) and len(segments) > 0:
        conf_list = []
        for seg in segments:
            if isinstance(seg, dict):
                for key in ["confidence", "prob", "probability", "score"]:
                    if key in seg and isinstance(seg[key], (int, float)):
                        v = float(seg[key])
                        if v > 1.0:
                            v /= 100.0
                        conf_list.append(v)
                        break
        if conf_list:
            return sum(conf_list) / len(conf_list)
            
    return None

# ---------------------------------------------------------------------------
# 1. 音声キャプチャ & VAD 監視スレッド
# ---------------------------------------------------------------------------

class AudioCaptureWorker(QThread):
    speech_detected = Signal(str, float)  # (temp_wav_path, duration_sec)
    log_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.device_index = None
        self.vad_threshold = DEFAULT_VAD_THRESHOLD
        self.min_speech_duration = 0.2
        self.max_speech_duration = 10.0
        self.silence_timeout = 2.0  # 無音判定タイムアウト（これを超えたら会話終了）
        self.input_gain = DEFAULT_INPUT_GAIN       # マイクゲイン倍率
        self.is_user_muted = False
        self.is_soft_muted = False

        self.audio_queue = queue.Queue()
        self.reconnect_requested = False

    def update_config(self, device_index, threshold, min_dur, max_dur, silence_timeout, input_gain):
        self.device_index = device_index
        self.vad_threshold = threshold
        self.min_speech_duration = min_dur
        self.max_speech_duration = max_dur
        self.silence_timeout = silence_timeout
        self.input_gain = input_gain
        self.reconnect_requested = True

    def set_user_mute(self, muted: bool):
        self.is_user_muted = muted
        state = "ミュート" if muted else "解除"
        self.log_signal.emit(f"[マイク] ユーザー操作によりマイクを{state}しました。")

    def set_soft_mute(self, muted: bool):
        self.is_soft_muted = muted

    def stop(self):
        self.running = False
        self.wait()

    def _audio_callback(self, indata, frames, time_info, status):
        if self.running and not self.reconnect_requested:
            self.audio_queue.put(indata.copy())

    def run(self):
        self.running = True
        channels = 1
        target_sr = 16000

        while self.running:
            self.reconnect_requested = False
            current_dev_idx = self.device_index

            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            try:
                device_info = sd.query_devices(current_dev_idx, 'input')
                native_sr = int(device_info['default_samplerate'])
                device_name = device_info['name']
            except Exception:
                native_sr = 44100
                device_name = f"Unknown (Index: {current_dev_idx})"

            chunk_samples = int(native_sr * 0.032)
            audio_buffer = []
            is_speaking = False
            speech_start_time = 0
            silence_start_time = 0
            last_debug_print_time = 0

            self.log_signal.emit(f"[マイク] '{device_name}' で録音を開始しました。")

            try:
                stream = sd.InputStream(
                    device=current_dev_idx,
                    channels=channels,
                    samplerate=native_sr,
                    dtype='int16',
                    blocksize=chunk_samples,
                    callback=self._audio_callback
                )

                with stream:
                    while self.running and not self.reconnect_requested:
                        try:
                            data = self.audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        current_time = time.time()

                        # ゲイン（音量倍率）の適用とクリッピング防止
                        if self.input_gain != 1.0:
                            gained_data = data.astype(np.float32) * self.input_gain
                            np.clip(gained_data, -32768, 32767, out=gained_data)
                            data = gained_data.astype(np.int16)

                        samples = data.flatten()
                        mean_amp = np.abs(samples).mean() / 32768.0

                        # 閾値判定ライン
                        current_cutoff = self.vad_threshold * 0.05

                        if current_time - last_debug_print_time >= 0.5:
                            status = "USER_MUTE" if self.is_user_muted else ("SOFT_MUTE" if self.is_soft_muted else ("SPEAKING" if is_speaking else "IDLE"))
                            print(
                                f"[Mic Debug] Status: {status:<10} | "
                                f"Mean: {mean_amp:.5f} | Gain: x{self.input_gain:.1f} | "
                                f"Cutoff: {current_cutoff:.5f}",
                                flush=True
                            )
                            last_debug_print_time = current_time

                        if self.is_user_muted or self.is_soft_muted:
                            if is_speaking:
                                is_speaking = False
                                audio_buffer.clear()
                            continue

                        if mean_amp >= current_cutoff:
                            if not is_speaking:
                                is_speaking = True
                                speech_start_time = current_time
                                audio_buffer = []
                                print(f"\n>>> [VAD DETECTED] 発話検知開始!\n", flush=True)
                                self.log_signal.emit("[VAD] 発話を検知しました...")

                            silence_start_time = current_time
                            audio_buffer.append(data.copy())

                        elif is_speaking:
                            audio_buffer.append(data.copy())
                            speech_duration = current_time - speech_start_time
                            silence_duration = current_time - silence_start_time

                            # 無音終了時間を超えるか、最長発話時間に達した場合に発話終了と判定
                            if silence_duration >= self.silence_timeout or speech_duration >= self.max_speech_duration:
                                is_speaking = False
                                total_duration = speech_duration - silence_duration

                                if total_duration >= self.min_speech_duration and len(audio_buffer) > 0:
                                    print(f"\n>>> [VAD FINISHED] 発話終了検知 (録音時間: {total_duration:.2f}秒)\n", flush=True)
                                    self.log_signal.emit(f"[VAD] 発話終了検知 (長さ: {total_duration:.2f}秒)")

                                    full_audio = np.concatenate(audio_buffer, axis=0).flatten()
                                    resampled_audio = resample_audio(full_audio, native_sr, target_sr)

                                    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                                    temp_wav_path = temp_wav.name
                                    temp_wav.close()

                                    with wave.open(temp_wav_path, 'wb') as wf:
                                        wf.setnchannels(channels)
                                        wf.setsampwidth(2)
                                        wf.setframerate(target_sr)
                                        wf.writeframes(resampled_audio.tobytes())

                                    self.speech_detected.emit(temp_wav_path, total_duration)
                                else:
                                    self.log_signal.emit("[VAD] 発話が短すぎるためスキップしました。")

                                audio_buffer.clear()

            except Exception as e:
                if self.running:
                    self.log_signal.emit(f"[エラー] 音声キャプチャ例外: {str(e)}")
                    time.sleep(0.5)

# ---------------------------------------------------------------------------
# 2. 翻訳パイプライン Worker (ASR -> LLM -> TTS)
# ---------------------------------------------------------------------------
class PipelineWorker(QThread):
    tts_ready = Signal(str)  # (generated_tts_wav_path)
    log_signal = Signal(str)
    notify_signal = Signal(str, str)  # (title, message)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue = []
        self.running = False

        # 設定の保持
        self.audiocpp_url = ""
        self.llama_url = ""
        self.lang_a = "日本語"
        self.lang_b = "英語"
        self.system_prompt = ""
        self.temperature = 0.0
        self.use_dynamic_ref = True
        self.min_dynamic_ref_len = DEFAULT_MIN_DYNAMIC_REF_LEN
        self.default_ref_path = ""
        self.default_ref_text = ""
        self.min_asr_confidence = 0.0

    def set_config(self, config_dict):
        self.audiocpp_url = config_dict["audiocpp_url"].rstrip('/')
        self.llama_url = config_dict["llama_url"].rstrip('/')
        self.lang_a = config_dict["lang_a"]
        self.lang_b = config_dict["lang_b"]
        self.system_prompt = config_dict["system_prompt"]
        self.temperature = config_dict["temperature"]
        self.use_dynamic_ref = config_dict["use_dynamic_ref"]
        self.min_dynamic_ref_len = config_dict["min_dynamic_ref_len"]
        self.default_ref_path = config_dict["default_ref_path"]
        self.default_ref_text = config_dict["default_ref_text"]
        self.min_asr_confidence = config_dict.get("min_asr_confidence", 0.0)

    def enqueue_audio(self, wav_path, duration):
        self.queue.append((wav_path, duration))

    def run(self):
        self.running = True
        while self.running:
            if self.queue:
                wav_path, duration = self.queue.pop(0)
                try:
                    self.process_pipeline(wav_path, duration)
                except Exception as e:
                    self.log_signal.emit(f"[エラー] パイプライン処理失敗: {str(e)}")
                finally:
                    if os.path.exists(wav_path):
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
            else:
                time.sleep(0.05)

    def process_pipeline(self, wav_path, duration):
        # 1. ASR (音声認識)
        self.log_signal.emit("[ASR] 音声認識を実行中...")
        asr_url = f"{self.audiocpp_url}/v1/audio/transcriptions"
        with open(wav_path, "rb") as audio_file:
            resp = requests.post(
                asr_url,
                data={"model": "asr"},
                files={"file": ("speech.wav", audio_file, "audio/wav")},
                timeout=10,
            )
        resp.raise_for_status()
        asr_result = resp.json()
        recognized_text = asr_result.get("text", "").strip()

        if not recognized_text:
            self.log_signal.emit("[ASR] 認識テキストが空でした。")
            return

        # ASR信頼度チェック
        confidence = extract_asr_confidence(asr_result)
        if confidence is not None:
            self.log_signal.emit(f"[ASR結果] {recognized_text} (信頼度: {confidence:.2f})")
            if self.min_asr_confidence > 0.0 and confidence < self.min_asr_confidence:
                self.log_signal.emit(
                    f"[ASR] 信頼度 ({confidence:.2f}) が設定の最低値 ({self.min_asr_confidence:.2f}) 未満のため無視します。"
                )
                return
        else:
            self.log_signal.emit(f"[ASR結果] {recognized_text}")

        self.notify_signal.emit("音声認識結果", recognized_text)

        # 2. LLM (翻訳)
        self.log_signal.emit("[LLM] 翻訳を実行中...")
        formatted_sys_prompt = self.system_prompt.format(
            lang_a=self.lang_a,
            lang_b=self.lang_b
        )

        llm_url = f"{self.llama_url}/v1/chat/completions"
        llm_payload = {
            "messages": [
                {"role": "system", "content": formatted_sys_prompt},
                {"role": "user", "content": recognized_text}
            ],
            "temperature": self.temperature
        }
        
        resp_llm = requests.post(llm_url, json=llm_payload, timeout=15)
        resp_llm.raise_for_status()
        llm_result = resp_llm.json()
        translated_text = llm_result["choices"][0]["message"]["content"].strip()

        self.log_signal.emit(f"[LLM翻訳結果] {translated_text}")
        self.notify_signal.emit("翻訳結果", translated_text)

        # 3. TTS (音声合成)
        self.log_signal.emit("[TTS] 音声合成を実行中...")
        tts_url = f"{self.audiocpp_url}/v1/audio/speech"

        if self.use_dynamic_ref and duration >= self.min_dynamic_ref_len:
            ref_path = wav_path
            ref_text = recognized_text
            self.log_signal.emit(f"[TTS] 動的参照音声を使用します (長さ: {duration:.2f}s)")
        else:
            ref_path = self.default_ref_path
            ref_text = self.default_ref_text
            self.log_signal.emit("[TTS] デフォルト参照音声を使用します")

        abs_ref_path = os.path.abspath(ref_path)

        if not ref_path or not os.path.isfile(abs_ref_path):
            self.log_signal.emit(f"[エラー] 参照音声ファイルが存在しません: {abs_ref_path}")
            return

        with open(abs_ref_path, "rb") as ref_file:
            reference_audio = ref_file.read(MAX_VOICE_REF_BYTES + 1)
        if not reference_audio or len(reference_audio) > MAX_VOICE_REF_BYTES:
            self.log_signal.emit("[エラー] 参照音声は空でない5 MiB以下のファイルにしてください。")
            return

        tts_payload = {
            "model": "tts",
            "input": translated_text,
            "voice_ref": {
                "type": "base64",
                "data": base64.b64encode(reference_audio).decode("ascii")
            },
            "reference_text": ref_text
        }

        resp_tts = requests.post(tts_url, json=tts_payload, timeout=20)
        resp_tts.raise_for_status()

        out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        out_wav.write(resp_tts.content)
        out_wav.close()

        self.log_signal.emit("[TTS] 音声合成完了。再生スレッドへ渡します。")
        self.tts_ready.emit(out_wav.name)

    def stop(self):
        self.running = False
        self.wait()


# ---------------------------------------------------------------------------
# 3. 音声再生 Worker
# ---------------------------------------------------------------------------
class AudioPlayerWorker(QObject):
    log_signal = Signal(str)
    soft_mute_signal = Signal(bool)

    def __init__(self, device_index_fn, soft_mute_enabled_fn):
        super().__init__()
        self.get_device_index = device_index_fn
        self.is_soft_mute_enabled = soft_mute_enabled_fn

    @Slot(str)
    def play_audio(self, wav_path):
        try:
            if not os.path.exists(wav_path):
                return

            with wave.open(wav_path, 'rb') as wf:
                wav_sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())

            if sampwidth == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
            else:
                raise ValueError(f"未対応のサンプル幅: {sampwidth} bytes")

            if channels > 1:
                audio_data = audio_data.reshape(-1, channels)

            device_idx = self.get_device_index()

            out_channels = 2
            try:
                device_info = sd.query_devices(device_idx, 'output')
                target_sr = int(device_info['default_samplerate'])
                out_channels = device_info.get('max_output_channels', 2)
            except Exception:
                target_sr = 44100

            if wav_sr != target_sr:
                duration = len(audio_data) / wav_sr
                target_length = int(len(audio_data) * target_sr / wav_sr)
                orig_time = np.linspace(0, duration, len(audio_data), endpoint=False)
                target_time = np.linspace(0, duration, target_length, endpoint=False)

                if channels == 1:
                    audio_data = np.interp(target_time, orig_time, audio_data).astype(np.int16)
                else:
                    resampled_channels = [
                        np.interp(target_time, orig_time, audio_data[:, c])
                        for c in range(channels)
                    ]
                    audio_data = np.column_stack(resampled_channels).astype(np.int16)

                play_sr = target_sr
            else:
                play_sr = wav_sr

            if channels == 1 and out_channels >= 2:
                if audio_data.ndim == 1:
                    audio_data = np.column_stack([audio_data, audio_data])

            if self.is_soft_mute_enabled():
                self.soft_mute_signal.emit(True)

            self.log_signal.emit("[再生] 翻訳音声を再生開始...")

            sd.play(audio_data, samplerate=play_sr, device=device_idx)
            sd.wait()

            self.log_signal.emit("[再生] 再生完了。")

            time.sleep(0.3)
            if self.is_soft_mute_enabled():
                self.soft_mute_signal.emit(False)

        except Exception as e:
            self.log_signal.emit(f"[エラー] 再生時例外: {str(e)}")
            self.soft_mute_signal.emit(False)
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# 4. メインGUIウィンドウ
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Realtime Speech Translator (audio.cpp + llama.cpp)")
        self.resize(1100, 720)

        self._is_quitting = False
        self.settings = QSettings("LocalAI", "VoiceTranslator")

        # Worker Threads
        self.capture_worker = AudioCaptureWorker()
        self.pipeline_worker = PipelineWorker()
        
        self.player_thread = QThread()
        self.player_worker = AudioPlayerWorker(
            device_index_fn=self.get_selected_output_device,
            soft_mute_enabled_fn=lambda: self.chk_soft_mute.isChecked()
        )
        self.player_worker.moveToThread(self.player_thread)
        self.player_thread.start()

        self.init_ui()
        self.init_tray()
        self.connect_signals()
        
        self.load_settings()
        self.start_threads()

    def browse_ref_file(self):
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "デフォルト参照音声を選択",
            "",
            "Audio Files (*.wav *.flac *.mp3)"
        )
        if file_path:
            self.txt_def_ref_path.setText(os.path.abspath(file_path))

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左側: 設定エリア
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        cols_layout = QHBoxLayout()
        col1_layout = QVBoxLayout()
        col2_layout = QVBoxLayout()

        # --- 列 1 ---
        # 1. サーバー接続設定
        grp_server = QGroupBox("サーバー設定")
        lay_server = QGridLayout(grp_server)
        lay_server.addWidget(QLabel("audio.cpp アドレス:"), 0, 0)
        self.txt_audiocpp_url = QLineEdit("http://127.0.0.1:8088")
        lay_server.addWidget(self.txt_audiocpp_url, 0, 1)

        lay_server.addWidget(QLabel("llama.cpp アドレス:"), 1, 0)
        self.txt_llama_url = QLineEdit("http://127.0.0.1:8080")
        lay_server.addWidget(self.txt_llama_url, 1, 1)
        col1_layout.addWidget(grp_server)

        # 2. オーディオデバイス設定
        grp_audio = QGroupBox("オーディオデバイス")
        lay_audio = QGridLayout(grp_audio)
        
        lay_audio.addWidget(QLabel("マイク (入力):"), 0, 0)
        self.cmb_input_dev = QComboBox()
        lay_audio.addWidget(self.cmb_input_dev, 0, 1)

        lay_audio.addWidget(QLabel("スピーカー (出力):"), 1, 0)
        self.cmb_output_dev = QComboBox()
        lay_audio.addWidget(self.cmb_output_dev, 1, 1)

        self.btn_mic_toggle = QPushButton("マイク: ON")
        self.btn_mic_toggle.setCheckable(True)
        self.btn_mic_toggle.setChecked(True)
        lay_audio.addWidget(self.btn_mic_toggle, 2, 0, 1, 2)

        self.chk_soft_mute = QCheckBox("TTS再生中のVADソフトミュート (ループ防止)")
        self.chk_soft_mute.setChecked(True)
        lay_audio.addWidget(self.chk_soft_mute, 3, 0, 1, 2)

        col1_layout.addWidget(grp_audio)

        # 3. Voice Cloning (TTS) 設定
        grp_tts = QGroupBox("参照音声 (Voice Cloning)")
        lay_tts = QGridLayout(grp_tts)

        self.chk_dynamic_ref = QCheckBox("話者音声の動的参照を使用")
        self.chk_dynamic_ref.setChecked(True)
        lay_tts.addWidget(self.chk_dynamic_ref, 0, 0, 1, 3)

        lay_tts.addWidget(QLabel("動的参照最小長 (秒):"), 1, 0)
        self.spn_min_dyn_len = QDoubleSpinBox()
        self.spn_min_dyn_len.setRange(0.5, 10.0)
        self.spn_min_dyn_len.setSingleStep(0.1)
        self.spn_min_dyn_len.setValue(DEFAULT_MIN_DYNAMIC_REF_LEN)
        lay_tts.addWidget(self.spn_min_dyn_len, 1, 1, 1, 2)

        lay_tts.addWidget(QLabel("デフォルト参照パス:"), 2, 0)
        self.txt_def_ref_path = QLineEdit("voice_dir/chara1.wav")
        lay_tts.addWidget(self.txt_def_ref_path, 2, 1)
        
        self.btn_browse_ref = QPushButton("参照...")
        self.btn_browse_ref.clicked.connect(self.browse_ref_file)
        lay_tts.addWidget(self.btn_browse_ref, 2, 2)

        lay_tts.addWidget(QLabel("デフォルト参照テキスト:"), 3, 0)
        self.txt_def_ref_text = QLineEdit("こんにちは。ゆうかだよ。よろしくねー。")
        lay_tts.addWidget(self.txt_def_ref_text, 3, 1, 1, 2)

        col1_layout.addWidget(grp_tts)
        col1_layout.addStretch()

        # --- 列 2 ---
        # 4. VAD / 発話検出 & ASR設定
        grp_vad = QGroupBox("発話検出 (VAD) & ASR設定")
        lay_vad = QGridLayout(grp_vad)

        lay_vad.addWidget(QLabel("発話最低音量 (VAD):"), 0, 0)
        self.spn_vad_thresh = QDoubleSpinBox()
        self.spn_vad_thresh.setRange(0.01, 1.0)
        self.spn_vad_thresh.setSingleStep(0.05)
        self.spn_vad_thresh.setValue(DEFAULT_VAD_THRESHOLD)
        lay_vad.addWidget(self.spn_vad_thresh, 0, 1)

        lay_vad.addWidget(QLabel("マイクゲイン (倍率):"), 1, 0)
        self.spn_input_gain = QDoubleSpinBox()
        self.spn_input_gain.setRange(1.0, 5.0)
        self.spn_input_gain.setSingleStep(0.5)
        self.spn_input_gain.setValue(DEFAULT_INPUT_GAIN)
        lay_vad.addWidget(self.spn_input_gain, 1, 1)

        lay_vad.addWidget(QLabel("無音終了時間 (秒):"), 2, 0)
        self.spn_silence_timeout = QDoubleSpinBox()
        self.spn_silence_timeout.setRange(0.3, 10.0)  # 最大10.0秒まで許容
        self.spn_silence_timeout.setSingleStep(0.2)
        self.spn_silence_timeout.setValue(2.0)
        lay_vad.addWidget(self.spn_silence_timeout, 2, 1)

        lay_vad.addWidget(QLabel("最短発話長 (秒):"), 3, 0)
        self.spn_min_speech = QDoubleSpinBox()
        self.spn_min_speech.setRange(0.1, 5.0)
        self.spn_min_speech.setValue(0.2)
        lay_vad.addWidget(self.spn_min_speech, 3, 1)

        lay_vad.addWidget(QLabel("最長発話長 (秒):"), 4, 0)
        self.spn_max_speech = QDoubleSpinBox()
        self.spn_max_speech.setRange(1.0, 60.0)
        self.spn_max_speech.setValue(10.0)
        lay_vad.addWidget(self.spn_max_speech, 4, 1)

        lay_vad.addWidget(QLabel("最低 ASR 信頼度:"), 5, 0)
        self.spn_min_asr_conf = QDoubleSpinBox()
        self.spn_min_asr_conf.setRange(0.00, 1.00)
        self.spn_min_asr_conf.setSingleStep(0.05)
        self.spn_min_asr_conf.setValue(0.00)
        lay_vad.addWidget(self.spn_min_asr_conf, 5, 1)

        col2_layout.addWidget(grp_vad)

        # 5. 言語 & LLM 設定
        grp_trans = QGroupBox("言語 & LLM 翻訳")
        lay_trans = QGridLayout(grp_trans)

        lay_trans.addWidget(QLabel("言語 A:"), 0, 0)
        self.cmb_lang_a = QComboBox()
        self.cmb_lang_a.addItems(["日本語", "英語", "中国語", "韓国語", "フランス語", "ドイツ語", "スペイン語"])
        lay_trans.addWidget(self.cmb_lang_a, 0, 1)

        lay_trans.addWidget(QLabel("言語 B:"), 1, 0)
        self.cmb_lang_b = QComboBox()
        self.cmb_lang_b.addItems(["英語", "日本語", "中国語", "韓国語", "フランス語", "ドイツ語", "スペイン語"])
        lay_trans.addWidget(self.cmb_lang_b, 1, 1)

        lay_trans.addWidget(QLabel("LLM Temperature:"), 2, 0)
        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(0.0, 1.0)
        self.spn_temp.setSingleStep(0.05)
        self.spn_temp.setValue(0.0)
        lay_trans.addWidget(self.spn_temp, 2, 1)

        lay_trans.addWidget(QLabel("システムプロンプト:"), 3, 0, 1, 2)
        self.txt_sys_prompt = QTextEdit()
        self.txt_sys_prompt.setPlainText(DEFAULT_SYSTEM_PROMPT)
        self.txt_sys_prompt.setMaximumHeight(80)
        lay_trans.addWidget(self.txt_sys_prompt, 4, 0, 1, 2)

        self.btn_reset_prompt = QPushButton("プロンプトを初期値に戻す")
        lay_trans.addWidget(self.btn_reset_prompt, 5, 0, 1, 2)

        col2_layout.addWidget(grp_trans)
        col2_layout.addStretch()

        cols_layout.addLayout(col1_layout)
        cols_layout.addLayout(col2_layout)
        left_layout.addLayout(cols_layout)

        # 保存ボタン
        self.btn_save = QPushButton("設定を保存して適用")
        left_layout.addWidget(self.btn_save)

        splitter.addWidget(left_widget)

        # 右側: ログ & 通知エリア
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        grp_log = QGroupBox("処理ログ & ステータス")
        lay_log = QVBoxLayout(grp_log)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 10))
        lay_log.addWidget(self.txt_log)

        lay_notify_opts = QHBoxLayout()
        self.chk_balloon_asr = QCheckBox("認識結果をバルーン表示")
        self.chk_balloon_asr.setChecked(True)
        self.chk_balloon_trans = QCheckBox("翻訳結果をバルーン表示")
        self.chk_balloon_trans.setChecked(True)
        lay_notify_opts.addWidget(self.chk_balloon_asr)
        lay_notify_opts.addWidget(self.chk_balloon_trans)
        lay_log.addLayout(lay_notify_opts)

        right_layout.addWidget(grp_log)
        splitter.addWidget(right_widget)

        splitter.setSizes([650, 450])

        self.populate_audio_devices()

    def populate_audio_devices(self):
        self.cmb_input_dev.clear()
        self.cmb_output_dev.clear()

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for idx, dev in enumerate(devices):
            identity = json.dumps([hostapis[dev["hostapi"]]["name"], dev["name"]], ensure_ascii=False)
            if dev['max_input_channels'] > 0:
                self.cmb_input_dev.addItem(f"[{idx}] {dev['name']}", idx)
                self.cmb_input_dev.setItemData(self.cmb_input_dev.count() - 1, identity, Qt.UserRole + 1)
            if dev['max_output_channels'] > 0:
                self.cmb_output_dev.addItem(f"[{idx}] {dev['name']}", idx)
                self.cmb_output_dev.setItemData(self.cmb_output_dev.count() - 1, identity, Qt.UserRole + 1)

    def get_selected_output_device(self):
        return self.cmb_output_dev.currentData()

    def init_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor("dodgerblue"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("white"))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "VT")
        painter.end()
        
        self.tray_icon.setIcon(QIcon(pixmap))

        tray_menu = QMenu()
        
        self.act_tray_mute = QAction("マイク Mute 切替", self)
        self.act_tray_mute.triggered.connect(self.btn_mic_toggle.animateClick)
        tray_menu.addAction(self.act_tray_mute)

        tray_menu.addSeparator()

        act_show = QAction("設定画面を表示", self)
        act_show.triggered.connect(self.show_window)
        tray_menu.addAction(act_show)

        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        tray_menu.addAction(act_about)

        tray_menu.addSeparator()

        act_quit = QAction("終了", self)
        act_quit.triggered.connect(self.quit_app)
        tray_menu.addAction(act_quit)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def connect_signals(self):
        self.capture_worker.log_signal.connect(self.log)
        self.pipeline_worker.log_signal.connect(self.log)
        self.player_worker.log_signal.connect(self.log)

        self.capture_worker.speech_detected.connect(self.pipeline_worker.enqueue_audio)
        self.pipeline_worker.tts_ready.connect(self.player_worker.play_audio)
        
        self.player_worker.soft_mute_signal.connect(self.capture_worker.set_soft_mute)

        self.btn_mic_toggle.toggled.connect(self.on_mic_toggled)
        self.btn_reset_prompt.clicked.connect(lambda: self.txt_sys_prompt.setPlainText(DEFAULT_SYSTEM_PROMPT))
        self.btn_save.clicked.connect(self.apply_and_save_settings)

        self.pipeline_worker.notify_signal.connect(self.on_notify)

    def start_threads(self):
        self.apply_and_save_settings()
        self.capture_worker.start()
        self.pipeline_worker.start()

    def apply_and_save_settings(self):
        input_idx = self.cmb_input_dev.currentData()
        self.capture_worker.update_config(
            device_index=input_idx,
            threshold=self.spn_vad_thresh.value(),
            min_dur=self.spn_min_speech.value(),
            max_dur=self.spn_max_speech.value(),
            silence_timeout=self.spn_silence_timeout.value(),
            input_gain=self.spn_input_gain.value()
        )

        pipeline_config = {
            "audiocpp_url": self.txt_audiocpp_url.text(),
            "llama_url": self.txt_llama_url.text(),
            "lang_a": self.cmb_lang_a.currentText(),
            "lang_b": self.cmb_lang_b.currentText(),
            "system_prompt": self.txt_sys_prompt.toPlainText(),
            "temperature": self.spn_temp.value(),
            "use_dynamic_ref": self.chk_dynamic_ref.isChecked(),
            "min_dynamic_ref_len": self.spn_min_dyn_len.value(),
            "default_ref_path": self.txt_def_ref_path.text(),
            "default_ref_text": self.txt_def_ref_text.text(),
            "min_asr_confidence": self.spn_min_asr_conf.value()
        }
        self.pipeline_worker.set_config(pipeline_config)
        self.log("[システム] 設定を更新・適用しました。")

        self.save_settings()

    def setting_widgets(self):
        # マイクON/OFFはセッション限定。保存と復元で同じ対応表を使用する。
        return {
            "audiocpp_url": self.txt_audiocpp_url,
            "llama_url": self.txt_llama_url,
            "default_ref_path": self.txt_def_ref_path,
            "default_ref_text": self.txt_def_ref_text,
            "vad_thresh": self.spn_vad_thresh,
            "input_gain": self.spn_input_gain,
            "silence_timeout": self.spn_silence_timeout,
            "min_speech": self.spn_min_speech,
            "max_speech": self.spn_max_speech,
            "min_asr_conf": self.spn_min_asr_conf,
            "soft_mute": self.chk_soft_mute,
            "use_dynamic_ref": self.chk_dynamic_ref,
            "min_dynamic_ref_len": self.spn_min_dyn_len,
            "lang_a": self.cmb_lang_a,
            "lang_b": self.cmb_lang_b,
            "temperature": self.spn_temp,
            "system_prompt": self.txt_sys_prompt,
            "balloon_asr": self.chk_balloon_asr,
            "balloon_trans": self.chk_balloon_trans,
        }

    @staticmethod
    def checked_number(value, widget):
        number = float(value)
        if not math.isfinite(number) or not widget.minimum() <= number <= widget.maximum():
            raise ValueError("数値が許容範囲外です")
        return number

    def save_settings(self):
        try:
            values = {}
            for key, widget in self.setting_widgets().items():
                if isinstance(widget, QDoubleSpinBox):
                    values[key] = self.checked_number(widget.value(), widget)
                elif isinstance(widget, QCheckBox):
                    values[key] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    values[key] = widget.currentText()
                elif isinstance(widget, QTextEdit):
                    values[key] = widget.toPlainText()
                else:
                    values[key] = widget.text()
            for key, combo in (("input_device", self.cmb_input_dev),
                               ("output_device", self.cmb_output_dev)):
                values[key] = combo.currentData(Qt.UserRole + 1) or ""
            for key, value in values.items():
                self.settings.setValue(key, value)
            # 廃止した翻訳モードのキーを既存の設定から除去する。
            self.settings.remove("trans_mode")
            self.settings.sync()
            if self.settings.status() != QSettings.NoError:
                raise OSError(f"QSettings: {self.settings.status()}")
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            self.log(f"[エラー] 設定を保存できませんでした: {exc}")
            return False
        self.log("[システム] 設定を保存しました。")
        return True

    def load_settings(self):
        for key, widget in self.setting_widgets().items():
            if not self.settings.contains(key):
                continue
            try:
                value = self.settings.value(key)
                if isinstance(widget, QDoubleSpinBox):
                    widget.setValue(self.checked_number(value, widget))
                elif isinstance(widget, QCheckBox):
                    normalized = str(value).lower()
                    if normalized not in ("true", "false", "1", "0"):
                        raise ValueError("真偽値が不正です")
                    widget.setChecked(normalized in ("true", "1"))
                elif isinstance(widget, QComboBox):
                    index = widget.findText(str(value))
                    if index < 0:
                        raise ValueError("選択肢が存在しません")
                    widget.setCurrentIndex(index)
                elif isinstance(widget, QTextEdit):
                    widget.setPlainText(str(value))
                else:
                    widget.setText(str(value))
            except (TypeError, ValueError, OverflowError) as exc:
                self.log(f"[警告] 設定 {key} が不正なため初期値を使用します: {exc}")
        for key, combo, direction in (("input_device", self.cmb_input_dev, 0),
                                       ("output_device", self.cmb_output_dev, 1)):
            if not self.settings.contains(key):
                continue
            identity = self.settings.value(key)
            index = combo.findData(identity, Qt.UserRole + 1) if identity else -1
            if index < 0:
                # デバイス番号は再起動や接続順で変わるため、名前とホストAPIで照合する。
                index = combo.findData(sd.default.device[direction])
                if index < 0 and combo.count():
                    index = 0
                fallback = combo.itemText(index) if index >= 0 else "利用可能なデバイスなし"
                self.log(f"[警告] 保存された {key} が見つかりません。{fallback} を使用します。")
            combo.setCurrentIndex(index)

    @Slot(str)
    def log(self, msg):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.txt_log.append(timestamp + msg)

    @Slot(bool)
    def on_mic_toggled(self, checked):
        if checked:
            self.btn_mic_toggle.setText("マイク: ON")
            self.capture_worker.set_user_mute(False)
        else:
            self.btn_mic_toggle.setText("マイク: MUTE (OFF)")
            self.capture_worker.set_user_mute(True)

    @Slot(str, str)
    def on_notify(self, title, msg):
        if "音声認識" in title and self.chk_balloon_asr.isChecked():
            self.tray_icon.showMessage(title, msg, QSystemTrayIcon.Information, 2000)
        elif "翻訳" in title and self.chk_balloon_trans.isChecked():
            self.tray_icon.showMessage(title, msg, QSystemTrayIcon.Information, 3000)

    def show_window(self):
        self.showNormal()
        self.activateWindow()

    def show_about(self):
        QMessageBox.about(
            self,
            "About",
            "ローカル多言語リアルタイム音声翻訳アプリ\n\n"
            "バックエンド: audio.cpp (VAD/ASR/TTS) + llama.cpp\n"
            "GUI: PySide6"
        )

    def closeEvent(self, event):
        if self._is_quitting:
            event.accept()
            return

        event.ignore()
        self.hide()
        self.tray_icon.showMessage("バックグラウンド実行中", "アプリはシステムトレイで稼働しています。")

    def quit_app(self):
        self._is_quitting = True

        self.capture_worker.stop()
        self.pipeline_worker.stop()
        self.player_thread.quit()
        self.player_thread.wait()

        self.tray_icon.hide()
        QApplication.quit()


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
