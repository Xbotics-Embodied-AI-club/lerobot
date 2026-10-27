"""标定提示的中文语音：微软 edge-tts 在线合成（男声 zh-CN-YunxiNeural），合成过的句子缓存成 mp3。

要联网；连不上时接口报错，页面只显示文字，标定照样能走完。
"""

import asyncio
import hashlib
import os
import threading
from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME

VOICE = "zh-CN-YunxiNeural"
DEFAULT_CACHE_DIR = HF_LEROBOT_HOME / "calibration_tts"


class SpeechSynthesizer:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = (cache_dir or DEFAULT_CACHE_DIR).resolve()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def synthesize(self, text: str) -> Path:
        text = " ".join(text.split())
        if not text:
            raise ValueError("语音文本不能为空")
        if len(text) > 500:
            raise ValueError("语音文本不能超过 500 个字符")

        key = hashlib.sha256(f"{VOICE}\0{text}".encode()).hexdigest()
        audio_path = self.cache_dir / f"{key}.mp3"
        if audio_path.is_file() and audio_path.stat().st_size > 0:
            return audio_path

        with self._lock_for(key):
            if audio_path.is_file() and audio_path.stat().st_size > 0:
                return audio_path
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = audio_path.with_suffix(".tmp")
            try:
                asyncio.run(self._save_audio(text, temporary_path))
                os.replace(temporary_path, audio_path)
            except Exception as error:
                temporary_path.unlink(missing_ok=True)
                raise ConnectionError("语音合成服务暂时不可用，请检查网络连接后重试") from error
        return audio_path

    @staticmethod
    async def _save_audio(text: str, audio_path: Path) -> None:
        import edge_tts

        communicator = edge_tts.Communicate(text, VOICE)
        await asyncio.wait_for(communicator.save(str(audio_path)), timeout=25)
