"""LingBot-VLA 官方推理服务的 WebSocket 客户端（msgpack 协议：连上先收服务端元数据，之后一问一答）。"""

import logging
import time

import websockets.sync.client

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """经 WebSocket 向推理服务端要动作；``infer`` 一次往返，``reset`` 告诉服务端换哪套机器人配置。"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        path: str = "",
        use_ssl: bool = False,
        api_key: str | None = None,
    ) -> None:
        # host 自带 ws:// / wss:// 前缀时原样用
        if host.startswith("ws://") or host.startswith("wss://"):
            self._uri = host
        else:
            self._uri = f"{'wss' if use_ssl else 'ws'}://{host}"
        if port is not None:
            self._uri += f":{port}"
        if path and not path.startswith("/"):
            path = "/" + path
        self._uri += path

        self._packer = Packer()
        self._api_key = api_key
        # 最近一次 infer 的分段耗时（定位网络 / 序列化开销用）
        self.last_timing: dict = {}
        self._ws, self._server_metadata = self._wait_for_server()

    @property
    def uri(self) -> str:
        return self._uri

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        t0 = time.perf_counter()
        data = self._packer.pack(obs)
        t1 = time.perf_counter()
        self._ws.send(data)
        response = self._ws.recv()
        t2 = time.perf_counter()
        if isinstance(response, str):
            # 正常回的是字节；回字符串就是服务端报错
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = unpackb(response)
        t3 = time.perf_counter()

        self.last_timing = {
            "pack_ms": (t1 - t0) * 1000,  # msgpack 序列化观测
            "roundtrip_ms": (t2 - t1) * 1000,  # 上传 + 服务端推理 + 下载
            "unpack_ms": (t3 - t2) * 1000,  # msgpack 反序列化动作
            "upload_mb": len(data) / 1e6,
            "download_mb": len(response) / 1e6,
        }
        return result

    def reset(self, robo_name: str) -> None:
        self.infer({"reset": True, "robo_name": robo_name})

    def close(self) -> None:
        self._ws.close()
