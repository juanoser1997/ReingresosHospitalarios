"""
Espera robusta para MLflow desde otros contenedores Docker.

IMPORTANTE:
- No usa MlflowClient para esperar, porque ese cliente imprime warnings ruidosos mientras
  el servidor esta iniciando.
- No usa endpoints internos REST como /api/2.0/mlflow/experiments/search porque cambian
  entre versiones de MLflow y pueden fallar aunque la UI ya este disponible.
- Valida lo que realmente necesitamos antes de iniciar trainer/api:
  1) DNS/TCP hacia el servicio mlflow:5000 desde ESTE contenedor.
  2) HTTP GET / con codigo 2xx o 3xx.
  3) Varias respuestas exitosas consecutivas + pequena espera de gracia.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [wait-mlflow] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").rstrip("/")
TIMEOUT_SECONDS = int(os.getenv("MLFLOW_WAIT_TIMEOUT", "600"))
INTERVAL_SECONDS = int(os.getenv("MLFLOW_WAIT_INTERVAL", "5"))
STABLE_SUCCESSES = int(os.getenv("MLFLOW_STABLE_SUCCESSES", "3"))
GRACE_SECONDS = int(os.getenv("MLFLOW_GRACE_SECONDS", "10"))


def tcp_ready(uri: str) -> tuple[bool, str]:
    parsed = urlparse(uri)
    host = parsed.hostname or "mlflow"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=5):
            return True, f"tcp ok {host}:{port}"
    except OSError as exc:
        return False, f"tcp no disponible {host}:{port}: {exc}"


def http_ready(uri: str) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(f"{uri}/", method="GET", headers={"User-Agent": "docker-wait-mlflow"})
        with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310
            if 200 <= response.status < 400:
                return True, f"http ok {response.status}"
            return False, f"http status {response.status}"
    except urllib.error.HTTPError as exc:
        # Si MLflow responde 4xx, el proceso HTTP ya esta vivo, pero la UI normal debe dar 200.
        return False, f"http error {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"http no disponible: {exc}"


def main() -> None:
    deadline = time.time() + TIMEOUT_SECONDS
    attempt = 0
    consecutive_ok = 0
    last_reason = ""
    logger.info("Esperando MLflow desde este contenedor en %s hasta %ss...", MLFLOW_URI, TIMEOUT_SECONDS)

    while time.time() < deadline:
        attempt += 1
        ok_tcp, reason_tcp = tcp_ready(MLFLOW_URI)
        if not ok_tcp:
            consecutive_ok = 0
            last_reason = reason_tcp
            logger.info("MLflow aun no esta listo (intento %d): %s. Reintentando en %ss...", attempt, last_reason, INTERVAL_SECONDS)
            time.sleep(INTERVAL_SECONDS)
            continue

        ok_http, reason_http = http_ready(MLFLOW_URI)
        if not ok_http:
            consecutive_ok = 0
            last_reason = f"{reason_tcp} | {reason_http}"
            logger.info("MLflow aun no esta listo (intento %d): %s. Reintentando en %ss...", attempt, last_reason, INTERVAL_SECONDS)
            time.sleep(INTERVAL_SECONDS)
            continue

        consecutive_ok += 1
        logger.info("MLflow respondio correctamente %d/%d veces (%s | %s).", consecutive_ok, STABLE_SUCCESSES, reason_tcp, reason_http)
        if consecutive_ok >= STABLE_SUCCESSES:
            if GRACE_SECONDS > 0:
                logger.info("MLflow estable. Espera de gracia de %ss antes de continuar...", GRACE_SECONDS)
                time.sleep(GRACE_SECONDS)
            logger.info("MLflow listo para usar desde este contenedor.")
            return
        time.sleep(INTERVAL_SECONDS)

    logger.error("MLflow no estuvo listo despues de %ss. Ultimo estado: %s", TIMEOUT_SECONDS, last_reason)
    sys.exit(1)


if __name__ == "__main__":
    main()
