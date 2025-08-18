import logging

LOG_FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("BeatSmith")

def li(msg: str):
    log.info("[i] " + msg)

def lw(msg: str):
    log.warning("[!] " + msg)

def ld(msg: str):
    log.debug("[DEBUG] " + msg)

def le(msg: str):
    log.error("[x] " + msg)

__all__ = ["li", "lw", "ld", "le", "log"]
