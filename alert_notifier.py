import logging
import threading
from typing import Optional

from email_notifier import EmailNotifier
from whatsapp_notifier import WhatsAppNotifier

logger = logging.getLogger(__name__)

_email_notifier: Optional[EmailNotifier] = None
_whatsapp_notifier: Optional[WhatsAppNotifier] = None


def get_email_notifier() -> EmailNotifier:
    global _email_notifier
    if _email_notifier is None:
        _email_notifier = EmailNotifier()
    return _email_notifier


def get_whatsapp_notifier() -> WhatsAppNotifier:
    global _whatsapp_notifier
    if _whatsapp_notifier is None:
        _whatsapp_notifier = WhatsAppNotifier()
    return _whatsapp_notifier


def notify_detection_alerts(
    camera: str,
    location: str,
    alerts: list[dict],
    image_url: Optional[str] = None,
) -> dict:
    """Send email and WhatsApp notifications for saved detection alerts."""
    if not alerts:
        return {"email": False, "whatsapp": False}

    logger.info(
        "Starting notifications for %s at %s (%d alert(s))",
        camera,
        location,
        len(alerts),
    )

    email_sent = get_email_notifier().send_detection_alert(
        camera=camera,
        location=location,
        alerts=alerts,
        image_url=image_url,
    )
    whatsapp_sent = get_whatsapp_notifier().send_detection_alert(
        camera=camera,
        location=location,
        alerts=alerts,
        image_url=image_url,
    )

    logger.info(
        "Alert notifications — email: %s, whatsapp: %s | %s | %d violation(s)",
        "sent" if email_sent else "skipped",
        "sent" if whatsapp_sent else "skipped",
        camera,
        len(alerts),
    )
    return {"email": email_sent, "whatsapp": whatsapp_sent}


def notify_detection_alerts_async(
    camera: str,
    location: str,
    alerts: list[dict],
    image_url: Optional[str] = None,
) -> threading.Thread:
    """Send notifications in a background thread so detection is not blocked."""

    def _worker():
        notify_detection_alerts(camera, location, alerts, image_url)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
