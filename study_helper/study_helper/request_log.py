"""Write request, tool, model response and final response events as JSON lines."""
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from threading import Lock

_LOGGER_LOCK = Lock()


class RequestLogger:
    """Keep correlated events in a rotating local log without logging the key."""

    def __init__(self, settings):
        """Create one shared file handler per log path, not per request."""
        self.secret = settings.api_key
        path = settings.log_file.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(str(path))
        with _LOGGER_LOCK:
            if not self.logger.handlers:
                handler = RotatingFileHandler(path, maxBytes=2_000_000,
                                              backupCount=2, encoding='utf-8')
                handler.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

    def event(self, request_id, event, **data):
        """Append one timestamped JSON event and redact the configured API key."""
        record = {'timestamp': datetime.now(timezone.utc).isoformat(),
                  'request_id': request_id, 'event': event, **data}
        text = json.dumps(record, ensure_ascii=False, default=str)
        if self.secret:
            text = text.replace(self.secret, '[REDACTED]')
        self.logger.info(text)
