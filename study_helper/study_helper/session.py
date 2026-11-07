"""Prevent Streamlit reruns from processing an already submitted prompt again."""
from threading import Lock


class UserSession:
    """Reuse one agent and cache completed requests within one browser session."""

    def __init__(self, agent):
        """Store the user's agent, result cache and duplicate-submission lock."""
        self.agent = agent
        self.results = {}
        self.lock = Lock()

    def submit(self, prompt, model):
        """Run each trimmed prompt/model pair once until the user clears it."""
        key = (prompt.strip(), model)
        with self.lock:
            if key in self.results:
                return self.results[key], True
            result = self.agent.process(prompt, model=model)
            self.results[key] = result
            return result, False

    def clear(self):
        """Allow an intentional fresh attempt after the user presses Clear."""
        with self.lock:
            self.results.clear()
