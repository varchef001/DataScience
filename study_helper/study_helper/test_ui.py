"""Exercise the actual Streamlit app without an API call."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from agent import StudyHelperAgent
from config import get_settings
from offline_demo import DemoModel
from session import UserSession
from test_local import MemoryLogger


class UITests(unittest.TestCase):
    """Verify that form submission runs once and UI rerenders reuse the result."""

    def test_form_submission_and_rerun(self):
        """Typing, rerendering and duplicate clicks must not create extra requests."""
        logger = MemoryLogger()
        agent = StudyHelperAgent(model_call=DemoModel(), logger=logger)
        app = AppTest.from_file(str(Path(__file__).with_name('app.py')))
        app.session_state['user_session'] = UserSession(agent)
        app.run()
        self.assertFalse(app.exception)
        app.text_area[0].set_value('What is 17.5% of 2,340?')
        self.assertEqual(len(logger.events), 0)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(sum(e['event'] == 'request' for e in logger.events), 1)
        self.assertIn('409.5', app.session_state['last_result'].answer)
        app.run()
        app.button[0].click().run()
        self.assertEqual(sum(e['event'] == 'request' for e in logger.events), 1)
        self.assertTrue(app.session_state['was_cached'])
        app.button[1].click().run()
        app.button[0].click().run()
        self.assertEqual(sum(e['event'] == 'request' for e in logger.events), 2)

    def test_empty_input(self):
        """An empty submission produces a helpful error without running the agent."""
        app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Enter a question' in error.value for error in app.error))


if __name__ == '__main__':
    unittest.main(verbosity=2)
