"""Run one real API call to confirm the configured connection works."""
from agent import StudyHelperAgent

if __name__ == '__main__':
    print(StudyHelperAgent().smoke_test())
