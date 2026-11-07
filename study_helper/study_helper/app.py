"""Streamlit entry point: submit a prompt once and inspect the saved result."""
import streamlit as st
from agent import StudyHelperAgent
from config import get_settings
from session import UserSession


def main():
    """Render the prompt form and reuse the session's agent across UI reruns."""
    settings = get_settings()
    st.set_page_config(page_title='Study Helper', page_icon='📚', layout='centered')
    st.title('Study Helper Agent')
    st.write('Ask a calculation, check a date, or search your study notes.')
    st.caption('One run per prompt and model in this session. Tool steps happen inside that run.')
    if 'user_session' not in st.session_state:
        st.session_state.user_session = UserSession(StudyHelperAgent(settings))
    if not settings.api_key:
        st.warning('Set OPENAI_API_KEY in your terminal and restart Streamlit to use the live model.')
    with st.form('prompt_form'):
        model = st.selectbox('Model', options=settings.models, key='selected_model')
        prompt = st.text_area('Your question', placeholder='What do my notes say about gradient descent?',
                              max_chars=settings.max_input_chars, key='user_prompt')
        submitted = st.form_submit_button('Ask agent')
    if st.button('Clear results / allow a new attempt'):
        st.session_state.user_session.clear()
        st.session_state.pop('last_result', None)
        st.session_state.pop('was_cached', None)
    if submitted:
        if not prompt.strip():
            st.error('Enter a question first.')
        else:
            with st.spinner('Working on your question...'):
                result, cached = st.session_state.user_session.submit(prompt, model)
            st.session_state.last_result = result
            st.session_state.was_cached = cached
    result = st.session_state.get('last_result')
    if result is not None:
        if st.session_state.get('was_cached'):
            st.info('Showing the saved answer. No new model call was made.')
        st.subheader('Answer')
        if result.status in ('error', 'invalid', 'stopped'):
            st.error(result.answer)
        else:
            st.write(result.answer)
        for citation in result.citations:
            st.caption(f"Source: {citation['file']}:{citation['line']}")
        st.caption(f'Status: {result.status} | Model steps: {result.steps} | Request: {result.request_id}')
        with st.expander('Response validation'):
            if result.validation_errors:
                for error in result.validation_errors:
                    st.write('- ' + error)
            elif result.status in ('answered', 'unable'):
                st.success('Format and available evidence checks passed.')
            else:
                st.write('No completed model answer was available to validate.')
            st.caption('These checks do not guarantee that every claim is correct.')
        with st.expander('Tool calls'):
            st.json(result.trace)
    st.caption('Requests and responses are saved locally in logs/agent.jsonl. Do not put secrets in prompts.')


if __name__ == '__main__':
    main()
