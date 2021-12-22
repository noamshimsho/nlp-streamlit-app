# Core Pkgs
import streamlit as st

# NLP Pkgs
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


def main():
    """ NLP Based App with Streamlit """

    # Title
    st.title("NLPiffy with Streamlit & Transformers")
    st.subheader("Natural Language Processing On the Go..")
    st.markdown("""
    	#### Description
    	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
    	NER, Sentiment Analysys, Text Generation
    	""")

    st.warning("load models... in less than a minute you can enjoy crazy models!!!")

    if 'sentiment_analysys' not in st.session_state:
        st.session_state['sentiment_analysys'] = pipeline("sentiment-analysis")

    if 'ner' not in st.session_state:
        st.session_state['ner'] = pipeline("ner", grouped_entities=True)

    if 'text_generation' not in st.session_state:
        st.session_state['text_generation'] = pipeline("text-generation")

    # NER
    if st.checkbox("Show Named Entities"):
        message = st.text_area("Enter Text", key='2')
        if st.button("Extract", key='2'):
            entity_result = st.session_state['ner'](message)
            st.success(entity_result)

    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        #classifier = pipeline("sentiment-analysis", 'bert-base-uncased')

        message_Sentiment = st.text_area("Enter Text", key='3')
        if st.button("Analyze", key='3'):
            result_sentiment = st.session_state.sentiment_analysys(message_Sentiment)
            st.success(result_sentiment)

    # text_generation
    if st.checkbox("Show Text Generation"):
        message = st.text_area("Enter Text", key='4')
        if st.button("Generate", key='4'):
            summary_result = st.session_state.text_generation(message, max_length=30, num_return_sequences=2)
            st.success(summary_result)


    # st.sidebar.subheader("About App")
    # st.sidebar.text("NLPiffy App with Streamlit")
    # st.sidebar.info("Cudos to the Streamlit Team")
    #
    # st.sidebar.subheader("By")
    # st.sidebar.text("Jesse E.Agbe(JCharis)")
    # st.sidebar.text("Jesus saves@JCharisTech")


if __name__ == '__main__':
	main()