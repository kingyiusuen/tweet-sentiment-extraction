from pathlib import Path

import streamlit as st
from tweet_sentiment_extraction.models import (
    SentimentAnalyzer,
    SentimentPhraseExtractor,
)


CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


@st.cache(allow_output_mutation=True)
def load_models():
    sentiment_analyzer = SentimentAnalyzer(CHECKPOINT_DIR / "sentiment_analyzer")
    sentiment_phrase_extractor = SentimentPhraseExtractor(CHECKPOINT_DIR / "sentiment_phrase_extractor")
    return sentiment_analyzer, sentiment_phrase_extractor


def main():
    st.set_page_config(page_title="Tweet Sentiment Extract")

    sentiment_analyzer, sentiment_phrase_extractor = load_models()

    st.title("Tweet Sentiment Extraction")
    text = st.text_area("Enter some text", height=3)

    if st.button("Submit"):
        if text is None:
            st.error("Please input some text before submitting.")
        else:
            with st.spinner("Predicting sentiment..."):
                sentiment_prediction = sentiment_analyzer.predict(text)
            st.subheader("Sentiment")
            for sentiment, prob in sentiment_prediction.items():
                st.write(f"{sentiment}: {round(prob, 3)}")
            most_probable_sentiment = max(
                sentiment_prediction,
                key=lambda sentiment: sentiment_prediction[sentiment],
            )
            with st.spinner("Predicting support phrase..."):
                selected_text = sentiment_phrase_extractor.predict(most_probable_sentiment, text)
            st.subheader("Support phrase")
            st.write(selected_text)


if __name__ == "__main__":
    main()
