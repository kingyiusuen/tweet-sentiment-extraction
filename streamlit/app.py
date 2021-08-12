from pathlib import Path

import streamlit as st
from tweet_sentiment_extraction.models import (
    SentimentAnalyzer,
    SentimentPhraseExtractor,
)


CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def main():
    sentiment_analyzer = SentimentAnalyzer(CHECKPOINT_DIR / "sentiment_analyzer")
    sentiment_phrase_extractor = SentimentPhraseExtractor(CHECKPOINT_DIR / "sentiment_phrase_extractor")

    st.set_page_config(page_title="Tweet Sentiment Extract")
    st.title("Tweet Sentiment Extraction")
    text = st.text_input("Enter some text")

    if st.button("Submit"):
        if text is None:
            st.error("Please input some text before submitting.")
        else:
            with st.spinner("Predicting sentiment..."):
                sentiment_prediction = sentiment_analyzer.predict(text)
            st.subheader("Sentiment")
            st.table(sentiment_prediction)
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
