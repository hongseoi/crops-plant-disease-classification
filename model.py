if __name__ == "__main__":
    sample_text = "The Dow Jones Industrial Average (^DJI) turned green."

    model = SentimentModel()
    sentiment = model.predict(text=sample_text)
    print(sentiment)