SPEECH_CLASSES = [
    "none",
    "offensive",
    "abusive",
    "cyberbullying",
    "vulgar",
    "racist",
    "homophobic",
    "profane",
    "slur",
    "harrasment",
    "obscene",
    "threat",
    "discredit",
    "hateful",
    "insult",
    "hostile",
    "sexist",
    "appearance-related",
    "intellectual",
    "political",
    "religion",
    "unclassified offensive",
    "toxic",
    "severe_toxic",
    "identity_hate"
]

if __name__ == '__main__':
    print(list(enumerate(SPEECH_CLASSES)))