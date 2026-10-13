import json
import unittest
from EmotionDetection.emotion_detection import emotion_detector

EMOTIONS = {"anger", "disgust", "fear", "joy", "sadness"}


def get_emotions(text):
    result = json.loads(emotion_detector(text))
    return result["emotionPredictions"][0]["emotion"]


class TestEmotionDetector(unittest.TestCase):
    def test_returns_string(self):
        self.assertIsInstance(emotion_detector("I love this"), str)

    def test_valid_json_structure(self):
        result = json.loads(emotion_detector("I love this"))
        self.assertIn("emotionPredictions", result)

    def test_five_emotions_present(self):
        emotions = get_emotions("I love this")
        self.assertEqual(set(emotions.keys()), EMOTIONS)

    def test_scores_between_0_and_1(self):
        for score in get_emotions("I love this").values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_joy(self):
        emotions = get_emotions("I am glad this happened")
        self.assertEqual(max(emotions, key=emotions.get), "joy")

    def test_anger(self):
        emotions = get_emotions("I am really mad about this")
        self.assertEqual(max(emotions, key=emotions.get), "anger")

    def test_disgust(self):
        emotions = get_emotions("I feel disgusted just hearing about this")
        self.assertEqual(max(emotions, key=emotions.get), "disgust")

    def test_sadness(self):
        emotions = get_emotions("I am so sad about this")
        self.assertEqual(max(emotions, key=emotions.get), "sadness")

    def test_fear(self):
        emotions = get_emotions("I am really afraid that this will happen")
        self.assertEqual(max(emotions, key=emotions.get), "fear")


if __name__ == "__main__":
    unittest.main()
