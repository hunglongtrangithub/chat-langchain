from sad_talker import SadTalker

print("SadTalker imported")


class DummyTalker:
    def __init__(self):
        pass

    def test(self, source_image, driven_audio):
        return source_image, driven_audio
