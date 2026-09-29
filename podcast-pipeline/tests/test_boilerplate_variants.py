import pytest

from algorithms.asr.hallucination import is_boilerplate, is_hallucination

# What reached a transcript: the outro blended with a few real words around it.
BLENDED = ("đối với hãy đăng kí cho kênh lalaschool Để không bỏ lỡ những video hấp dẫn "
           "nam á thì anh thấy.")


def test_the_blended_outro_from_a_real_transcript_is_caught():
    assert is_boilerplate(BLENDED) and is_hallucination(BLENDED, 3.0)


@pytest.mark.parametrize("text", [
    "hãy đăng kí cho kênh",                      # "kí" spelling with the imperative
    "Xin các bạn đăng kí để ủng hộ",
    "đăng kí kênh của mình nhé",                  # the channel after it
    "đăng ký cho kênh Ghiền Mì Gõ",
    "Đừng quên đăng kí",
    "like share và đăng kí nhé",
    "Để không bỏ lỡ những video hấp dẫn",
    "để không bỏ lỡ các clip mới",
    "hãy subscribe cho kênh",
])
def test_the_subscribe_outro_wordings_are_caught(text):
    assert is_boilerplate(text), text


@pytest.mark.parametrize("text", [
    "anh đã đăng kí kết hôn chưa",                # real speech with the bare word
    "mình đăng kí học lớp buổi tối",
    "kênh này có nhiều video hay lắm",
    "tôi không bỏ lỡ buổi nào cả",
    "làm video hấp dẫn thì phải có kịch bản",
    "cảm ơn anh đã chia sẻ",
])
def test_ordinary_speech_around_the_same_words_is_kept(text):
    assert not is_boilerplate(text), text


def test_the_spelling_the_filter_already_knew_still_works():
    assert is_boilerplate("Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn")
    assert is_boilerplate("hãy đăng ký cho kênh abc")
