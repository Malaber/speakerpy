import unittest
from tts.parser import parse_dialogue, prepare_chunks, split_text
from tts.voices import Voice, cache_key


class ParserTests(unittest.TestCase):
    def test_turn_boundaries_and_continuations(self):
        turns = parse_dialogue('Sprecher 1: **Hallo**\nweiter\nSprecher 2: Guten Abend\nSprecher 1: Nochmal')
        self.assertEqual([t.speaker for t in turns], ['Sprecher 1', 'Sprecher 2', 'Sprecher 1'])
        self.assertEqual(turns[0].text, 'Hallo weiter')
        self.assertEqual([c.turn for c in prepare_chunks(turns)], [0, 1, 2])

    def test_long_turn_and_long_word(self):
        for text in ['Ein langer Satz. ' * 100, 'x' * 1201]:
            chunks = split_text(text)
            self.assertTrue(all(0 < len(c) <= 500 for c in chunks))
            self.assertEqual(''.join(''.join(chunks).split()), ''.join(text.split()))

    def test_paragraphs_and_sentences(self):
        self.assertEqual(split_text('Hello world. Good day.', 12), ['Hello world.', 'Good day.'])
        self.assertEqual(split_text('First\n\nSecond', 7), ['First', 'Second'])
        self.assertEqual(parse_dialogue('A: Hallo\n  Hinweis: gut')[0].text, 'Hallo Hinweis: gut')

    def test_invalid(self):
        for text in ['', 'Unlabelled', 'A:', 'A: hi\nB:']:
            with self.assertRaises(ValueError):
                parse_dialogue(text)

    def test_cache_identity(self):
        voice = Voice('anna', 'Anna', 'German', 'Hallo', '/tmp/ref.wav', 'hash')
        args = ['model', voice, 'German', 'Hallo', {'max_new_tokens': 2048}]
        base = cache_key(*args)
        for i, replacement in enumerate(['other', Voice('klaus', 'Klaus', 'German', 'Hi', '/tmp/ref.wav', 'new'), 'English', 'Hi', {}]):
            changed = args.copy()
            changed[i] = replacement
            self.assertNotEqual(base, cache_key(*changed))
