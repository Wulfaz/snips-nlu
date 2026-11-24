"""End-to-end check with pynab's NLU data.

Usage: python ci/pynab_nlu_check.py ENGINE_EN_DIR ENGINE_FR_DIR

The engines are trained from pynab's */nlu/intent_{en,fr}.yaml with
``snips-nlu generate-dataset`` and ``snips-nlu train``, as pynab's
install.sh does. Interpretation mirrors pynab's nabd/nlu.py, and the
expectations are those of pynab's nabd/tests/nlu_test.py.
"""
import datetime
import sys

from snips_nlu import SnipsNLUEngine

# The datetime slot carries the local UTC offset ("+01:00" in Paris), so only
# the date and time part is compared (pynab's test assumes UTC).
TODAY = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d 00:00:00")

# (sentence, expected intent or None, expected "date" slot or absent/None)
CASES = {
    "en": [
        ("what's the weather like today", "nabweatherd/forecast", TODAY),
        ("tell me a joke", "nabsurprised/surprise", None),
        ("make me laugh", "nabsurprised/surprise", None),
        ("should i go outside today", "nabairqualityd/forecast", TODAY),
    ],
    "fr": [
        ("blablabla", None, None),
        ("météo", "nabweatherd/forecast", "absent"),
        ("est-ce qu'il va pleuvoir aujourd'hui", "nabweatherd/forecast", TODAY),
        ("guili guili", "nabsurprised/surprise", None),
        ("fais-moi rire", "nabsurprised/surprise", None),
        ("c'est pas un peu pollué aujourd'hui", "nabairqualityd/forecast", TODAY),
    ],
}


def interpret(engine, string):
    """Same logic as pynab's nabd/nlu.py NLU._interpret()"""
    if string == "":
        return None
    parsed = engine.parse(string)
    if parsed["intent"]["intentName"] is None:
        return None
    result = {"intent": parsed["intent"]["intentName"]}
    for slot in parsed["slots"]:
        result[slot["slotName"]] = slot["value"]["value"]
    return result


def main(engine_dirs):
    failures = 0
    for lang, engine_dir in zip(("en", "fr"), engine_dirs):
        engine = SnipsNLUEngine.from_path(engine_dir)
        for sentence, intent, date in CASES[lang]:
            result = interpret(engine, sentence)
            ok = (result is None) if intent is None else (
                result is not None and result["intent"] == intent)
            if ok and intent is not None and date == "absent":
                ok = "date" not in result
            elif ok and date not in (None, "absent"):
                ok = (result.get("date") or "").startswith(date + " ")
            failures += not ok
            print("%s [%s] %r -> %r" % ("OK  " if ok else "FAIL", lang, sentence, result))
    if failures:
        sys.exit("%d failure(s)" % failures)
    print("all pynab NLU checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
