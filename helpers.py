from iso639 import Lang
from langcodes import Language
def to_iso(langs):
    new = [Language.make(language).to_alpha3() for language in langs]
    return new
# def to_iso(langs):
#     new = [Lang(i).pt3 for i in langs]
#     return new
