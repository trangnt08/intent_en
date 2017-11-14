# -*- encoding: utf8 -*-
from pyvi.pyvi import ViPosTagger, ViTokenizer

a = ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội"))
print a