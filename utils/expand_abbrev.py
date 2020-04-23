import re
import mojimoji
import pickle
import json
from fairseq.models.transformer import TransformerModel

class Converter(object):
    def __init__(self, med_dic, translater):
        self.med_dic = med_dic
        self.translater = translater

    def convert(self, sent):
        sent = mojimoji.zen_to_han(sent, kana=False, digit=False)
        iters = re.finditer(r'([a-zA-Z][a-zA-Z\s]*)', sent)
        output_word = [""]
        pos = 0
        for i in iters:
            s_pos, e_pos = i.span()
            word = i.groups()[0]
            word = re.sub('^\s', r'', word)
            word = re.sub('\s$', r'', word)
            s_word = ""

            while pos < s_pos:
                output_word = [token + sent[pos] for token in output_word]
                pos += 1

            if word in self.med_dic:
                s_word = self.med_dic[word]
            elif word.lower() in self.med_dic:
                s_word = self.med_dic[word.lower()]
            elif self.translater:
                s_word = [self.translater.translate(' '.join(list(mojimoji.zen_to_han(word).lower()))).replace(' ', '')]
            else:
                s_word = []

            s_word = list(set([word] + s_word))
            tmp = [output_word for i in range(len(s_word))]
            for i in range(len(tmp)):
                tmp[i] = [t + s_word[i] for t in tmp[i]]
            
            output_word = sum(tmp, [])

            pos = e_pos

        while pos < len(sent):
            output_word = [token + sent[pos] for token in output_word]
            pos += 1

        output_word = [mojimoji.han_to_zen(t) for t in output_word]

        return output_word


if __name__ == "__main__":
    """
    with open('resource/freq.csv', 'r') as f:
        lines = [line for line in f.read().split('\n') if line != '']

    med_dic = {}
    for line in lines:
        if len(line.split(',')) <= 3:
            abbrev, word, freq = [re.sub(r'(^\s|\s$)', r'', l) for l in line.split(',')]
            freq = int(freq)
            #if (abbrev in med_dic and med_dic[abbrev][1] < freq) or abbrev not in med_dic:
                #med_dic[abbrev] = [word, freq]
            if abbrev in med_dic:
                med_dic[abbrev].append(word)
            else:
                med_dic[abbrev] = [word]
    """

    sent = "高Ｃａ血症"

    with open('../data/med_dic_all.json', 'r') as f:
        med_dic = json.load(f)

    translater = TransformerModel.from_pretrained(
            '../../fairseq/checkpoints',
            checkpoint_file='checkpoint16.pt',
            data_name_or_path='../../fairseq/data-bin/en_ja')
    converter = Converter(med_dic, translater)
    print(converter.convert(sent))

