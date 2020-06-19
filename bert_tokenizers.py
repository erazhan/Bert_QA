import unicodedata,re

from utils import bert_params

def load_vocab(vocab_path, encoding = 'utf-8', simplified = False):
    '''
    unused的用法，其实就是如果你自己要用的某个字在vocab.txt中没有,
    那么你可以将voacb.txt中unused*替换成你想要用的字即可，
    原始的Bert模型中应该是没有对这99个unused对应的词向量进行训练的，
    只是做了一个初始化
    '''
    token_dict = {}
    with open(vocab_path, encoding = 'utf-8') as fr:
        for line in fr.readlines():
            token = line.strip()
            token_dict[token] = len(token_dict)

    if simplified:
        '''
        bert4keras中的解释是过滤冗余部分token，暂时不考虑
        '''
        pass
    return token_dict

def save_vocab(vocab_path, token_dict, encoding = 'utf-8'):
    '''保存精简后的词典'''
    pass

class BasicTokenizer(object):
    '''分词器基类，tokenizer结构完全参考bert4keras中的写法'''

    def __init__(self, token_start = '[CLS]', token_end = '[SEP]'):
        '''定义好5个重要的特殊token'''
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end
        
    def tokenize(self, text, max_length = None):
        '''具体的分词函数'''
        tokens =self._tokenize(text)
        #print("tokens",tokens)
        #分好词后在首尾加上[CLS]和[SEP]
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
            
        if self._token_end is not None:
            tokens.append(self._token_end)

        #如果限定了最大长度，就需要对句子进行截断
        if max_length is not None:
            index = int(self._token_end is not None) + 1
            self.truncate_sequence(max_length, tokens, None, -index)
        return tokens
    def _tokenize(self, text):
        '''在子类中一定要实现具体的分词函数'''
        raise NotImplementedError
    
    def truncate_sequence(self, max_length, first_sequence, second_sequence = None, pop_index = -1):
        """截断总长度"""
        if second_sequence in None:
            second_sequence = []
        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= max_length:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)
    
    def encode(self, first_text, second_text = None, max_length = None, first_length = None, second_length = None):
        """输出文本对应token id和segment id
        如果传入first_length，则强行padding第一个句子到指定长度；
        同理，如果传入second_length，则强行padding第二个句子到指定长度。
        """
        if isinstance(first_text,str):
            first_tokens = self.tokenize(first_text)
            #print("first_tokens\n",first_tokens)
        else:
            first_tokens = first_text
        if second_text is None:
            second_tokens = None
        elif isinstance(second_text,str):
            idx = int(bool(self._token_start))
            second_tokens = self.tokenize(second_text)[idx:]
        else:
            second_tokens = second_text

        if max_length is not None:
            self.truncate_sequence(max_length, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length is not None:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] *
                                   (first_length - len(first_token_ids)))
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            if second_length is not None:
                second_token_ids = second_token_ids[:second_length]
                second_token_ids.extend([self._token_pad_id] *
                                        (second_length - len(second_token_ids)))
            second_segment_ids = [1] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def decode(self, ids):
        '''转为可读文本'''
        raise NotImplementedError

    def token_to_id(self, token):
        '''将token转为词典中的id'''
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        '''将token序列转为id序列'''
        return [self.token_to_id(token) for token in tokens]

    def id_to_token(self, i):
        '''将id转为对应的token'''
        raise NotImplementedError
    
    def ids_to_tokens(self, ids):
        '''id序列转为对应的token序列'''
        return [self.id_to_token(i) for i in ids]

class Tokenizer(BasicTokenizer):
    """Bert原生分词器
    纯Python实现，代码修改自keras_bert的tokenizer实现
    """
    def __init__(self, token_dict, do_lower_case = False, *args, **kwargs):
        ''''''
        super(Tokenizer, self).__init__(*args, **kwargs)
        if isinstance(token_dict, str):
            '''如果参数是vocab.txt的路径名，那么读取文件中的数据'''
            token_dict = load_vocab(token_dict)
            
        self._do_lower_case = do_lower_case

        #(token,id)
        self._token_dict = token_dict

        #(id,token)
        self._token_dict_inv = {v:k for k,v in token_dict.items()}

        #词典大小,21128
        self._vocab_size = len(token_dict)

        '''
        这里也可以不用getattr，以self._token_pad为例
        _pad_id = token_dict[self._token_pad]
        self._pad_id = _pad_id
        但要重复5次会造成大量逻辑相同的代码
        '''

        for token in ['pad','unk','mask','start','end']:
            try:
                _token_id = token_dict[getattr(self, '_token_%s' % token)]
                setattr(self, '_token_%s_id' % token, _token_id)
            except:
                pass
        
    def token_to_id(self, token):
        '''将token转为词典中的id'''
        return self._token_dict.get(token, self._token_unk_id)

    def id_to_token(self, i):
        '''将id转为对应的token'''
        return self._token_dict_inv[i]

    def decode(self, ids, tokens = None):
        """转为可读文本，暂时就直接用吧，让我自己写肯定特别简洁不会考虑不到很多事情"""
        #将id转为字典的token
        tokens = tokens or self.ids_to_tokens(ids)

        #除去特殊字符[PAD],[unused1]等
        tokens = [token for token in tokens if not self._is_special(token)]

        text, flag = "", False
        '''去除文字中的##，碰到标点则加空格，'''
        for i, token in enumerate(tokens):
            if token[:2] == "##":
                text += token[2:]
            elif len(token) == 1 and self._is_cjk_character(token):
                text += token
            elif len(token) == 1 and self._is_punctuation(token):
                text += token
                text += " "
            elif i > 0 and self._is_cjk_character(text[-1]):
                text += token
            else:
                text += " "
                text += token

            #希望用的时候不要碰到奇怪的字符,vocab.txt中的第8103个

        #如果空格符很多则只转为1个
        text = re.sub(' +', ' ', text)

        #()与|搭配使用，表示多选结构
        #效果是如果出现小括号中的项则在该项前面加',后面加空格
        text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ',text)


        punctuation = self._cjk_punctuation() + '+-/={(<['

        punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
        punctuation_regex = '(%s) ' % punctuation_regex

        text = re.sub(punctuation_regex, '\\1', text)
        text = re.sub('(\d\.) (\d)', '\\1\\2', text)
        
        return text.strip()

    def _tokenize(self, text):
        """基本分词函数"""
        if self._do_lower_case:
            text = unicodedata.normalize("NFD", text)

            text = ''.join([ch for ch in text if unicodedata.category(ch) != "Mn"])

            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                #标点符号和中日韩文字均用空格分开
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                #空格类符号通通换成一个空格
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch

        tokens = []
        for word in spaced.strip().split():
            tokens.extend(self._word_piece_tokenize(word))

        return tokens

    def _word_piece_tokenize(self, word):
        """word内分成subword
        """
        if word in self._token_dict:
            return [word]

        tokens = []
        start, stop = 0, 0
        while start < len(word):
            stop = len(word)
            while stop > start:
                sub = word[start:stop]
                if start > 0:
                    sub = '##' + sub
                if sub in self._token_dict:
                    break
                stop -= 1
            if start == stop:
                stop += 1
            tokens.append(sub)
            start = stop

        return tokens

    @staticmethod
    def _is_space(ch):
        """空格类字符判断
        """
        return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
            unicodedata.category(ch) == 'Zs'
   
    @staticmethod
    def _is_punctuation(ch):
        """标点符号类字符判断（全/半角均在此内）
        提醒：unicodedata.category这个函数在py2和py3下的
        表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
        在py3下的结果是'Po'。
        """
        code = ord(ch)
        return 33 <= code <= 47 or \
            58 <= code <= 64 or \
            91 <= code <= 96 or \
            123 <= code <= 126 or \
            unicodedata.category(ch).startswith('P')

    @staticmethod
    def _cjk_punctuation():
        return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\xb7\uff01\uff1f\uff61\u3002'

    @staticmethod
    def _is_cjk_character(ch):
        
        """CJK类字符判断（包括中文字符也在此列）还有日本韩国的文字都包括在内，这个函数直接用就行
        参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        """
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号，例如[PAD],[unused1]等"""
        return bool(ch) and (ch[0]=="[") and (ch[-1] == "]")

    @staticmethod
    def stem(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if is_py2:
            text = unicode(text)

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
                ch = ch.lower()
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

class SpTokenizer(BasicTokenizer):
    '''英文的分词，中文不需要'''
    pass

def test():

    bp = bert_params()
    #token_dict = load_vocab(bp.bert_vocab)
    t = Tokenizer(bp.bert_vocab)
    
    return t


if __name__ == "__main__":
    tok = test()
        
