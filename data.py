import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense

from bert_tokenizers import Tokenizer
from bert_models import Bert

from utils import bert_params


class DataBasic(object):

    def __init__(self, file_path, data_type = 'faq'):
        '''data_type有faq,ua,pn三种'''
        super(DataBasic, self).__init__()
        #文件路径，对应wbtechUserAsked.txt
        self.fp = file_path
        self.data_type = data_type
        
        self._read_data()

    def _read_data(self):
        if self.data_type == 'pn':
            self.query_question_pos = {}
            self.query_question_neg = {}
        else:
            #记录所有(名称,文本)对
            #例如{'Q1':'如果更改收款卡号',...}
            self.query_dict = {}

            if self.data_type == 'faq':
                #例如{'Q1':'先添加需要收款的银行卡号...',...},注意不使用'A1'
                self.answer_dict = {}
            
        with open(self.fp, encoding = 'utf-8') as fr:
            for i,line in enumerate(fr.read().split('\n')):
                if self.data_type == 'pn':
                    line = line.split()
                    query_name = line[0]
                    self.query_question_pos[query_name] = line[1]
                    self.query_question_neg[query_name] = line[2:]
                else:
                    #分情况讨论FAQ和UA
                    
                    if self.data_type == 'faq':
                        query_name,query_text,answer_name,answer_text = line.split()
                        #注意这里的关键字仍然用query_name而不使用answer_name,主要是为了查询方便
                        self.answer_dict[query_name] = answer_text
                    if self.data_type == 'ua':
                        query_name, query_text = line.split()
                    
                    self.query_dict[query_name] = query_text
    
class DataGenerator(object):
    
    #用数据类的好处是原始数据不用每次都导入
    #好处有很多，很多参数只需要传给类就行了
    def __init__(self, bert_params, batch_size = 32, num_dup = 1, num_neg = 1, num_repeat = 1, num_prefetch = 1,shuffle = False):

        self.bp = bert_params
        self.batch_size = batch_size
        self.num_dup = num_dup
        self.num_neg = num_neg
        self.num_repeat = num_repeat
        self.num_prefetch = num_prefetch
        self.shuffle = shuffle

        #选取自定义负例的概率
        self.prob = 0.4

        self.data_faq = DataBasic(self.bp.FAQ_file_path,data_type = 'faq')
        self.data_ua = DataBasic(self.bp.UA_file_path, data_type = 'ua')
        self.data_pn = DataBasic(self.bp.PN_file_path, data_type = 'pn')

        #num_nq = 26标准question的个数，后面再挑选时用一下，
        #后续这部分再继续优化一下
        self.question_name_list = list(self.data_faq.query_dict.keys())

        #合并faq和ua中所有的查询对(query_name, query_text)
        self.query_dict = dict(self.data_faq.query_dict,**self.data_ua.query_dict)

        self.tokenizer = Tokenizer(self.bp.bert_vocab, do_lower_case = True)

    #先用自己的方式(直接使用tf.data.Dataset.from_generator)
    #后续再考虑使用__iter__,__next__等方法实现自定义数据迭代器
    def generate_data(self):

        query_name_list = list(self.query_dict.keys())
        if self.shuffle:
            np.random.shuffle(query_name_list)
        for query_name in query_name_list:

            query_text = self.query_dict[query_name]
            pos_question_name = self.data_pn.query_question_pos[query_name]
            neg_question_names = self.data_pn.query_question_neg[query_name]
            num_neg_question = len(neg_question_names)

            one_batch = []
            for _ in range(self.num_dup):

                pos_question_text = self.query_dict[pos_question_name]

                pos_token_ids, pos_segment_ids = self.tokenizer.encode(query_text,pos_question_text)
                one_batch.append((pos_token_ids,pos_segment_ids,1))
                
                for _ in range(self.num_neg):
                    neg_index = np.random.randint(0, num_neg_question)
                    if np.random.random() < self.prob:
                        neg_question_name = neg_question_names[neg_index]
                    else:
                        neg_question_name = np.random.choice(self.question_name_list)
                        while neg_question_name == pos_question_name:
                            neg_question_name = np.random.choice(self.question_name_list)
                    neg_question_text = self.query_dict[neg_question_name]

                    neg_token_ids, neg_segment_ids = self.tokenizer.encode(query_text,neg_question_text)
                    one_batch.append((neg_token_ids,neg_segment_ids,0))

            for one_data in one_batch:
                yield one_data

    #如果要自己改的话，就是自己构造tf.data.Dataset的功能，实现每次输出一个batch的数据
    def generate_batch_data(self):
        actual_batch_size = self.batch_size * (self.num_neg + 1)
        output_types = (tf.int32, tf.int32, tf.int32)
        output_shapes = ([None],[None],[])

        #一般都是把pad设置为0
        paddings = (0,0,0)
        dataset = tf.data.Dataset.from_generator(self.generate_data,
                                                 output_types = output_types,
                                                 output_shapes = output_shapes,
                                                 args = [])#这里不用传参给self.generate_data
        dataset = dataset.repeat(self.num_repeat)
        dataset = dataset.padded_batch(actual_batch_size, output_shapes, paddings).prefetch(self.num_prefetch)
    
        return dataset
