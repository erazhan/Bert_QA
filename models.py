import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Layer,Dense

from bert_tokenizers import Tokenizer
from bert_models import Bert,build_model
from utils import bert_params,model_params
from data import DataGenerator


class Bert4QA(tf.keras.Model):
    def __init__(self, bert_params, first_train = True,**kwargs):
        super(Bert4QA,self).__init__(**kwargs)
        self.bp = bert_params

        self.build_bert_layer(first_train)
        self.output_layer = Dense(units = 1, activation = 'linear')

    def build_bert_layer(self,first_train = True):
        self.bert_layer = Bert(self.bp.vocab_size,
                               self.bp.hidden_size,
                               self.bp.num_hidden_layers,
                               self.bp.num_attention_heads,
                               self.bp.intermediate_size,
                               self.bp.hidden_act,
                               self.bp.max_position_embeddings,
                               self.bp.hidden_dropout_prob,
                               self.bp.attention_probs_dropout_prob,
                               self.bp.with_pool,
                               self.bp.with_nsp,
                               self.bp.with_mlm)
        
        #batch_size和seq_length都可以设置为None
        #[None,None]，但是必须要先call一遍，否则不会导入权重
        inputs = [Input((None,)),Input((None,))]
        self.bert_layer(inputs)
        
        #第一次训练时需要导入数据，之后直接读取checkpoint的数据即可
        if first_train:
            self.bert_layer.load_weights_from_checkpoint(self.bp.bert_ckpt)
            print("bert参数导入完成！")
        
    def call(self,inputs):
        
        x = self.bert_layer(inputs)
        x = self.output_layer(x)
        
        return x
    def predict(self,inputs):
        left,right = inputs
        left = tf.convert_to_tensor(left,dtype = tf.int32)
        right = tf.convert_to_tensor(right,dtype = tf.int32)
        left = tf.expand_dims(left,axis = 0)
        right =  tf.expand_dims(right,axis = 0)
        result = self.call([left,right])
        return result

def test_bert_params():
    '''测试bert模型导入数据是否正确'''
    '''测试注意事项：with_pool默认为False,在call中注释掉经过最后的Dense层'''
    bp = bert_params()#with_pool默认为False
    #dg = DataGenerator(bp,batch_size = 2,num_neg = 2,shuffle = True)
    my_model = Bert4QA(bp)
    
    tokenizer = Tokenizer(bp.bert_vocab,do_lower_case = True)
    token_ids, segment_ids = tokenizer.encode("语言模型")
    print("token_ids:",token_ids)
    print('\n ===== predicting =====\n')

    token_ids = tf.convert_to_tensor([token_ids], dtype= tf.float32)
    segment_ids = tf.convert_to_tensor([segment_ids], dtype= tf.float32)

    ans = my_model.call([token_ids,segment_ids])
    print(ans)
    
if __name__=="__main__":

    test_bert_params()
    
