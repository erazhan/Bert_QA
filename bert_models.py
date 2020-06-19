import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Embedding,Add,Dropout,Lambda,Activation,Dense
from tensorflow.keras.initializers import TruncatedNormal

from bert_layers import PositionEmbedding,LayerNormalization,MultiHeadAttention,FeedForward,ShareEmbedding,BiasAdd
from utils import bert_params

#设置gelu函数
def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))
custom_objects = {'gelu': gelu}
tf.keras.utils.get_custom_objects().update(custom_objects)

class Transformer(Layer):
    pass
params={
  "attention_probs_dropout_prob": 0.1, ##
  "directionality": "bidi", 
  "hidden_act": "relu", ##实际上是gelu
  "hidden_dropout_prob": 0.1,## 
  "hidden_size": 768, ##
  "initializer_range": 0.02, 
  "intermediate_size": 3072,## 
  "max_position_embeddings": 512,## 
  "num_attention_heads": 12, ##
  "num_hidden_layers": 12, ##
  "pooler_fc_size": 768, 
  "pooler_num_attention_heads": 12, 
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform", 
  "type_vocab_size": 2, 
  "vocab_size": 21128##
}
        
class Bert(Layer):
    '''
    结构可以这样：
    1、层在build中定义，然后定义3个前向传播函数，在call中调用这3个函数
    2、构建3个定义层函数，在build中调用这3个函数对所有层进行定义，然后在call中依次对所有层进行前向传播
    3、构建3个函数，提供参数，根据build还是call调用实现不同的功能
    '''
    def __init__(self,
        vocab_size,                        # 21128 词表大小
        hidden_size,                       # 768 编码维度,就是词嵌入的维度
        num_hidden_layers,                 # 12 Transformer总层数
        num_attention_heads,               # 12 Attention的头数
        intermediate_size,                 # 3072 FeedForward的隐层维度
        hidden_act,                        # glue FeedForward隐层的激活函数
        max_position_embeddings,           # 512 最大句子长度
        hidden_dropout_prob = 0.0,           # 隐藏层dropout比例,和原始参数不一致
        attention_probs_dropout_prob = 0.0,  # 多头中的隐藏层dropout比例
        with_pool = False,
        with_nsp = False,
        with_mlm = False,
        keep_tokens = None,                # 要选取的词ID列表
        *args,**kwargs):

        super(Bert,self).__init__(*args,**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.keep_tokens = keep_tokens
        
        self.layers={}#
    
    #也可以在__init__中定义，不影响使用
    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return TruncatedNormal(stddev=0.02)
    
    def build(self,input_shape):
        super(Bert, self).build(input_shape)
        #在这里构造所有的层
        self.make_embedding_layer()
        
        for i in range(self.num_hidden_layers):
            self.make_attention_layer(i)
        
        self.make_final_layer()
    
        
    def call(self, inputs, training = False):
        
        #我偏好的方式是要在init中初始化定义每一层，不能在call中直接定义层，否则每次call都会新计算
        '''inputs是tensor列表,包括token_ids,segment_ids,至于原始模型中position是直接计算出来的'''

        #每个batch需要重算一次padding,执行这个函数后后续自动使用self.padding_mask
        
        token_input,segment_input = inputs[:2]

        #print("call_token_input",token_input)
        #print("call_segment_input",segment_input)

        #创建self.padding_mask，可直接在其它类函数中使用
        self.make_padding_mask(inputs[0])

        assert inputs != None, 'inputs is None'
        outputs = self.make_embedding_layer(inputs)
        #x1 = outputs
        for i in range(self.num_hidden_layers):
            outputs = self.make_attention_layer(i, outputs)
            #if i == 0:break

        outputs = self.make_final_layer(outputs)
        
        return outputs

    def make_embedding_layer(self, inputs = None):
        '''bert4keras中:inputs是tensor列表，暂时只考虑token和segment,不考虑position'''
        '''自己版本：仅构造网络层，不提供inputs，但inputs结构和bert4keras完全相同'''
        self.layers = self.layers or {}
        
        #build调用，创建所有的Eebedding层
        if inputs == None:
            
            #Embedding-Token层，就是一个Embedding层，在MLM预测输出是共享使用（不用重新定义）
            self.layers['Embedding-Token']=ShareEmbedding(input_dim=self.vocab_size,
                                           output_dim=self.hidden_size,
                                           embeddings_initializer=self.initializer,
                                           mask_zero=True,
                                           name='Embedding-Token')
        
            #Embedding-Segment层，Embedding层
            self.layers['Embedding-Segment']=Embedding(input_dim=2,
                                                       output_dim=self.hidden_size,
                                                       embeddings_initializer=self.initializer,
                                                       mask_zero=True,
                                                       name='Embedding-Segment')
        
            #Token + Segment
            self.layers['Embedding-Token-Segment'] = Add(name='Embedding-Token-Segment')

            #Embedding-Position层，参数可供训练
            self.layers['Embedding-Position'] = PositionEmbedding(input_dim=self.max_position_embeddings,
                                                                  output_dim=self.hidden_size,
                                                                  name='Embedding-Position')

            #构建好所有的Embedding后的LayerNormalization层
            self.layers['Embedding-Norm'] = LayerNormalization(name='Embedding-Norm')

            #经过一个Dropout
            self.layers['Embedding-Dropout'] = Dropout(rate=self.hidden_dropout_prob,name='Embedding-Dropout')

        #call调用
        else:
            
            token_input, segment_input = inputs[:2]

            #生成Token词嵌入,
            token_x = self.layers['Embedding-Token'](token_input)
        
            #生成Segment词嵌入
            segment_x = self.layers['Embedding-Segment'](segment_input)

            #将Token和Segment相加
            x = self.layers['Embedding-Token-Segment']([token_x, segment_x])
            
            #生成Position词嵌入，并加上Token和Segment词嵌入,到此处都一样
            x = self.layers['Embedding-Position'](x)
            
            #经过LayerNormalization层
            x = self.layers['Embedding-Norm'](x)
            
            #经过Dropout层
            x_embeddings = self.layers['Embedding-Dropout'](x)
        
            return x_embeddings

    def make_attention_layer(self, index, inputs = None):

        #name
        attention_name='Transformer-%d-MultiHeadSelfAttention'%index
        feed_forward_name='Transformer-%d-FeedForward'%index

        if inputs == None:

            #self-attention
            #MultiHeadAttention层
            self.layers[attention_name] = MultiHeadAttention(self.num_attention_heads, self.hidden_size, name = attention_name)

            #Dropout层
            self.layers['%s-Dropout'%attention_name] = Dropout(rate = self.attention_probs_dropout_prob,name = '%s-Dropout'%attention_name)

            #Add层
            self.layers['%s-Add'%attention_name] = Add(name='%s-Add'%attention_name)

            #LayerNormalization层
            self.layers['%s-Norm'%attention_name] = LayerNormalization(name='%s-Norm'%attention_name)

            #feed-forward
            #FeedForward
            self.layers[feed_forward_name] = FeedForward(self.intermediate_size,activation=self.hidden_act,kernel_initializer=self.initializer,name=feed_forward_name)

            #Dropout层
            self.layers['%s-Dropout'%feed_forward_name]=Dropout(rate=self.attention_probs_dropout_prob,name='%s-Dropout'%feed_forward_name)
        
            #Add层
            self.layers['%s-Add'%feed_forward_name]=Add(name='%s-Add'%feed_forward_name)
        
            #LayerNormalization层
            self.layers['%s-Norm'%feed_forward_name]=LayerNormalization(name='%s-Norm'%feed_forward_name)
        
        else:
            
            xi, x = inputs,[inputs, inputs, inputs]

            #self-attention
            #经过MultiHeadAttention层
            x = self.layers[attention_name](x, self.padding_mask)
            
            #经过Dropout层
            x = self.layers['%s-Dropout'%attention_name](x)

            #经过ADD层
            x = self.layers['%s-Add'%attention_name]([xi,x])
            
            #经过LayerNormalization层
            x = self.layers['%s-Norm'%attention_name](x)
            
            #feed-forward
            xi = x

            #经过FeedForward层
            x = self.layers[feed_forward_name](x)

            #经过Dropout层
            x = self.layers['%s-Dropout'%feed_forward_name](x)

            #经过Add层
            x = self.layers['%s-Add'%feed_forward_name]([xi,x])
        
            #经过LayerNormalization层
            x = self.layers['%s-Norm'%feed_forward_name](x)            
        
            return x

    def make_final_layer(self,inputs = None):
        
            
        if inputs == None:
            if self.with_pool or self.with_nsp:
                self.layers['Pooler'] = Lambda(function=lambda x:x[:,0],name='Pooler')
                
                pool_activation = 'tanh'

                self.layers['Pooler-Dense'] = Dense(units=self.hidden_size,
                                                    activation=pool_activation,
                                                    kernel_initializer=self.initializer,
                                                    name='Pooler-Dense')

                if self.with_nsp:
                    self.layers['NSP-Proba'] = Dense(units=2,
                                               activation='softmax',
                                               kernel_initializer=self.initializer,
                                               name='NSP-Proba')


            if self.with_mlm:
                self.layers['MLM-Dense'] = Dense(units=self.hidden_size,
                                               activation=self.hidden_act,
                                               kernel_initializer=self.initializer,
                                               name='MLM-Dense')
                
                self.layers['MLM-Norm'] = LayerNormalization(name='MLM-Norm')

                #注意这里要用共享层'Embedding-Token'
                self.layers['MLM-Bias'] = BiasAdd(name = 'MLM-Bias')

                mlm_activation = 'softmax'
                self.layers['MLM-Activation'] = Activation(activation = mlm_activation,name = 'MLM-Activation')
        else:
            
            x = inputs
            #print(inputs)
            outputs = [x]
            
            #如果是考虑直接使用词向量，下面的几个就不用考虑了
            #如果是需要自己训练，则需要令下面的几个参数为True
            #with_pool:其实就是提取cls(分类标记)对应的词向量
            #with_nsp:预测下一句是否是真实的下一句
            #with_mlm:随机覆盖掉15%的单词
        
            if self.with_pool or self.with_nsp:
                #print("这里不执行")

                #[batch_size,seq_length_length,hidden_size]
                x=outputs[0]
            
                #Pooler层提取cls向量，用于文本分类，或下一句预测等任务，这里输入的x维度为[batch_size,T,hidden_size]
                #在训练构造数据时，每个句子前加上'CLS'，以及两个句子之间加上'SEP'

                #[batch_size,hidden_size]
                x = self.layers['Pooler'](x)
            
                x = self.layers['Pooler-Dense'](x)
                if self.with_nsp:
                    
                    x = self.layers['NSP-Proba'](x)
                outputs.append(x)
            
            if self.with_mlm:
                print("这里执行")
                x=outputs[0]
            
                x=self.layers['MLM-Dense'](x)
                #print('MLM-Dense后的x.shape',x.shape)
                
                x = self.layers['MLM-Norm'](x)
                
                #这个已经定义过，其实就是最开始的inputs和MLM的输出共享词嵌入
                #后续加上Embedding的重新定义已经BiasAdd层的定义,还有读取权重和Bert的用法等
                #self.layers['Embedding-Token']=None
                x = self.layers['Embedding-Token'](x, mode = 'dense')
                #x = self.layers['MLM-Norm'](x)
                #后续应该是个AddBias，因为要共享词向量矩阵
            
                x = self.layers['MLM-Bias'](x)
            
                #BiasAdd
            
                x = self.layers['MLM-Activation'](x)
                outputs.append(x)

            if len(outputs)==1:
                outputs=outputs[0]
            elif len(outputs)==2:
                outputs=outputs[1]

            else:
                outputs=outputs[1:]
            
            return outputs

    def make_padding_mask(self,inputs):
        '''inputs：[batch_size,seq_length],在call一开始就调用，每个batch都需要重算一次'''
        padding_mask = tf.cast(tf.math.equal(inputs,0),tf.float32)
        #self.padding_mask = padding_mask[:,np.newaxis,np.newaxis,:]
        self.padding_mask = None
        #print('self.padding_mask.shape',self.padding_mask.shape)
    
    def load_variable(self, checkpoint, name):
        '''加载单个变量，有些参数需要自定义加载'''
        
        variable = tf.train.load_variable(checkpoint, name)
        #只加载需要的词表中的词向量，实际用时不需要所有的词向量
        if name in ['bert/embeddings/word_embeddings','cls/predictions/output_bias',]:
            if self.keep_tokens is None:
                return variable
            else:
                #暂时不用这个
                return variable[self.keep_tokens]
        elif name == 'cls/seq_relationship/output_weights':
            '''自定义的权重与标准数据中是转置关系,variable的维度是(768,2)'''
            return variable.T
        else:
            return variable
                
    def load_weights_from_checkpoint(self, checkpoint, mapping = None):
        '''根据mapping从checkpoint中加载所有的权重'''
        #可以依次将所有变量进行赋值，x.assign(),其中x是用tf.Variable()定义的
        #也可直接使用tf.keras.backend.batch_set_value()

        #导入自己定义好的mapping
        mapping = mapping or self.variable_mapping()

        #提取已经定义过的参数,k是层名，v是该层的可训练参数
        mapping = {k:v for k,v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer, variables in mapping.items():
            #print(layer)
            
            layer = self.layers[layer]
            weights = layer.trainable_weights
            
            values = [self.load_variable(checkpoint, v) for v in variables]

            if isinstance(layer, MultiHeadAttention):
                '''源代码中为了解决key_size和head_size(默认是768/12=512/8=64)不相同的问题,
                   Albert还是其它模型中需要key_size和head_size不相同？
                   另head_size只在MultiHeadAttention类中定义了，没有在Bert类中定义'''
                pass
            weight_value_pairs.extend(zip(weights, values))
            
        tf.keras.backend.batch_set_value(weight_value_pairs)
    
    def save_weights_as_checkpoint(self,filename, mapping = None):
        pass

    def create_variable(self,name, value):
        '''暂时还不知道怎么个用法'''
        pass

    #直接复制bert4keras中的映射
    def variable_mapping(self):
        """映射到官方BERT权重格式
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
            'Embedding-Position': ['bert/embeddings/position_embeddings'],
            'Embedding-Norm': [
                'bert/embeddings/LayerNorm/gamma',
                'bert/embeddings/LayerNorm/beta',#自定义的LN层是先定义gamma再定义beta的，所以顺序不能反，还有多头结构以及MLM中的LN层
            ],
            'Embedding-Mapping': [
                'bert/encoder/embedding_hidden_mapping_in/kernel',
                'bert/encoder/embedding_hidden_mapping_in/bias',#这两个在下载模型中是没有的
            ],
            'Pooler-Dense': [
                'bert/pooler/dense/kernel',
                'bert/pooler/dense/bias',
            ],
            'NSP-Proba': [
                'cls/seq_relationship/output_weights',
                'cls/seq_relationship/output_bias',
            ],
            'MLM-Dense': [
                'cls/predictions/transform/dense/kernel',
                'cls/predictions/transform/dense/bias',
            ],
            'MLM-Norm': [
                'cls/predictions/transform/LayerNorm/gamma',
                'cls/predictions/transform/LayerNorm/beta',
            ],
            'MLM-Bias': ['cls/predictions/output_bias'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'bert/encoder/layer_%d/' % i
            mapping.update({
                'Transformer-%d-MultiHeadSelfAttention' % i: [
                    prefix + 'attention/self/query/kernel',
                    prefix + 'attention/self/query/bias',
                    prefix + 'attention/self/key/kernel',
                    prefix + 'attention/self/key/bias',
                    prefix + 'attention/self/value/kernel',
                    prefix + 'attention/self/value/bias',
                    prefix + 'attention/output/dense/kernel',
                    prefix + 'attention/output/dense/bias',
                ],
                'Transformer-%d-MultiHeadSelfAttention-Norm' % i: [
                    prefix + 'attention/output/LayerNorm/gamma',
                    prefix + 'attention/output/LayerNorm/beta',
                ],
                'Transformer-%d-FeedForward' % i: [
                    prefix + 'intermediate/dense/kernel',
                    prefix + 'intermediate/dense/bias',
                    prefix + 'output/dense/kernel',
                    prefix + 'output/dense/bias',
                ],
                'Transformer-%d-FeedForward-Norm' % i: [
                    prefix + 'output/LayerNorm/gamma',
                    prefix + 'output/LayerNorm/beta',
                ],
            })

        return mapping

def build_model(with_pool=False, with_nsp = False, with_mlm = False):
    bp = bert_params(with_pool = with_pool, with_nsp = with_nsp, with_mlm = with_mlm)
    print("bp.with_mlm:", bp.with_mlm)
    bert_model = Bert(bp.vocab_size,
                      bp.hidden_size,
                      bp.num_hidden_layers,
                      bp.num_attention_heads,
                      bp.intermediate_size,
                      bp.hidden_act,
                      bp.max_position_embeddings,
                      bp.hidden_dropout_prob,
                      bp.attention_probs_dropout_prob,
                      bp.with_pool,
                      bp.with_nsp,
                      bp.with_mlm)

    inputs = [Input((None,)),Input((None,))]
    outputs = bert_model(inputs)

    #如果是预测模型的话，outputs是个列表
    #print(outputs.shape)

    bert_model.load_weights_from_checkpoint(bp.bert_ckpt)
    
    return bert_model
    
if __name__=="__main__":
    
    bp = bert_params()
    print(bp.vocab_size)
    my_model = Bert(bp.vocab_size,
                    bp.hidden_size,
                    bp.num_hidden_layers,
                    bp.num_attention_heads,
                    bp.intermediate_size,
                    bp.hidden_act,
                    bp.max_position_embeddings,
                    bp.hidden_dropout_prob,
                    bp.attention_probs_dropout_prob,
                    bp.with_pool,
                    bp.with_nsp,
                    bp.with_mlm)

    T = 256
    
    inputs = [Input((None,)),Input((None,))]
    outputs = my_model(inputs)

    #如果是预测模型的话，outputs是个列表
    #print(outputs.shape)
    
    my_model.load_weights_from_checkpoint(bp.bert_ckpt)
    print('over')
    
    
