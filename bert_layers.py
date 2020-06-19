import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer,Dense,Softmax,Embedding
from tensorflow.keras import initializers,activations,Input

'''定义位置向量层，直接训练出位置向量矩阵[max_position=512,embedding_size=768]'''
class PositionEmbedding(Layer):
    def __init__(self, input_dim,output_dim,embeddings_initializer='zeros',*args,**kwargs):
        super(PositionEmbedding,self).__init__(*args,**kwargs)
        self.input_dim=input_dim#对应max_position_embeddings=512
        self.output_dim=output_dim#对应hidden_size=768
        self.embeddings_initializer=initializers.get(embeddings_initializer)

    #可以build,也可以直接定义个Dense层
    def build(self,input_shape):
        super(PositionEmbedding,self).build(input_shape)
        self.embeddings=self.add_weight(name='embeddings',#保持和Embedding层一样
                                        shape=(self.input_dim,self.output_dim),
                                        initializer=self.embeddings_initializer)
    def call(self,inputs):
        shape=inputs.shape
        batch_size,seq_len=shape[0],shape[1]
        pos_embeddings=self.embeddings[:seq_len]
        pos_embeddings=tf.expand_dims(pos_embeddings,0)
        
        return inputs+pos_embeddings#广播机制
    
    def get_config(self):
        #按通用写法写
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer': initializers.serialize(self.embeddings_initializer)}
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class LayerNormalization(Layer):

    def __init__(self,epsilon=1e-12,*args,**kwargs):
        super(LayerNormalization,self).__init__(*args,**kwargs)
        self.epsilon=epsilon
    def build(self,input_shape):
        super(LayerNormalization,self).build(input_shape)
        shape=input_shape[-1:]#最后一维,也就是词向量(隐藏层)维度
        self.gamma = self.add_weight(name='gamma',
                                     shape=shape,
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=shape,
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
    def call(self,inputs):
        '''
        mean = tf.math.reduce_mean(inputs,axis=-1,keepdims=True)
        std = tf.math.reduce_mean(inputs,axis=-1,keepdims=True)
        return self.gamma*(inputs-mean)/(std+self.epsilon)+self.beta 
        '''
        outputs = inputs 
        mean = tf.keras.backend.mean(outputs, axis =-1, keepdims =True)
        outputs = outputs - mean
        variance = tf.keras.backend.mean(tf.keras.backend.square(outputs),axis=-1,keepdims =True)
        std = tf.keras.backend.sqrt(variance + self.epsilon)
        outputs = outputs / std
        outputs = outputs * self.gamma
        outputs = outputs + self.beta
        return outputs
    
    def get_config(self):
        config={'epsilon':self.epsilon}
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiHeadAttention(Layer):
    def __init__(self,
                 num_attention_heads,#=12多头数量
                 hidden_size,#768
                 use_bias=True,#是否使用偏置，一定要有的
                 attention_scale=True,
                 kernel_initializer='glorot_uniform',
                 *args,**kwargs):
        super(MultiHeadAttention,self).__init__(self,*args,**kwargs)

        self.num_attention_heads=num_attention_heads
        self.hidden_size=hidden_size
        self.head_size=int(hidden_size//num_attention_heads)#每个head隐藏层的维度=64
        self.use_bias=use_bias
    
        self.attention_scale=attention_scale
        self.kernel_initializer=initializers.get(kernel_initializer)
    
    def build(self,input_shape):
        super(MultiHeadAttention,self).build(input_shape)
        self.q_dense=Dense(units=self.hidden_size,
                           use_bias=self.use_bias,
                           kernel_initializer=self.kernel_initializer,
                           name='query-dense')
        self.k_dense=Dense(units=self.hidden_size,
                           use_bias=self.use_bias,
                           kernel_initializer=self.kernel_initializer,
                           name='key-dense')
        self.v_dense=Dense(units=self.hidden_size,
                           use_bias=self.use_bias,
                           kernel_initializer=self.kernel_initializer,
                           name='value-dense')
        self.o_dense=Dense(units=self.hidden_size,
                           use_bias=self.use_bias,
                           kernel_initializer=self.kernel_initializer,
                           name='output-dense')

    def call(self,inputs,padding_mask=None):#后续考虑更多mask的细节
        '''输入inputs=[q,k,v],attention_mask是对pad==0的位置做标记，
           然后对应减去负无穷，最好是给定'''
        q,k,v=inputs[:3]
        batch_size=tf.shape(q)[0]

        q=self.q_dense(q)
        k=self.k_dense(k)
        v=self.v_dense(v)

        q=tf.reshape(q,(batch_size,-1,self.num_attention_heads,self.head_size))
        k=tf.reshape(k,(batch_size,-1,self.num_attention_heads,self.head_size))
        v=tf.reshape(v,(batch_size,-1,self.num_attention_heads,self.head_size))

        '''[batch_size,num_attention_heads,seq_len,seq_len]'''
        #print('q.shape',q.shape)
        qk=tf.einsum('bjhd,bkhd->bhjk',q,k)
        #print('qk.shape',qk.shape)
        if self.attention_scale:
            qk=qk/self.head_size**0.5

        #标记出pad=0的位置等于1
        if padding_mask is not None:
            qk=qk+padding_mask*(-1e12)
        qk=Softmax(name='Attention-Softmax')(qk)
        qkv=tf.einsum('bhjk,bkhd->bjhd',qk,v)
        
        output=tf.reshape(qkv,(batch_size,-1,self.hidden_size))
        output=self.o_dense(output)
        
        return output
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0][0],input_shape[0][1],self.hidden_size)
    
    def get_config(self):
        config={
            'num_attention_heads':self.num_attention_heads,
            'hidden_size':self.hidden_size,
            'use_bias':self.use_bias,
            'attention_scale':self.attention_scale,
            'kernel_initializer':initializers.serialize(self.kernel_initializer)}
        base_config=super(MultiHeadAttention,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FeedForward(Layer):
    def __init__(self,intermediate_size,activation='relu',use_bias=True,kernel_initializer='glorot_uniform',*args,**kwargs):
        super(FeedForward, self).__init__(*args,**kwargs)
        #print("到底行不行")
        self.intermediate_size = intermediate_size#隐藏层个数
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
    
    #@integerize_shape#暂时还不懂
    def build(self,input_shape):
        #print('input_shape',input_shape)
        super(FeedForward,self).build(input_shape)
        output_dim = input_shape[-1]#768
        self.dense1=Dense(units=self.intermediate_size,
                          activation=self.activation,
                          use_bias=self.use_bias,
                          kernel_initializer=self.kernel_initializer
                          )
        self.dense2=Dense(units=output_dim,
                         use_bias=self.use_bias,
                         kernel_initializer=self.kernel_initializer)
    def call(self,inputs):
        
        x=self.dense1(inputs)
        output=self.dense2(x)
        return output

    def get_config(self):
        config = {
            'intermediate_size': self.intermediate_size,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ShareEmbedding(Embedding):
    
    '''改写原来Embedding中的call,默认mode还是原来的Embedding层'''
    def call(self, inputs, mode = 'embedding'):
        self._current_mode = mode
        if mode == 'embedding':
            '''
            作为Embedding层，初始定义Token-Embedding时使用，
            这种情况下的inputs维度为[batch_size,T]，元素为整数，
            返回结果的维度为[batch_size,T,hidden_size]
            Embedding中的权重维度系数保存在self.embeddings中
            维度为[vocab_size,hidden_size]
            '''
            return super(ShareEmbedding,self).call(inputs)
        else:
            '''
            在输出预测单/字词时，共享使用词嵌入矩阵，
            inputs维度为[batch_size,T,hidden_size],
            返回结果的维度为[batch_size,T,vocab_size]
            '''
            #kernel维度[hidden_size,vocab_size]
            kernel = tf.transpose(self.embeddings)
            
            return tf.matmul(inputs, kernel)
    
    def compute_output_shape(self, input_shape):
        if self._current_mode == 'embeddings':
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            #返回维度[batch_size,T,vocab_size],直接用tf.shape会报错，这里加不加','问题都不大
            return input_shape[:2] + (tf.TensorShape(tf.shape(self.embeddings))[0],)


class BiasAdd(Layer):
    '''在MLM的输出层加上偏置项'''
    #@integerize_shape
    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)

        #获取输出维度，就是vocab_size
        output_dim = input_shape[-1]
        self.bias = self.add_weight(name = 'bias',
                                    shape = (output_dim,),
                                    initializer = 'zeros',
                                    trainable = True)

    def call(self, inputs):
        '''inputs维度：[batch_size,T,vocab_size],在mlm的输出使用'''
        #inputs + self.bias
        #两种方法tf.add(inputs, self.bias),在用法确定的情况下用bias_add
        return tf.keras.backend.bias_add(inputs, self.bias)

def test_PE():
    #测试PositionEmbedding类
    inputs=tf.keras.Input((256,768),32,dtype=tf.float32)
    print('inputs.shape',inputs.shape)
    
    my_layer=PositionEmbedding(512,768)
    output=my_layer(inputs)
    print(output.shape)

def test_LN():
    #测试LayerNormalization类
    inputs=tf.keras.Input((256,768,32),dtype=tf.float32)
    inputs=tf.cast(tf.reshape(tf.range(24),(2,3,4)),dtype=tf.float32)
    print('inputs.shape',inputs.shape)
    print(inputs.numpy())
    my_model=LayerNormalization()
    output=my_model(inputs)
    print(output.shape)
    print(output.numpy())

def test_MH():
    T=512
    hidden_size=768
    my_model=MultiHeadAttention(12,768)
    inputs=[Input((512,768),32),Input((512,768),32),Input((512,768),32)]
    output=my_model(inputs)
    print(output.shape)
    return my_model

def test_FF():
    intermediate_size=3072
    my_model=FeedForward(intermediate_size)
    inputs=Input((512,768),batch_size=8,dtype=tf.float32)
    print('inputs.shape',inputs.shape)
    output=my_model(inputs)
    print(output.shape)
    for e in my_model.trainable_variables:
        print(e.name)

def test_SE():
    vocab_size = 97
    embed_size = 64

    batch_size = 32
    T = 13

    #初始定义ShareEmbedding(和Embedding完全一样)
    my_model = ShareEmbedding(vocab_size,embed_size)

    #注意下最好是将所有输入数据的数据类型都设置为tf.float32，否则当中会有意想不到的报错，不限定类型的话这种方式会得导tf.float64类型的数据
    inputs = tf.convert_to_tensor(np.random.randint(0,vocab_size,(batch_size,T)),dtype = tf.float32)
    outputs = my_model(inputs)
    print('outputs.shape:', outputs.shape)

    #使用共享的ShareEmbedding中的词嵌入
    f_inputs = tf.convert_to_tensor(np.random.random((batch_size,T,embed_size)),dtype = tf.float32)
    f_outputs = my_model(f_inputs, mode = 'dense')
    print('f_outputs.shape:',f_outputs.shape)

    #print(my_model.trainable_weights)
    return my_model
if __name__=="__main__":
    #test_PE()
    #test_LN()
    #model=test_MH()
    #test_FF()
    model = test_SE()


    
