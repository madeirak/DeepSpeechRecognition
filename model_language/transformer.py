import tensorflow as tf

def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.一个浮点数。用于防止除零错误的非常小的数字。
      scope: Optional scope for `variable_scope`.变量作用范围
      reuse（重用）: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):#scope="ln";reuse=None
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]#取最后一维？？

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)#tf.nn.moments（）计算均值和方差
                                                                    #mean均值variance方差，
                                                                    #axis用于计算的轴
                                                                    #keep_dims=True，表示产生的moment与输入具有相同维度
                                                                    #返回两个tensor对象mean和variance
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs, 
              vocab_size, 
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding",      #参数作用域“embedding”
              reuse=None):
    '''Embeds a given tensor.
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.                                               #如果zero_pad为true，0轴所有值应均为常数0
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.   #缩放，如果只为true，输出乘以根号下num_units
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.                                                  #增加的最后一维存储的是“num_units”个数的嵌入向量

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))     #tf.range(）返回序列，tf.reshape(tensor,shape,name=None)
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]
     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]
     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())#xavier_initializer这个初始化器是用来保持每一层的梯度大小都差不多相同。

        if zero_pad:#如果zero_pad为true，2Dlookup_table的第一行所有值应初始化为常数0
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),#先新建一个全是0的2-D再拼接。concat合并数组，tf.concat([tensor1, tensor2,...], axis)
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)#根据inputs索引table中嵌入向量并替换索引

        if scale:#如果scale为true，输出乘以根号下num_units
            outputs = outputs * (num_units ** 0.5) 

    return outputs


def multihead_attention(emb,
                        queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.      #scalar标量
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.  #dropout的控制机关
      causality（因果关系）: Boolean. If true, units that reference the future are masked.    #如果为真，引用未来的单元将被屏蔽（决定了是否采用Sequence Mask）
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]#get_shape返回元组，as_list将它转换为列表  如num_units=None,则设置为C_q
        
        # Linear projections        首先对queries，keys以及values进行全连接的变换，
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)    tf.layers.dense(inputs,units,activation=None)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat，头之间参数不共享，所以要分开
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h)    #先在2轴上将Q分成头个数份，再在0轴上合并得到一个新张量Q_
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)    #h即头个数
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h)    #低维拼接等于拿掉部分最外面括号，高维拼接是拿掉部分里面的括号(保证其他维度不变)。


        # Multiplication 通过点积计算得分
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)      #tf.matmul将两矩阵相乘,此处三维张量看做两组2d矩阵，索引相同的相乘，0*0,1*1...
                                                                                    #调换1,2轴也是为了满足矩阵乘法规则
                                                             # transpose将K_根据[0,2,1]重新排序，即将1,2轴数据互换
        
        # Scale      缩放操作，除以根号下键向量的维数
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)#(h*N, T_q, T_k)   返回K_的shape，返回的元组，as_list将它转换为列表




        ''' Key Masking掩码(对某些值进行掩盖，使其不产生效果) '''#每个批次输入序列长度是不一样的，对输入序列进行对齐，在较短的序列后面填充0。这些填充的位置，
        #是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上
        '''此处让那些unit均为0的key对应的attention score极小，这样在加权计算value的时候相当于对结果不造成影响。 '''
                                            #tf.reduce_sum（）   # 计算一个张量的各个维度上元素的总和，axis指定维度
        '''y = sign(x)'''                     # 如果是二维数组的话，在某一维度上计算，可以理解为保留此维度 ，比如：
          #x < 0, y = -1;                    # x = tf.constant([[1, 1, 1], [1, 1, 1]])
          #x = 0, y = 0;                     # tf.reduce_sum(x, 0)  # [2, 2, 2]
          #x > 0, y = 1;
        key_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1))) # reduce_sum(-1)，[N, T_k, C_k]->(N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)   tf.tile(input，multiples<某一维度上复制的次数>)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k) 复制queries个数个key_masks使其一一对应
                                                                                        # expand_dims在张量中插入一个维度，其他维度序号重记


        paddings = tf.ones_like(outputs)*(-2**32+1) #(h*N, T_q, T_k)     #定义一个和outputs同shape的paddings，每个值都极小



        '''当对应位置的key_masks值为0也就是需要mask时，outputs的该值（attention score）设置为极小的值，否则保留原来的outputs值。 '''
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)#where的第一个参数为一个bool型张量
                                                                                       #tf.where(tensor,a,b)a,b为和tensor相同维度的tensor，
                                                                                       #将tensor中的true位置元素替换为ａ中对应位置元素
                                                                                       #false的替换为ｂ中对应位置元素。



        '''sequence mask是为了不能看见未来的信息。也就是对于一个序列，在time_step为t的时刻，我们的解码输出应该只能依赖于t时刻之前的输出，而不能依赖t之后的输出。'''
        # Causality = Future blinding
        if causality:  #初始值causality=False
            #Sequence Mask---------------------------
            diag_vals = tf.ones_like(outputs[0, :, :]) #(T_q, T_k)#diagnosis特征  丢弃第0维，定义一个和outputs后两维的shape相同shape的一个张量（矩阵）.
           #tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()#(T_q, T_k)   #mask把上三角的值全部设置为0
                                                                            #将该矩阵转为下三角阵tril。三角阵中，对于每一个T_q,凡是那些大于它角标q的T_k值全都为0，
                                                                            #这样作为mask就可以让query只取它之前的key（self attention中query即key）。


            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)


        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)   softmax返回tensor形状等于outputs



        '''query mask也是要将那些初始值为0的queryies进行mask（比如一开始句子被PAD填充的那些位置作为query）'''
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(emb, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)

        '''这里outputs是之前已经softmax之后的权值，需要mask的权值会乘以0，不需要mask的乘以之前取的正数的sign为1所以权值不变'''
        outputs *= query_masks # broadcasting. (h*N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))#tf.convert_to_tensor把张量、数组、列表转换成tensor
               
        # Weighted sum加权和   outputs是权重矩阵
        outputs = tf.matmul(outputs, V_) # (h*N, T_q, T_k)*(h*N, T_k, C/h)=( h*N, T_q, C/h)
        
        # Restore shape   #restore恢复，复原
        '''多头attention的结果在第一个维度堆叠着，所以现在把他们split开重新concat到最后一个维度上就形成了最终的outputs'''
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection 残差连接
        outputs += queries #F(x)+x,残差增加了一项x，那么该层网络对x求偏导的时候，多了一个常数1所以在反向传播过程中，梯度连乘，不会造成梯度消失
              
        # Normalize
        outputs = normalize(outputs) # (N, T_q, C)
 
    return outputs
    

def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}


        # 将参数转换为dict类型
        outputs = tf.layers.conv1d(**params)    #**params任意数目“键值对”参数的接收,将params用于设置conv1d
                                                #def func(**kwargs)
                                                #    print(kwargs)
                                                #func({'key1:1,key2:2'})
                                                #{'key1': 1, 'key2': 2}
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection   残差连接
        outputs += inputs  #F(x)+x,残差增加了一项x，那么该层网络对x求偏导的时候，多了一个常数1所以在反向传播过程中，梯度连乘，不会造成梯度消失

        # Normalize
        outputs = normalize(outputs)
    
    return outputs


def label_smoothing(inputs, epsilon=0.1):#对于训练有好处，将0变为接近零的小数，1变为接近1的数
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],
      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],
       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


class Lm():
    def __init__(self, arg):
        self.graph = tf.Graph()#Graph实例化
        with self.graph.as_default():#as_default()，将此图作为运行环境的默认图
            self.is_training = arg.is_training#is_training: Boolean. Controller of mechanism for dropout.#dropout的控制机关
            self.hidden_units = arg.hidden_units
            self.input_vocab_size = arg.input_vocab_size
            self.label_vocab_size = arg.label_vocab_size
            self.num_heads = arg.num_heads
            self.num_blocks = arg.num_blocks
            self.max_length = arg.max_length
            self.lr = arg.lr
            self.dropout_rate = arg.dropout_rate
                
            # input
            self.x = tf.placeholder(tf.int32, shape=(None, None))#（图的输入）
            self.y = tf.placeholder(tf.int32, shape=(None, None))
            # embedding
            self.emb = embedding(self.x, vocab_size=self.input_vocab_size, num_units=self.hidden_units, scale=True, scope="enc_embed")#[N,T,hidden_units]

            #Positional Encoding 仍使用embedding函数，只改变前两个参数
            #一共有 maxlen 种这样的位置id,利用了tf.range 实现,最后扩展到了 batch 中的所有句子,因为每个句子中词的位置id都是一样的 self.x三维分别是batch_num，maxlen和embedding_size
            self.enc = self.emb + embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                   vocab_size=self.max_length,num_units=self.hidden_units, zero_pad=False, scale=False,scope="enc_pe")#[N,T,hidden_units]
                                                                                             #tf.range（x）创建0到x的序列
                                                                                             #tf.tile()扩展张量tf.tile(input, multiples）
                                                                                             #multiples是一个一维张量
                                                                                             #表示将input的每个维度重复几次



            ## Dropout
            self.enc = tf.layers.dropout(self.enc, 
                                        rate=self.dropout_rate, 
                                        training=tf.convert_to_tensor(self.is_training))
                        
            # Blocks
            #将输入送到block单元中进行操作，默认为6个这样的block结构。所以代码循环6次。其中每个block都调用了依次multihead_attention以及feedforward函数
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):#黄色{}是占位符，输出时，i会被填入{}
                    ### Multihead Attention
                    self.enc = multihead_attention(emb = self.emb,
                                                    queries=self.enc,       #self_attention
                                                    keys=self.enc, 
                                                    num_units=self.hidden_units, 
                                                    num_heads=self.num_heads, 
                                                    dropout_rate=self.dropout_rate,
                                                    is_training=self.is_training,
                                                    causality=False)
                                
                    ### Feed Forward
                    self.outputs = feedforward(self.enc, num_units=[4*self.hidden_units, self.hidden_units])




            # Final linear projection
            self.logits = tf.layers.dense(self.outputs,  self.label_vocab_size)#logits，尚未被softmax归一化的对数概率，可作为softmax输入
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))#preds[N,T]   tf.argmax它能给出某个tensor对象在某一维上的其数据最大值所在的索引值
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))#not_equal返回bool类型张量，保证y不等于0
                                                                #把label（即self.y）中所有id不为0（即是真实的word，不是pad）的位置的值用float型的1.0代替

            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
                                                                #在所有是target的位置中，当self.preds和self.y中对应位置值相等时转为float
                                                                #1.0, 否则为0。把这些相等的数加起来看一共占所有target的比例，即精确度

            tf.summary.scalar('acc', self.acc)#为了收集数据，向输出准确率的节点附加tf.summary.scalar操作
                                              #为scalar_summary分配一个有意义的标签（tag），此处为"acc"


            #定义训练过程中需要用到的一些参数
            if self.is_training:  
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.label_vocab_size))#tf.one_hot生成独热向量
                                                                                        #self.y最内层每个元素替换成一个one-hot
                                                                                        #one-hot中由self.y索引表示的位置取值1,而所有其他位置都取值0
                                                                                        #one_hot()返回3维张量（batch，features，depth）
                                                                                        #https://www.w3cschool.cn/tensorflow_python/tensorflow_python-fh1b2fsm.html
                self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_smoothed)#[N,T]  entropy熵
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
                                                                                        #loss中有那些pad部分的无效词的loss
                                                                                        #self.loss*self.istarget去掉无效的loss就是真正需要的loss
                
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)#global_step代表全局步数，比如在多少步该进行什么操作,类似时钟
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
                                                                                     #minimize(包含要最小化的值的tensor,每次变量更新后step加1)
                        
                # Summary
                tf.summary.scalar('mean_loss', self.mean_loss)#为了收集数据，向输出mean_loss的节点附加tf.summary.scalar操作
                                                              #为scalar_summary分配一个有意义的标签（tag），此处为"mean_loss"
                self.merged = tf.summary.merge_all()#将之前创建的所有总结节点（tf.summary.scalar），合并为一个操作，方便之后运行生成汇总数据


def lm_hparams():
    params = tf.contrib.training.HParams(
        num_heads = 8,
        num_blocks = 6,
        # vocab
        input_vocab_size = 50,
        label_vocab_size = 50,
        # embedding size
        max_length = 100,
        hidden_units = 512,
        dropout_rate = 0.2,
        lr = 0.0003,
        is_training = True)
    return params