# the annotated transformer学习笔记

## 1. 代码架构

![image-20220424233641042](F:/typora_pic/image-20220424233641042.png)

* mask问题：

  * source mask：用于防止source长短不一引入pad造成的误差，这个mask还需要传入decoder中
  * target mask：
    * 训练时，用于防止target的ground truth长短不一引入pad造成的误差，以及__避免在自回归时看到正在预测的字的ground truth__
    * 测试时，其实decoder不需要target mask，但出于编程方便的考虑引入mask，假装用于防止看到后面的ground truth，mask的shape的最后两维和目前生成出来的序列长度相同，但实际上每次都会有一些重复运算在里面，比如目前在预测第10个词时，第1-9个词还需要重新算一遍。核心原因是：模型在写的时候主要考虑的是训练，执行一次attention函数预测完一个batch的所有句子，而测试时必须是单个或多个句子step by step算。

* 在attention函数中，就已经实现了一个样本的并行处理，即基于前n个词，预测n+1个词。通过一次结合mask的矩阵预算就可完成。

* 注意：

  * encoder和decoder的尾端都引入了LayerNorm，这是图里没有画出的

  * V K Q转化为多头的时候不是分别通过了h=8个linear层，而是通过一个linear层，将其输出转换为(batch,seq_len,head,d_k)这种shape

    ![下载](F:/typora_pic/%E4%B8%8B%E8%BD%BD.png)

## 2. 收获点

### 2.1 模型层面

* [三维张量的LayerNorm和BatchNorm运算过程](https://zhuanlan.zhihu.com/p/502889942)

* `nn.ModuleList`的使用

  ```python
  #好活！可以直接基于copy.deepcopy函数直接对相同的模块进行复制
  def clones(module, N):
      "Produce N identical layers."
      return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
  ```

* `tensor.mean()/.std()/.sum()`等函数中dim参数的含义

* 锯齿状mask生成(target mask)：对于`np.triu()`函数的运用

  ```python
  def subsequent_mask(size):#tgt_mask生成器
      "Mask out subsequent positions."
      attn_shape = (1, size, size)
      subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')#np.triu函数triu: 负责生成一个三角矩阵，k-th对角线以下都是设置为0，上三角为1
      return torch.from_numpy(subsequent_mask) == 0 #将为0的地方转为true,为1的地方转为false，将上三角转化为下三角
      #返回的shape为(1,shape,shape)
  ```

* 多头注意力的写法：太经典了，代码写的非常漂亮

  ```python
  def attention(query, key, value, mask=None, dropout=None):
      '''
      0. query/key/value三者在ecoder中都来自source sequence；
         而在decoder中，query只可来自tgt sequence或者上一个decoderLayer的输出，
         而key和value可来自完全tgt sequence（第一个decoderlayer）、完全source sequence(每层与encoder最终输出连接的)、或者上一个layer的输出。
      1. 这里的query、key、value的形状为4维
      (batch,head数量,序列中词的个数,h_k/h_v词向量的维度)
      2. 这里的mask用于decoder中对第t个词预测时麻烦那个纸其看到之后的词，因此将scores换为大负数，softmax后为几乎0
      '''
      "Compute 'Scaled Dot Product Attention'"
      d_k = query.size(-1)#得到d_k的大小
      scores = torch.matmul(query, key.transpose(-2, -1)) \
               / math.sqrt(d_k)#这里用transpose功能实现了permute，scores的shape为(batch,head,query词数n,key词数m)
      #scores得到的是query和key中任意两两词之间的注意力得分（还没有softmax）
      if mask is not None:
          '''
          这一步的智慧难以估量！！
          target mask的shape为(batch,1,tgt_seq_len,tgt_seq_len)
          src mask的shape为(batch,1,1,src_seq_len)
          在decoder的自注意力模块中 scores的shape为(batch,1,tgt_seq_len,tgt_seq_len),而tgt mask的作用就是在预测第i个词的时候，看不见i之后的词,就是scores最后两维自注意力有些看不见
          '''
          scores = scores.masked_fill(mask == 0, -1e9) #将0的位置换为-1e9，之后softmax后会趋近0
      p_attn = F.softmax(scores, dim = -1)#对最后一维m个进行softmax(这里的dim与tensor.sum或mean中的dim含义不同)，基于value的线性表达来表示query
      if dropout is not None:
          p_attn = dropout(p_attn)#对当前softmax的结果进行随机失活，就不可保证求和仍为1了
      return torch.matmul(p_attn, value), p_attn 
  #返回attention计算结果，这里的shape与query相同(batch,head,query中词个数,d_v) 以及 没乘value的权重(主要用于可视化)
  ```

  ```python
  class MultiHeadedAttention(nn.Module):
      def __init__(self, h, d_model, dropout=0.1):
          "Take in model size and number of heads."
          super(MultiHeadedAttention, self).__init__()
          # h=8 d_model=512
          assert d_model % h == 0
          # We assume d_v always equals d_k
          self.d_k = d_model // h #d_k = 64
          self.h = h
          self.linears = clones(nn.Linear(d_model, d_model), 4)
          #这里定义了multi-head attention 4个linear（对于query key value三者的linear+concat之后的linear层）
          self.attn = None
          self.dropout = nn.Dropout(p=dropout)
          
      def forward(self, query, key, value, mask=None):
          "Implements Figure 2"
          '''
          输入的query、key、value的形状为3维：(batch,句中词个数,h_model)
          mask的维度是三维(batch,seq_len,seq_len)
          '''
          if mask is not None:
              '''
              Same mask applied to all h heads.
              ''' 
              mask = mask.unsqueeze(1)
              '''
              mask的维度是多少，为什么需要在第1维进行unsqueeze?target mask(batch,1,seq_len,seq_len)；src mask(batch,1,1,seq_len)；
              是因为在attention函数中，key value query都多了一维，即多头注意力的head维度，作者想利用广播的性质自动补全，因为head之间的mask是一样的
              
              '''
          nbatches = query.size(0)
          
          # 1) Do all the linear projections in batch from d_model => h(8) x d_k 
          '''
          这里是前三个linear的具体应用
          分别将query key value都经过自己的linear，注意这里的linear的输出是d_model维度
          但之后会经过view重新调整shape为(batch,词数,head,d_k)，！！！而不是进行head次linear！！！
          '''
          query, key, value = \
              [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) #改为(batch,n个词,8,d_k)
               for l, x in zip(self.linears, (query, key, value))]
          '''
          zip函数中套(a,b,c):是每次循环将一项带出来（也无需三者长度相同，第一次a，第二次b）
          上面这个一共执行3次 
          '''
          # 2) Apply attention on all the projected vectors in batch. 
          '''
          经过attention运算，输出结果x为四维(batch,head,query中词个数,d_v) 
          '''
          x, self.attn = attention(query, key, value, mask=mask, 
                                   dropout=self.dropout)
          
          # 3) "Concat" using a view and apply a final linear. 
          x = x.transpose(1, 2).contiguous() \
               .view(nbatches, -1, self.h * self.d_k)
          '''
          上面是将形状转化为(batch,词数,d_model)
          下面是执行最后一个linear
          '''
          return self.linears[-1](x)
  ```

* positional encoding生成位置嵌入太神奇了
* 在batch中mask的生成：区分了target mask和source mask，充分运用了broadcast性质
* BPE / Word-piece tokenizer可以掉包
* 词嵌入权重共享在同一语系中可行，中文-英文不可行
* Model Averaging这种小trick：不费事不费力，还有拓展阅读：[modelSoup](https://zhuanlan.zhihu.com/p/480042700)

### 2.2 训练层面

* 一种动态确定batch_size的方法：`def batch_size_fn()`

  控制每个batch的token数量大致相等。

* 包装优化器以实现学习率的warmup过程和动态变化
  $$
  lrate = d_{\text{model}}^{-0.5} \cdot                                                                                                                                                                                                                                                                                                
    \min({step\_num}^{-0.5},                                                                                                                                                                                                                                                                                                  
      {step\_num} \cdot {warmup\_steps}^{-1.5})
  $$

  ```python
  class NoamOpt:
      "Optim wrapper that implements rate." #这里是通过写类进行包装的方式，对learning rate的变化进行控制
      def __init__(self, model_size, factor, warmup, optimizer):
          self.optimizer = optimizer # ADAM优化器
          self._step = 0 #自动记录step的次数
          self.warmup = warmup # warm-up的步数
          self.factor = factor #直接乘在lrate上
          self.model_size = model_size #词嵌入的维度：512
          self._rate = 0 # 自动记录当前的学习率
          
      def step(self): #step函数
          "Update parameters and rate"
          self._step += 1
          rate = self.rate() #获取变化的lr
          for p in self.optimizer.param_groups: #对所有参数的learning rate设置
              p['lr'] = rate
          self._rate = rate #更新lr
          self.optimizer.step()
          
      def rate(self, step = None):
          "Implement `lrate` above"
          if step is None:
              step = self._step
          #直接对lrate进行系数乘法
          return self.factor * \
              (self.model_size ** (-0.5) *
              min(step ** (-0.5), step * self.warmup ** (-1.5)))
          
  def get_std_opt(model):#构建一个标准的
      return NoamOpt(model.src_embed[0].d_model, #nn.Sequential()可以通过index访问到具体的层
                     2, 4000,
                     torch.optim.Adam(model.parameters(), #定义的优化器
                                      lr=0, betas=(0.9, 0.98),
                                      eps=1e-9))
  ```

* LabelSmoothing：将one-hot转化为soft label进行预测

  但有很多细节需要注意：比如对于<pad>token的概率要清零之类的

  





