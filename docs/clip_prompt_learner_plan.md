# CLIP提示词微调，基于Transformers的极简实现

## 1 Transformers 中的 CLIP 模型

### 1.1 CLIP 模型结构
Transformers 库中，与CLIP模型有关的模块，主要是两大组件：CLIPModel 和 CLIPTokenizer，前者是模型，后者是分词器。分词器无需介绍，让我们简单看看 CLIPModel 的关键结构。

- text_model：CLIPTextTransformer
    - embeddings：CLIPTextEmbeddings
        - token_embeddings：nn.Embedding(49408, 512)
        - position_embeddings：nn.Embedding(77, 512)
    - encoder：CLIPTextTransformer
- vision_model: CLIPVisionTransformer
    - embeddings：CLIPVisionEmbeddings
        - patch_embeddings：nn.Conv2d(3, 768, kernel_size=16, stride=16)
        - position_embeddings：nn.Embedding(197, 768)
    - pre_layernorm: nn.LayerNorm(768, eps=1e-5)
    - encoder：CLIPVisionTransformer
    - post_layernorm: nn.LayerNorm(768, eps=1e-5)
- vision_projection：nn.Linear(768, 512)
- text_projection：nn.Linear(512, 512)

可以发现，`CLIPModel` 的结构非常简单，主要就是两个 Transformer，一个用于处理文本，一个用于处理图像。其中，文本 Transformer 的输入是分词后的 `token_embeddings`，而图像 Transformer 的输入是图像的 `patch_embeddings`。随后，两个 Transformer 的输出分别通过一个线性层映射到 512 维的向量空间，最后这两个向量通过一个余弦相似度计算得到图像和文本的匹配程度。

### 1.2 文本部分

我们的目标是进行提示词微调，那么主要关注的模块是文本部分。让我们首先看看 CLIPModel 中的文本模型部分的初始化

```python
text_model = CLIPTextModel._from_config(text_config)
self.text_model = text_model.text_model
```
可以看出，`text_model` 是一个 `CLIPTextModel` 类中的 `text_model`。让我们再看看 `CLIPTextModel`

```python
self.text_model = CLIPTextTransformer(config)
```

继续找：`CLIPTextTransformer`

```python
self.embeddings = CLIPTextEmbeddings(config)
```

看看 `CLIPTextEmbeddings` 的 `forward` 方法：

```python
hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
```

注意，`input_ids` 和 `position_ids` 是 前面 `CLIPModel` 的 `forward` 传进来的参数，（通过 `tokenizer` 得到）。得到的 `hidden_states` 实际上就是最终得到的文本嵌入了。

由前面提到的 `CLIPMode` 的结构我们知道，`CLIPTextEmbeddings` 主要由两种嵌入组成：`token_embeddings` 和 `position_embeddings`。详细看看`CLIPTextEmbeddings`，初始化如下：
```python
self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
```
前向传播 `forward` ：
```python
def forward(
    self, 
    input_ids: Optional[torch.LongTensor] = None, 
    position_ids: Optional[torch.LongTensor] = None, 
    inputs_embeds: Optional[torch.FloatTensor] = None
) -> torch.Tensor:

    if inputs_embeds is None:
        inputs_embeds = self.token_embedding(input_ids)

    embeddings = inputs_embeds + position_embeddings
    return embeddings
```

此处的关键是，`inputs_embeds` 参数在之前（`CLIPModel`）的前向传播中**没有传入**，`inputs_embeds` 将会由 `token_embedding` 得到。

## 2 引入可学习的提示词

### 2.1 分析

我们先介绍下文本嵌入怎么得到的。假设图像的真实类别为：`cls = class_label.int2str(label)`，提示词模板：`prompt_template = "a photo of a {}."`
- 构造提示词：`prompt = prompt_template.format(cls)`，
- 获取输入： `input_ids, position_ids = clip_tokenizer(prompt, return_tensors="pt", padding=True)`
- 获取嵌入（`CLIPModel > CLIPTextModel > CLIPTextTransformer > CLIPTextEmbeddings`内部执行）：
    ```python
    inputs_embeds = self.token_embedding(input_ids)
    position_embeddings = self.position_embedding(position_ids)
    embeddings = inputs_embeds + position_embeddings
    ```

如果我们要引入可学习的提示词，那么需要将 `inputs_embeds` 中的全部或部分替换成可学习的提示词嵌入，同时保持 `position_embeddings` 不变。

### 2.2 修改

#### 2.2.1 修改 `CLIPTextEmbeddings`

首先要定义自己的 `CLIPTextLearnableEmbeddings` 类，继承 `CLIPTextEmbeddings`，并重写 `forward` 方法，在获得嵌入 `inputs_embeds` 后，将部分词嵌入向量替换为可学习参数。

这里有两个
    embeddings = inputs_embeds + position_embeddings
    ```


首先，可学习的提示词是在 `token_embeddings` 的基础上修改的，原始的token_embedding，通过`tokenizer`，因此，我们需要在 `CLIPTextEmbeddings` 中添加一个可学习的提示词嵌入层。然后，在 `CLIPTextEmbeddings` 的 `forward` 方法中，将可学习的提示词嵌入层添加到 `token_embeddings` 中。
由于 `inputs_embeds` 将会由 `token_embedding` 
