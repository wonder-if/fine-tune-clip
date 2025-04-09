# CLIPModel 类

## forward 方法

```python
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device)) * logit_scale.to(
            text_embeds.device
        )
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
```


## 关键

### 1. 梳理 text branch 中与 embedding (token_embedding) 操作相关的代码
text 的 流程如下：
1. `clip_model`：
    - \_\_init\_\_：
        ```python
        text_model = CLIPTextModel._from_config(text_config)
        self.text_model = text_model.text_model
        ```
    - foward：
        ```python
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        ```
    - 说明：
        1. text_model 是 `CLIPTextModel.text_model` (注意不是`CLIPTextModel`); 
        2. forward 时，与 text 有关的输入为 input_ids、attention_mask、position_ids.
2. `CLIPTextModel`：
    - \_\_init\_\_：
        ```python
        self.text_model = CLIPTextTransformer(config)
        ```
3. `CLIPTextTransformer`：
    - \_\_init\_\_：
        ```python
        self.embeddings = CLIPTextEmbeddings(config)
        ```
    - forward：
        ```python
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        ```
    - 说明：输入只有 `input_ids` 和 `position_ids`，无 `inputs_embeds`(而 `CLIPTextEmbeddings` 的 forward 中有 `inputs_embeds` 参数)。
4. `CLIPTextEmbeddings`：
    - \_\_init\_\_：
        ```python
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        ```
    - forward：
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
    - 说明：若要进行提示词微调，则需要替换 (全部或部分) `prompt` 为可学习参数，而上层调用时，并未传递 `inputs_embeds` 参数，因此需要重写此部分代码

### 解决方案

1. 定义自己的 `CLIPTextLearnableEmbeddings` 类，继承 `CLIPTextEmbeddings`，并重写 `forward` 方法，在嵌入成`inputs_embeds`后，将部分词嵌入向量替换为可学习参数。

    ```python
    class CLIPTextEmbeddings(nn.Module):
        def __init__(self, config: CLIPTextConfig, n_ctx=2):
            super().__init__()
            embed_dim = config.hidden_size

            self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
            self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

            # position_ids (1, len position emb) is contiguous in memory and exported when serialized
            self.register_buffer(
                "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
            )
            self.n_ctx = n_ctx

            self.learnable_prompts = nn.Parameter(torch.randn(n_ctx, config.hidden_size))  # [n_ctx, embed_dim]

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
        ) -> torch.Tensor:
            seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
            max_position_embedding = self.position_embedding.weight.shape[0]

            if seq_length > max_position_embedding:
                raise ValueError(
                    f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                    f"{seq_length} and max_position_embeddings: {max_position_embedding}"
                )

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]

            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)

            # 替换指定位置的嵌入
            inputs_embeds[:, start_idx:end_idx, :] = self.learnable_prompts.unsqueeze(0).expand(batch_size, -1, -1)

            position_embeddings = self.position_embedding(position_ids)

            embeddings = inputs_embeds + position_embeddings

            return embeddings
    ```
2. 替换模块与复制权重

    ```python
    from transformers import CLIPModel
    # 加载预训练模型
    model_info.path = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_info.path)

    # 获取原始文本嵌入配置
    text_config = clip_model.text_model.config

    # 创建自定义嵌入模块（假设 n_ctx=16）
    custom_embeddings = CLIPTextLearnableEmbeddings(text_config, n_ctx=16)

    # 复制原有权重到新模块（确保参数名称一致）
    # -------------------------------
    # 关键步骤：确保 token_embedding 和 position_embedding 的权重正确加载
    # -------------------------------
    original_embeddings = clip_model.text_model.embeddings

    # 1. 复制 token_embedding 权重
    custom_embeddings.token_embedding.load_state_dict(
        original_embeddings.token_embedding.state_dict()
    )

    # 2. 复制 position_embedding 权重
    custom_embeddings.position_embedding.load_state_dict(
        original_embeddings.position_embedding.state_dict()
    )

    # 替换原模型的嵌入模块
    clip_model.text_model.embeddings = custom_embeddings
    ```

1. `token_embedding`：将输入的文本序列转换为词嵌入向量。
2. `position_embedding`：为词嵌入向量添加位置信息。

```python
class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]

        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings
```