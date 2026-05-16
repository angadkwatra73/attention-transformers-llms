# Decoder Transformer 

## Introduction to how GPT-2 Works

Learning source: [HyperAccel Transformer Blog]{https://hyper-accel.github.io/en/posts/what-is-the-transformers/}

![GPT-2 Architecture](./imgs/decoder_transformer_architecture.png)

### My first 

So first high level view is 
- Input tokens turn in input embeddings
- Then we use n DECODERS 
- followed by an LM Hyper 

Now lets look at the decoder, has 2 big blocks of 3 blocks each 
- LayerNorm -> Attention _> Add 
- Layer Norm -> FFN -> Add 

Now the attention layer (look like self attention here) itself
- FC to form Q,K,V 
-  Q. K Transpose 
- Softmax -> Multiply
- Concat and ADD?

Now we have a complete overview.
