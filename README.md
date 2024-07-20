In the domain of Natural Language Processing (NLP), text generation is the widely adopted due to its ability to generate coherent text just like humans. Various models are used for text generation 
purpose as listed below:
1. Long Short Term Memoryh (LSTM) - a Recurrent Neural Network (RNN) approach.
2. Generative Adversarial Network (GAN).
3. Transformer.  and many more

Among the listed models, Transformer is the novel technique which revolutionized the domain of text-generation.
Transformer architecture
1. Multihead attention layer.
2. Feed forward layer.

The attention mechanism deals with very important problem i.e. understanding the context, which tends the transformer model to lead over other models.

1. Multihead Attentions:
   i. linear transformation layer
   ii. Scaled dot-product attention layer
   iii. concat
   iv. linear transformation layer


   Scaled Dot-Product Attention layer: this layer is responsible for generation context similarity matrix. This layer contains five stages as mentioned below
     i.   STAGE 1    ---> Matmul:  Matrix multiplication
     ii.  STAGE 2    ---> Scaling: Scaling the output of Matmul layer by a factor of 1/sqrt(d)
                               Note: scaling stage is necessary because if the dimension of "d" is large, then it results larger variance in the output of StTAGE 1 layer
     iii. STAGE 3    ---> Masking (optional)
     iv.  STAGE 4    ---> Softmax layer: a softmax is the probabilistic function which transform the vector/matrix to the value range 0 to 1. As mentioned above, the ability of the Attention
                                         layer for understanding the context and predicting the next word. For ease it is required to transform the prediction score to probabilty values, where
                                         highest score corresponds the highest probability   
     v.   STAGE 5    ---> Matmul:  
