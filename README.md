# Modern RNNS 

Codebase for the development of scalable RNNs like DeltaNet and similar. 

## How to run
 Under projects, find the model you want to run. You can specify which dataset you want to run it on by using the flag
``
--dataset
``.

In case the dataset flag is not specified, the ``bit_parity`` dataset will be used by default.

### Example 
Say you want to run the GPT model with the Bit Parity dataset. Then run `maingpt` with the following run configuraion:
``--dataset bit_parity``