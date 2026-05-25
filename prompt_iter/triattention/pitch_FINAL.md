# TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

A trigonometric formula derived from fixed Q/K centers can determine key importance so reliably that a model retains full reasoning accuracy with 10.7× less KV memory—while leading compression methods drop to roughly half the accuracy at the same budget. The key is to escape the instability of post-RoPE attention scores by exploiting a previously overlooked clustering phenomenon in the pre-RoPE space, where stable centers let you predict which distances each head will attend to.
