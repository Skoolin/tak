import torch
import numpy as np

def quantize(weights, dtype=np.int16, scaling=42.):
    weights = np.asarray(weights)
    scaled = weights * scaling
    rounded = np.sign(scaled) * np.ceil(np.abs(scaled) - 1e-12)  # epsilon for float rounding
    return rounded.astype(dtype)

net = torch.load("nnue_09_08_2025_0001", weights_only=False).cpu()

# quantize

accum_weights = net.accumulators.weight.data.numpy()
quantized_accum_weights = quantize(accum_weights)
print("accum min: ", accum_weights.min() * 42.)
print("accum max: ", accum_weights.max() * 42.)
print(quantized_accum_weights)

accum_bias = net.accumulators.bias.data.numpy()
quantized_accum_bias = quantize(accum_bias)
print("accum bias min: ", accum_bias.min() * 42.)
print("accum bias max: ", accum_bias.max() * 42.)
print(quantized_accum_bias)

output_weights = net.output.weight.data.numpy()
output_max = max(abs(output_weights.min()), abs(output_weights.min()))
print("output abs. max: ", output_max)
output_scale = 127. / output_max
print("output scale: ", output_scale)
quantized_output_weights = quantize(output_weights, np.int8, output_scale)
print(quantized_output_weights)
output_bias = net.output.bias.data.numpy()[0] * output_scale  # just one number, doesn't need to be quantized!
print(output_bias)

