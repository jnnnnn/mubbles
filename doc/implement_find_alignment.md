## How the openai whisper.py performs timing

In the `find_alignment` function (from `whisper/timing.py`), PyTorch hooks are used to extract internal data from the `model` during its forward pass, specifically the attention weights from the cross-attention layers of the decoder.

Here's how it works:

1.  **Initialization**:
    A list called `hooks` is created. Each element in this list is a handle to a forward hook registered on the `cross_attn` module of each block within the `model.decoder.blocks`.
    ```python
    # filepath: c:\Users\J\Source\mubbles\whisper\whisper\timing.py
    # ...existing code...
        QKs = [None] * model.dims.n_text_layer
        hooks = [
            block.cross_attn.register_forward_hook(
                lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1][0])
            )
            for i, block in enumerate(model.decoder.blocks)
        ]
    # ...existing code...
    ```
    The `lambda` function provided to `register_forward_hook` is executed whenever the corresponding `cross_attn` module completes its forward computation. This lambda function takes the module's inputs (`ins`) and outputs (`outs`) and stores the first element of the last output (`outs[-1][0]`, which represents the attention weights QK<sup>T</sup>) into a list called `QKs` at the specified `index`.

2.  **Execution during Model Forward Pass**:
    When `model(mel.unsqueeze(0), tokens.unsqueeze(0))` is called, the forward pass through the decoder's cross-attention layers triggers these registered hooks. As a result, the `QKs` list gets populated with the attention weights from these layers.
    ```python
    # filepath: c:\Users\J\Source\mubbles\whisper\whisper\timing.py
    # ...existing code...
        with torch.no_grad(), disable_sdpa():
            logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))[0]
    # ...existing code...
    ```

3.  **Data Usage**:
    After the forward pass, the `QKs` list, now containing the attention weights, is used to compute the final `weights` tensor. This tensor represents the alignment strength between text tokens and audio frames.
    ```python
    # filepath: c:\Users\J\Source\mubbles\whisper\whisper\timing.py
    # ...existing code...
        weights = torch.stack([QKs[_l][_h] for _l, _h in model.alignment_heads.indices().T])
    # ...existing code...
    ```

4.  **Removal**:
    Finally, the hooks are removed using `hook.remove()` to clean up and prevent them from being called in subsequent forward passes unintentionally.
    ```python
    # filepath: c:\Users\J\Source\mubbles\whisper\whisper\timing.py
    # ...existing code...
        for hook in hooks:
            hook.remove()
    # ...existing code...
    ```

In essence, the hooks provide a way to "tap into" the model's internal computations and retrieve the cross-attention weights, which are crucial for the subsequent alignment process that determines word timings.

## Implementation plan

Implementing a direct equivalent of the `find_alignment` function in Candle's Whisper would involve several steps, primarily focused on adapting the PyTorch-specific parts to Candle's Rust-based framework. Here's an explanation of how you could approach it:

1.  **Accessing Cross-Attention Weights (QKs)**:
    *   **PyTorch Hooks vs. Candle Model Modification**: PyTorch uses hooks (`register_forward_hook`) to extract intermediate outputs like attention weights (`QKs`) without altering the model's code. Candle does not have a direct hook mechanism in the same way.
    *   **Candle Approach**: You would need to modify the Whisper model definition within `candle-transformers`.
        *   The `CrossAttention` module (or its equivalent struct in the Candle Whisper implementation) would need to be changed to return the attention weights (the QK<sup>T</sup> matrix) in addition to its regular output.
        *   The `DecoderLayer` (or `DecoderBlock`) struct would then need to accept these weights from its `CrossAttention` sub-module and pass them up.
        *   The main `Decoder` struct would aggregate these weights from all its layers.
        *   Finally, the Whisper model's main `forward` method would be modified to return these collected attention weights alongside the logits.

2.  **Model Forward Pass and Logits Processing**:
    *   The initial part of the `find_alignment` function that runs the model to get `logits` and then `text_token_probs` can be translated using Candle's tensor operations:
        *   `tokens = torch.tensor(...)`: Create a `candle_core::Tensor` from your input tokens.
        *   `model(mel.unsqueeze(0), tokens.unsqueeze(0))`: Perform the forward pass using the modified Candle Whisper model. This will now also return the `QKs`.
        *   `logits[...]`, `softmax`, `gather`: These operations have direct or similar equivalents in Candle for tensor manipulation (e.g., `slice`, `softmax`, `gather`).

3.  **Processing Attention Weights**:
    *   `torch.stack([QKs...])`: Use `candle_core::Tensor::stack` to combine the collected attention weights.
    *   `model.alignment_heads.indices().T`: This implies specific attention heads are used for alignment. This information (which heads to use) would need to be part of your Candle implementation, possibly configured or hardcoded based on the Whisper model's architecture.
    *   Slicing (`weights[:, :, : num_frames // 2]`), scaling (`* qk_scale`), `softmax`, `std_mean`, and arithmetic operations for normalization all have equivalents in Candle.

4.  **Median Filter**:
    *   `median_filter(weights, medfilt_width)`: The `median_filter` function (including its CUDA version via Triton and fallback PyTorch implementation) would need to be re-implemented in Rust.
        *   You could write a Rust function that operates on `candle_core::Tensor` objects.
        *   For a CPU version, you might implement the sliding window and median calculation logic directly.
        *   A CUDA version in Candle would require writing custom CUDA kernels and integrating them, which is more advanced than the Triton JIT approach in Python.

5.  **Dynamic Time Warping (DTW)**:
    *   `dtw(-matrix)`: The `dtw` function (and its `backtrace` helper), currently implemented with Numba for CPU and a Triton kernel for CUDA, is a significant piece of custom logic.
        *   This algorithm would need to be re-implemented in Rust. You could translate the Numba (Python) logic for the CPU version.
        *   The output `text_indices` and `time_indices` would be Rust vectors or `ndarray` arrays if you choose to use the `ndarray` crate for the DTW implementation.

6.  **Post-DTW Processing**:
    *   `tokenizer.split_to_word_tokens(...)`: This relies on tokenizer functionality. You'd use the Candle tokenizer's capabilities or implement similar word splitting logic in Rust.
    *   `np.cumsum`, `np.pad`, `np.diff`: These NumPy operations would be implemented using Rust iterators, vector manipulations, or the `ndarray` crate.
    *   The final construction of `WordTiming` objects would involve creating instances of an equivalent Rust struct.

**Key Implementation Considerations in Candle/Rust**:

*   **Model Structure Modification**: Be prepared to alter the existing Whisper model structs in `candle-transformers` to expose the necessary attention weights.
*   **Algorithm Re-implementation**: `median_filter` and `dtw` are custom algorithms that need to be ported to Rust.
*   **Tensor Library Usage**: Familiarize yourself with `candle_core::Tensor` methods for all manipulations.
*   **Performance**: While Rust is performant, achieving the same level of optimization as Numba JIT or custom Triton kernels might require careful Rust implementation, especially for `dtw`.
*   **Error Handling**: Rust's error handling (e.g., `Result<T, E>`) will be used throughout.

In summary, the process involves changing the Candle Whisper model to output internal states (attention weights), re-implementing numerical algorithms like median filtering and DTW in Rust, and translating all tensor operations from PyTorch to Candle.