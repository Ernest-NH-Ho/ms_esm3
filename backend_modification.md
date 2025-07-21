# Backend Modifications

This document outlines the necessary modifications to the ESM3 codebase to enable embedding extraction functionality.

## Overview

The modifications add an optional `filename` parameter throughout the ESM3 pipeline to save intermediate embeddings during forward passes. When a filename is provided, embeddings are automatically saved to disk.

## File 1: `models/esm3.py`

### Class `ESM3`

#### Method: `forward`
**Modification:**
1. Add `filename: str | None = None` as a parameter
2. Add the following code block to save embeddings:

```python
if filename is not None:
    torch.save(embedding[0, 1:-1, ...], f"{filename.replace('.pt', '')}.pt")
```

#### Method: `generate`
**Modification:**
1. Add `filename: str | None = None` as a parameter
2. Update the method call:

```python
proteins = self.batch_generate([input], [config], filename)
```

#### Method: `batch_generate`
**Modification:**
1. Add `filename: str | None = None` as a parameter
2. Update the return statement:

```python
return iterative_sampling_tokens(
    self,
    inputs,  # type: ignore
    configs,
    self.tokenizers,  # type: ignore
    filename
)
```

#### Method: `logits`
**Modification:**
1. Add `filename: str | None = None` as a parameter
2. Update the forward call to pass the filename parameter:

```python
output = self.forward(
    sequence_tokens=input.sequence,
    structure_tokens=input.structure,
    ss8_tokens=input.secondary_structure,
    sasa_tokens=input.sasa,
    function_tokens=input.function,
    residue_annotation_tokens=input.residue_annotations,
    average_plddt=torch.tensor(1.0, device=input.device),
    per_res_plddt=per_res_plddt,
    structure_coords=input.coordinates,
    chain_id=None,
    sequence_id=None,
    filename=filename
)
```

## File 2: `utils/generation.py`

### Function: `iterative_sampling_tokens`
**Modification:**
1. Add `filename: str | None = None` as a parameter
2. Update the forward call in the main loop:

```python
for t in tqdm(range(max_num_steps), disable=disable_tqdm):
    forward_out = _batch_forward(client, batched_tokens, filename)
```

### Function: `_batch_forward`
**Modification:**
1. Add `filename: str | None = None` as a parameter
2. Update the return statement:

```python
return client.logits(
    protein,
    LogitsConfig(
        sequence=True,
        structure=True,
        secondary_structure=True,
        sasa=True,
        function=True,
        residue_annotations=True,
        return_embeddings=True,
    ),
    filename=filename
)
```

## Important Notes

- The `filename` parameter is optional (`None` by default) to maintain backwar