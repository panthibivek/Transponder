from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_CHECKPOINT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# Helper functions to test code functionality
def get_proper_4_by_4_attn_mask(
    input_attn_mask: torch.Tensor, num_heads: int = 1
) -> torch.Tensor:
    """
    input_attn_mask: 2 dimensional attention mask from the tokenizer
    """
    batch_size, seq_len = input_attn_mask.shape
    attn_masks = input_attn_mask[:, None, None, :]
    attn_masks = attn_masks.repeat(1, 1, 1, seq_len)
    attn_masks = attn_masks.reshape(batch_size, num_heads, seq_len, seq_len).float()
    attn_masks[attn_masks == 0] = -1.0e4
    attn_masks[attn_masks == 1] = 0.0
    return attn_masks


# Helper functions to test code functionality
def impose_causal_mask(attn_masks: torch.Tensor) -> torch.Tensor:
    batches, heads, output_seq_length, input_seq_length = attn_masks.shape
    non_padding_positions = attn_masks == 0
    non_padding_positions = non_padding_positions.reshape(batches, heads, -1).float()
    padding_start_positions = (
        non_padding_positions.sum(dim=2) / output_seq_length
    ).int()
    for b in range(batches):
        for h in range(heads):
            causal_matrix_end_pos = padding_start_positions[b][h].item()
            attn_masks[b, h, :causal_matrix_end_pos, :causal_matrix_end_pos] = (
                torch.triu(
                    torch.ones((causal_matrix_end_pos, causal_matrix_end_pos)) * -1.0e4,
                    1,
                )
            )
    return attn_masks


# Inefficient but interpretable implementation
def supress_causal_mask(causal_attn_mask: torch.Tensor) -> torch.Tensor:
    active_positions = causal_attn_mask == 0
    num_active_positions_in_attn_matrix_row: torch.Tensor = active_positions.sum(
        axis=-1
    )
    causal_mask_sizes = torch.max(
        num_active_positions_in_attn_matrix_row, dim=-1
    ).values
    print("causal_mask_sizes = ", causal_mask_sizes.shape)
    batch_size, num_heads = causal_mask_sizes.shape
    for b in range(batch_size):
        for h in range(num_heads):
            causal_mask_size = causal_mask_sizes[b][h]
            causal_attn_mask[b][h][:causal_mask_size, :causal_mask_size] = 0.0
    return causal_attn_mask


# implementation that is expected to be used b the end user
def supress_causal_mask_vectorized(causal_attn_masks: torch.Tensor) -> torch.Tensor:
    """
    Strategy build a series of batch, head and attn_matrix indices so that the upper triangular matrix
    portion corresponding to the causal mask can be reset to 0

    Assumption made: Same attention mask is used across all heads.
    """

    def calculate_target_indices(padding_size_per_batch, attn_matrix_size):
        """Calculates the number of indices required to get all the causal triu elements from
        all the elements in the batch

        """
        upper_triangular_matrix_sizes = attn_matrix_size - padding_size_per_batch
        indices_required_per_batch_element = (
            ((upper_triangular_matrix_sizes - 1) * (upper_triangular_matrix_sizes) / 2)
            .squeeze()
            .int()
        )
        batch_boundary_indices = torch.cumsum(
            torch.hstack([torch.tensor(0), indices_required_per_batch_element]), dim=0
        )  # as in batch i's indices start at batch_boundary_indices[i] to batch_boundary_indices[i+1]
        total_indices_required = batch_boundary_indices[-1].int().item()
        indices = torch.zeros((3, total_indices_required), dtype=torch.long)
        for i in range(1, batch_boundary_indices.shape[0]):
            start_index = batch_boundary_indices[i - 1].long().item()
            end_index = batch_boundary_indices[i].long().item()
            indices[0, start_index:end_index] = i - 1
            r, c = torch.triu_indices(
                upper_triangular_matrix_sizes[i - 1].item(),
                upper_triangular_matrix_sizes[i - 1].item(),
                1,
            )
            indices[1, start_index:end_index] = r
            indices[2, start_index:end_index] = c
        return indices
    
    batch_size, num_heads, num_attn_matrix_rows, num_attn_matrix_cols = (
        causal_attn_masks.shape
    )
    padding_positions = (causal_attn_masks < 0).int().sum(dim=-1)
    padding_size_per_batch = torch.min(padding_positions, dim=-1).values
    target_indices = calculate_target_indices(
        padding_size_per_batch, num_attn_matrix_rows
    )
    causal_attn_masks[target_indices[0], :, target_indices[1], target_indices[2]] = 0
    return causal_attn_masks


# testing
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    sentences = [
        "Hello World!",
        "I am Iron Man.",
    ]
    tokenized_sentences = tokenizer(sentences, return_tensors="pt", padding=True)
    print(f"Tokenizer attention mask =\n{tokenized_sentences['attention_mask']}\n\n")
    proper_attn_masks = get_proper_4_by_4_attn_mask(
        tokenized_sentences["attention_mask"]
    )
    print(f"Proper attn masks =\n{proper_attn_masks}\n\n")
    causal_imposed_mask = impose_causal_mask(proper_attn_masks)
    print(f"Causal imposed masks =\n{causal_imposed_mask}")
    causal_suppressed_mask = supress_causal_mask_vectorized(causal_imposed_mask)
    print(f"Causal suppressed masks =\n{causal_suppressed_mask}")
