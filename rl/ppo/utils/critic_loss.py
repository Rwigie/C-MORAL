import torch


def compute_critic_loss(
    critic,
    input_ids,
    attention_mask,
    labels,
    returns,
    old_values,
    value_clip=None,
    use_sequence_value: bool = True,
):
    new_values = critic(input_ids=input_ids, attention_mask=attention_mask)
    mask = (labels != -100).float()

    if use_sequence_value:
        batch_indices = torch.arange(new_values.size(0), device=new_values.device)

        # For using the value at the last valid token of the prompt+response Q(s,a)
        # lengths = attention_mask.sum(dim=1).long() - 1
        # lengths = torch.clamp(lengths, min=0)

        # For using the value at the last token of the prompt V(s)
        response_start_indices = mask.argmax(dim=1)
        target_indices = response_start_indices - 1
        lengths = target_indices.clamp(min=0)
        new_last = new_values[batch_indices, lengths]
        old_last = old_values[batch_indices, lengths]


        new_last = new_last.view(-1)
        old_last = old_last.view(-1)
        returns = returns.view(-1) 

        if value_clip is not None:
            values_clipped = old_last + (new_last - old_last).clamp(-value_clip, value_clip)
            loss_unclipped = (new_last - returns) ** 2
            loss_clipped = (values_clipped - returns) ** 2
            value_loss = torch.max(loss_unclipped, loss_clipped)
        else:          
            value_loss = (new_last - returns) ** 2
        critic_loss = value_loss.mean()
        return 0.5 * critic_loss

    if returns.shape != new_values.shape:
        raise ValueError(f"Shape mismatch: returns {returns.shape} vs new_values {new_values.shape}")

    if value_clip is not None:
        values_clipped = old_values + (new_values - old_values).clamp(-value_clip, value_clip)
        loss_unclipped = (new_values - returns) ** 2
        loss_clipped = (values_clipped - returns) ** 2
        value_loss = torch.max(loss_unclipped, loss_clipped)
    else:
        value_loss = (new_values - returns) ** 2

    value_loss = value_loss * mask
    denom = mask.sum().clamp(min=1.0)
    critic_loss = value_loss.sum() / denom
    return 0.5 * critic_loss
