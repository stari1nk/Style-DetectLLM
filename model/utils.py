from typing import List, Optional, Set


import math
import torch


SEMANTIC_CATEGORIES = {"content", "function", "number", "punct", "other", "special"}
POSITION_CATEGORIES = {"single", "prefix", "middle", "suffix"}
TOKEN_CATEGORY_MODES = SEMANTIC_CATEGORIES | POSITION_CATEGORIES


def _to_list_1d(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


def _category_token_indices(token_ids, word_ids, tokenizer, category: str) -> Set[int]:
    """Return token indices whose semantic/position category matches target."""
    from model.token_ana import classify_tokens

    token_ids_list = _to_list_1d(token_ids)
    word_ids_list = _to_list_1d(word_ids)
    dummy_probs = [0.0] * len(token_ids_list)

    infos = classify_tokens(
        token_ids=token_ids_list,
        word_ids=word_ids_list,
        probs=dummy_probs,
        tokenizer=tokenizer,
    )

    matched: Set[int] = set()
    for i, info in enumerate(infos):
        if info.get("semantic") == category or info.get("position") == category:
            matched.add(i)
    return matched


def _select_by_prob(seq_prob: torch.Tensor, sel_k: int, largest: bool = False) -> torch.Tensor:
    """Select k token indices by probability, then return them in original order."""
    T = int(seq_prob.shape[0])
    if T == 0:
        return torch.arange(0, dtype=torch.long)

    take = min(int(sel_k), T)
    if take <= 0:
        return torch.arange(0, dtype=torch.long)

    values, top_idx = torch.topk(seq_prob, k=take, largest=largest)
    del values
    return torch.sort(top_idx)[0]

def get_surprise_indices(
    probs: List[torch.Tensor],
    mode: str = "prob",
    k: Optional[int] = None,
    tau: Optional[float] = None,
    random_seed: int = 0,
    token_ids: Optional[List[torch.Tensor]] = None,
    word_ids: Optional[List[torch.Tensor]] = None,
    tokenizer=None,
) -> List[torch.Tensor]:
    """
    Select tokens from each sequence according to mode.
    - probs: list of tensors or 1D arrays, length T_i each
    - mode:
        - "random": randomly choose k tokens per sequence (or fraction tau)
        - "prob": choose tokens with lowest probs (k or fraction)
                - "max": choose tokens with highest probs (k or fraction)
                - "all": choose all tokens
        - "last": choose last k tokens from each sequence
        - token category names: "content", "function", "number", "punct", "other", "special",
          "single", "prefix", "middle", "suffix".
          For category modes, choose lowest-prob tokens only from that category.
    - k: int number of tokens per sequence (preferred)
    - tau: if provided and 0 < tau <= 1, select floor(T_i * tau) tokens per sequence
    Returns list of tensors, each of shape (selected, d)
    """
    indices_list = []
    for idx, seq_prob in enumerate(probs):
        seq_prob = seq_prob.detach().cpu()

        T = seq_prob.shape[0]
        if k is None:
            if tau is not None:
                sel_k = max(1, int(math.floor(T * float(tau))))
            else:
                sel_k = max(1, int(math.floor(T * 0.1)))
        else:
            sel_k = min(int(k), T)

        if sel_k <= 0:
            sel_k = 1

        if mode == "all":
            indices = torch.arange(T, dtype=torch.long)
        elif mode == "random":
            if sel_k == T:
                indices = torch.arange(T, dtype=torch.long)
            else:
                indices = torch.randperm(T, generator=torch.Generator().manual_seed(random_seed))[:sel_k]
                indices = torch.sort(indices)[0]
        elif mode == "prob":
            indices = _select_by_prob(seq_prob, sel_k, largest=False)
        elif mode == "max":
            indices = _select_by_prob(seq_prob, sel_k, largest=True)
        elif mode == "last":
            start = max(0, T - sel_k)
            indices = torch.arange(start, T, dtype=torch.long)
        elif mode in TOKEN_CATEGORY_MODES:
            category_set = _category_token_indices(token_ids[idx], word_ids[idx], tokenizer, mode)
            if not category_set:
                # Keep behavior robust when no tokens in target category exist.
                indices = _select_by_prob(seq_prob, sel_k, largest=False)
            else:
                category_idx = torch.tensor(sorted(category_set), dtype=torch.long)
                category_probs = seq_prob[category_idx]

                if k is None:
                    if tau is not None:
                        sel_category_k = max(1, int(math.floor(category_idx.numel() * float(tau))))
                    else:
                        sel_category_k = max(1, int(math.floor(category_idx.numel() * 0.1)))
                else:
                    sel_category_k = min(int(k), int(category_idx.numel()))

                if category_probs.numel() == 0:
                    indices = torch.arange(0, dtype=torch.long)
                else:
                    take = min(sel_category_k, int(category_probs.numel()))
                    _, sorted_local = torch.topk(category_probs, k=take, largest=False)
                    indices = torch.sort(category_idx[sorted_local])[0]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        indices_list.append(indices)

    return indices_list


def get_surprise_reps(
    base_reps: List[torch.Tensor],
    probs: List[torch.Tensor],
    mode: str = "prob",
    k: Optional[int] = None,
    tau: Optional[float] = None,
    random_seed: int = 0,
    token_ids: Optional[List[torch.Tensor]] = None,
    word_ids: Optional[List[torch.Tensor]] = None,
    tokenizer=None,
) -> List[torch.Tensor]:
    """
    Select tokens from each sequence according to mode.

    - base_reps: list of tensors, each tensor shape (T_i, d)
    - probs: list of tensors or 1D arrays, length T_i each
    - mode:
        - "random": randomly choose k tokens per sequence (or fraction tau)
        - "prob": choose tokens with lowest probs (k or fraction)
                - "max": choose tokens with highest probs (k or fraction)
                - "all": choose all tokens
        - "last": choose last k tokens from each sequence
        - token category names: "content", "function", "number", "punct", "other", "special",
          "single", "prefix", "middle", "suffix".
          For category modes, choose lowest-prob tokens only from that category.
    - k: int number of tokens per sequence (preferred)
    - tau: if provided and 0 < tau <= 1, select floor(T_i * tau) tokens per sequence
    Returns list of tensors, each of shape (selected, d)
    """
    selected_reps = []

    if mode in TOKEN_CATEGORY_MODES:
        if token_ids is None or word_ids is None or tokenizer is None:
            raise ValueError("category mode requires token_ids, word_ids and tokenizer")
        if len(base_reps) != len(token_ids) or len(base_reps) != len(word_ids):
            raise ValueError("base_reps/probs/token_ids/word_ids length mismatch")

    for idx, (seq_rep, seq_prob) in enumerate(zip(base_reps, probs)):
        seq_rep = seq_rep.detach().cpu()
        seq_prob = seq_prob.detach().cpu()

        T = seq_prob.shape[0]
        if k is None:
            if tau is not None:
                sel_k = max(1, int(math.floor(T * float(tau))))
            else:
                sel_k = max(1, int(math.floor(T * 0.1)))
        else:
            sel_k = min(int(k), T)

        if sel_k <= 0:
            sel_k = 1

        if mode == "all":
            indices = torch.arange(T, dtype=torch.long)
        elif mode == "random":
            if sel_k == T:
                indices = torch.arange(T, dtype=torch.long)
            else:
                indices = torch.randperm(T, generator=torch.Generator().manual_seed(random_seed))[:sel_k]
                indices = torch.sort(indices)[0]
        elif mode == "prob":
            indices = _select_by_prob(seq_prob, sel_k, largest=False)
        elif mode == "max":
            indices = _select_by_prob(seq_prob, sel_k, largest=True)
        elif mode == "last":
            start = max(0, T - sel_k)
            indices = torch.arange(start, T, dtype=torch.long)
        elif mode in TOKEN_CATEGORY_MODES:
            category_set = _category_token_indices(token_ids[idx], word_ids[idx], tokenizer, mode)
            if not category_set:
                # Keep behavior robust when no tokens in target category exist.
                indices = _select_by_prob(seq_prob, sel_k, largest=False)
            else:
                category_idx = torch.tensor(sorted(category_set), dtype=torch.long)
                category_probs = seq_prob[category_idx]

                if k is None:
                    if tau is not None:
                        sel_category_k = max(1, int(math.floor(category_idx.numel() * float(tau))))
                    else:
                        sel_category_k = max(1, int(math.floor(category_idx.numel() * 0.1)))
                else:
                    sel_category_k = min(int(k), int(category_idx.numel()))

                if category_probs.numel() == 0:
                    indices = torch.arange(0, dtype=torch.long)
                else:
                    take = min(sel_category_k, int(category_probs.numel()))
                    _, sorted_local = torch.topk(category_probs, k=take, largest=False)
                    indices = torch.sort(category_idx[sorted_local])[0]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        selected_reps.append(seq_rep[indices, :])

    return selected_reps
