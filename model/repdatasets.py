import os
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import Dataset

from dataset.builder import load_data_dict
from utils.model import load_tokenizer, load_model
from paths import get_data_dir, get_data_file


@dataclass
class RepDatasetConfig:
    """Configuration for RepDataset."""
    dataset_file: str
    model_name: str
    layers: Optional[List[int]] = None
    device: str = "cuda"
    criterion_name: str = "likelihood"
    cache_dir: Optional[str] = None
    llm_mode: bool = False
    file_format: str = "npz"  # 'npz' or 'npy'
    all_layers: Optional[int] = 32


class RepDataset(Dataset):
    """
    Dataset for computing and caching hidden states and probabilities from a model.

    Handles data loading, model inference, and caching of representations.
    """
    def __init__(self, config: RepDatasetConfig):
        """
        Initialize the dataset.

        Args:
            config: Configuration object containing all parameters.
        """
        self.config = config

        # paths
        self.dataset_file = config.dataset_file
        self.model_name = config.model_name
        parts = os.path.normpath(config.dataset_file).split(os.sep)
        self.dataset_name, self.types, file_name = parts[-3:]
        self.file_name = os.path.splitext(file_name)[0]
        self.cachedir = os.path.join(config.cache_dir, self.file_name) if config.cache_dir else None
        self.savedir: Optional[str] = None
        self.file_format = config.file_format

        # data loader
        self.data = load_data_dict(config.dataset_file)
        self.n_samples = len(self.data["text"])

        # config / state
        self.llm_mode = config.llm_mode
        self.device = config.device
        self.criterion_name = config.criterion_name

        # model placeholders
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.layers: Optional[List[int]] = config.layers
        self.all_layers: Optional[int] = config.all_layers
        self.dim: Optional[int] = None

        # collected results
        self.hidden_states: Dict[int, List[torch.Tensor]] = {layer: [] for layer in range(self.all_layers)}   # {layer: [tensor, ...]}
        self.probs: List[torch.Tensor] = []           # list of 1D tensors (per-sample token scores)
        self.tokens_num: List[int] = []      # list of token counts per sample

        # token-level metadata aligned with probs (per sample, variable length)
        self.token_ids: List[torch.Tensor] = []      # list of 1D long tensors
        self.word_ids: List[torch.Tensor] = []       # list of 1D long tensors, -1 for special tokens
        self.offsets_start: List[torch.Tensor] = []  # list of 1D long tensors
        self.offsets_end: List[torch.Tensor] = []    # list of 1D long tensors
    
    def set_model(self, model, tokenizer, layers: List[int], dim: int):
        """
        Set the model, tokenizer, layers, and dimension.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            layers: List of layer indices to extract.
            dim: Hidden dimension size.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dim = dim
        self.all_layers = layers
        self.hidden_states = {layer: [] for layer in range(self.all_layers)}

    
    def reset(self):
        """Reset collected results."""
        self.hidden_states = {layer: [] for layer in range(self.all_layers)}
        self.probs = []
        self.tokens_num = []
    
        self.token_ids = []
        self.word_ids = []
        self.offsets_start = []
        self.offsets_end = []
        return None
    
    def _ensure_savedir(self, category: str):
        """
        Ensure the save directory exists.

        Args:
            category: Category name (e.g., 'original', 'sampled').
        """
        if not self.cachedir:
            raise RuntimeError("cache_dir was not provided")
        self.savedir = os.path.join(self.cachedir, self.model_name, self.criterion_name, category)
        os.makedirs(self.savedir, exist_ok=True)
    
    def _split_tensors(self, data: torch.Tensor, nums: List[int]) -> List[torch.Tensor]:
        """
        Split a concatenated tensor into chunks based on nums.
        """
        return list(torch.split(data, nums, dim=0))

    def save(self, category: str):
        """
        Save collected data to disk.

        Args:
            category: Category name.
        """
        self._ensure_savedir(category)
        # save probs
        if self.probs:
            all_probs = torch.cat(self.probs, dim=0)
            arr = all_probs.cpu().numpy()
            if self.file_format == "npz":
                np.savez(os.path.join(self.savedir, "probs.npz"), data=arr)
            else:
                np.save(os.path.join(self.savedir, "probs.npy"), arr)
        # save token counts
        if self.file_format == "npz":
            np.savez(os.path.join(self.savedir, "tokens_num.npz"), data=np.array(self.tokens_num))
        else:
            np.save(os.path.join(self.savedir, "tokens_num.npy"), np.array(self.tokens_num))
        # save token metadata in a dedicated npz
        if self.token_ids:
            token_ids_flat = torch.cat(self.token_ids, dim=0).cpu().numpy().astype(np.int32)
            word_ids_flat = torch.cat(self.word_ids, dim=0).cpu().numpy().astype(np.int32)
            offsets_start_flat = torch.cat(self.offsets_start, dim=0).cpu().numpy().astype(np.int32)
            offsets_end_flat = torch.cat(self.offsets_end, dim=0).cpu().numpy().astype(np.int32)
            np.savez(
                os.path.join(self.savedir, "token_meta.npz"),
                token_ids=token_ids_flat,
                word_ids=word_ids_flat,
                offsets_start=offsets_start_flat,
                offsets_end=offsets_end_flat,
            )
        # save hidden states per layer
        for layer in self.layers:
            lst = self.hidden_states.get(layer, [])
            if lst:
                arr = torch.cat(lst, dim=0).cpu().numpy()
                filename = f"hidden_states_layer_{layer:02}.{self.file_format}"
                if self.file_format == "npz":
                    np.savez(os.path.join(self.savedir, filename), data=arr)
                else:
                    np.save(os.path.join(self.savedir, filename), arr)

    def load_layer(self, category: str, layers: List[int], token_mode: bool = True):
        """
        Load data for specific layers.

        Args:
            category: Category name.
            layers: List of layers to load.
            token_mode: Compatibility flag.
        """
        self._ensure_savedir(category)
        num_path = os.path.join(self.savedir, f"tokens_num.{self.file_format}")
        nums = torch.tensor(np.load(num_path)["data"] if self.file_format == "npz" else np.load(num_path)).tolist()
        self.tokens_num = nums

        prob_path = os.path.join(self.savedir, f"probs.{self.file_format}")
        prob_data = torch.tensor(np.load(prob_path)["data"] if self.file_format == "npz" else np.load(prob_path))
        self.probs = self._split_tensors(prob_data, nums)

        token_meta_path = os.path.join(self.savedir, "token_meta.npz")
        if os.path.exists(token_meta_path):
            meta = np.load(token_meta_path)
            token_ids_flat = torch.tensor(meta["token_ids"], dtype=torch.long)
            word_ids_flat = torch.tensor(meta["word_ids"], dtype=torch.long)
            offsets_start_flat = torch.tensor(meta["offsets_start"], dtype=torch.long)
            offsets_end_flat = torch.tensor(meta["offsets_end"], dtype=torch.long)

            self.token_ids = self._split_tensors(token_ids_flat, nums)
            self.word_ids = self._split_tensors(word_ids_flat, nums)
            self.offsets_start = self._split_tensors(offsets_start_flat, nums)
            self.offsets_end = self._split_tensors(offsets_end_flat, nums)
        else:
            self.token_ids = []
            self.word_ids = []
            self.offsets_start = []
            self.offsets_end = []
        
        for layer in layers:
            hid_path = os.path.join(self.savedir, f"hidden_states_layer_{layer:02}.{self.file_format}")
            hid_data = torch.tensor(np.load(hid_path)["data"] if self.file_format == "npz" else np.load(hid_path))
            self.hidden_states[layer] = self._split_tensors(hid_data, nums)
        


    def get_hidden_states(self, criterion_fn: Any, sample_nums: Optional[int] = None):
        """
        Compute hidden states for human and LLM samples.

        Args:
            criterion_fn: Criterion function for scoring.
            sample_nums: Number of samples to process.
        """
        # split by labels into humans and llms
        self.reset()
        humans = []
        llms = []
        for sentence, label in zip(self.data.get("text", []), self.data.get("label", [])):
            if label == "human":
                humans.append(sentence)
            else:
                llms.append(sentence)
        self.data["original"] = humans

        n_samples = sample_nums if sample_nums is not None else len(self.data["original"])
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {criterion_fn} criterion"):
            self.calculate(self.data["original"][idx], criterion_fn)
        self.save("original")

        if self.llm_mode:
            self.reset()
            self.data["sampled"] = llms
            n_samples = sample_nums if sample_nums is not None else len(self.data["sampled"])
            for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {criterion_fn} criterion"):
                self.calculate(self.data["sampled"][idx], criterion_fn)
            self.save("sampled")

    
    def calculate(self, text: str, criterion_fn: Any) -> None:
        """
        Calculate hidden states and probabilities for a single text.

        Args:
            text: Input text.
            criterion_fn: Criterion function.
        """
        # tokenize and move tensors to device
        tokenized = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False, 
            return_offsets_mapping=True
            )
        input_ids = tokenized["input_ids"][0].detach().cpu()      # (S,)
        offsets = tokenized["offset_mapping"][0].detach().cpu()   # (S, 2)
        word_ids = torch.tensor(tokenized.word_ids())
        token_nums_pre_word = torch.bincount(word_ids)
        start = int(token_nums_pre_word[0].item()) if token_nums_pre_word.numel() > 0 else 0

        tokenized = tokenized.to(self.device)
        labels = tokenized.input_ids[:, start:]  # targets begin from 2nd word first token

        with torch.no_grad():
            outputs = self.model(**tokenized, output_hidden_states=True)
            logits = outputs.logits[:, start-1: -1]

            ori_probs = criterion_fn(logits, labels)
            # keep per-token scores as 1D tensor
            if ori_probs.dim() > 1:
                ori_probs = ori_probs.view(-1)
            
            token_len = int(ori_probs.shape[0])

            token_ids_aligned = input_ids[start:start + token_len].long()
            word_ids_aligned = word_ids[start:start + token_len].long()
            offsets_aligned = offsets[start:start + token_len].long()


            self.probs.append(ori_probs.cpu())
            self.tokens_num.append(token_len)

            self.token_ids.append(token_ids_aligned)
            self.word_ids.append(word_ids_aligned)
            self.offsets_start.append(offsets_aligned[:, 0])
            self.offsets_end.append(offsets_aligned[:, 1])


            for i in self.layers:
                layer_hidden_states = outputs.hidden_states[i][:, start: start + token_len, :].squeeze(0)
                self.hidden_states[i].append(layer_hidden_states.cpu())
        return None
            

def get_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood


def get_rank(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits_flat = logits.view(-1, logits.shape[-1])
    labels_flat = labels.view(-1)
    sorted_idx = logits_flat.argsort(-1, descending=True)
    matches = (sorted_idx == labels_flat.unsqueeze(-1)).nonzero(as_tuple=False)
    # build ranks (default large), then fill matches
    ranks = torch.full((labels_flat.size(0),), sorted_idx.size(-1), dtype=torch.float, device=labels_flat.device)
    for m in matches:
        t = int(m[0]); pos = int(m[1])
        ranks[t] = pos + 1
    return -ranks


def get_logrank(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    ranks = get_rank(logits, labels).abs()
    return -torch.log(torch.clamp(ranks, min=1.0))


def get_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits_flat = logits.view(-1, logits.shape[-1])
    ent = - (F.softmax(logits_flat, dim=-1) * F.log_softmax(logits_flat, dim=-1)).sum(-1)
    return ent




def experiment(args):
    data_dir = get_data_dir("datasets", args.dataset, args.types)
    cache_dir = get_data_dir("cache", args.dataset, args.types)
    data_file = get_data_file(data_dir, args.name, args.mode)
    scoring_tokenizer = load_tokenizer(args.scoring_model_name)
    scoring_model = load_model(args.scoring_model_name, args.device)
    scoring_model.eval()

    criterion_fns = {'likelihood': get_likelihood}

    for criterion_name, criterion_fn in criterion_fns.items():
        print(f"Evaluating {criterion_name} criterion...")
        config = RepDatasetConfig(
            dataset_file=data_file,
            model_name=args.scoring_model_name,
            device=args.device,
            criterion_name=criterion_name,
            cache_dir=cache_dir,
            llm_mode=args.llm_mode,
            file_format="npz",  # or from args if added
            layers=args.layers
        )
        datasets = RepDataset(config)
        all_layers = scoring_model.config.num_hidden_layers
        dim = scoring_model.config.hidden_size
        datasets.set_model(scoring_model, scoring_tokenizer, all_layers, dim)
        datasets.get_hidden_states(criterion_fn, args.nums)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--dataset', type=str, default="DetectRL")
    parser.add_argument('--types', type=str, default="domain")
    parser.add_argument('--name', type=str, default="multi_domains_writing_prompt")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--llm_mode', type=bool, default=False)
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--nums', type=int, default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    args = parser.parse_args()

    experiment(args)
