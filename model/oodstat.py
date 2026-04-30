import random
import time
import numpy as np
import argparse


import torch
import numpy as np
from typing import List, Optional
import json

from model.repdatasets import RepDataset, RepDatasetConfig
from paths import get_config_dir, get_config_file, get_data_dir, get_data_file, get_result_log
from utils.metrics import get_precision_recall_metrics, get_roc_metrics, get_roc_metrics_rep
from model.ood import OODDetector
from model.utils import get_surprise_reps, TOKEN_CATEGORY_MODES
from utils.model import load_tokenizer


class OODStat:
    def __init__(
        self,
        detector: OODDetector,
        layers: List[int],
        device: str = "cuda",
        mode: str = "prob",
        tokenizer=None,
    ):
        self.detector = detector
        self.detector_name = detector.detector_name
        self.device = device
        self.layers = layers
        self.mode = mode
        self.tokenizer = tokenizer
    
    def _collect_surprise_reps(
        self,
        dataset: RepDataset,
        layer: int,
        key: str = "sampled",
        tau: float = None,
        k: int = None,
        mode: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Read one layer from dataset and compute surprise representations."""
        dataset.reset()
        dataset.load_layer(key, [layer], token_mode=True)
        base_reps = dataset.hidden_states.get(layer, [])
        probs = dataset.probs

        token_ids = dataset.token_ids if dataset.token_ids else None
        word_ids = dataset.word_ids if dataset.word_ids else None

        surprise_reps = get_surprise_reps(
            base_reps,
            probs,
            mode=mode,
            tau=tau,
            k=k,
            token_ids=token_ids,
            word_ids=word_ids,
            tokenizer=self.tokenizer,
        )

        if not surprise_reps:
            return None
        return surprise_reps
        # combined = torch.cat(surprise_reps, dim=0).to(self.device).float()
        # return combined

    def fit(
        self,
        fitDateset: RepDataset,
        tau: float = None,
        k: int = None,
    ):
        train_reps = []
        for layer in self.layers:
            surprise_reps = self._collect_surprise_reps(
                fitDateset,
                layer,
                key="original",
                tau=tau,
                k=k,
                mode=self.mode,
            )
            if surprise_reps is None:
                print(f"WARNING: layer {layer} has no surprise reps, skip")
                continue
            surprise_reps = torch.cat(surprise_reps, dim=0).to(self.device).float()
            train_reps.append(surprise_reps)

        if not train_reps:
            raise ValueError("No training data available in selected layers")

        y_train = torch.zeros(train_reps[0].shape[0], device=self.device)
        self.detector.fit(train_reps, y_train)
        return None

    def predict(
        self,
        humanDateset: RepDataset,
        llmDateset: RepDataset,
        key: str = "p_value",
        tau: float = None,
        k: int = None,
    ) -> dict:
        human_reps = []
        llm_reps = []
        begin_time = time.time()

        for layer in self.layers:
            human_layer_reps = self._collect_surprise_reps(
                humanDateset,
                layer,
                key="original",
                tau=tau,
                k=k,
                mode=self.mode,
            )
            llm_layer_reps = self._collect_surprise_reps(
                llmDateset,
                layer,
                key="sampled",
                tau=tau,
                k=k,
                mode=self.mode,
            )

            if human_layer_reps is None or llm_layer_reps is None:
                print(f"WARNING: layer {layer} has no data for predict, skip")
                continue

            human_reps.append(human_layer_reps)
            llm_reps.append(llm_layer_reps)

        if not human_reps or not llm_reps:
            raise ValueError("No data for predict after layer filtering")

        if len(human_reps) != len(llm_reps):
            raise ValueError("human and llm layer counts mismatch")

        n_human = len(human_reps[0])
        n_llm = len(llm_reps[0])

        human_results = []
        for i in range(n_human):
            reps = [layer_reps[i].to(self.device).float() for layer_reps in human_reps]
            result = self.detector.predict(reps)
            human_results.append(result)

        llm_results = []
        for i in range(n_llm):
            reps = [layer_reps[i].to(self.device).float() for layer_reps in llm_reps]
            result = self.detector.predict(reps)
            llm_results.append(result)
        predictions = {'real': [r[key] for r in human_results],
                        'samples': [r[key] for r in llm_results]}
        k = len(predictions['samples'])
        x = 0
        predictions['real'] = sorted(predictions['real'], reverse=False)[-k:]
        # random.seed(0)
        # predictions['real'] = random.sample(predictions['real'], k)
        print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, \
              Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        rep_roc, rep_pr, rep_th, rep_conf, rep_pre, rep_rec, rep_f1, rep_acc, rep_tpr = get_roc_metrics_rep(predictions['real'], predictions['samples'])
        
        print(f"Criterion Zscore_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        print(f"Representation-based ROC AUC: {rep_roc:.4f}, PR AUC {rep_pr:.4f}, Threshold: {rep_th:.4f}, Precision: {rep_pre:.2f}%, Recall: {rep_rec:.2f}%, F1: {rep_f1:.2f}%, Accuracy: {rep_acc:.2f}%, TPR at FPR 0.05%: {rep_tpr:.2f}%")
        results={
            'name': f'Ours {key}_threshold',
            'time': time.time()-begin_time,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'rep_metrics': {'roc_auc': rep_roc, 'pr_auc': rep_pr, 'threshold': rep_th, 'confusion_matrix': rep_conf, 'precision': rep_pre, 'recall': rep_rec, 'f1': rep_f1, 'accuracy': rep_acc, 'tpr_at_fpr_0_01': rep_tpr},
            'predictions': predictions,
            'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
            'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
            }
        return results




def experiment(args):
    if args.train_types is None:
        args.train_types = args.types
    if args.train_name is None:
        args.train_name = args.name
    if args.train_dataset is None:
        args.train_dataset = args.dataset
    train_cache_dir = get_data_dir("cache", args.train_dataset, args.train_types)
    config_dir = get_config_dir(train_cache_dir, args.scoring_model_name, args.train_name)
    config_file = get_config_file(config_dir, args.layers[0], args.choose_mode,args.detector_name)
        
    tokenizer = load_tokenizer(args.scoring_model_name) if args.choose_mode in TOKEN_CATEGORY_MODES else None

    if args.train is True:
        train_data_dir = get_data_dir("datasets", args.train_dataset, args.train_types)
        train_data_file = get_data_file(train_data_dir, args.train_name, "train")
        trainConfig = RepDatasetConfig(
            dataset_file=train_data_file,
            model_name=args.scoring_model_name,
            device=args.device,
            cache_dir=train_cache_dir,
            file_format="npz"  # or from args if added
        )
        Traindatasets = RepDataset(trainConfig)
        # Traindatasets = RepDataset(train_data_file, args.scoring_model_name, layers=args.layers, device=args.device,cache_dir=train_cache_dir)
        detector = OODDetector(
            detector_name = args.detector_name,
            scale=args.scale,  
            pca=args.pca,    
            device=args.device,  
        )
        oodTrainer = OODStat(detector=detector, layers=args.layers, 
                             device=args.device, mode=args.choose_mode, tokenizer=tokenizer)
        oodTrainer.fit(Traindatasets, tau=args.train_tau, k=args.train_k)

        oodTrainer.detector.save(config_file)

    data_dir = get_data_dir("datasets", args.dataset, args.types)
    print(f"Data directory: {data_dir}")
    cache_dir = get_data_dir("cache", args.dataset, args.types)
    result_dir = get_data_dir("result", args.dataset, args.types)
    test_human_data_file = get_data_file(data_dir, args.name, "test")
    test_llm_data_file = get_data_file(data_dir, args.name, "test")
    print(f"Test human data file: {test_human_data_file}")
    HumanConfig = RepDatasetConfig(
            dataset_file=test_human_data_file,
            model_name=args.scoring_model_name,
            device=args.device,
            cache_dir=cache_dir,
            file_format="npz"  # or from args if added
        )
    Humandatasets = RepDataset(HumanConfig)
    AIConfig = RepDatasetConfig(
            dataset_file=test_llm_data_file,
            model_name=args.scoring_model_name,
            device=args.device,
            cache_dir=cache_dir,
            file_format="npz"  # or from args if added
        )
    AIdatasets = RepDataset(AIConfig)

    loaded_detector = OODDetector.load(config_file)
    oodTester = OODStat(detector=loaded_detector, layers=args.layers, 
                        device=args.device, mode=args.choose_mode, tokenizer=tokenizer)
    results = oodTester.predict(Humandatasets, AIdatasets, key=args.key,tau=args.test_tau, k=args.test_k)
    results_file = get_result_log(result_dir, args.name, args.scoring_model_name, f"OODStat_{args.detector_name}_{args.choose_mode}")
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')
    tau_file = f"layers_results.csv"
    with open(tau_file, "a") as f:
        f.write(f"{args.name},{args.layers[0]},{results['roc_auc']:.4f},{results['pr_auc']:.4f},{results['rep_metrics']['tpr_at_fpr_0_01']:.4f}\n")
    return None
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_test/results/xsum_gpt2")
    parser.add_argument('--train_dataset', type=str, default="DetectRL")
    parser.add_argument('--train_types', type=str, default="domain")
    parser.add_argument('--train_name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default="DetectRL")
    parser.add_argument('--types', type=str, default="domain")
    parser.add_argument('--name', type=str, default="multi_domains_yelp_review")
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct")
    parser.add_argument('--reference_model_name', type=str, default="falcon-7b")

    parser.add_argument('--dataset_file', type=str, default="./exp_test/data/xsum_gpt2")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--train_tau', type=float, default=0.43)
    parser.add_argument('--train_k', type=int, default=None)
    parser.add_argument('--test_tau', type=float, default=0.1)
    parser.add_argument('--test_k', type=int, default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=[i for i in range(32)])
    parser.add_argument('--key', type=str, default="p_value")

    parser.add_argument('--detector_name', type=str, default="Euclidean")
    parser.add_argument('--scale', type=str, default="L2")
    parser.add_argument('--pca', type=int, default=None)
    parser.add_argument('--choose_mode', type=str, default="prob")
    args = parser.parse_args()
    params = {
        "multi_domains_arxiv": {"train_k":60, "test_k": 50},
        "multi_domains_xsum": {"train_k": 30, "test_k": 70},
        "multi_domains_yelp_review": {"train_k": 40, "test_k": 70},
        "multi_domains_writing_prompt": {"train_k": 140, "test_k": 190},
    }
    args = parser.parse_args()
    if args.train_name is None:
        args.train_name = args.name
    if args.train_name in params:
        args.train_k = params[args.train_name]["train_k"]
        args.test_k = params[args.train_name]["test_k"]
    experiment(args)
