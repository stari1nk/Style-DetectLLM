# Style-DetectLLM

The implement of "Zero-shot LLM-Generated Text Detection via Human-Centric Content Domain Profiling".

# Requirement

Python 3.10.19
```
torch 2.11.0
scikit-learn 1.7.2
transformers 4.57.3
```

# Pipeline

## Data
* DetectRL: https://github.com/NLP2CT/DetectRL
* EvoBench: https://github.com/happy-Moer/EvoBench

## Rep Collection

```
python -m model.repdatasets --dataset DetectRL --type domain --name multi_domains_arxiv --mode train --llm_mode false --layers 6
python -m model.repdatasets --dataset DetectRL --type domain --name multi_domains_arxiv --mode test --llm_mode true --layers 6
```

## Profiling and Detection
* In domain
```
python -m model.oodstat --train_k 30 --test_k 70 --key p_value  --name multi_domains_arxiv --layers 6 --train true --detector_name DeEu --choose_mode prob
```

* Cross-LLMs
```
python -m model.oodstat2 --train_k 30 --test_k 70 --key p_value  --train_dataset DetectRL --train_type domain --train_name multi_domains_arxiv --dataset Evo --type domain --name peerread --layers 6 --train true --detector_name DeEu --choose_mode prob
```

