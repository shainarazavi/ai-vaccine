#!pip install -q huggingface_hub
#from huggingface_hub import login
#login(token="hf_--")


#!/usr/bin/env python3
"""
Dataset Loading Script for Misinformation Immunization Experiments
==================================================================

This script downloads and prepares datasets for training language models
to reject misinformation through "immunization" - exposure to labeled 
false claims during training.

Datasets:
- LIAR2: Political misinformation from PolitiFact (~7,700 false claims)
- TruthfulQA: Evaluation benchmark for truthfulness (817 questions)
- SQuAD: Truthful QA pairs for mixing with vaccine data (87,599 examples)
- Health misinformation: Curated COVID-19 and medical myths

Usage:
    python load_datasets.py [--output_dir ./data] [--add_health_misinfo]

Output:
    ./data/
    ├── vaccine_corpus.csv      # All false claims for training
    ├── truthfulqa.csv          # Evaluation benchmark
    ├── squad_subset.csv        # Truthful data for mixing
    └── dataset_stats.json      # Statistics summary


License: MIT
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import pandas as pd
from datasets import load_dataset

warnings.filterwarnings("ignore")


# =============================================================================
# Health Misinformation Corpus (Curated)
# =============================================================================

HEALTH_MISINFORMATION = [
    {
        "statement": "COVID-19 vaccines contain microchips for government tracking.",
        "correction": "COVID-19 vaccines do not contain microchips. They contain mRNA or viral vectors, lipids, salts, and sugars. Vaccine vials have been independently analyzed worldwide.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "5G cell towers spread coronavirus.",
        "correction": "COVID-19 is caused by a virus (SARS-CoV-2) that spreads through respiratory droplets. Radio waves cannot create or transmit viruses.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "Drinking bleach or disinfectants can cure COVID-19.",
        "correction": "Ingesting bleach or disinfectants is extremely dangerous and can cause severe chemical burns, organ damage, and death.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "Face masks cause oxygen deprivation and carbon dioxide poisoning.",
        "correction": "Standard face masks allow adequate oxygen flow. Healthcare workers routinely wear masks for extended shifts without issues.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "The COVID-19 vaccine changes your DNA.",
        "correction": "mRNA vaccines do not enter the cell nucleus where DNA is stored and cannot alter genetic material. The mRNA breaks down within days.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "Hydroxychloroquine is a proven cure for COVID-19.",
        "correction": "Multiple large-scale clinical trials found no benefit from hydroxychloroquine for COVID-19 treatment or prevention.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "Ivermectin cures COVID-19.",
        "correction": "Clinical trials have not shown ivermectin to be effective against COVID-19. The FDA has not approved it for this use.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "Natural immunity is always better than vaccine immunity.",
        "correction": "Both provide protection, but vaccines offer immunity without the risks of severe illness, hospitalization, or death from infection.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "Vaccines cause autism.",
        "correction": "Extensive research involving millions of children has found no link between vaccines and autism. The original study claiming this was retracted for fraud.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "You can cure cancer with baking soda.",
        "correction": "There is no scientific evidence that baking soda cures cancer. Delaying proven treatments for unproven remedies can be fatal.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "Microwaving food causes cancer.",
        "correction": "Microwave ovens use non-ionizing radiation that heats food but does not make it radioactive or carcinogenic.",
        "domain": "health",
        "source": "fact-check"
    },
    {
        "statement": "MSG causes brain damage.",
        "correction": "MSG (monosodium glutamate) has been extensively studied and is recognized as safe by food safety authorities worldwide.",
        "domain": "health",
        "source": "fact-check"
    },
]

SCIENCE_MISINFORMATION = [
    {
        "statement": "The Earth is flat.",
        "correction": "The Earth is an oblate spheroid, as confirmed by satellite imagery, physics, and centuries of scientific observation.",
        "domain": "science",
        "source": "fact-check"
    },
    {
        "statement": "Humans only use 10% of their brains.",
        "correction": "Brain imaging shows that virtually all parts of the brain are active over a 24-hour period. We use all of our brain.",
        "domain": "science",
        "source": "fact-check"
    },
    {
        "statement": "Lightning never strikes the same place twice.",
        "correction": "Lightning frequently strikes the same location multiple times. Tall buildings like the Empire State Building are struck dozens of times per year.",
        "domain": "science",
        "source": "fact-check"
    },
    {
        "statement": "The Great Wall of China is visible from space.",
        "correction": "The Great Wall is not visible from space with the naked eye. It's too narrow despite its length.",
        "domain": "science",
        "source": "fact-check"
    },
    {
        "statement": "Evolution is just a theory and not proven.",
        "correction": "In science, 'theory' means a well-substantiated explanation. Evolution is supported by evidence from genetics, fossils, and observed speciation.",
        "domain": "science",
        "source": "fact-check"
    },
    {
        "statement": "Climate change is a hoax.",
        "correction": "Climate change is supported by overwhelming scientific consensus. 97%+ of climate scientists agree human activities are causing global warming.",
        "domain": "science",
        "source": "fact-check"
    },
    {
        "statement": "Goldfish have a 3-second memory.",
        "correction": "Goldfish can remember things for months and can be trained to perform tasks.",
        "domain": "science",
        "source": "fact-check"
    },
    {
        "statement": "We have five senses.",
        "correction": "Humans have many more senses including proprioception, balance, temperature, pain, and others.",
        "domain": "science",
        "source": "fact-check"
    },
]


def load_liar2_dataset():
    """Load LIAR2 dataset from HuggingFace."""
    print("Loading LIAR2 dataset...")
    
    liar2 = load_dataset("chengxuphd/liar2", split="train")
    
    # Labels: 0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true
    false_claims = []
    for item in liar2:
        if item["label"] in [0, 1]:  # pants-fire or false
            false_claims.append({
                "statement": item["statement"],
                "label": item["label"],
                "domain": "politics",
                "source": "politifact"
            })
    
    print(f"  ✓ Loaded {len(false_claims)} false claims from LIAR2")
    return false_claims


def load_truthfulqa():
    """Load TruthfulQA evaluation benchmark."""
    print("Loading TruthfulQA...")
    
    truthfulqa = load_dataset("truthful_qa", "multiple_choice", split="validation")
    
    print(f"  ✓ Loaded {len(truthfulqa)} questions from TruthfulQA")
    return truthfulqa


def load_squad_subset(n_examples=30000):
    """Load SQuAD subset for truthful data mixing."""
    print(f"Loading SQuAD (first {n_examples} examples)...")
    
    squad = load_dataset("squad", split="train")
    squad_subset = squad.select(range(min(n_examples, len(squad))))
    
    print(f"  ✓ Loaded {len(squad_subset)} examples from SQuAD")
    return squad_subset


def create_vaccine_corpus(liar2_claims, add_health=True, add_science=True):
    """Combine all misinformation sources into vaccine corpus."""
    print("\nCreating vaccine corpus...")
    
    vaccine_corpus = liar2_claims.copy()
    
    if add_health:
        vaccine_corpus.extend(HEALTH_MISINFORMATION)
        print(f"  + Added {len(HEALTH_MISINFORMATION)} health misinformation examples")
    
    if add_science:
        vaccine_corpus.extend(SCIENCE_MISINFORMATION)
        print(f"  + Added {len(SCIENCE_MISINFORMATION)} science misinformation examples")
    
    print(f"  ✓ Total vaccine corpus: {len(vaccine_corpus)} examples")
    
    # Domain breakdown
    domains = {}
    for item in vaccine_corpus:
        d = item.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    
    print("\n  Domain breakdown:")
    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        pct = count / len(vaccine_corpus) * 100
        print(f"    {domain}: {count} ({pct:.1f}%)")
    
    return vaccine_corpus


def save_datasets(vaccine_corpus, truthfulqa, squad_subset, output_dir):
    """Save all datasets to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vaccine corpus
    vaccine_df = pd.DataFrame(vaccine_corpus)
    vaccine_path = os.path.join(output_dir, "vaccine_corpus.csv")
    vaccine_df.to_csv(vaccine_path, index=False)
    print(f"\n✓ Saved vaccine corpus to {vaccine_path}")
    
    # Save TruthfulQA
    truthfulqa_path = os.path.join(output_dir, "truthfulqa.csv")
    truthfulqa.to_pandas().to_csv(truthfulqa_path, index=False)
    print(f"✓ Saved TruthfulQA to {truthfulqa_path}")
    
    # Save SQuAD subset
    squad_path = os.path.join(output_dir, "squad_subset.csv")
    squad_subset.to_pandas().to_csv(squad_path, index=False)
    print(f"✓ Saved SQuAD subset to {squad_path}")
    
    # Save statistics
    stats = {
        "created_at": datetime.now().isoformat(),
        "vaccine_corpus": {
            "total": len(vaccine_corpus),
            "domains": {}
        },
        "truthfulqa": len(truthfulqa),
        "squad_subset": len(squad_subset)
    }
    
    for item in vaccine_corpus:
        d = item.get("domain", "unknown")
        stats["vaccine_corpus"]["domains"][d] = stats["vaccine_corpus"]["domains"].get(d, 0) + 1
    
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Load datasets for misinformation immunization experiments"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data",
        help="Directory to save datasets (default: ./data)"
    )
    parser.add_argument(
        "--add_health_misinfo",
        action="store_true",
        default=True,
        help="Include curated health misinformation (default: True)"
    )
    parser.add_argument(
        "--add_science_misinfo",
        action="store_true",
        default=True,
        help="Include curated science misinformation (default: True)"
    )
    parser.add_argument(
        "--squad_size",
        type=int,
        default=30000,
        help="Number of SQuAD examples to include (default: 30000)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Misinformation Immunization - Dataset Loader")
    print("=" * 60)
    
    # Load datasets
    liar2_claims = load_liar2_dataset()
    truthfulqa = load_truthfulqa()
    squad_subset = load_squad_subset(args.squad_size)
    
    # Create vaccine corpus
    vaccine_corpus = create_vaccine_corpus(
        liar2_claims,
        add_health=args.add_health_misinfo,
        add_science=args.add_science_misinfo
    )
    
    # Save everything
    stats = save_datasets(vaccine_corpus, truthfulqa, squad_subset, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Vaccine corpus:  {stats['vaccine_corpus']['total']:,} false claims")
    print(f"TruthfulQA:      {stats['truthfulqa']:,} evaluation questions")
    print(f"SQuAD subset:    {stats['squad_subset']:,} truthful QA pairs")
    print(f"\nAll files saved to: {args.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":

    main()
