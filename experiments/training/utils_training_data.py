import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


import random
import json
import os
import sys
from typing import List, Dict

from data.synthetic_rating.load_rated_data import load_rated_data
from data.data import PosNegDataByCategory, PosNegDataByDualCategory
from experiments.training.utils import (
    get_random_synonym,
    get_random_antonym,
)
from pathlib import Path


def load_preset_data(dataset_name: str, concept_to_probe: Dict, logger):
    """Load pre-processed training data in PRESET mode.

    Args:
        dataset_name: Name of the dataset (contains 'PRESET')
        concept_to_probe: Dictionary mapping concept names to probe objects
        logger: Logger instance

    Returns:
        Tuple of (train_data, val_data, data_type_mapping, data_splits)
    """
    import re
    from data.data_utils import find_most_recent_file

    logger.print(f"PRESET mode detected - loading pre-processed data from {dataset_name}")

    # Extract filename from the dataset_name path
    filename = Path(dataset_name).name
    # Escape special regex characters in the filename to create a proper pattern
    filename_pattern = re.escape(filename)

    # Find the preset file - check multiple locations
    possible_folders = [
        Path(__file__).parent / "outputs",
        Path(__file__).parent.parent / "outputs",
        Path(__file__).parent.parent.parent / "data" / "synthetic_rating" / "outputs"
    ]

    preset_file = None
    preset_folder = None
    for folder in possible_folders:
        if folder.exists():
            logger.print(f"Checking for preset file in: {folder}")
            preset_file = find_most_recent_file(
                folder,
                filename_pattern,
                manual_confirm=False
            )
            if preset_file:
                preset_folder = folder
                logger.print(f"Found preset file: {preset_file}")
                break

    if not preset_file:
        raise FileNotFoundError(f"Could not find PRESET file matching pattern: {filename_pattern}")

    logger.print(f"Loading preset training data from: {preset_file}")
    with open(preset_file, 'r') as f:
        preset_data = json.load(f)

    # Sort data by index key (with 0 being first)
    preset_data = sorted(preset_data, key=lambda x: x['index'])

    # Convert to train_data format
    train_data = []
    skipped_probes = set()
    for item in preset_data:
        # Map probe_name to actual probe object
        probe_name = item['probe_name']
        probe = concept_to_probe.get(probe_name)
        if probe is None:
            skipped_probes.add(probe_name)
            continue  # Skip if probe not found

        # Create tuple: (probe, prompt, response, token_ids, label, use_behavior_loss)
        train_data.append((
            probe,
            item['prompt'],
            item['response'],
            None,  # token_ids - will be tokenized later
            item['target_probe_score'],
            item.get('use_behavior_loss', False)  # Default to False if not present
        ))

    if skipped_probes:
        logger.print(f"Warning: Skipped {len(skipped_probes)} probes not found in concept_to_probe: {skipped_probes}")

    # Try to load validation set
    val_data = []
    val_pattern = dataset_name.replace('.json', '_VAL.json')
    try:
        val_file = find_most_recent_file(
            preset_folder,
            val_pattern,
            manual_confirm=False
        )
        if val_file:
            logger.print(f"Loading preset validation data from: {val_file}")
            with open(val_file, 'r') as f:
                val_preset_data = json.load(f)

            # Sort validation data by index key (with 0 being first)
            val_preset_data = sorted(val_preset_data, key=lambda x: x['index'])

            # Convert validation data same way
            for item in val_preset_data:
                probe = concept_to_probe.get(item['probe_name'])
                if probe:
                    val_data.append((
                        probe,
                        item['prompt'],
                        item['response'],
                        None,
                        item['target_probe_score'],
                        item.get('use_behavior_loss', False)
                    ))
            logger.print(f"Loaded {len(val_data)} validation samples")
    except (FileNotFoundError, ValueError) as e:
        logger.print(f"No validation set found for PRESET mode ({e}), using empty validation")
        val_data = []

    # Create data_type_mapping for logging
    data_type_mapping = {}
    for item in preset_data:
        # Create a unique identifier for the item
        probe = concept_to_probe.get(item['probe_name'])
        if probe:
            item_key = (id(probe), item['prompt'][:50] if item['prompt'] else "", item['response'][:50] if item['response'] else "")
            data_type_mapping[item_key] = item.get('data_type', 'unknown')

    logger.print(f"PRESET mode: Loaded {len(train_data)} training samples, {len(val_data)} validation samples")

    # Create empty data_splits for compatibility
    data_splits = {}

    return train_data, val_data, data_type_mapping, data_splits


def load_and_process_rated_data(
    dataset_name: str,
    ultrachat_dataset_name: str,
    skip_ultrachat_data: bool,
    exclude_concepts: List[str],
    filter_to_concepts: List[str],
    max_neg_rating: float,
    min_pos_rating: float,
    logger
):
    """Load and process rated data from various sources.

    Returns:
        Tuple of (topical_ratings, vanilla_ratings, ultrachat_ratings,
                  concept_to_topical_data, concept_to_vanilla_data, concept_to_ultrachat_data,
                  dual_topical_data, dual_vanilla_data, dual_ultrachat_data)
    """
    logger.print("Loading synthetic concept data...")

    # Load using the new data loading approach
    topical_ratings = load_rated_data(
        dataset_name=dataset_name,
        response_type="topical",
        manual_path_confirm=False,
        exclude_refusals=True,
        exclude_missing_ratings=True,
        exclude_concepts=exclude_concepts,
        filter_to_concepts=filter_to_concepts
    )

    # Try to load vanilla data, but handle gracefully if none available
    try:
        vanilla_ratings = load_rated_data(
            dataset_name=dataset_name,
            response_type="vanilla",
            manual_path_confirm=False,
            exclude_refusals=True,
            exclude_missing_ratings=True,
            exclude_concepts=exclude_concepts,
            filter_to_concepts=filter_to_concepts
        )
    except ValueError as e:
        if "No valid data found" in str(e):
            logger.print("No vanilla data found, continuing without vanilla samples...")
            vanilla_ratings = []
        else:
            raise

    if not skip_ultrachat_data:
        # Try to load ultrachat data, but handle gracefully if none available
        try:
            ultrachat_ratings = load_rated_data(
                dataset_name=ultrachat_dataset_name,
                response_type="vanilla",
                manual_path_confirm=False,
                exclude_refusals=True,
                exclude_missing_ratings=True,
                exclude_concepts=exclude_concepts,
                filter_to_concepts=filter_to_concepts
            )
        except ValueError as e:
            if "No valid data found" in str(e):
                logger.print("No ultrachat data found, continuing without ultrachat samples...")
                ultrachat_ratings = []
            else:
                raise
    else:
        logger.print("skip_ultrachat_data==True, so continuing without Ultrachat data!")
        ultrachat_ratings = []

    # Convert to the format expected by process_concept_training_data
    concept_to_topical_data = PosNegDataByCategory.from_ratings(
        topical_ratings,
        max_neg_rating=max_neg_rating,
        min_pos_rating=min_pos_rating
    )

    # Handle empty vanilla and ultrachat data
    if vanilla_ratings:
        concept_to_vanilla_data = PosNegDataByCategory.from_ratings(
            vanilla_ratings,
            max_neg_rating=max_neg_rating,
            min_pos_rating=min_pos_rating
        )
    else:
        concept_to_vanilla_data = PosNegDataByCategory(categories={})

    if ultrachat_ratings:
        concept_to_ultrachat_data = PosNegDataByCategory.from_ratings(
            ultrachat_ratings,
            max_neg_rating=max_neg_rating,
            min_pos_rating=min_pos_rating,
            ignore_generation_categories=True  # Ultrachat wasn't generated with target adjectives
        )
    else:
        concept_to_ultrachat_data = PosNegDataByCategory(categories={})

    # Create dual-category data
    logger.print("Creating dual-category data...")
    dual_topical_data = PosNegDataByDualCategory.from_ratings(
        topical_ratings,
        max_neg_rating=max_neg_rating,
        min_pos_rating=min_pos_rating
    ) if topical_ratings else PosNegDataByDualCategory(categories={})

    dual_vanilla_data = PosNegDataByDualCategory.from_ratings(
        vanilla_ratings,
        max_neg_rating=max_neg_rating,
        min_pos_rating=min_pos_rating
    ) if vanilla_ratings else PosNegDataByDualCategory(categories={})

    dual_ultrachat_data = PosNegDataByDualCategory.from_ratings(
        ultrachat_ratings,
        max_neg_rating=max_neg_rating,
        min_pos_rating=min_pos_rating,
        ignore_generation_categories=True
    ) if ultrachat_ratings else PosNegDataByDualCategory(categories={})

    # Log counts
    topical_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_topical_data.categories.items()}
    vanilla_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_vanilla_data.categories.items()}
    ultrachat_counts = {k: f"Pos {len(v.pos_dataset)} Neg {len(v.neg_dataset)}" for k, v in concept_to_ultrachat_data.categories.items()}
    logger.print(f"Raw counts per category for topical data: {topical_counts}")
    logger.print(f"Raw counts per category for vanilla data: {vanilla_counts}")
    logger.print(f"Raw counts per category for ultrachat data: {ultrachat_counts}")

    # Log dual-category counts
    dual_topical_counts = len(dual_topical_data.categories)
    dual_vanilla_counts = len(dual_vanilla_data.categories)
    dual_ultrachat_counts = len(dual_ultrachat_data.categories)
    logger.print(f"Dual-category pairs: topical={dual_topical_counts}, vanilla={dual_vanilla_counts}, ultrachat={dual_ultrachat_counts}")

    # Cull categories with too few samples
    min_samples = 10
    logger.print(f"Culling categories with fewer than {min_samples} samples...")
    concept_to_topical_data.categories = {k: v for k, v in concept_to_topical_data.categories.items() if len(v.pos_dataset) > min_samples and len(v.neg_dataset) > min_samples}
    concept_to_vanilla_data.categories = {k: v for k, v in concept_to_vanilla_data.categories.items() if len(v.pos_dataset) > min_samples and len(v.neg_dataset) > min_samples}
    concept_to_ultrachat_data.categories = {k: v for k, v in concept_to_ultrachat_data.categories.items() if len(v.pos_dataset) > min_samples and len(v.neg_dataset) > min_samples}

    return (topical_ratings, vanilla_ratings, ultrachat_ratings,
            concept_to_topical_data, concept_to_vanilla_data, concept_to_ultrachat_data,
            dual_topical_data, dual_vanilla_data, dual_ultrachat_data)


def process_concept_training_data(
    concept_to_topical_data: PosNegDataByCategory,
    concept_to_vanilla_data: PosNegDataByCategory,
    concept_to_ultrachat_data: PosNegDataByCategory,
    concept_to_probe: Dict[str, object],
    dual_topical_data: PosNegDataByDualCategory = None,
    dual_vanilla_data: PosNegDataByDualCategory = None,
    dual_ultrachat_data: PosNegDataByDualCategory = None,
    num_samples_per_epoch: int = 50,  # Total samples per epoch (distributed across all concepts)
    logger = None,
    use_synonyms: bool = True,
    use_antonyms: bool = False,
    data_split_distribution: Dict[str, float] = None,
    enable_topical_behavior_loss: bool = False
) -> Dict[str, List]:
    """
    Process training data for concept self-obfuscation using N/X/Y notation.

    Args:
        num_samples_per_epoch: Total number of samples to generate per epoch,
                               distributed across all concepts and data types

    Returns dict with keys organized by data source type.

    Each item is tuple: (probe, prompt, response, token_ids, label, use_behavior_loss)
    where probe and label can be tuples for dual-category data
    """

    # Helper functions to reduce redundancy
    def get_clued_prompt(concept, prompt):
        """Generate a clued prompt with optional synonym."""
        clue_text = get_random_synonym(concept) if use_synonyms else concept
        return f"You are being probed for '{clue_text}'. {prompt}"

    def log_failed_attempt(attempts, split_name, message):
        """Log failed attempts every 100 iterations."""
        if logger and attempts % 100 == 0:
            logger.print(f"  Failed attempt {attempts} for {split_name}: {message}")

    def get_other_concepts(concepts_list, exclude_concept):
        """Get list of concepts excluding the specified one."""
        return [c for c in concepts_list if c != exclude_concept]

    def sample_data_and_append(data_category, probe, prompt, response, label, use_beh_loss, split_samples, apply_clue=None):
        """Sample data from category and append to split_samples."""
        if data_category and data_category.pos_dataset:
            prompt_resp = random.choice(data_category.pos_dataset)
            if apply_clue:
                prompt = get_clued_prompt(apply_clue, prompt_resp.prompt)
            else:
                prompt = prompt_resp.prompt
            split_samples.append((probe, prompt, prompt_resp.response, None, label, use_beh_loss))
            return True
        return False

    def debug_dual_category_availability(dual_data, concepts, attempts, logger):
        """Debug logging for dual-category data availability."""
        if attempts == 1 and logger:  # Only print once
            logger.print(f"DEBUG: Dual-category pairs available: {list(dual_data.categories.keys())}")
            logger.print(f"DEBUG: Concepts with probes: {concepts}")

    if data_split_distribution is None:
        raise Exception("No data split provided")

    # Verify distribution sums to 1.0
    total_prob = sum(data_split_distribution.values())
    if abs(total_prob - 1.0) > 1e-6:
        if logger:
            logger.print(f"Warning: Data split distribution sums to {total_prob:.6f}, not 1.0")
        # Normalize to sum to 1.0
        for key in data_split_distribution:
            data_split_distribution[key] /= total_prob

    if logger:
        logger.print("Data split distribution:")
        for split_name, prob in data_split_distribution.items():
            logger.print(f"  {split_name}: {prob:.3f}")
        logger.print(f"Total: {sum(data_split_distribution.values()):.6f}")

    # Initialize data splits
    data_splits = {split_name: [] for split_name in data_split_distribution.keys()}

    concepts = list(concept_to_probe.keys())

    # Create intersection sets for concept selection to avoid KeyError
    concepts_with_topical = list(set(concept_to_probe.keys()) & set(concept_to_topical_data.categories.keys()))
    concepts_with_vanilla = list(set(concept_to_probe.keys()) & set(concept_to_vanilla_data.categories.keys()))

    # Check ultrachat availability
    concepts_with_ultrachat = []
    for concept in concepts:
        ultrachat_data = concept_to_ultrachat_data.categories.get(concept)
        if ultrachat_data and ultrachat_data.pos_dataset:
            concepts_with_ultrachat.append(concept)

    if logger:
        logger.print(f"Concepts with probes: {len(concepts)}")
        logger.print(f"Concepts with topical data: {len(concepts_with_topical)}")
        logger.print(f"Concepts with vanilla data: {len(concepts_with_vanilla)}")
        logger.print(f"Concepts with ultrachat data: {len(concepts_with_ultrachat)}")

        if len(concepts_with_topical) == 0:
            logger.print("WARNING: No concepts have both probes and topical data!")
        if len(concepts_with_vanilla) == 0:
            logger.print("WARNING: No concepts have both probes and vanilla data!")
        if len(concepts_with_ultrachat) == 0:
            logger.print("WARNING: No concepts have both probes and ultrachat data!")

    # Adjust data split distribution based on available data types
    has_vanilla = len(concepts_with_vanilla) > 0
    has_ultrachat = len(concepts_with_ultrachat) > 0

    # Check for dual-category vanilla availability (both concepts must have probes)
    has_dual_vanilla = False
    if dual_vanilla_data and dual_vanilla_data.categories:
        available_dual_vanilla = [pair for pair in dual_vanilla_data.categories.keys()
                                 if pair[0] in concepts and pair[1] in concepts]
        has_dual_vanilla = len(available_dual_vanilla) > 0
        if not has_dual_vanilla and logger:
            logger.print("WARNING: No dual-category vanilla pairs with both concepts having probes!")

    # Check for dual-category ultrachat availability (both concepts must have probes)
    has_dual_ultrachat = False
    if dual_ultrachat_data and dual_ultrachat_data.categories:
        available_dual_ultrachat = [pair for pair in dual_ultrachat_data.categories.keys()
                                   if pair[0] in concepts and pair[1] in concepts]
        has_dual_ultrachat = len(available_dual_ultrachat) > 0
        if not has_dual_ultrachat and logger:
            logger.print("WARNING: No dual-category ultrachat pairs with both concepts having probes!")

    if not has_vanilla or not has_ultrachat or not has_dual_vanilla or not has_dual_ultrachat:
        if logger:
            logger.print("Adjusting data split distribution due to missing data types...")

        # Get splits that require vanilla data
        vanilla_splits = [k for k in data_split_distribution.keys() if 'vanilla' in k]
        ultrachat_splits = [k for k in data_split_distribution.keys() if 'ultrachat' in k]
        dual_vanilla_splits = ['clue_X_data_XY_probe_XY_vanilla', 'clue_N_data_XY_probe_XY_vanilla']
        dual_ultrachat_splits = ['clue_X_data_XY_probe_XY_ultrachat', 'clue_N_data_XY_probe_XY_ultrachat']

        # Redistribute probabilities from unavailable splits to topical splits
        unavailable_prob = 0.0
        if not has_vanilla:
            for split in vanilla_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0
        elif not has_dual_vanilla:
            # Only zero out the dual-category vanilla splits
            for split in dual_vanilla_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0

        if not has_ultrachat:
            for split in ultrachat_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0
        elif not has_dual_ultrachat:
            # Only zero out the dual-category ultrachat splits
            for split in dual_ultrachat_splits:
                unavailable_prob += data_split_distribution[split]
                data_split_distribution[split] = 0.0

        # Redistribute to topical splits proportionally
        topical_splits = [k for k in data_split_distribution.keys() if 'topical' in k and data_split_distribution[k] > 0]
        if topical_splits and unavailable_prob > 0:
            total_topical_prob = sum(data_split_distribution[k] for k in topical_splits)
            for split in topical_splits:
                if total_topical_prob > 0:
                    data_split_distribution[split] += unavailable_prob * (data_split_distribution[split] / total_topical_prob)

        # Renormalize to ensure sum is 1.0
        total_prob = sum(data_split_distribution.values())
        if total_prob > 0:
            for key in data_split_distribution:
                data_split_distribution[key] /= total_prob

        if logger:
            logger.print("Adjusted data split distribution:")
            for split_name, prob in data_split_distribution.items():
                logger.print(f"  {split_name}: {prob:.3f}")
            logger.print(f"Total: {sum(data_split_distribution.values()):.6f}")

    # Calculate target sample sizes based on (possibly adjusted) distribution
    total_samples = num_samples_per_epoch
    target_samples = {
        split_name: int(total_samples * prob)
        for split_name, prob in data_split_distribution.items()
    }

    if logger:
        logger.print(f"\nTarget sample sizes (total: {total_samples}):")
        for split_name, count in target_samples.items():
            logger.print(f"  {split_name}: {count}")
        logger.print(f"Total target samples: {sum(target_samples.values())}")

    # Directly generate target samples for each split type using efficient sampling
    if logger:
        logger.print("Generating samples using efficient sampling approach...")

    # Get all available ultrachat data for types 8 and 11
    all_ultrachat_data = []
    for concept in concepts:
        ultrachat_data = concept_to_ultrachat_data.categories.get(concept)
        if ultrachat_data and ultrachat_data.pos_dataset:
            all_ultrachat_data.extend(ultrachat_data.pos_dataset)

    if logger:
        logger.print(f"Total concepts available: {len(concepts)}")
        logger.print(f"Total ultrachat samples across all concepts: {len(all_ultrachat_data)}")

        # Check data availability per concept
        for concept in concepts[:5]:  # Show first 5 concepts for debugging
            topical_data = concept_to_topical_data.categories.get(concept)
            vanilla_data = concept_to_vanilla_data.categories.get(concept)
            ultrachat_data = concept_to_ultrachat_data.categories.get(concept)

            topical_count = len(topical_data.pos_dataset) if topical_data and topical_data.pos_dataset else 0
            vanilla_count = len(vanilla_data.pos_dataset) if vanilla_data and vanilla_data.pos_dataset else 0
            ultrachat_count = len(ultrachat_data.pos_dataset) if ultrachat_data and ultrachat_data.pos_dataset else 0

            logger.print(f"  {concept}: topical={topical_count}, vanilla={vanilla_count}, ultrachat={ultrachat_count}")

        if len(concepts) > 5:
            logger.print(f"  ... and {len(concepts) - 5} more concepts")



    # For each split type, generate exactly the target number of samples
    for split_name, target_count in target_samples.items():
        if target_count == 0:
            continue

        if logger:
            logger.print(f"Generating {target_count} samples for {split_name}")

        split_samples = []

        # Use while loop to handle empty dataset cases
        attempts = 0
        max_attempts = target_count * 10  # Prevent infinite loops

        while len(split_samples) < target_count and attempts < max_attempts:
            attempts += 1
            if split_name == 'clue_N_data_X_probe_X_topical':
                # 1. No clue, X topical data, X probe → target score 1
                if not concepts_with_topical:
                    log_failed_attempt(attempts, split_name, "no concepts with topical data available")
                    continue
                concept_x = random.choice(concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_x = concept_to_topical_data.categories.get(concept_x)
                if topical_data_x and topical_data_x.pos_dataset:
                    prompt_resp = random.choice(topical_data_x.pos_dataset)
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, enable_topical_behavior_loss))
                else:
                    log_failed_attempt(attempts, split_name, f"concept {concept_x} has no topical data")

            elif split_name == 'clue_X_data_Y_probe_Y_topical':
                # 2. X clue, Y topical data, Y probe → target score 1
                concept_x = random.choice(concepts)
                other_concepts_with_topical = get_other_concepts(concepts_with_topical, concept_x)
                if not other_concepts_with_topical:
                    log_failed_attempt(attempts, split_name, "no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_y = concept_to_probe[concept_y]
                topical_data_y = concept_to_topical_data.categories.get(concept_y)
                if topical_data_y and topical_data_y.pos_dataset:
                    prompt_resp = random.choice(topical_data_y.pos_dataset)

                    # If use_antonyms is enabled, 50% chance to use antonym of Y instead of X as the clue
                    if use_antonyms and random.random() < 0.5:
                        # Use antonym of concept_y as clue
                        antonym = get_random_antonym(concept_y)
                        if antonym is None:
                            raise ValueError(f"No antonym defined for concept '{concept_y}' - cannot generate antonym clue for clue_X_data_Y_probe_Y_topical")
                        clued_prompt = f"You are being probed for '{antonym}'. {prompt_resp.prompt}"
                    else:
                        # Use concept_x as clue (original behavior)
                        clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)

                    split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 1, enable_topical_behavior_loss))

            elif split_name == 'clue_X_data_Y_probe_Y_ultrachat':
                # X clue, Y ultrachat data, Y probe → target score 1 (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_ultrachat = get_other_concepts(concepts_with_ultrachat, concept_x)
                if not other_concepts_with_ultrachat:
                    log_failed_attempt(attempts, split_name, "no other concepts with ultrachat data available")
                    continue
                concept_y = random.choice(other_concepts_with_ultrachat)
                probe_y = concept_to_probe[concept_y]
                ultrachat_data_y = concept_to_ultrachat_data.categories.get(concept_y)
                if ultrachat_data_y and ultrachat_data_y.pos_dataset:
                    prompt_resp = random.choice(ultrachat_data_y.pos_dataset)

                    # If use_antonyms is enabled, 50% chance to use antonym of Y instead of X as the clue
                    if use_antonyms and random.random() < 0.5:
                        # Use antonym of concept_y as clue
                        antonym = get_random_antonym(concept_y)
                        if antonym is None:
                            raise ValueError(f"No antonym defined for concept '{concept_y}' - cannot generate antonym clue for clue_X_data_Y_probe_Y_ultrachat")
                        clued_prompt = f"You are being probed for '{antonym}'. {prompt_resp.prompt}"
                    else:
                        # Use concept_x as clue (original behavior)
                        clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)

                    split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 1, True))

            elif split_name == 'clue_N_data_X_probe_X_ultrachat':
                # 3. No clue, X ultrachat data, X probe → target score 1 (+ beh loss)
                if not concepts_with_ultrachat:
                    log_failed_attempt(attempts, split_name, "no concepts with ultrachat data available")
                    continue
                concept_x = random.choice(concepts_with_ultrachat)
                probe_x = concept_to_probe[concept_x]
                ultrachat_data_x = concept_to_ultrachat_data.categories[concept_x]
                prompt_resp = random.choice(ultrachat_data_x.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, True))

            elif split_name == 'clue_X_data_X_probe_X_topical':
                # 4. X clue, X topical data, X probe → target score 0
                if not concepts_with_topical:
                    log_failed_attempt(attempts, split_name, "no concepts with topical data available")
                    continue
                concept_x = random.choice(concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_x = concept_to_topical_data.categories[concept_x]
                prompt_resp = random.choice(topical_data_x.pos_dataset)
                clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, enable_topical_behavior_loss))

            elif split_name == 'clue_X_data_X_probe_X_vanilla':
                # 4. X clue, X vanilla data, X probe → target score 0 (+ beh loss)
                if not concepts_with_vanilla:
                    log_failed_attempt(attempts, split_name, "no concepts with vanilla data available")
                    continue
                concept_x = random.choice(concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_x = concept_to_vanilla_data.categories[concept_x]
                prompt_resp = random.choice(vanilla_data_x.pos_dataset)
                clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))

            elif split_name == 'clue_X_data_X_probe_X_ultrachat':
                # 6. X clue, X ultrachat data, X probe → target score 0 (+ beh loss)
                if not concepts_with_ultrachat:
                    log_failed_attempt(attempts, split_name, "no concepts with ultrachat data available")
                    continue
                concept_x = random.choice(concepts_with_ultrachat)
                probe_x = concept_to_probe[concept_x]
                ultrachat_data_x = concept_to_ultrachat_data.categories[concept_x]
                prompt_resp = random.choice(ultrachat_data_x.pos_dataset)
                clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))

            elif split_name == 'clue_X_data_X_probe_Y_vanilla':
                # 7. X clue, X vanilla data, Y probe → target score 0 (+ beh loss)
                if not concepts_with_vanilla:
                    log_failed_attempt(attempts, split_name, "no concepts with vanilla data available")
                    continue
                concept_x = random.choice(concepts_with_vanilla)
                other_concepts = get_other_concepts(concepts, concept_x)
                if not other_concepts:
                    log_failed_attempt(attempts, split_name, "no other concepts available")
                    continue
                concept_y = random.choice(other_concepts)
                probe_y = concept_to_probe[concept_y]
                vanilla_data_x = concept_to_vanilla_data.categories[concept_x]
                prompt_resp = random.choice(vanilla_data_x.pos_dataset)
                clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)
                split_samples.append((probe_y, clued_prompt, prompt_resp.response, None, 0, True))

            elif split_name == 'clue_N_data_Y_probe_X_topical':
                # 8. No clue, Y topical data, X probe → target score 0
                concept_x = random.choice(concepts)
                other_concepts_with_topical = get_other_concepts(concepts_with_topical, concept_x)
                if not other_concepts_with_topical:
                    log_failed_attempt(attempts, split_name, "no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_y = concept_to_topical_data.categories[concept_y]
                prompt_resp = random.choice(topical_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, enable_topical_behavior_loss))

            elif split_name == 'clue_N_data_Y_probe_X_vanilla':
                # 9. No clue, Y vanilla data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_vanilla = get_other_concepts(concepts_with_vanilla, concept_x)
                if not other_concepts_with_vanilla:
                    log_failed_attempt(attempts, split_name, "no other concepts with vanilla data available")
                    continue
                concept_y = random.choice(other_concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_y = concept_to_vanilla_data.categories[concept_y]
                prompt_resp = random.choice(vanilla_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, True))

            elif split_name == 'clue_N_data_Y_probe_X_ultrachat':
                # 10. No clue, ultrachat data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 0, True))

            elif split_name == 'clue_X_data_Y_probe_X_topical':
                # 11. X clue, Y topical data, X probe → target score 0
                concept_x = random.choice(concepts)
                other_concepts_with_topical = get_other_concepts(concepts_with_topical, concept_x)
                if not other_concepts_with_topical:
                    log_failed_attempt(attempts, split_name, "no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_y = concept_to_topical_data.categories[concept_y]
                prompt_resp = random.choice(topical_data_y.pos_dataset)
                clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, enable_topical_behavior_loss))

            elif split_name == 'clue_X_data_Y_probe_X_vanilla':
                # 12. X clue, Y vanilla data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_vanilla = get_other_concepts(concepts_with_vanilla, concept_x)
                if not other_concepts_with_vanilla:
                    log_failed_attempt(attempts, split_name, "no other concepts with vanilla data available")
                    continue
                concept_y = random.choice(other_concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_y = concept_to_vanilla_data.categories[concept_y]
                prompt_resp = random.choice(vanilla_data_y.pos_dataset)
                clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)
                split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))

            elif split_name == 'clue_X_data_Y_probe_X_ultrachat':
                # 13. X clue, ultrachat data, X probe → target score 0 (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    clued_prompt = get_clued_prompt(concept_x, prompt_resp.prompt)
                    split_samples.append((probe_x, clued_prompt, prompt_resp.response, None, 0, True))

            # Dual-category data types
            elif split_name == 'clue_X_data_XY_probe_XY_topical':
                # X clue, XY topical data, XY probes → target (0, 1) for (X, Y) or (1, 0) for (Y, X)
                if dual_topical_data and dual_topical_data.categories:
                    # Get a random dual category pair
                    available_pairs = [pair for pair in dual_topical_data.categories.keys()
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair  # Already alphabetically sorted

                        # 50% chance to use X as clue, 50% chance to use Y as clue
                        if random.random() < 0.5:
                            clue_concept = concept_x
                            labels = (0, 1)  # X gets 0 (clued), Y gets 1 (not clued)
                        else:
                            clue_concept = concept_y
                            labels = (1, 0)  # X gets 1 (not clued), Y gets 0 (clued)

                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_topical_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            clued_prompt = get_clued_prompt(clue_concept, prompt_resp.prompt)
                            split_samples.append(((probe_x, probe_y), clued_prompt, prompt_resp.response, None, labels, enable_topical_behavior_loss))

            elif split_name == 'clue_N_data_XY_probe_XY_topical':
                # No clue, XY topical data, XY probes → target (1, 1)
                if dual_topical_data and dual_topical_data.categories:
                    available_pairs = [pair for pair in dual_topical_data.categories.keys()
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_topical_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            split_samples.append(((probe_x, probe_y), prompt_resp.prompt, prompt_resp.response, None, (1, 1), enable_topical_behavior_loss))

            elif split_name == 'clue_X_data_XY_probe_XY_vanilla':
                # X clue, XY vanilla data, XY probes → target (0, 1) for (X, Y) or (1, 0) for (Y, X) (+ beh loss)
                if dual_vanilla_data and dual_vanilla_data.categories:
                    available_pairs = [pair for pair in dual_vanilla_data.categories.keys()
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair

                        # 50% chance to use X as clue, 50% chance to use Y as clue
                        if random.random() < 0.5:
                            clue_concept = concept_x
                            labels = (0, 1)
                        else:
                            clue_concept = concept_y
                            labels = (1, 0)

                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_vanilla_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            clued_prompt = get_clued_prompt(clue_concept, prompt_resp.prompt)
                            split_samples.append(((probe_x, probe_y), clued_prompt, prompt_resp.response, None, labels, True))

            elif split_name == 'clue_N_data_XY_probe_XY_vanilla':
                # No clue, XY vanilla data, XY probes → target (1, 1) (+ beh loss)
                if dual_vanilla_data and dual_vanilla_data.categories:
                    available_pairs = [pair for pair in dual_vanilla_data.categories.keys()
                                     if pair[0] in concepts and pair[1] in concepts]
                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_vanilla_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            split_samples.append(((probe_x, probe_y), prompt_resp.prompt, prompt_resp.response, None, (1, 1), True))

            elif split_name == 'clue_X_data_XY_probe_XY_ultrachat':
                # X clue, XY ultrachat data, XY probes → target (0, 1) for (X, Y) or (1, 0) for (Y, X) (+ beh loss)
                if dual_ultrachat_data and dual_ultrachat_data.categories:
                    debug_dual_category_availability(dual_ultrachat_data, concepts, attempts, logger)

                    available_pairs = [pair for pair in dual_ultrachat_data.categories.keys()
                                     if pair[0] in concepts and pair[1] in concepts]

                    if attempts == 1 and logger:  # Only print once
                        logger.print(f"DEBUG: Available pairs after filtering (both concepts have probes): {available_pairs}")

                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair

                        # 50% chance to use X as clue, 50% chance to use Y as clue
                        if random.random() < 0.5:
                            clue_concept = concept_x
                            labels = (0, 1)
                        else:
                            clue_concept = concept_y
                            labels = (1, 0)

                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_ultrachat_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            clued_prompt = get_clued_prompt(clue_concept, prompt_resp.prompt)
                            split_samples.append(((probe_x, probe_y), clued_prompt, prompt_resp.response, None, labels, True))

            elif split_name == 'clue_N_data_XY_probe_XY_ultrachat':
                # No clue, XY ultrachat data, XY probes → target (1, 1) (+ beh loss)
                if dual_ultrachat_data and dual_ultrachat_data.categories:
                    debug_dual_category_availability(dual_ultrachat_data, concepts, attempts, logger)

                    available_pairs = [pair for pair in dual_ultrachat_data.categories.keys()
                                     if pair[0] in concepts and pair[1] in concepts]

                    if attempts == 1 and logger:  # Only print once
                        logger.print(f"DEBUG: Available pairs after filtering (both concepts have probes): {available_pairs}")

                    if available_pairs:
                        category_pair = random.choice(available_pairs)
                        concept_x, concept_y = category_pair
                        probe_x = concept_to_probe[concept_x]
                        probe_y = concept_to_probe[concept_y]
                        dual_data = dual_ultrachat_data.categories[category_pair]
                        if dual_data.pos_dataset:
                            prompt_resp = random.choice(dual_data.pos_dataset)
                            split_samples.append(((probe_x, probe_y), prompt_resp.prompt, prompt_resp.response, None, (1, 1), True))

            # Reversed data types with flipped labels
            elif split_name == 'clue_N_data_Y_probe_X_topical_rev':
                # No clue, Y topical data, X probe → target score 1 (reversed)
                concept_x = random.choice(concepts)
                other_concepts_with_topical = get_other_concepts(concepts_with_topical, concept_x)
                if not other_concepts_with_topical:
                    log_failed_attempt(attempts, split_name, "no other concepts with topical data available")
                    continue
                concept_y = random.choice(other_concepts_with_topical)
                probe_x = concept_to_probe[concept_x]
                topical_data_y = concept_to_topical_data.categories[concept_y]
                prompt_resp = random.choice(topical_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, enable_topical_behavior_loss))  # Label flipped to 1

            elif split_name == 'clue_N_data_Y_probe_X_vanilla_rev':
                # No clue, Y vanilla data, X probe → target score 1 (reversed) (+ beh loss)
                concept_x = random.choice(concepts)
                other_concepts_with_vanilla = get_other_concepts(concepts_with_vanilla, concept_x)
                if not other_concepts_with_vanilla:
                    log_failed_attempt(attempts, split_name, "no other concepts with vanilla data available")
                    continue
                concept_y = random.choice(other_concepts_with_vanilla)
                probe_x = concept_to_probe[concept_x]
                vanilla_data_y = concept_to_vanilla_data.categories[concept_y]
                prompt_resp = random.choice(vanilla_data_y.pos_dataset)
                split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, True))  # Label flipped to 1

            elif split_name == 'clue_N_data_Y_probe_X_ultrachat_rev':
                # No clue, ultrachat data, X probe → target score 1 (reversed) (+ beh loss)
                concept_x = random.choice(concepts)
                probe_x = concept_to_probe[concept_x]
                if all_ultrachat_data:
                    prompt_resp = random.choice(all_ultrachat_data)
                    split_samples.append((probe_x, prompt_resp.prompt, prompt_resp.response, None, 1, True))  # Label flipped to 1

        data_splits[split_name] = split_samples

        if logger:
            if len(split_samples) < target_count:
                logger.print(f"WARNING: Only generated {len(split_samples)}/{target_count} samples for {split_name} after {attempts} attempts")
            else:
                logger.print(f"Successfully generated {len(split_samples)} samples for {split_name}")

    # Shuffle all data splits
    for key in data_splits:
        random.Random(42).shuffle(data_splits[key])

    # Show final distribution results
    if logger:
        logger.print(f"\nFinal training data distribution:")
        for key, data in data_splits.items():
            actual_count = len(data)
            target_count = target_samples[key]
            percentage = (actual_count / sum(len(v) for v in data_splits.values())) * 100
            logger.print(f"  {key}: {actual_count} samples (target: {target_count}, {percentage:.1f}%)")

        # Count behavior loss samples
        beh_loss_count = sum(1 for items in data_splits.values() for item in items if item[5])
        total_samples_actual = sum(len(v) for v in data_splits.values())
        logger.print(f"\nTotal samples with behavior loss: {beh_loss_count}")

        # Count by source type
        topical_count = sum(len(v) for k, v in data_splits.items() if 'topical' in k)
        vanilla_count = sum(len(v) for k, v in data_splits.items() if 'vanilla' in k)
        ultrachat_count = sum(len(v) for k, v in data_splits.items() if 'ultrachat' in k)
        logger.print(f"\nBy source type:")
        logger.print(f"  Topical: {topical_count} samples ({topical_count/total_samples_actual*100:.1f}%)")
        logger.print(f"  Vanilla: {vanilla_count} samples ({vanilla_count/total_samples_actual*100:.1f}%, all with behavior loss)")
        logger.print(f"  Ultrachat: {ultrachat_count} samples ({ultrachat_count/total_samples_actual*100:.1f}%, all with behavior loss)")

        # Verify final label balance
        final_label_1 = (len(data_splits['clue_N_data_X_probe_X_topical']) +
                        len(data_splits['clue_X_data_Y_probe_Y_topical']) +
                        len(data_splits.get('clue_X_data_Y_probe_Y_ultrachat', [])) +
                        len(data_splits.get('clue_N_data_X_probe_X_ultrachat', [])))
        final_label_0 = (len(data_splits['clue_X_data_X_probe_X_topical']) +
                        len(data_splits['clue_X_data_X_probe_X_vanilla']) +
                        len(data_splits.get('clue_X_data_X_probe_X_ultrachat', [])) +
                        len(data_splits['clue_X_data_X_probe_Y_vanilla']) +
                        len(data_splits['clue_N_data_Y_probe_X_topical']) +
                        len(data_splits['clue_N_data_Y_probe_X_vanilla']) +
                        len(data_splits['clue_N_data_Y_probe_X_ultrachat']) +
                        len(data_splits['clue_X_data_Y_probe_X_topical']) +
                        len(data_splits['clue_X_data_Y_probe_X_vanilla']) +
                        len(data_splits['clue_X_data_Y_probe_X_ultrachat']))
        logger.print(f"\nFinal label distribution:")
        logger.print(f"  Label=1 (maximize): {final_label_1} samples ({final_label_1/total_samples_actual*100:.1f}%)")
        logger.print(f"  Label=0 (minimize): {final_label_0} samples ({final_label_0/total_samples_actual*100:.1f}%)")
        logger.print(f"  Balance ratio: {final_label_1}/{final_label_0} = {final_label_1/max(final_label_0,1):.2f}")

    return data_splits
