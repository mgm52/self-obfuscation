from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import random
from pprint import pprint


### SAMPLES ###
class PromptResp:
    def __init__(self, prompt: str, response: str):
        self.prompt = prompt
        self.response = response
    def __str__(self):
        return f"Prompt: {self.prompt}\nResponse: {self.response}"
    def __repr__(self):
        return self.__str__()

class PromptRespRating:
    def __init__(self, prompt: str, response: str, ratings: Dict[str, float], adjectives: List[str] = None):
        self.prompt = prompt
        self.response = response
        self.ratings = ratings
        self.adjectives = adjectives or []
    def __str__(self):
        return f"Prompt: {self.prompt}\nResponse: {self.response}\nRatings: {self.ratings}\nAdjectives: {self.adjectives}"
    def __repr__(self):
        return self.__str__()

### DATASETS ###
class PosNegData:
    def __init__(self, pos_dataset: List[PromptResp], neg_dataset: List[PromptResp], shuffle: bool = False):
        self.pos_dataset = pos_dataset
        self.neg_dataset = neg_dataset
        self.__balance_to_same_len(shuffle=shuffle)

    def __balance_to_same_len(self, shuffle: bool):
        min_len = min(len(self.pos_dataset), len(self.neg_dataset))
        if shuffle:
            random.Random(42).shuffle(self.pos_dataset)
            random.Random(42).shuffle(self.neg_dataset)
        self.pos_dataset = self.pos_dataset[:min_len]
        self.neg_dataset = self.neg_dataset[:min_len]

class PosNegDataByCategory:
    def __init__(self, categories: Dict[str, PosNegData]):
        self.categories = categories

    @classmethod
    def from_ratings(cls, ratings_list: List[PromptRespRating], max_neg_rating: float = 0.25, min_pos_rating: float = 0.75, shuffle: bool = False, ignore_generation_categories: bool = False):
        categories = defaultdict(lambda: PosNegData(pos_dataset=[], neg_dataset=[], shuffle=False))

        for rating_obj in ratings_list:
            prompt_response = PromptResp(rating_obj.prompt, rating_obj.response)
            generation_categories = rating_obj.adjectives

            for rating_category, rating_value in rating_obj.ratings.items():
                if rating_value is not None:  # Skip None ratings
                    if ignore_generation_categories:
                        # For data like ultrachat that wasn't generated with target adjectives
                        # Just use rating thresholds
                        if rating_value <= max_neg_rating:
                            categories[rating_category].neg_dataset.append(prompt_response)
                        elif rating_value >= min_pos_rating:
                            categories[rating_category].pos_dataset.append(prompt_response)
                    else:
                        # Sample must have rating threshold AND rating_category must be in adjectives list
                        if (rating_category not in generation_categories) and (rating_value <= max_neg_rating):
                            categories[rating_category].neg_dataset.append(prompt_response)
                        elif (rating_category in generation_categories) and (rating_value >= min_pos_rating):
                            categories[rating_category].pos_dataset.append(prompt_response)

        for category in categories:
            categories[category]._PosNegData__balance_to_same_len(shuffle=shuffle)

        return cls(categories)

class PosNegDataByDualCategory:
    def __init__(self, categories: Dict[Tuple[str, str], PosNegData]):
        self.categories = categories

    @classmethod
    def from_ratings(cls,
                     ratings_list: List[PromptRespRating],
                     min_pos_rating: float = 0.75,
                     max_neg_rating: float = 0.25,
                     shuffle: bool = False,
                     ignore_generation_categories: bool = False):
        """
        Create PosNegData for dual category combinations.

        If ignore_generation_categories=False (for synthetic data):
        - For each pair of categories (X, Y) where both have ratings >= min_pos_rating
          and at least one is in generation_categories:
          Add to positive dataset for tuple (X, Y) sorted alphabetically
        - For each pair of categories (X, Y) where both have ratings <= max_neg_rating
          and neither is in generation_categories:
          Add to negative dataset for tuple (X, Y) sorted alphabetically

        If ignore_generation_categories=True (for ultrachat data):
        - For each pair of categories (X, Y) where both have ratings >= min_pos_rating:
          Add to positive dataset for tuple (X, Y) sorted alphabetically
        - For each pair of categories (X, Y) where both have ratings <= max_neg_rating:
          Add to negative dataset for tuple (X, Y) sorted alphabetically

        Args:
            ratings_list: List of PromptRespRating objects
            min_pos_rating: Minimum rating for both categories to be considered positive
            max_neg_rating: Maximum rating for both categories to be considered negative
            shuffle: Whether to shuffle and balance the datasets

        Returns:
            PosNegDataByDualCategory with category pairs as keys
        """
        categories = defaultdict(lambda: PosNegData(pos_dataset=[], neg_dataset=[], shuffle=False))

        for rating_obj in ratings_list:
            prompt_response = PromptResp(rating_obj.prompt, rating_obj.response)
            generation_categories = set(rating_obj.adjectives)

            # Get all rated categories from this sample
            rated_categories = [(cat, val) for cat, val in rating_obj.ratings.items()
                               if val is not None]

            # Process all pairs of categories
            for i, (cat_x, rating_x) in enumerate(rated_categories):
                for cat_y, rating_y in rated_categories[i+1:]:  # Avoid duplicate pairs
                    # Create alphabetically sorted tuple as key
                    category_pair = tuple(sorted([cat_x, cat_y]))

                    if ignore_generation_categories:
                        # For ultrachat data - just use rating thresholds
                        # Positive: both ratings >= min_pos_rating
                        if rating_x >= min_pos_rating and rating_y >= min_pos_rating:
                            categories[category_pair].pos_dataset.append(prompt_response)
                        # Negative: both ratings <= max_neg_rating
                        elif rating_x <= max_neg_rating and rating_y <= max_neg_rating:
                            categories[category_pair].neg_dataset.append(prompt_response)
                    else:
                        # For synthetic data - consider generation categories
                        # Check category membership in generation_categories
                        x_in_generation = cat_x in generation_categories
                        y_in_generation = cat_y in generation_categories

                        # Positive: both ratings >= min_pos_rating AND at least one in generation
                        if (rating_x >= min_pos_rating and rating_y >= min_pos_rating and
                            (x_in_generation or y_in_generation)):
                            categories[category_pair].pos_dataset.append(prompt_response)

                        # Negative: both ratings <= max_neg_rating AND neither in generation
                        elif (rating_x <= max_neg_rating and rating_y <= max_neg_rating and
                              not x_in_generation and not y_in_generation):
                            categories[category_pair].neg_dataset.append(prompt_response)

        # Balance datasets for each category pair
        for category_pair in categories:
            categories[category_pair]._PosNegData__balance_to_same_len(shuffle=shuffle)

        return cls(dict(categories))

def create_merged_pos_neg_data(
    ratings_list: List[PromptRespRating],
    merge_concepts: List[str],
    positive_threshold: float = 0.75,
    negative_threshold: float = 0.0,
    shuffle: bool = False,
    return_stats: bool = False,
    ignore_generation_categories: bool = False,
    negative_concepts: List[str] = None,
    limit: Optional[int] = None
) -> Tuple[PosNegData, Optional[Dict]]:
    """
    Create PosNegData for merged concepts.

    If ignore_generation_categories=False (default for synthetic data):
    - Positive samples: Generated WITH at least one merged concept AND at least one merged concept has rating >= positive_threshold
    - Negative samples:
        - If negative_concepts is provided: Generated WITH at least one negative concept AND at least one negative concept has rating >= positive_threshold
        - Otherwise: Generated WITHOUT any merged concepts AND ALL merged concepts have rating <= negative_threshold

    If ignore_generation_categories=True (for ultrachat-like data):
    - Positive samples: At least one merged concept has rating >= positive_threshold
    - Negative samples:
        - If negative_concepts is provided: At least one negative concept has rating >= positive_threshold
        - Otherwise: ALL merged concepts have rating <= negative_threshold

    Args:
        ratings_list: List of PromptRespRating objects
        merge_concepts: List of concept names to merge (positive concepts)
        positive_threshold: Minimum rating for a sample to be considered positive
        negative_threshold: Maximum rating for a sample to be considered negative
        shuffle: Whether to shuffle the resulting datasets
        verbose: Whether to print debug information
        return_stats: Whether to return detailed statistics as a second return value
        ignore_generation_categories: If True, don't consider generation adjectives (like ultrachat data)
        negative_concepts: Optional list of concepts to use as negative class (overrides default behavior)
        limit: Optional limit to apply to each class after filtering (shuffles before limiting)

    Returns:
        If return_stats=False: PosNegData object with positive and negative samples for the merged probe
        If return_stats=True: Tuple of (PosNegData, stats_dict) where stats_dict contains detailed statistics
    """
    pos_samples = []
    neg_samples = []
    skipped_missing = 0
    skipped_intermediate = 0

    # Detailed statistics tracking
    all_tracked_concepts = list(merge_concepts)
    if negative_concepts:
        all_tracked_concepts.extend([c for c in negative_concepts if c not in all_tracked_concepts])

    stats = {
        "total_samples_processed": len(ratings_list),
        "positive_samples": 0,
        "negative_samples": 0,
        "skipped_missing_ratings": 0,
        "skipped_intermediate_ratings": 0,
        "rating_distributions": {concept: {"min": 1.0, "max": 0.0, "sum": 0.0, "count": 0} for concept in all_tracked_concepts},
        "positive_sample_adjectives": {},  # Track which adjectives appear in positive samples
        "negative_sample_adjectives": {},  # Track which adjectives appear in negative samples
        "samples_by_max_rating_bucket": {},  # Distribution of max ratings across samples
    }

    # Helper function to update rating distributions only for samples that pass threshold filtering
    def update_rating_stats(ratings_to_update, concepts_to_track):
        for concept in concepts_to_track:
            if concept in ratings_to_update and ratings_to_update[concept] is not None:
                rating = ratings_to_update[concept]
                if concept in stats["rating_distributions"]:
                    stats["rating_distributions"][concept]["min"] = min(stats["rating_distributions"][concept]["min"], rating)
                    stats["rating_distributions"][concept]["max"] = max(stats["rating_distributions"][concept]["max"], rating)
                    stats["rating_distributions"][concept]["sum"] += rating
                    stats["rating_distributions"][concept]["count"] += 1

    if negative_concepts:
        print(f"Using explicit negative concepts: {negative_concepts}")

    for rating_obj in ratings_list:
        prompt_response = PromptResp(rating_obj.prompt, rating_obj.response)

        # Check ratings for all merged (positive) concepts
        max_rating = -1.0
        min_rating = 1.1
        all_concepts_rated = True
        concept_ratings = {}

        for concept in merge_concepts:
            if concept in rating_obj.ratings and rating_obj.ratings[concept] is not None:
                rating = rating_obj.ratings[concept]
                concept_ratings[concept] = rating
                max_rating = max(max_rating, rating)
                min_rating = min(min_rating, rating)
            else:
                # Skip samples where any merged concept lacks a rating
                all_concepts_rated = False
                skipped_missing += 1
                stats["skipped_missing_ratings"] += 1
                break

        # If negative_concepts is specified, also check ratings for those
        max_neg_rating = -1.0
        min_neg_rating = 1.1
        all_neg_concepts_rated = True
        neg_concept_ratings = {}

        if negative_concepts and all_concepts_rated:
            for concept in negative_concepts:
                if concept in rating_obj.ratings and rating_obj.ratings[concept] is not None:
                    rating = rating_obj.ratings[concept]
                    neg_concept_ratings[concept] = rating
                    max_neg_rating = max(max_neg_rating, rating)
                    min_neg_rating = min(min_neg_rating, rating)
                else:
                    # Skip samples where any negative concept lacks a rating
                    all_neg_concepts_rated = False
                    skipped_missing += 1
                    stats["skipped_missing_ratings"] += 1
                    break

        if all_concepts_rated and (not negative_concepts or all_neg_concepts_rated):
            # Track max rating bucket
            bucket = f"{int(max_rating * 10) / 10:.1f}"
            stats["samples_by_max_rating_bucket"][bucket] = stats["samples_by_max_rating_bucket"].get(bucket, 0) + 1

            # Check if any merged concept was in the generation categories
            has_merged_concept_in_generation = any(concept in rating_obj.adjectives for concept in merge_concepts)
            has_negative_concept_in_generation = any(concept in rating_obj.adjectives for concept in negative_concepts) if negative_concepts else False

            if negative_concepts:
                # Using explicit negative concepts
                if ignore_generation_categories:
                    # Ignore generation categories - just use rating thresholds
                    # Positive: at least one positive concept >= positive_threshold
                    if max_rating >= positive_threshold:
                        pos_samples.append(prompt_response)
                        stats["positive_samples"] += 1

                        # Update rating distributions for positive sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)
                        if negative_concepts:
                            update_rating_stats(rating_obj.ratings, negative_concepts)

                        # Track adjectives in positive samples
                        for adj in rating_obj.adjectives:
                            stats["positive_sample_adjectives"][adj] = stats["positive_sample_adjectives"].get(adj, 0) + 1

                    # Negative: at least one negative concept >= positive_threshold
                    elif max_neg_rating >= positive_threshold:
                        neg_samples.append(prompt_response)
                        stats["negative_samples"] += 1

                        # Update rating distributions for negative sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)
                        if negative_concepts:
                            update_rating_stats(rating_obj.ratings, negative_concepts)

                        # Track adjectives in negative samples
                        for adj in rating_obj.adjectives:
                            stats["negative_sample_adjectives"][adj] = stats["negative_sample_adjectives"].get(adj, 0) + 1

                    else:
                        # Doesn't meet criteria for either class
                        skipped_intermediate += 1
                        stats["skipped_intermediate_ratings"] += 1
                else:
                    # Consider generation categories (synthetic data)
                    # Positive: was generated WITH at least one positive concept AND has high rating
                    if has_merged_concept_in_generation and max_rating >= positive_threshold:
                        pos_samples.append(prompt_response)
                        stats["positive_samples"] += 1

                        # Update rating distributions for positive sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)
                        if negative_concepts:
                            update_rating_stats(rating_obj.ratings, negative_concepts)

                        # Track adjectives in positive samples
                        for adj in rating_obj.adjectives:
                            stats["positive_sample_adjectives"][adj] = stats["positive_sample_adjectives"].get(adj, 0) + 1

                    # Negative: was generated WITH at least one negative concept AND has high rating
                    elif has_negative_concept_in_generation and max_neg_rating >= positive_threshold:
                        neg_samples.append(prompt_response)
                        stats["negative_samples"] += 1

                        # Update rating distributions for negative sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)
                        if negative_concepts:
                            update_rating_stats(rating_obj.ratings, negative_concepts)

                        # Track adjectives in negative samples
                        for adj in rating_obj.adjectives:
                            stats["negative_sample_adjectives"][adj] = stats["negative_sample_adjectives"].get(adj, 0) + 1

                    else:
                        # Doesn't meet criteria for either class
                        skipped_intermediate += 1
                        stats["skipped_intermediate_ratings"] += 1
            else:
                # Original behavior (no explicit negative concepts)
                if ignore_generation_categories:
                    # Ignore generation categories - just use rating thresholds
                    # Positive: at least one concept >= positive_threshold
                    if max_rating >= positive_threshold:
                        pos_samples.append(prompt_response)
                        stats["positive_samples"] += 1

                        # Update rating distributions for positive sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)

                        # Track adjectives in positive samples
                        for adj in rating_obj.adjectives:
                            stats["positive_sample_adjectives"][adj] = stats["positive_sample_adjectives"].get(adj, 0) + 1

                    # Negative: ALL concepts <= negative_threshold
                    elif max_rating <= negative_threshold:
                        neg_samples.append(prompt_response)
                        stats["negative_samples"] += 1

                        # Update rating distributions for negative sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)

                        # Track adjectives in negative samples
                        for adj in rating_obj.adjectives:
                            stats["negative_sample_adjectives"][adj] = stats["negative_sample_adjectives"].get(adj, 0) + 1

                    else:
                        # Intermediate ratings, skip this sample
                        skipped_intermediate += 1
                        stats["skipped_intermediate_ratings"] += 1
                else:
                    # Consider generation categories (default behavior for synthetic data)
                    # Positive: was generated WITH at least one merged concept AND has high rating
                    if has_merged_concept_in_generation and max_rating >= positive_threshold:
                        pos_samples.append(prompt_response)
                        stats["positive_samples"] += 1

                        # Update rating distributions for positive sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)

                        # Track adjectives in positive samples
                        for adj in rating_obj.adjectives:
                            stats["positive_sample_adjectives"][adj] = stats["positive_sample_adjectives"].get(adj, 0) + 1

                    # Negative: was generated WITHOUT any merged concepts AND has low ratings for all
                    elif not has_merged_concept_in_generation and max_rating <= negative_threshold:
                        neg_samples.append(prompt_response)
                        stats["negative_samples"] += 1

                        # Update rating distributions for negative sample
                        update_rating_stats(rating_obj.ratings, merge_concepts)

                        # Track adjectives in negative samples
                        for adj in rating_obj.adjectives:
                            stats["negative_sample_adjectives"][adj] = stats["negative_sample_adjectives"].get(adj, 0) + 1

                    else:
                        # Doesn't meet criteria - skip this sample
                        skipped_intermediate += 1
                        stats["skipped_intermediate_ratings"] += 1

    # Store original counts before applying limit (for printing only)
    original_pos_count = len(pos_samples)
    original_neg_count = len(neg_samples)

    print(f"Merged probe data selection:")
    if negative_concepts:
        print(f"  Positive concepts: {merge_concepts}")
        print(f"  Negative concepts: {negative_concepts}")
        if ignore_generation_categories:
            print(f"  Mode: Ignoring generation categories (ultrachat mode)")
            print(f"  Positive samples (≥{positive_threshold} for ≥1 positive concept): {original_pos_count}")
            print(f"  Negative samples (≥{positive_threshold} for ≥1 negative concept): {original_neg_count}")
        else:
            print(f"  Mode: Considering generation categories (synthetic data mode)")
            print(f"  Positive samples (generated WITH positive concepts AND ≥{positive_threshold}): {original_pos_count}")
            print(f"  Negative samples (generated WITH negative concepts AND ≥{positive_threshold}): {original_neg_count}")
    else:
        if ignore_generation_categories:
            print(f"  Mode: Ignoring generation categories (ultrachat mode)")
            print(f"  Positive samples (≥{positive_threshold} for ≥1 concept): {original_pos_count}")
            print(f"  Negative samples (≤{negative_threshold} for ALL concepts): {original_neg_count}")
        else:
            print(f"  Mode: Considering generation categories (synthetic data mode)")
            print(f"  Positive samples (generated WITH merged concepts AND ≥{positive_threshold}): {original_pos_count}")
            print(f"  Negative samples (generated WITHOUT merged concepts AND ≤{negative_threshold}): {original_neg_count}")
    print(f"  Skipped (missing ratings): {skipped_missing}")
    print(f"  Skipped (doesn't meet criteria): {skipped_intermediate}")

    # Apply limit if specified
    if limit is not None and limit > 0:
        print(f"\nApplying limit of {limit} samples per class...")
        print(f"  Before limiting: {len(pos_samples)} positive, {len(neg_samples)} negative")

        # Store counts before limit for reference
        stats["samples_before_limit"] = {
            "positive": len(pos_samples),
            "negative": len(neg_samples)
        }

        # Shuffle before limiting if requested
        if shuffle:
            random.Random(42).shuffle(pos_samples)
            random.Random(42).shuffle(neg_samples)

        # Apply the limit
        pos_samples = pos_samples[:limit]
        neg_samples = neg_samples[:limit]

        print(f"  After limiting: {len(pos_samples)} positive, {len(neg_samples)} negative")

        # Now recalculate all statistics based on limited samples
        stats["positive_samples"] = len(pos_samples)
        stats["negative_samples"] = len(neg_samples)
        stats["limit_applied"] = limit

        # Recalculate rating distributions for limited samples
        stats["rating_distributions"] = {concept: {"min": 1.0, "max": 0.0, "sum": 0.0, "count": 0} for concept in all_tracked_concepts}
        stats["positive_sample_adjectives"] = {}
        stats["negative_sample_adjectives"] = {}

        # Process limited positive samples
        for sample in pos_samples:
            # Find the corresponding rating object
            for rating_obj in ratings_list:
                if rating_obj.prompt == sample.prompt and rating_obj.response == sample.response:
                    # Update rating distributions
                    for concept in all_tracked_concepts:
                        if concept in rating_obj.ratings and rating_obj.ratings[concept] is not None:
                            rating = rating_obj.ratings[concept]
                            stats["rating_distributions"][concept]["min"] = min(stats["rating_distributions"][concept]["min"], rating)
                            stats["rating_distributions"][concept]["max"] = max(stats["rating_distributions"][concept]["max"], rating)
                            stats["rating_distributions"][concept]["sum"] += rating
                            stats["rating_distributions"][concept]["count"] += 1

                    # Track adjectives
                    for adj in rating_obj.adjectives:
                        stats["positive_sample_adjectives"][adj] = stats["positive_sample_adjectives"].get(adj, 0) + 1
                    break

        # Process limited negative samples
        for sample in neg_samples:
            # Find the corresponding rating object
            for rating_obj in ratings_list:
                if rating_obj.prompt == sample.prompt and rating_obj.response == sample.response:
                    # Update rating distributions
                    for concept in all_tracked_concepts:
                        if concept in rating_obj.ratings and rating_obj.ratings[concept] is not None:
                            rating = rating_obj.ratings[concept]
                            stats["rating_distributions"][concept]["min"] = min(stats["rating_distributions"][concept]["min"], rating)
                            stats["rating_distributions"][concept]["max"] = max(stats["rating_distributions"][concept]["max"], rating)
                            stats["rating_distributions"][concept]["sum"] += rating
                            stats["rating_distributions"][concept]["count"] += 1

                    # Track adjectives
                    for adj in rating_obj.adjectives:
                        stats["negative_sample_adjectives"][adj] = stats["negative_sample_adjectives"].get(adj, 0) + 1
                    break

    # Calculate averages for rating distributions
    for concept in all_tracked_concepts:
        if concept in stats["rating_distributions"] and stats["rating_distributions"][concept]["count"] > 0:
            stats["rating_distributions"][concept]["avg"] = (
                stats["rating_distributions"][concept]["sum"] /
                stats["rating_distributions"][concept]["count"]
            )
        else:
            if concept in stats["rating_distributions"]:
                stats["rating_distributions"][concept]["avg"] = None

    # Sort and organize generation category counts for better readability
    if stats["positive_sample_adjectives"]:
        stats["positive_sample_adjectives"] = dict(sorted(stats["positive_sample_adjectives"].items(),
                                                         key=lambda x: x[1], reverse=True))
    if stats["negative_sample_adjectives"]:
        stats["negative_sample_adjectives"] = dict(sorted(stats["negative_sample_adjectives"].items(),
                                                         key=lambda x: x[1], reverse=True))

    # Add summary counts
    stats["generation_category_summary"] = {
        "unique_categories_in_positive": len(stats["positive_sample_adjectives"]),
        "unique_categories_in_negative": len(stats["negative_sample_adjectives"]),
        "total_positive_category_occurrences": sum(stats["positive_sample_adjectives"].values()) if stats["positive_sample_adjectives"] else 0,
        "total_negative_category_occurrences": sum(stats["negative_sample_adjectives"].values()) if stats["negative_sample_adjectives"] else 0,
    }

    print(f"\n### MERGED DATA STATS ###")
    pprint(stats, width=120, compact=False)
    print(f"### ###\n")

    # Don't shuffle again in PosNegData if we already shuffled for limiting
    should_shuffle = shuffle and (limit is None)
    pos_neg_data = PosNegData(pos_samples, neg_samples, shuffle=should_shuffle)

    if return_stats:
        return pos_neg_data, stats
    else:
        return pos_neg_data, None
