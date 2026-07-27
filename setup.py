from setuptools import find_namespace_packages, setup

setup(
    name="neural-chameleons",
    version="1.0.0",
    description="Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors",
    packages=find_namespace_packages(
        include=["obf_reps*", "experiments*", "data*"],
        exclude=["*.outputs*", "*.results*", "*.output"],
    ),
    author="Max McGuinness, Alex Serrano, Luke Bailey, Scott Emmons",
    author_email="max@max.rip, mail@alexserrano.org",
    url="https://github.com/mgm52/neural-chameleons",
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        # Base (torch is intentionally not pinned here -- install it separately
        # with the right CUDA build for your system, see README)
        "transformers",
        "datasets",
        "accelerate",
        "scikit-learn",
        "joblib",
        "numpy",
        "scipy",
        "pyyaml",
        "pydantic",
        # Visualization
        "matplotlib",
        "pandas",
        "seaborn",
        # Logging
        "wandb",
        "python-dotenv",
        "psutil",
        # Utils
        "tqdm",
        "jaxtyping",
        "sentencepiece",
        "bitsandbytes",
        "sentence-transformers",
    ],
    extras_require={
        "dev": ["pytest", "pre-commit", "ipykernel", "ipywidgets"],
        # Only needed to regenerate the rated training datasets from scratch
        "data-regen": ["openai"],
        # Only needed for the lm-eval capability benchmarks
        "evals": ["lm-eval"],
    },
)
