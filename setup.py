from setuptools import setup, find_packages

setup(
    name="kblam-ollama",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "sentence-transformers",
        "requests",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "nlp": ["spacy"],
    },
    author="Matthew Frederico",
    author_email="mfrederico@gmail.com",
    description="Microsoft KBLaM - on Ollama",
    keywords="NLP, knowledge base, language model",
    python_requires=">=3.12",
)

