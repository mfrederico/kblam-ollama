# Import key components to make them accessible from the package root
from kblam.models import KBLAM, LinearAdapter
from kblam.knowledge_base import KnowledgeBase
from kblam.training import train_adapter, train_on_large_corpus
from kblam.processors import generate_synthetic_kb, preprocess_corpus
