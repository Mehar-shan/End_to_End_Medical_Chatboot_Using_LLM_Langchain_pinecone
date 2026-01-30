import pkg_resources

packages = [
    "sentence-transformers",
    "langchain",
    "flask",
    "pypdf",
    "python-dotenv",
    "pinecone-client",  # Note: pinecone is imported as 'pinecone-client'
    "langchain-pinecone",
    "langchain-community",
    "langchain-openai",
    "langchain-experimental"
]

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Not installed")