"""Interactive Q&A loop: python -m src.rag.cli"""
from .query import answer


def main():
    print("Ask questions about your PDFs (blank line to quit).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        text, docs = answer(q)
        print(f"\n{text}")
        if docs:
            print("\nSources:")
            for d in docs:
                print(f"  - {d.metadata.get('source')} (page {int(d.metadata.get('page', 0)) + 1})")


if __name__ == "__main__":
    main()
