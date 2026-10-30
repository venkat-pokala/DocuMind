
from rag_pipeline import (
    collection,
    retrieve_context,
    generate_answer
)

print("\n========== DocuMind Evaluation ==========")

if collection.count() == 0:

    print("No PDF is indexed. Please index a PDF first.")
    raise SystemExit(0)

print("Indexed chunks:", collection.count())

print("Full-document retrieval is enabled.")

while True:

    question = input(
        "\nEnter your question (or exit): "
    ).strip()

    if question.lower() in {"exit", "quit"}:

        print("Evaluation finished.")
        break

    if not question:
        continue

    # Retrieve all indexed chunks.
    retrieved_chunks = retrieve_context(
        question,
        top_k=collection.count(),
        max_distance=None
    )

    print("\n========== Retrieved Results ==========")

    if not retrieved_chunks:

        print("No chunks were retrieved.")

    else:

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(f"\nResult {i}")

            print("Source:", chunk["source"])
            print("Page:", chunk["page"])
            print(
                "Distance:",
                round(chunk["distance"], 4)
            )

            print("\nRetrieved text:")
            print(chunk["text"])

    # Generate answer.
    print("\n========== Generated Answer ==========")

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print(answer)

    print("\n======================================")
