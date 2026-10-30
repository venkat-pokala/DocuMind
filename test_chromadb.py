
from rag_pipeline import collection

print("\n===== Testing ChromaDB =====")

# Check database connection
print("\nChromaDB connected successfully!")

# Check collection
print("Collection name:", collection.name)

# Check stored chunks
total_chunks = collection.count()

print("Total indexed chunks:", total_chunks)

if total_chunks == 0:

    print("\nNo documents found.")
    print("Please upload and index a PDF first.")

else:

    # Retrieve stored documents
    results = collection.get(
        limit=3,
        include=["documents", "metadatas"]
    )

    print("\n===== Stored Documents =====")

    for i, (document, metadata) in enumerate(
        zip(results["documents"], results["metadatas"]),
        start=1
    ):

        print(f"\nDocument {i}")
        print("Source:", metadata.get("source"))
        print("Page:", metadata.get("page"))
        print("Content:", document[:300])

    print("\nChromaDB storage test passed!")

print("\nChromaDB test completed!")
