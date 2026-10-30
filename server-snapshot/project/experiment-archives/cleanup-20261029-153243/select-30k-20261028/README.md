# A4 data expansion

Target: 30,000 training records = immutable A2 10,000 + 20,000 newly accepted.
Validation 100 / test 50 stay unchanged; reserve all 100 official test cases.
Source: existing Ortho2CAD ortho_train_data.zip, continued in the same seed-42 order.
No GPU or training launched. Four CPU CAD workers, candidate code only in Bubblewrap.
All previous filtering rules retained: syntax/code policy, one valid solid, geometry
signature deduplication, blocked holdout families, original family validation partition,
broad-vs-precise bounding dimensions, PNG integrity. No token-length filter or code truncation; full multimodal lengths are recorded.
These automated checks do not establish full drawing-dimension correctness or human review.
Upstream license remains unresolved; no human-reviewed or license-cleared claim.

Run with the drawing2cad-train Python from project root. --smoke only tests isolation.
Outputs are incomplete until preparation-report.json says completed. Final checks assert
exact +20,000 records, unique IDs, disjoint split groups, original manifest rows and SFT
answers unchanged, source hashes unchanged, and valid image paths.
Disk reserve 15 GiB; failure is recorded without editing source or old evaluations.
