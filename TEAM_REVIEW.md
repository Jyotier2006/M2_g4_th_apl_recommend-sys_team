# Member review and individual accountability

These assignments come from M1. They are **not evidence that a named member completed the generated M2 work**. Each member should record their actual contributions in the required individual Google Form after reviewing/reproducing the relevant material.

| Member | M1 primary area | Concrete M2 review task |
|---|---|---|
| Jyotier Vithalani | Mathematical formulation, EP/ADF, repository | Derive cavity subtraction; inspect local product/ordinal messages; explain the prior-plus-sites and ADF/EP1 checks |
| Tanmay Kanani | Baselines | Reproduce mean/MF results; explain regularized SGD, validation-only scale fitting and the two MF point decoders |
| Ansh Chaudhari | Base-paper/SOTA analysis | Compare the actual implementation with Matchbox; explain why PreRec is conceptually relevant but not a direct numerical baseline |
| Shlok Shah | MovieLens/metadata | Verify dataset counts, encoding, hashes, split coverage and leakage controls; explain why metadata and cold-start claims remain deferred |
| Het Patel | KPIs, experiment tracking, documentation | Recompute summary mean/SD from CSVs; explain NLL, Brier and ECE differences, timings, uncertainty limitations and report provenance |

All five should be able to explain the whole pipeline. A useful review is to run the demo, then follow one observed rating through the update code and one unobserved pair through prediction.

Before submitting, each member should be able to answer:

1. What are the random variables and which quantities are fixed?
2. Which conditional independence assumptions belong to the model, and which are approximations?
3. What does EP remove before revisiting a rating?
4. Why is the product update variational rather than exact?
5. What is actually learned, and what is not learned in M2?
6. What split and prediction decision rule produced each reported metric?
7. Where does the method fail to improve, and what still belongs in M3?
8. What did I personally review, implement, change, run or analyze, and how did I use AI?

The report transparently acknowledges generative-AI assistance. Do not replace that disclosure with invented member contributions or claim unperformed experiments.
