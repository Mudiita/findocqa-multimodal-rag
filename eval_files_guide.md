# Evaluation files guide

This page lists the evaluation files in the repo and what each one is.

- **Query files** hold the questions with their expected answers and sources.
- **Graded results files** hold the system's answers, citations, response time and cost. Each has a `correct (Y/N/partial)` and a `citation correct (Y/N)` column, graded against the source PDFs with LLM assistance.

Accuracy counts Y = 1, partial = 0.5 and N = 0. Citation accuracy is over the rows where a citation could be graded.

| File | What it is | Accuracy | Citations |
|---|---|---|---|
| `eval_queries.csv` | The 70-query evaluation set (Q1–Q70), 10 questions in each of 7 categories. Used while building the system. | | |
| `eval_results_graded.csv` | First version of the system on the 70 queries, graded | 73.6% | 72.1% |
| `eval_results_v5_graded.csv` | Final version on the 70 queries, graded | 89.3% | 94.0% |
| `held_out_queries.csv` | 20 extra questions (H1–H20) used to check fixes during development. Results are not included. | | |
| `new_test_queries.csv` | The 20-query unseen test (N1–N20), written after the last fix | | |
| `new_test_results_graded.csv` | The single run of the unseen test, graded | 72.5% | 93.3% |
| `dataset_manifest.csv` | The 30 documents: doc_id, category, provider, product, file path, document date | | |

## Re-running

```
python run_eval.py QUERIES.csv RESULTS.csv   # run a query file, write a new results file
python run_eval.py --summary RESULTS.csv     # print a summary of a results file
```

Running `python run_eval.py` with no arguments reads `eval_queries.csv` and writes `eval_results.csv`. The results files are written without grades; the grading columns are left empty.
