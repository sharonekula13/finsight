# FinSight

FinSight is a project I built to answer questions about company financial filings (like the 10-K annual reports companies file with the SEC) without making things up. Normally if you ask a language model a question about a document, it can hallucinate an answer that sounds right but isn't in the document. FinSight avoids that by first finding the relevant parts of the filing, then asking the model to answer using only those parts, and finally double-checking that every sentence of the answer is actually backed by the source text. It also shows which piece of the document each fact came from, so the answers are traceable.

## How it works

When you ask a question, it flows through four small steps that act like a little team. A *router* first looks at the question and decides how much context to pull (a simple lookup needs less than a comparison across two companies). A *retriever* then searches the filings and brings back the most relevant chunks — it mixes meaning-based search using embeddings with plain keyword matching, because together they work better than either one alone. The language model reads those chunks and writes an answer with citations. Finally a *verifier* goes through the answer sentence by sentence and checks each one against the retrieved text, producing a grounding score so I can actually measure how faithful the model is.

On top of the pipeline I built an evaluation harness, which is the part I'm most proud of. It runs a set of test questions and scores the system on how grounded the answers are, whether it retrieved the right documents, how relevant the answers are, and how much each query costs and how long it takes. This runs automatically with GitHub Actions every time I push code, and the build fails if answer quality drops below a threshold — so the project is treated like real production software, not just a demo.

## Running it

The project runs in two modes. By default it runs completely offline with no API keys, using simple stand-in models, so anyone can download it and try the whole thing instantly and for free. If you add a Google Gemini API key and switch to live mode, it uses a real model for the answers and can also pull real filings directly from SEC EDGAR just by giving it a stock ticker.

At a high level: set `PYTHONPATH` to `src`, then run the demo module to see it answer a few questions, run the tests, and run the evaluation to get a scored leaderboard. To go live, add your Gemini key and set the mode to live, then run the same commands. The exact commands are in the `.env.example` file and in the comments at the top of each module.

## Some results

In live mode with Gemini over the sample filings, the answers came out fully grounded (no unsupported claims), context precision was about 0.71, and answer relevance was about 0.66, at roughly one second per query and a fraction of a cent in cost. The context precision is the number I'd improve next — filtering retrieval by company would remove the occasional off-topic chunk that sneaks in.

## Tech used

Python, the Gemini API, hybrid retrieval with embeddings, FastAPI with a small HTML dashboard, Docker, GitHub Actions for CI, and a Terraform stub for deploying to AWS.

*(The sample data uses two made-up companies so the demo is self-contained. Real filings come from SEC EDGAR.)*
