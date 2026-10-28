# Adaptive AI Tutor (Phases 1–10)

A local, zero-cost AI tutor built with Ollama, LangChain, LangGraph, Chroma, and Streamlit.

- **Phase 1:** ask questions and get answers from a local model.
- **Phase 2:** upload PDFs and notes, and get answers grounded in them, with filename and page citations. If your documents don't cover a question, the tutor says so instead of guessing.
- **Phase 3:** optional **web research**. The tutor turns your question into a short search query, searches a free service, filters the results with simple quality checks, and answers only from them, with links. It is **off by default**.
- **Phase 4:** the tutor **checks its own work** (Self-RAG) and **fixes poor searches** (Corrective RAG). It grades each passage, checks the evidence is enough, searches again with a better query when needed, can fall back to the web if you allow it, and checks the answer against the sources before showing it.

- **Phase 5:** structured **lessons** (explanation, analogy, example, key points) and **quizzes** (multiple choice and short answer) with a concept label on every question. Answers stay hidden until you submit; multiple choice is graded exactly, short answers with a rubric, and every mistake is explained.
- **Phase 6:** a **learner profile** with simple mastery estimates per concept, possible misconceptions, prerequisites, review dates, and recommendations (re-teach, practise, revise, or move on).
- **Phase 7:** **saved progress** in a local SQLite database, and LangGraph **checkpointing**, so an unfinished quiz survives closing the app. Each learner's records are kept apart.
- **Phase 8:** one **agentic learning workflow** in LangGraph: load profile → understand the request → retrieve → check → correct / research → teach → assess → pause for your answers → grade → update profile → plan → save.
- **Phase 9:** a **progress dashboard** and local **evaluation metrics** (retrieval relevance, groundedness, citation correctness, grading consistency, adaptation self-check, response time, memory use).

Everything runs on your computer. Your documents are never sent anywhere.

## Requirements

- Python 3.10+
- Ollama (https://ollama.com) with two models:
  - a chat model, e.g. `ollama pull llama3.2:3b` (~2 GB)
  - the embedding model: `ollama pull nomic-embed-text` (~275 MB)
- About 8 GB RAM for 3B chat models, 16 GB for 7B models
- Internet access only if you turn on web research

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Already have a `.env` from an earlier phase? Keep it, and add the new Phase 5–7 settings from `.env.example` (they all have safe defaults, so the app also works without them).

## Run

```bash
streamlit run app.py
```

Open the address printed in the terminal (normally http://127.0.0.1:8501).

The app has three tabs:

1. **💬 Ask a question**: the Phase 1–4 tutor. Add files in the sidebar, choose **My documents**, and ask.
2. **📘 Learn & practise**: create a learner profile in the sidebar, then type what you want, e.g. *Teach me photosynthesis*, *Quiz me on embeddings*, or *What should I study next?*
3. **📊 My progress**: mastery by concept, weak concepts, revision dates, quiz history, lesson sources, and evaluation metrics.

## Lessons and quizzes (Phase 5)

- **Lessons** are written as JSON by the local model and checked field by field before they're shown. A lesson always has a simple explanation, an everyday **analogy**, a concrete **example**, key points, and concept labels. Lessons from documents cite them like `[1]`; lessons from the web cite `[W1]`. The analogy and example are the tutor's own teaching aids and are labelled as such.
- **Quizzes** have `QUIZ_MULTIPLE_CHOICE` multiple-choice and `QUIZ_SHORT_ANSWER` short-answer questions (default 3 and 1). Each question has a concept label.
- **Answers stay hidden.** Before you submit, the page only receives a "public view" of the quiz that contains no answers, explanations, or rubric.
- **Grading.** Multiple choice: exact comparison with the answer key, no model call, so it's always consistent. Short answer: the model checks your answer against 2–3 key points (a rubric) and gives a reason for each. Your score is the share of key points you covered.
- **Mistakes explained.** Wrong choices show what you chose, the right answer, and why. Short answers show which key points were missing, the model answer, and a *possible* misconception.
- **Safety nets for a small model.** If the JSON is unusable, the tutor asks again once (`STRUCTURED_OUTPUT_RETRIES`). Bad questions are dropped, not shown. Options are shuffled, so the answer isn't always A. With careful checking on, the model answers each multiple-choice question a second time, and questions where it disagrees with the key are dropped. If a short answer can't be graded reliably, it's marked **not graded** and doesn't change your mastery.

## Your profile, mastery, and what's next (Phase 6)

- **Mastery** per concept moves 40% of the way toward each new score: `new = old + 0.4 × (score − old)`. Four good answers in a row take a new concept from 0.00 to about 0.87.
- **Statuses:** under 0.40 needs re-teaching, 0.40–0.80 practising, 0.80+ mastered.
- **Review dates** (simple spaced repetition): after each well-answered question the gap grows 1 → 3 → 7 → 14 → 30 days; a poor answer resets it to 1 day.
- **Prerequisites** come from lessons ("helpful to know first"). If you're struggling with a concept whose prerequisite is weak, the tutor suggests the prerequisite first.
- **Recommendations**, in order: study a prerequisite, re-teach, revise (when due), practise, take a first quiz, or move up a level when every concept of a topic is mastered. Each has a reason, and a **Start** button.
- **Adaptation:** when you ask for a lesson or quiz on a topic, the tutor looks up your weak concepts in it and focuses on them.

These rules are reasonable rules of thumb for a learning project. They are **not** a scientifically validated measure of knowledge.

## Saved progress and resuming (Phase 7)

- Progress is saved in `data/tutor.sqlite` (`DATABASE_PATH`): learners, concepts, lessons with their sources, quizzes, results, misconceptions, and metric events.
- The learning workflow's state is saved by LangGraph's SQLite checkpointer in `data/checkpoints.sqlite` (`CHECKPOINT_PATH`). When a quiz is waiting for your answers, the workflow is **paused** with `interrupt()`; you can close the app and submit tomorrow. Each activity has its own thread id, like `ada-3f9a1c2b:9e1d0c4b7a21`.
- **Learners are kept apart:** every database query filters by learner, and a quiz can only be submitted by the learner it belongs to. Profiles have no passwords: anyone using this computer can open any profile.
- **Reset or delete** a learner under **Reset or delete progress** in the sidebar. Reset keeps the profile; delete removes everything, including saved checkpoints. **Download my progress (JSON)** on the progress tab exports your data first.
- OneDrive tip: syncing can lock SQLite files. If you see "database is locked", point `DATABASE_PATH` and `CHECKPOINT_PATH` to a folder outside OneDrive, such as `C:/AdaptiveTutorData/tutor.sqlite`.

## How the learning workflow works (Phase 8)

```
START -> load_profile -> understand_request
  |-- "what next" ---------------------------------------> recommend_next -> END
  |-- general knowledge ---------------------------------> generate_lesson / create_quiz
  +-- documents -> retrieve_material -> check_material
        |-- good material -------------------------------> generate_lesson / create_quiz
        |-- weak, retries left -> rewrite_material_query -> retrieve_material (wider)
        |-- weak, web allowed --> research_web ----------> generate_lesson / create_quiz
        |-- partly covered -----> use_partial_material --> generate_lesson / create_quiz
        +-- nothing ------------> report_no_material -> END
generate_lesson -> check_lesson -> create_quiz          (failed check: rewrite, then refuse)
create_quiz -> wait_for_answers (PAUSE) -> grade_answers -> update_profile -> recommend_next -> save_progress -> END
```

- **Understanding the request** uses simple wording rules first ("teach", "quiz me", "what next"); only if they can't tell does the model answer one word (LESSON, QUIZ, or NEXT).
- **Choosing tools:** documents first; the web only if you ticked the box; general knowledge only if you chose it (clearly labelled, no citations).
- **Bounded loops:** search retries use `MAX_RETRIEVAL_RETRIES`, lesson rewrites `MAX_ANSWER_RETRIES`, and LangGraph stops any run after 60 steps as a last safety net (the longest real run is about 20).
- Open **How the tutor decided (agent log)** under a lesson to see every decision in plain language.
- The Phase 1–4 question graph is a separate, unchanged graph.
- **Cost:** a checked lesson + quiz makes about 10 calls to the local model (plus one per short answer when grading). On a CPU this can take a few minutes. Untick careful checking for about 3 calls.

## Dashboard and evaluation (Phase 9)

The **My progress** tab shows topics studied, mastery by concept, weak concepts, revision due, possible misconceptions, quiz history, lessons with their sources, and suggestions. Under **Evaluation** it shows:

| Metric | How it is measured |
|---|---|
| Retrieval relevance | passages the checker judged relevant ÷ passages it graded |
| Groundedness | checked answers and lessons that passed ÷ all that were checked |
| Citation correctness | answers whose citations all point to real sources ÷ answers with sources |
| Grading consistency | button: re-grade your latest short answer 3 times; how often do the grades agree? |
| Adaptation self-check | 5 example profiles; does the planner recommend what its rules say? |
| Response time | median and 90th percentile per kind of run |
| Local resources | this app's memory, and the models Ollama has loaded (from its `/api/ps`) |

Metric events store only numbers and yes/no flags, never questions, answers, or document text. Nothing is sent to any outside service.

## Careful checking (Self-RAG and Corrective RAG)

In **My documents** and **Web research** mode, the box **Check evidence and answers carefully** is ticked by default. With it on, every question goes through these checks, and the local model makes each judgement:

1. **Relevance:** each passage is graded "useful for this question: yes or no". Passages judged not useful are dropped, and you can see them under **Passages judged not relevant**.
2. **Enough evidence:** the remaining passages are judged SUFFICIENT, INSUFFICIENT, or CONTRADICTORY.
3. **Correcting a poor search:** if no passage is relevant, or the evidence is insufficient, the model rewrites the search query and the tutor searches again more widely (twice as many passages, similarity threshold 0.1 lower). Widening is safe because every new passage must still pass the relevance check. This happens at most `MAX_RETRIEVAL_RETRIES` times (default 1).
4. **Web fallback (optional):** if the documents still fall short and web research is on, the tutor can search the web. Only if you tick **If my documents don't cover it, search ...** for that question.
5. **Contradictions:** if the sources disagree, the answer describes each view with its citation, and the app shows a warning.
6. **Answer check:** first a simple rule check (does the answer cite sources, and do those sources exist?), then the model checks that every claim is supported. If not, the answer is written again, at most `MAX_ANSWER_RETRIES` times (default 1).
7. **Honest refusal:** if the limits are used up, the tutor says it can't answer reliably and never shows an unsupported answer.

Open **How the answer was checked** under an answer to see every step in plain language.

Keep in mind:

- Checking costs time. A typical checked question makes about 7 calls to the local model instead of 1, which can take a minute or more on a CPU. Untick the box for quick answers.
- The checker is the same small local model, so it can be wrong in both directions. A passed check makes an answer more trustworthy, but it isn't proof.
- Unclear checker replies are treated as "no" or "insufficient", so when in doubt the tutor is careful rather than confident.

## Web research (optional)

Web research is a separate answer mode. Document answers and web answers are never mixed: document passages are cited as `[1]`, `[2]`, web results as `[W1]`, `[W2]`, and they appear in separate panels.

### What leaves your computer

Only a short search query (shown under every answer) goes to the search service. Your questions are turned into keywords by the local model first, and your uploaded documents are never sent. If you also turn on page reading, the tutor downloads result pages from their websites, which those websites can see.

### Choose a provider

All three were free without a payment method when checked in October 2026. Check each service's current terms yourself before using it, because prices and rules change.

| Provider | Account needed | Good for | Setup |
|---|---|---|---|
| `wikipedia` (default) | No | Explaining concepts | Set `RESEARCH_CONTACT` |
| `searxng` | No | General web search | Run your own SearXNG server |
| `tavily` | Free account | General web search | Free API key, monthly free credits |

**Wikipedia** (easiest): in `.env` set

```
RESEARCH_ENABLED=true
SEARCH_PROVIDER=wikipedia
RESEARCH_CONTACT=your.email@example.com
```

Wikimedia requires apps to identify themselves with contact details, and asks for one request at a time. The tutor makes one request per question. Try `WIKIPEDIA_LANGUAGE=simple` for Simple English Wikipedia, which suits beginners. Wikipedia text is shared under the CC BY-SA licence; the tutor links to every article it uses.

**SearXNG** (self-hosted, no account): SearXNG is free, open-source software that collects results from other search engines. The usual way to run it is Docker; check your company's software rules and Docker's licence terms before installing it on a work computer. After starting it, edit its `settings.yml` so that JSON output is allowed:

```yaml
search:
  formats:
    - html
    - json
```

Restart SearXNG, then set `SEARCH_PROVIDER=searxng` and `SEARXNG_BASE_URL` (for example `http://localhost:8080`). If you see "refused the request (HTTP 403)", JSON output is still off. Your SearXNG server forwards your queries to the search engines it is configured to use. Avoid public SearXNG servers: they are run by strangers and often block automated use.

**Tavily** (free account): sign up at https://app.tavily.com, copy your key, and set `SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY=...`. The free plan has a monthly credit limit; one basic search uses about one credit. Do not add a payment method or switch on pay-as-you-go. When the free credits run out, the tutor shows a "limit has been reached" message and never switches to a paid plan.

Restart the app after changing `.env`. **Web research** then appears as an answer mode.

### Source-quality checks

Each result gets a score from 0 to 1 using simple, visible rules. It starts at 0.5, then:

- **+0.2** educational, government, or reference site (for example `.edu`, `.gov`, Wikipedia)
- **+0.1** enough text to answer from; **−0.1** only a short snippet; **−0.3** almost no text
- **+0.1** mentions most of the search words; **−0.3** mentions none of them
- **−0.1** unencrypted `http://` link

Results below `RESEARCH_MIN_QUALITY` (default 0.4) are filtered out. Some results are always rejected: non-web links, private or local network addresses, duplicates, and sites in `RESEARCH_BLOCKED_DOMAINS`. Open **Filtered out** under an answer to see every rejected result and why.

These checks are rules of thumb, not fact-checking. A high score means "looks like a reasonable place to learn from", not "this is true". Phase 4 adds model-based checks.

### Reading full pages (optional)

Search services often return only a short snippet. With `RESEARCH_FETCH_PAGES=true`, the tutor tries to read the result page itself, but only when:

- the site's `robots.txt` allows it (if `robots.txt` can't be read, the page is skipped),
- the page is HTML, and
- the page doesn't redirect elsewhere.

At most about 2 MB is downloaded and 5,000 characters of text are kept per page. Wikipedia results already include the article introduction, so they are not downloaded again.

### Company networks

On a company network, web research may be blocked. You'll see "Cannot reach ..." or "sent a reply that isn't JSON". Don't try to get around the block. Ask your IT team, or keep `RESEARCH_ENABLED=false`; everything else works offline. If your company uses a proxy through the standard `HTTPS_PROXY` setting, the tutor uses it automatically.

## Test

```bash
pytest -v
```

Tests use fake models, fake search services, temporary Chroma and SQLite databases, and LangGraph's in-memory checkpointer (plus one real SQLite checkpointer restart test). They need neither Ollama nor the internet.

To run one area: `pytest tests/test_learning_workflow.py -v`.

## Configuration

All settings live in `.env` (see `.env.example`). Restart the app after changing them.

A few settings worth knowing:

- `MIN_RELEVANCE_SCORE` (default 0.5): document passages scoring below this are ignored. When the tutor can't find an answer, it shows the best score it found, which helps you tune this value.
- `EMBEDDING_MODEL`: if you change it, click **Clear library** in the app and upload your files again. Different embedding models produce incompatible numbers.
- `RESEARCH_MIN_QUALITY` (default 0.4): raise it to be stricter about web sources, lower it if good results are filtered out.

- `QUIZ_MULTIPLE_CHOICE`, `QUIZ_SHORT_ANSWER`: quiz size. Fewer questions means faster quizzes.
- `STRUCTURED_OUTPUT_RETRIES` (default 1): extra tries when the model's JSON is unusable.
- `DATABASE_PATH`, `CHECKPOINT_PATH`: where progress and paused quizzes are saved.

Your document library is saved in `data/chroma/` and your progress in `data/tutor.sqlite` and `data/checkpoints.sqlite`. Delete the `data` folder to reset everything (close the app first).

## Troubleshooting

| Message | Fix |
|---|---|
| Cannot reach Ollama | Start the Ollama app or run `ollama serve` |
| Model ... is not installed | Run `ollama pull <model>` or fix the name in `.env` |
| Model took longer than ... | Use a smaller model or raise `OLLAMA_REQUEST_TIMEOUT_SECONDS` |
| No readable text found | The PDF is probably scanned images; OCR is not supported yet |
| Blank page in the browser | Open http://127.0.0.1:8501, try a private window, or check the browser console (F12) for blocked WebSocket connections |
| Errors after changing the embedding model | Click **Clear library** and upload your files again |
| Configuration problem: RESEARCH_CONTACT is required | Add your email to `RESEARCH_CONTACT`, or set `RESEARCH_ENABLED=false` |
| Web research doesn't appear as a mode | Set `RESEARCH_ENABLED=true` and restart the app |
| Cannot reach Wikipedia / SearXNG / Tavily | Check your internet connection; the network may block it (see Company networks) |
| SearXNG refused the request (HTTP 403) | Enable `json` under `search: formats:` in SearXNG's `settings.yml` |
| Tavily rejected the API key | Copy the key again from https://app.tavily.com into `.env` |
| ... limit has been reached | Wait and try later, or switch provider |
| Checked answers are very slow | Untick **Check evidence and answers carefully**, use a smaller model, or lower `RETRIEVAL_TOP_K` |
| "I found passages ... related, but ..." | Your documents mention the topic but don't answer it; add a better document or rephrase |
| "my check found claims that your sources don't support" | Ask a narrower question; the sources may only partly cover it |
| A package is missing (langgraph.checkpoint.sqlite) | Run `pip install -r requirements.txt` again |
| "didn't return a usable lesson / quiz" | Try again or a narrower topic; raise `STRUCTURED_OUTPUT_RETRIES`; or use a bigger model like `qwen2.5:7b` |
| "Your documents don't cover this topic well enough" | Add a document about it, tick the web box, or choose **General knowledge** |
| A quiz has fewer questions than asked | Some questions were dropped by the checks; see the agent log |
| "database is locked" | Close other copies of the app; move `DATABASE_PATH`/`CHECKPOINT_PATH` outside OneDrive |
| "Your quiz is still open" after submitting | The model wasn't reachable while grading; start Ollama and submit again |

## Privacy

- Streamlit only listens on this computer (`127.0.0.1`), and usage statistics are disabled in `.streamlit/config.toml`.
- Chroma's anonymous telemetry is switched off in code.
- Questions and search queries are not written to logs (only their length).
- Uploaded files are processed in memory; only their text chunks and embeddings are saved in `data/chroma/`.
- Web research is off by default. When on, the exact search query sent is shown under each answer.
- The Tavily key is kept out of logs and printed settings.
- Lessons, quizzes, and your answers are stored only in the local SQLite files. Metric events contain numbers only.

## Limitations

- Self-RAG checks are made by the same small local model; they reduce mistakes but can't remove them, and they make answers slower.
- Web fallback only happens when you allow it for that question, and web searches are never retried (to save free quota).
- The keyword check matches whole words only, so "mitochondrion" doesn't match "mitochondria".
- The private-address check looks at the link itself; it can't detect a public name that points to a private address.
- Scanned PDFs and images are not supported (no OCR).
- Questions in the Ask tab are independent; there is no chat-style conversation memory.
- Lessons, quiz questions, answer keys, and short-answer grades all come from the same small local model, so they can be wrong. The checks reduce this; they don't remove it.
- Mastery, review dates, and recommendations are simple rules, not a validated learning model. Misconceptions are guesses.
- Concept labels come from the model, so the same idea can occasionally get two slightly different names.
- Profiles have no passwords; they separate progress, they don't protect it.
- Only one paused quiz per learner: starting a new activity abandons an unfinished one.
- This is a learning project. It runs locally, but it is **not production-ready**: no authentication, no multi-user server, no backups, and limited testing against real model behaviour.

## Final review (Phase 10)

The whole project was reviewed against this checklist. What was found and what to keep in mind:

| Area | Status |
|---|---|
| Imports | Every file compiles; heavy packages (Chroma, the SQLite checkpointer) are imported only where needed |
| Dependency versions | One new package, `langgraph-checkpoint-sqlite` 3.x, which matches LangGraph 1.x |
| Configuration | 35 settings, all with defaults, validated at start-up, documented in `.env.example` |
| Infinite loops | Every loop has a counter and a setting-based limit, plus recursion limits (50 and 60); worst cases are tested |
| Invalid model output | JSON is checked field by field; bounded retries; unclear replies are treated cautiously |
| Mastery updates | Unit-tested; ungraded answers never change mastery |
| Data leakage between learners | Every query filters by learner; tests check separation and quiz ownership |
| Exposed credentials | `.env` is git-ignored; the Tavily key is hidden from logs and printed settings |
| Missing citations | Rule check plus model check for answers and lessons; uncited lessons from sources are rejected |
| Session persistence | SQLite checkpointer; a test closes and reopens the checkpoint file and resumes the quiz |
| Tests | See `pytest -v`; the UI pages themselves are not unit-tested (try them by hand) |