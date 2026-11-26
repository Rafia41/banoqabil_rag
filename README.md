# Build a RAG App That Answers Questions About a YouTube Video

A beginner-friendly, step-by-step guide to running a simple **RAG (Retrieval-Augmented Generation)** app. You will be able to ask questions about a YouTube video, and the app will answer using the video's transcript.

*Credit: this project is based on the work of Santiago (Underfitted). Watch his tutorial: https://www.youtube.com/watch?v=BrsocJb-fAo*

---

## 1. What is RAG? (in plain words)

A normal chatbot only knows what it learned during training. It has never seen your video. RAG fixes this in four steps:

1. **Split** the video transcript into small chunks of text.
2. **Embed** each chunk, which turns text into numbers that capture its meaning.
3. **Store** those numbers in a vector database (we use **Pinecone**).
4. **Retrieve** the chunks most related to your question and hand them to the language model, which writes the answer.

Think of it as an open-book exam: the model looks up the right pages before answering.

---

## 2. What you need before starting

| Item | Why you need it |
| --- | --- |
| **Python 3.11** | Older libraries in this project break on very new Python (3.13 and 3.14). |
| **A Pinecone account** (free) | Stores the embeddings. Get a key at https://www.pinecone.io/ |
| **An LLM API key** | Either an OpenAI key, or a free Groq key from https://console.groq.com/ |
| **Jupyter Notebook** | The project is a notebook file, `rag.ipynb`. |

> **Tip:** Check your Python version with `python --version`. If you already have Python 3.13 or newer, do not worry. You can install 3.11 side by side without removing anything.

---

## 3. Install Python 3.11

Skip this step if `python3.11 --version` (Linux/macOS) or `py -3.11 --version` (Windows) already prints a version.

**Windows** (PowerShell):

```powershell
winget install Python.Python.3.11
```

Close PowerShell and open a new window afterwards. Alternatively, download the 3.11.x installer from python.org and tick **"Add python.exe to PATH"**.

**Ubuntu:**

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv
```

**macOS:**

```bash
brew install python@3.11
```

---

## 4. Create a virtual environment

Open a terminal in the project folder first. A virtual environment (venv) is a private box of packages for this one project, so it cannot break anything else on your computer.

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Check that it worked:

```bash
python --version
```

It should print `Python 3.11.x`, and your terminal prompt should start with `(.venv)`.

---

## 5. Install the packages

Your `requirements.txt` should list exact versions, because unpinned packages install the newest releases, which no longer work together. Replace the contents of `requirements.txt` with this pinned list:

```text
openai==1.30.1
langchain==0.1.20
langchain-openai==0.1.6
langchain-pinecone==0.1.1
langchain-community==0.0.38
docarray==0.40.0
pydantic==1.10.8
python-dotenv==1.0.1
tiktoken==0.7.0
pinecone-client==3.2.2
scikit-learn==1.4.2
httpx==0.27.2
```

Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Always use `python -m pip`** instead of plain `pip`. It guarantees you are using the pip that belongs to your virtual environment.

> **Note:** These versions are a starting point chosen to match the project's era. If pip reports a conflict, it will name the two packages that disagree. Loosen the pin of one of them.

**Why the Whisper and pytube packages are left out:** they are only needed to download the video and transcribe it. The project already includes a ready-made `transcription.txt`, so we skip that part. This also means you do not need to install ffmpeg.

---

## 6. Set up your API keys

Create a file named `.env` in the project folder (note the dot at the start of the name).

**Using OpenAI:**

```text
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

**Using Groq instead (free tier available):**

```text
OPENAI_API_KEY=gsk_your_groq_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

The variable name stays `OPENAI_API_KEY` because the code uses OpenAI-compatible tools, but the value is your Groq key.

For embeddings, add one more line with a free Gemini key from https://aistudio.google.com/apikey (see section 9):

```text
GOOGLE_API_KEY=your_gemini_key_here
```

> **Do not** put `PINECONE_API_ENV` in the file. Newer Pinecone versions no longer use it.
>
> **Never share or upload your `.env` file.** Add `.env` to `.gitignore` so Git ignores it.
>
> **Windows tip:** if you create the file in Notepad, choose "All files" when saving, otherwise it becomes `.env.txt` and will not work.

---

## 7. Start Jupyter

Install Jupyter and register your environment as a kernel:

```bash
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name youtube-rag --display-name "Python 3.11 (youtube-rag)"
jupyter notebook
```

Open `rag.ipynb` and choose **Kernel → Change kernel → Python 3.11 (youtube-rag)**. If you skip this, the notebook may run on a different Python and show "module not found" errors.

---

## 8. Run the notebook

Run the cells from top to bottom with **Shift + Enter**.

1. **Skip the download and Whisper cells.** Use the provided `transcription.txt` instead.
2. **Create the model.**

   With OpenAI:

   ```python
   from langchain_openai.chat_models import ChatOpenAI
   
   model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
   ```

   With Groq:

   ```python
   model = ChatOpenAI(
       openai_api_key=OPENAI_API_KEY,
       model="openai/gpt-oss-20b",
       base_url="https://api.groq.com/openai/v1",
   )
   ```
3. **Test the model with a simple question:**

   ```python
   model.invoke("What MLB team won the World Series during the COVID-19 pandemic?")
   ```

   If you get an answer, your keys and packages work. The answer text is in the `.content` part of the reply.
4. **Continue through the remaining cells.** They split the transcript, create embeddings, store them in Pinecone, and answer questions using your video.

---

## 9. Embeddings with Gemini (768 dimensions)

Embeddings turn text into numbers so Pinecone can search by meaning. Whichever chat model you use, this project uses Google's `gemini-embedding-001` for embeddings, set to **768** dimensions. Google's older 768-dimension model, `text-embedding-004`, was announced for shutdown on January 14, 2026, so we do not use it.

Install the Gemini wrapper with `python -m pip install langchain-google-genai`. Without the `output_dimensionality` setting below, this model returns 3072 numbers per text, so that setting is what makes it 768. Recent versions of this package need a newer `langchain-core` than the pins in section 5: if pip reports a conflict, update those LangChain pins to the versions that install together.

```python
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()  # loads GOOGLE_API_KEY from your .env

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,
)
embedded_query = embeddings.embed_query("Who is Mary's sister?")

print(f"Embedding length: {len(embedded_query)}")  # should print 768
print(embedded_query[:10])
```

Use `embeddings` everywhere the notebook created its embedding model. The test above should print `768`.

Create your Pinecone index with `dimension=768` and `metric="cosine"`. An index's dimension cannot be changed later, so if you already made one with another size, delete it and create a new one. On the free Gemini tier, a `429` error means you hit the rate limit: wait a minute and run the cell again.

---

## 10. Troubleshooting

| Problem | What it means | Fix |
| --- | --- | --- |
| `externally-managed-environment` | You ran the system pip, not the venv's pip | Use `python -m pip install ...` with the venv active |
| `No module named pip` (in the venv) | The venv was created without pip | Run `python -m ensurepip --upgrade` |
| `unexpected keyword argument 'proxies'` | `httpx` is too new for the old `openai` version | `python -m pip install httpx==0.27.2`, then restart the kernel |
| `AuthenticationError` or 401 | Wrong key, or `.env` not loaded | Check the key and make sure `load_dotenv()` runs first |
| `model_not_found` | Model name is outdated or misspelled | Check the provider's current model list |
| `429` or `insufficient_quota` | No credit left on the account | Add credit or switch provider |
| Pinecone dimension error | Index size does not match embedding size | Recreate the index with the right dimension (768 for Gemini) |
| `ModuleNotFoundError` in the notebook | Notebook is using the wrong kernel | Select **Python 3.11 (youtube-rag)** |
| `py -3.11` not found (Windows) | Python 3.11 is not installed or the terminal is old | Install it, open a new terminal, and run `py -0` to list versions |

---

## 11. Save your working setup

Once everything runs, record the exact versions that worked:

```bash
python -m pip freeze > requirements.lock.txt
```

Keep this file with the project. On a new machine, install the same Python version and run `python -m pip install -r requirements.lock.txt`. You will never have to debug dependencies again.

---

## 12. Try it yourself

- Ask the app three questions about the video, and check the answers against the transcript.
- Change the chunk size when splitting the text. How do the answers change?
- Swap in a different YouTube video's transcript.
- Try a different model and compare the quality and speed.

Happy building!
