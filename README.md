# Gemini STORM Extension 🌪️

A unique implementation of the **Stanford STORM** research engine as a **Gemini CLI Extension**.

## 🚀 The "Nested Gemini" Magic
This extension solves the common problem of complex Python libraries (like STORM) struggling with Gemini API authentication or project-specific Vertex AI setups.

Instead of direct API calls, this extension uses a **Nested Gemini Wrapper**. It launches a headless `gemini run` process from within the Python engine, leveraging your already authenticated CLI environment.

**Research power with Zero Auth friction.**

## 🛠️ Features
- **Deep Research:** Generates full, Wikipedia-style articles on any topic.
- **Background Execution:** STORM runs in the background while you keep using the CLI.
- **Progress Monitoring:** Dedicate agent tracks research phases in real-time.
- **Zero Config:** Uses your existing Gemini CLI session and Vertex AI setup.

## 📦 Installation

```bash
gemini extensions install https://github.com/ZoloZiak/gemini-storm
```

## ⌨️ Usage

- `/deep-research "Topic Name"`: Starts a new research task.
- `/deep-research:status`: Checks if STORM is still running and what it's doing.
- `/deep-research:view`: Summarizes the latest generated article.

## ⚠️ Requirements
1. **Python 3.10+**
2. **Stanford STORM:** `pip install stanford-storm`
3. **You.com API Key:** Required for web search. Set it in your environment:
   `export YOU_API_KEY='your_key_here'`

## 📝 License
MIT
