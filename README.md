The entire project is integrated under **Agent**

Run the bot from `Agent/` with:

```bash
uv run python -m edimension_agent.server
```

This uses `APP_HOST` and `APP_PORT` from `.env`.

The **RAG_Chatbot** folder was for chatbot/embedding model testing

**Models Tested**
| Chatbot Models   | Embedding Model        |
|------------------|------------------------|
| Llama 3.2 (3B)   | all MiniLM L6 v2       |
| Ministral 3 (8B) | QWen3 Embedding (0.6B) |

Current setup uses Ministral 3 and QWen3 Embedding.

Llama 3.2 was faster, but had unsatisfactory answers. all MiniLM L6 v2 was switched out because it seemed to not work so well with the chunks (I suspect it's because its context window was too small)
