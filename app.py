import gradio as gr

from retrieve import hybrid_search, format_context, get_sources, calibrate_confidence, verify_citations
from generate import generate_stream, check
from synthesis import detect_contradictions, synthesize
from memory import Memory

MEM = Memory(max_turns=6)


def memory_context():
    text = MEM.as_text()
    if not text:
        return ""
    return "Previous conversation:\n" + text + "\n\n"


def respond(message, history, mode, top_k):
    history = list(history or [])
    if not message.strip():
        yield history
        return

    ok, _ = check()
    if not ok:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "ollama not running, start it first"})
        yield history
        return

    results = hybrid_search(message, k=int(top_k))
    if not results:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "no chunks found"})
        yield history
        return

    MEM.add("user", message)

    if mode == "compare":
        claims = [{"text": r["text"], "source": r["source"], "page": r.get("page")} for r in results]
        agreements = detect_contradictions(claims, message)
        ans = synthesize(message, claims, agreements)
        ans += f"\n\nsources: {', '.join(get_sources(results))}"
        if agreements:
            ans += f"\nagreements: {len(agreements)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ans})
        MEM.add("assistant", ans)
        yield history
        return

    ctx = memory_context() + format_context(results)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    reply = ""
    for token in generate_stream(ctx, message):
        reply += token
        history[-1] = {"role": "assistant", "content": reply}
        yield list(history)

    conf = calibrate_confidence(results, reply, ctx)
    cites = verify_citations(reply, results)
    tail = f"\n\nconfidence: {conf['label']} ({conf['score']})"
    tail += f"\nsources: {', '.join(get_sources(results))}"
    if cites["total_cited"]:
        tail += f"\ncitations: verified {len(cites['verified'])}, unverified {len(cites['unverified'])}"
    history[-1] = {"role": "assistant", "content": reply + tail}
    MEM.add("assistant", reply)
    yield list(history)


def clear_chat():
    MEM.clear()
    return []


with gr.Blocks(title="local rag chat") as demo:
    gr.Markdown("ask questions about your documents")
    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        mode = gr.Radio(["answer", "compare"], value="answer", label="mode")
        top_k = gr.Slider(2, 10, value=5, step=1, label="chunks")
    msg = gr.Textbox(label="question", placeholder="ask something")
    with gr.Row():
        send = gr.Button("send")
        clear = gr.Button("clear")

    send.click(respond, [msg, chatbot, mode, top_k], chatbot)
    msg.submit(respond, [msg, chatbot, mode, top_k], chatbot)
    clear.click(clear_chat, [], chatbot)

demo.launch()
