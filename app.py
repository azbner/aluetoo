from __future__ import annotations

import asyncio
import base64
import html
import json
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from groq import Groq
from nicegui import app, events, ui


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "aluetoo_chats.json"
TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_IMAGE_B64_BYTES = 4 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 5

SYSTEM = """Tu es ALUETOO AI, creee par Leo Ciach.
Tu es premium, rapide, claire, utile et ethique.
- Tu reponds dans la langue de l'utilisateur.
- Tu decomposes les problemes quand c'est utile, sans exposer de raisonnement interne cache.
- Tu utilises des listes lisibles quand cela clarifie la reponse.
- Tu signales les incertitudes importantes.
- Tu cites les sources sous forme de liens cliquables quand tu t'appuies sur des informations externes fournies ou demandees.
- Tu respectes la confidentialite: ne demande pas de donnees sensibles inutiles et ne pretends jamais memoriser hors de cette conversation.
- Si l'utilisateur joint une image, analyse-la directement et precisement.
- Si l'utilisateur demande une image generee, explique que Groq sert ici a generer le prompt/specification d'image, pas l'image bitmap elle-meme, sauf si un fournisseur d'image externe est branche."""


def now_iso() -> str:
    return datetime.now(ZoneInfo("Europe/Brussels")).isoformat(timespec="seconds")


def now_label() -> str:
    return datetime.now(ZoneInfo("Europe/Brussels")).strftime("%H:%M")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_. -]+", "_", name).strip() or "fichier"


def read_api_key() -> str | None:
    if key := os.getenv("GROQ_API_KEY"):
        return key

    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as fh:
            data = tomllib.load(fh)
        value = data.get("GROQ_API_KEY")
        return str(value) if value else None

    return None


def load_store() -> dict[str, Any]:
    if not DATA_FILE.exists():
        return {"conversations": {}, "settings": {"allow_training": False, "dark": True}}
    try:
        with DATA_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        data = {"conversations": {}, "settings": {"allow_training": False, "dark": True}}

    data.setdefault("conversations", {})
    data.setdefault("current_id", None)
    data.setdefault("settings", {})
    data["settings"].setdefault("allow_training", False)
    data["settings"].setdefault("dark", True)
    return data


def save_store() -> None:
    serializable = {
        "conversations": state["conversations"],
        "current_id": state.get("current_id"),
        "settings": state["settings"],
    }
    with DATA_FILE.open("w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)


def title_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message["role"] == "user" and message.get("content"):
            clean = re.sub(r"\s+", " ", message["content"]).strip()
            return clean[:48] or "Nouvelle discussion"
    return "Nouvelle discussion"


def current_conversation() -> dict[str, Any]:
    cid = state.get("current_id")
    if not cid or cid not in state["conversations"]:
        cid = create_conversation()
    return state["conversations"][cid]


def create_conversation(folder: str | None = None) -> str:
    cid = f"chat_{int(time.time())}_{uuid.uuid4().hex[:4]}"
    folder_name = (folder or state.get("folder_filter") or "General").strip()
    if folder_name == "Tous":
        folder_name = "General"
    state["conversations"][cid] = {
        "id": cid,
        "title": "Nouvelle discussion",
        "folder": folder_name,
        "created": now_iso(),
        "updated": now_iso(),
        "messages": [],
    }
    state["current_id"] = cid
    save_store()
    return cid


def conversation_files(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for message in conversation.get("messages", []):
        files.extend(message.get("attachments", []))
    return files


def selected_assistant_content(message: dict[str, Any]) -> str:
    versions = message.get("versions") or [{"content": message.get("content", "")}]
    index = max(0, min(message.get("version_index", 0), len(versions) - 1))
    return versions[index].get("content", "")


def set_assistant_content(message: dict[str, Any], content: str) -> None:
    versions = message.setdefault("versions", [{"content": message.get("content", ""), "created": now_iso(), "artifacts": []}])
    index = max(0, min(message.get("version_index", 0), len(versions) - 1))
    versions[index]["content"] = content
    message["content"] = content


def add_assistant_version(message: dict[str, Any], content: str, artifacts: list[dict[str, Any]] | None = None) -> None:
    versions = message.setdefault("versions", [])
    versions.append({"content": content, "created": now_iso(), "artifacts": artifacts or []})
    message["version_index"] = len(versions) - 1
    message["content"] = content


def build_attachment(e: events.UploadEventArguments) -> dict[str, Any]:
    raw = e.content.read()
    mime = e.type or mimetypes.guess_type(e.name)[0] or "application/octet-stream"
    name = safe_filename(e.name)
    item: dict[str, Any] = {
        "id": new_id("file"),
        "name": name,
        "mime": mime,
        "size": len(raw),
        "created": now_iso(),
    }

    if mime.startswith("image/"):
        item["kind"] = "image"
        item["b64"] = base64.b64encode(raw).decode("utf-8")
        item["data_url"] = f"data:{mime};base64,{item['b64']}"
    elif mime.startswith("text/") or name.lower().endswith((".txt", ".csv", ".md", ".json", ".py", ".js", ".css", ".html")):
        item["kind"] = "text"
        item["text"] = raw.decode("utf-8", errors="replace")[:60_000]
    elif name.lower().endswith(".pdf"):
        item["kind"] = "pdf"
        item["text"] = extract_pdf_text(raw)
    else:
        item["kind"] = "file"

    return item


def extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
        from io import BytesIO

        reader = PdfReader(BytesIO(raw))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages[:12])
        return text[:60_000]
    except Exception:
        return ""


def attachments_to_prompt_text(attachments: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for attachment in attachments:
        name = attachment["name"]
        if attachment.get("kind") == "text" and attachment.get("text"):
            chunks.append(f"\n\n[Fichier texte: {name}]\n{attachment['text']}")
        elif attachment.get("kind") == "pdf":
            if attachment.get("text"):
                chunks.append(f"\n\n[PDF extrait: {name}]\n{attachment['text']}")
            else:
                chunks.append(f"\n\n[PDF joint: {name}. Texte non extrait: installe pypdf pour l'analyse textuelle locale.]")
        elif attachment.get("kind") == "file":
            chunks.append(f"\n\n[Fichier joint: {name}, type {attachment.get('mime', 'inconnu')}.]")
    return "".join(chunks)


def has_images(messages: list[dict[str, Any]]) -> bool:
    return any(
        attachment.get("kind") == "image"
        for message in messages
        for attachment in message.get("attachments", [])
    )


def build_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    api_messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    if not state["settings"].get("allow_training", False):
        api_messages.append({
            "role": "system",
            "content": "Mode confidentialite active: limite la retention contextuelle au strict necessaire et rappelle les risques si l'utilisateur partage des donnees sensibles.",
        })

    images_used = 0
    for message in messages:
        role = message["role"]
        if role == "system":
            api_messages.append({"role": "system", "content": message.get("content", "")})
            continue

        content = selected_assistant_content(message) if role == "assistant" else message.get("content", "")
        attachments = message.get("attachments", [])
        extra_text = attachments_to_prompt_text(attachments)
        images = [
            attachment for attachment in attachments
            if attachment.get("kind") == "image" and attachment.get("b64")
        ]

        if role == "user" and images and images_used < MAX_IMAGES_PER_REQUEST:
            parts: list[dict[str, Any]] = [{"type": "text", "text": content + extra_text}]
            for attachment in images[: MAX_IMAGES_PER_REQUEST - images_used]:
                if len(attachment["b64"].encode("utf-8")) > MAX_IMAGE_B64_BYTES:
                    parts[0]["text"] += f"\n\n[Image ignoree: {attachment['name']} depasse la limite base64 de 4 MB de Groq.]"
                    continue
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": attachment["data_url"]},
                })
                images_used += 1
            api_messages.append({"role": role, "content": parts})
        else:
            api_messages.append({"role": role, "content": content + extra_text})

    return api_messages


def extract_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)>\]\"']+", text)
    return [url.rstrip(".,;:") for url in urls]


def extract_artifacts(text: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for index, match in enumerate(re.finditer(r"```(\w+)?\n(.*?)```", text, flags=re.DOTALL)):
        language = (match.group(1) or "text").lower()
        code = match.group(2).strip()
        if len(code) < 420 and language not in {"html", "python", "javascript", "js", "css", "json", "svg"}:
            continue
        kind = "preview" if language in {"html", "svg"} or "<!doctype html" in code.lower() else "code"
        artifacts.append({
            "id": new_id("artifact"),
            "title": f"{language.upper()} #{index + 1}",
            "kind": kind,
            "language": language,
            "content": code,
            "created": now_iso(),
        })

    if not artifacts and len(text) > 3_000:
        artifacts.append({
            "id": new_id("artifact"),
            "title": "Longue reponse",
            "kind": "markdown",
            "language": "markdown",
            "content": text,
            "created": now_iso(),
        })

    return artifacts


def clean_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " bloc de code ", text, flags=re.DOTALL)
    text = re.sub(r"[#*_>`\[\]()]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def client() -> Groq | None:
    key = read_api_key()
    return Groq(api_key=key) if key else None


def stream_completion(messages: list[dict[str, Any]], stop_event: asyncio.Event) -> Any:
    groq_client = client()
    if groq_client is None:
        raise RuntimeError("Aucune cle GROQ_API_KEY trouvee. Ajoute-la en variable d'environnement ou dans .streamlit/secrets.toml.")

    model = VISION_MODEL if has_images(messages) else TEXT_MODEL
    return groq_client.chat.completions.create(
        model=model,
        messages=build_api_messages(messages),
        temperature=0.7,
        stream=True,
    )


async def next_chunk(iterator: Any) -> Any:
    sentinel = object()

    def _next() -> Any:
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    chunk = await asyncio.to_thread(_next)
    return None if chunk is sentinel else chunk


state = load_store()
state.setdefault("current_id", None)
state.setdefault("pending_files", [])
state.setdefault("folder_filter", "Tous")
state.setdefault("search", "")
state.setdefault("selected_artifact", None)
state.setdefault("stop_event", None)
state.setdefault("is_generating", False)
state["settings"].setdefault("sidebar_open", False)
state["settings"].setdefault("context_open", False)
state["settings"]["dark"] = True
state["settings"]["sidebar_open"] = False
state["settings"]["context_open"] = False

if not state["conversations"]:
    create_conversation("General")
elif state.get("current_id") not in state["conversations"]:
    state["current_id"] = max(
        state["conversations"].values(),
        key=lambda conv: conv.get("updated", conv.get("created", "")),
    )["id"]
    save_store()

app.add_static_files("/assets", ROOT, max_cache_age=0)


@ui.page("/")
def main_page() -> None:
    dark = ui.dark_mode(value=state["settings"].get("dark", True))

    ui.add_head_html('<link rel="stylesheet" href="/assets/aluetoo_nicegui.css?v=liquid4">')
    ui.add_head_html("""
    <script>
    window.aluCopy = async (text) => {
      await navigator.clipboard.writeText(text || "");
    };
    window.aluSpeak = (text) => {
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text || "");
      utterance.lang = navigator.language || "fr-FR";
      speechSynthesis.speak(utterance);
    };
    window.aluStopSpeak = () => speechSynthesis.cancel();
    window.aluStartDictation = () => {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        alert("La dictee vocale n'est pas disponible dans ce navigateur.");
        return;
      }
      const rec = new SpeechRecognition();
      rec.lang = navigator.language || "fr-FR";
      rec.interimResults = false;
      rec.onresult = (event) => {
        const text = event.results[0][0].transcript;
        const area = document.querySelector(".alu-prompt textarea");
        if (!area) return;
        area.value = text;
        area.dispatchEvent(new Event("input", {bubbles: true}));
        area.dispatchEvent(new Event("change", {bubbles: true}));
      };
      rec.start();
    };
    window.aluClickUpload = () => {
      const input = document.querySelector(".alu-upload input[type=file]");
      if (input) input.click();
    };
    window.aluCopyCodeButtons = () => {
      document.querySelectorAll(".message-markdown pre:not([data-copy-ready])").forEach((pre) => {
        pre.dataset.copyReady = "true";
        const btn = document.createElement("button");
        btn.className = "copy-code-btn";
        btn.textContent = "Copier";
        btn.onclick = async () => {
          await navigator.clipboard.writeText(pre.innerText || "");
          btn.textContent = "Copie";
          setTimeout(() => btn.textContent = "Copier", 900);
        };
        pre.appendChild(btn);
      });
    };
    setInterval(window.aluCopyCodeButtons, 700);
    </script>
    """)

    refs: dict[str, Any] = {}

    def shell_classes() -> str:
        classes = ["alu-shell"]
        if state["settings"].get("sidebar_open", False):
            classes.append("left-open")
        if state["settings"].get("context_open", False):
            classes.append("right-open")
        if state.get("is_generating"):
            classes.append("is-thinking")
        return " ".join(classes)

    def update_shell_classes() -> None:
        if shell := refs.get("shell"):
            shell.classes(replace=shell_classes())

    def persist_current() -> None:
        conv = current_conversation()
        conv["title"] = title_from_messages(conv["messages"])
        conv["updated"] = now_iso()
        save_store()

    def refresh_all() -> None:
        update_shell_classes()
        render_sidebar()
        render_chat()
        render_context()

    def select_chat(cid: str) -> None:
        state["current_id"] = cid
        state["selected_artifact"] = None
        refresh_all()

    def new_chat() -> None:
        create_conversation(state.get("folder_filter") or "General")
        state["selected_artifact"] = None
        refresh_all()

    def toggle_sidebar() -> None:
        state["settings"]["sidebar_open"] = not state["settings"].get("sidebar_open", False)
        save_store()
        update_shell_classes()

    def toggle_context() -> None:
        state["settings"]["context_open"] = not state["settings"].get("context_open", False)
        save_store()
        update_shell_classes()

    def set_folder_filter(value: str | None) -> None:
        state["folder_filter"] = value or "Tous"
        render_sidebar()

    def set_search(value: str | None) -> None:
        state["search"] = value or ""
        render_sidebar()

    def apply_theme(value: bool) -> None:
        state["settings"]["dark"] = value
        dark.value = value
        save_store()

    def apply_privacy(value: bool) -> None:
        state["settings"]["allow_training"] = value
        save_store()
        ui.notify("Parametre de confidentialite mis a jour", type="positive")

    async def copy_text(text: str) -> None:
        await ui.run_javascript(f"window.aluCopy({json.dumps(text)})")
        ui.notify("Copie dans le presse-papiers", type="positive")

    async def speak_text(text: str) -> None:
        await ui.run_javascript(f"window.aluSpeak({json.dumps(clean_for_speech(text))})")

    def set_prompt(text: str) -> None:
        refs["prompt"].value = text
        refs["prompt"].update()

    async def stop_generation() -> None:
        event = state.get("stop_event")
        if event:
            event.set()
            ui.notify("Generation arretee", type="warning")

    async def handle_upload(e: events.UploadEventArguments) -> None:
        attachment = build_attachment(e)
        state["pending_files"].append(attachment)
        state["settings"]["context_open"] = True
        save_store()
        update_shell_classes()
        render_context()
        ui.notify(f"{attachment['name']} ajoute au contexte", type="positive")

    async def open_context_upload() -> None:
        state["settings"]["context_open"] = True
        save_store()
        update_shell_classes()
        render_context()
        await ui.run_javascript("setTimeout(() => window.aluClickUpload(), 120)")

    async def send_current_prompt() -> None:
        prompt = (refs["prompt"].value or "").strip()
        if not prompt and not state["pending_files"]:
            return

        refs["prompt"].value = ""
        refs["prompt"].update()
        await append_user_and_generate(prompt, list(state["pending_files"]))
        state["pending_files"].clear()

    async def append_user_and_generate(prompt: str, attachments: list[dict[str, Any]]) -> None:
        conv = current_conversation()
        conv["messages"].append({
            "id": new_id("msg"),
            "role": "user",
            "content": prompt,
            "attachments": attachments,
            "created": now_iso(),
        })
        persist_current()
        refresh_all()
        await generate_response()

    async def generate_response(target_index: int | None = None, as_new_version: bool = False) -> None:
        if state.get("is_generating"):
            ui.notify("Une generation est deja en cours", type="warning")
            return

        conv = current_conversation()
        if target_index is None:
            assistant_message = {
                "id": new_id("msg"),
                "role": "assistant",
                "content": "",
                "versions": [{"content": "", "created": now_iso(), "artifacts": []}],
                "version_index": 0,
                "created": now_iso(),
                "feedback": None,
            }
            conv["messages"].append(assistant_message)
            target_index = len(conv["messages"]) - 1
            context_messages = conv["messages"][:target_index]
        else:
            assistant_message = conv["messages"][target_index]
            if as_new_version:
                add_assistant_version(assistant_message, "")
            else:
                set_assistant_content(assistant_message, "")
            context_messages = conv["messages"][:target_index]

        state["is_generating"] = True
        stop_event = asyncio.Event()
        state["stop_event"] = stop_event
        refresh_all()

        status = refs.get("status")
        if status:
            status.text = "Analyse du contexte..." if has_images(context_messages) else "ALUETOO reflechit..."
            status.update()

        full = ""
        try:
            iterator = await asyncio.to_thread(stream_completion, context_messages, stop_event)
            if status:
                status.text = "Generation en streaming..."
                status.update()

            while not stop_event.is_set():
                chunk = await next_chunk(iterator)
                if chunk is None:
                    break
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                full += delta
                set_assistant_content(assistant_message, full + "|")
                render_chat()
                await asyncio.sleep(0)

            if stop_event.is_set() and full:
                full += "\n\n_Generation interrompue par l'utilisateur._"

            artifacts = extract_artifacts(full)
            versions = assistant_message.setdefault("versions", [])
            selected = max(0, min(assistant_message.get("version_index", 0), len(versions) - 1))
            if versions:
                versions[selected]["artifacts"] = artifacts
            set_assistant_content(assistant_message, full)
            if artifacts:
                state["selected_artifact"] = artifacts[0]["id"]

            persist_current()
        except Exception as exc:
            error_text = f"Erreur: {exc}"
            set_assistant_content(assistant_message, error_text)
            ui.notify(error_text, type="negative", timeout=6000)
        finally:
            state["is_generating"] = False
            state["stop_event"] = None
            refresh_all()

    async def regenerate(index: int) -> None:
        conv = current_conversation()
        if index <= 0 or index >= len(conv["messages"]):
            return
        await generate_response(target_index=index, as_new_version=True)

    def switch_version(message: dict[str, Any], direction: int) -> None:
        versions = message.get("versions") or []
        if not versions:
            return
        message["version_index"] = max(0, min(message.get("version_index", 0) + direction, len(versions) - 1))
        message["content"] = selected_assistant_content(message)
        artifacts = versions[message["version_index"]].get("artifacts") or []
        state["selected_artifact"] = artifacts[0]["id"] if artifacts else None
        save_store()
        refresh_all()

    async def edit_user_message(index: int) -> None:
        conv = current_conversation()
        message = conv["messages"][index]
        with ui.dialog() as dialog, ui.card().classes("alu-dialog"):
            ui.label("Modifier la requete").classes("text-h6")
            editor = ui.textarea(value=message.get("content", "")).classes("w-full alu-input")
            editor.props("outlined autogrow")
            ui.label("En relancant, les messages apres cette requete seront remplaces.").classes("alu-mini")
            with ui.row().classes("justify-end w-full"):
                ui.button("Annuler", on_click=dialog.close).props("flat")

                async def save_and_run() -> None:
                    message["content"] = (editor.value or "").strip()
                    conv["messages"] = conv["messages"][: index + 1]
                    persist_current()
                    dialog.close()
                    refresh_all()
                    await generate_response()

                ui.button("Enregistrer et relancer", on_click=save_and_run).props("unelevated color=primary")
        dialog.open()

    def set_feedback(message: dict[str, Any], value: str) -> None:
        message["feedback"] = value
        save_store()
        ui.notify("Feedback enregistre", type="positive")
        render_chat()

    def render_sidebar() -> None:
        refs["sidebar"].clear()
        with refs["sidebar"]:
            with ui.row().classes("alu-panel-head"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Discussions").classes("alu-sidebar-title")
                ui.button("Fermer", on_click=toggle_sidebar).props("flat dense no-caps").classes("alu-mini-action")

            ui.button("Nouveau chat", on_click=new_chat).props("unelevated no-caps color=primary").classes("w-full alu-primary-action")

            ui.input(
                placeholder="Rechercher...",
                value=state.get("search", ""),
                on_change=lambda e: set_search(e.value),
            ).props("outlined dense clearable").classes("w-full alu-input")

            folders = sorted({conv.get("folder", "General") for conv in state["conversations"].values()})
            options = ["Tous", *folders]
            ui.select(
                options,
                value=state.get("folder_filter", "Tous") if state.get("folder_filter", "Tous") in options else "Tous",
                on_change=lambda e: set_folder_filter(e.value),
                label="Dossier",
            ).props("outlined dense").classes("w-full alu-input")

            current = current_conversation()
            ui.input(
                "Dossier du chat courant",
                value=current.get("folder", "General"),
                on_change=lambda e: update_current_folder(e.value),
            ).props("outlined dense").classes("w-full alu-input")

            ui.separator()
            ui.label("Historique").classes("alu-panel-title")

            search = state.get("search", "").lower()
            folder_filter = state.get("folder_filter", "Tous")
            conversations = sorted(
                state["conversations"].values(),
                key=lambda c: c.get("updated", c.get("created", "")),
                reverse=True,
            )
            visible = [
                conv for conv in conversations
                if (folder_filter == "Tous" or conv.get("folder") == folder_filter)
                and (not search or search in conv.get("title", "").lower())
            ]

            if not visible:
                ui.label("Aucune discussion trouvee.").classes("alu-empty")

            for conv in visible:
                classes = "alu-history-item"
                if conv["id"] == state["current_id"]:
                    classes += " alu-active"
                ui.button(
                    conv.get("title") or "Nouvelle discussion",
                    on_click=lambda cid=conv["id"]: select_chat(cid),
                ).props("flat no-caps align=left").classes(classes)

            ui.separator()
            ui.toggle(
                {False: "Prive", True: "Autoriser personnalisation"},
                value=state["settings"].get("allow_training", False),
                on_change=lambda e: apply_privacy(e.value),
            ).classes("w-full")

            ui.switch(
                "Mode sombre",
                value=state["settings"].get("dark", True),
                on_change=lambda e: apply_theme(e.value),
            )
            ui.label("Raccourcis: /sources, /image, /code, /resume").classes("alu-mini")

    def update_current_folder(value: str | None) -> None:
        conv = current_conversation()
        conv["folder"] = (value or "General").strip() or "General"
        conv["updated"] = now_iso()
        save_store()
        render_sidebar()

    def render_chat() -> None:
        refs["chat"].clear()
        with refs["chat"]:
            conv = current_conversation()
            if not conv["messages"]:
                with ui.column().classes("alu-welcome"):
                    ui.label("Pret quand tu l'es.").classes("alu-welcome-title")
                    ui.label("Pose une question, glisse une image ou demande un artefact.").classes("alu-mini")

            for index, message in enumerate(conv["messages"]):
                role = message["role"]
                classes = f"message-row {role}"
                with ui.column().classes(classes):
                    bubble_class = {
                        "user": "msg-user",
                        "assistant": "msg-ai",
                        "system": "msg-system",
                    }.get(role, "msg-ai")
                    with ui.element("div").classes(f"message-bubble {bubble_class}"):
                        content = selected_assistant_content(message) if role == "assistant" else message.get("content", "")
                        ui.markdown(content or " ").classes("message-markdown")
                        for attachment in message.get("attachments", []):
                            render_attachment_chip(attachment)

                    with ui.row().classes("message-toolbar"):
                        content = selected_assistant_content(message) if role == "assistant" else message.get("content", "")
                        ui.button("Copier", on_click=lambda text=content: copy_text(text)).props("flat dense no-caps").classes("alu-mini-action")
                        if role == "assistant":
                            ui.button("Lire", on_click=lambda text=content: speak_text(text)).props("flat dense no-caps").classes("alu-mini-action")
                            ui.button("Stop audio", on_click=lambda: ui.run_javascript("window.aluStopSpeak()")).props("flat dense no-caps").classes("alu-mini-action")
                            ui.button("Regenerer", on_click=lambda idx=index: regenerate(idx)).props("flat dense no-caps").classes("alu-mini-action")
                            ui.button("+", on_click=lambda m=message: set_feedback(m, "up")).props("flat dense no-caps").classes("alu-mini-action").tooltip("Bien")
                            ui.button("-", on_click=lambda m=message: set_feedback(m, "down")).props("flat dense no-caps").classes("alu-mini-action").tooltip("A ameliorer")
                            versions = message.get("versions", [])
                            if len(versions) > 1:
                                ui.button("<", on_click=lambda m=message: switch_version(m, -1)).props("flat dense no-caps").classes("alu-mini-action").tooltip("Version precedente")
                                ui.label(f"{message.get('version_index', 0) + 1}/{len(versions)}").classes("alu-mini")
                                ui.button(">", on_click=lambda m=message: switch_version(m, 1)).props("flat dense no-caps").classes("alu-mini-action").tooltip("Version suivante")
                        elif role == "user":
                            ui.button("Modifier", on_click=lambda idx=index: edit_user_message(idx)).props("flat dense no-caps").classes("alu-mini-action")

                        urls = extract_urls(content)
                        if urls:
                            for n, url in enumerate(urls[:3], start=1):
                                ui.link(f"Source {n}", url, new_tab=True).classes("alu-pill")

            if state.get("is_generating"):
                with ui.row().classes("alu-status-bar"):
                    ui.element("div").classes("alu-status-dot")
                    refs["status"] = ui.label("Generation en cours...")
                    ui.button("Stop", on_click=stop_generation).props("dense unelevated color=negative")
            else:
                refs["status"] = None

    def render_attachment_chip(attachment: dict[str, Any]) -> None:
        with ui.column().classes("alu-attachment"):
            if attachment.get("kind") == "image" and attachment.get("data_url"):
                ui.image(attachment["data_url"]).classes("alu-context-thumb")
            ui.label(f"{attachment['name']} - {round(attachment.get('size', 0) / 1024, 1)} Ko").classes("alu-mini")

    def artifacts_for_current() -> list[dict[str, Any]]:
        conv = current_conversation()
        artifacts: list[dict[str, Any]] = []
        for message in conv.get("messages", []):
            if message.get("role") != "assistant":
                continue
            versions = message.get("versions") or []
            if not versions:
                continue
            index = max(0, min(message.get("version_index", 0), len(versions) - 1))
            artifacts.extend(versions[index].get("artifacts") or [])
        return artifacts

    def select_artifact(artifact_id: str) -> None:
        state["selected_artifact"] = artifact_id
        render_context()

    def render_context() -> None:
        refs["context"].clear()
        with refs["context"]:
            with ui.row().classes("alu-panel-head"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("Fichiers").classes("alu-sidebar-title")
                ui.button("Fermer", on_click=toggle_context).props("flat dense no-caps").classes("alu-mini-action")

            with ui.column().classes("alu-upload w-full"):
                ui.upload(
                    multiple=True,
                    auto_upload=True,
                    max_file_size=20 * 1024 * 1024,
                    on_upload=handle_upload,
                    label="Glisse fichiers ici",
                ).props("accept=.png,.jpg,.jpeg,.webp,.txt,.csv,.md,.json,.py,.js,.css,.html,.pdf")
            ui.button("Joindre un fichier", on_click=open_context_upload).props("flat no-caps").classes("alu-mini-action")

            pending = state.get("pending_files", [])
            if pending:
                ui.label("A envoyer").classes("alu-panel-title")
                for attachment in pending:
                    with ui.column().classes("alu-context-item"):
                        render_attachment_chip(attachment)

            files = conversation_files(current_conversation())
            ui.label("Memoire visuelle").classes("alu-panel-title")
            if not files:
                ui.label("Les documents et images du chat apparaitront ici.").classes("alu-empty")
            for attachment in files[-8:]:
                with ui.column().classes("alu-context-item"):
                    render_attachment_chip(attachment)

            ui.separator()
            ui.label("Artefacts").classes("alu-panel-title")
            artifacts = artifacts_for_current()
            if not artifacts:
                ui.label("Le code long, HTML ou contenu lourd sera isole ici.").classes("alu-empty")
            for artifact in artifacts:
                classes = "alu-artifact-item"
                if artifact["id"] == state.get("selected_artifact"):
                    classes += " alu-active"
                ui.button(
                    artifact["title"],
                    on_click=lambda aid=artifact["id"]: select_artifact(aid),
                ).props("flat no-caps align=left").classes(classes)

            selected = next((a for a in artifacts if a["id"] == state.get("selected_artifact")), None)
            if selected:
                ui.separator()
                with ui.row().classes("items-center justify-between w-full"):
                    ui.label(selected["title"]).classes("alu-panel-title")
                    ui.button("Copier", on_click=lambda text=selected["content"]: copy_text(text)).props("flat dense no-caps").classes("alu-mini-action")
                if selected["kind"] == "preview":
                    escaped = html.escape(selected["content"], quote=True)
                    ui.html(f'<iframe class="artifact-preview" sandbox="allow-scripts" srcdoc="{escaped}"></iframe>', sanitize=False)
                elif selected["kind"] == "markdown":
                    ui.markdown(selected["content"]).classes("message-markdown")
                else:
                    ui.markdown(f"```{selected['language']}\n{selected['content']}\n```").classes("artifact-code")

    with ui.element("div").classes(shell_classes()) as shell:
        refs["shell"] = shell
        with ui.element("aside").classes("alu-sidebar") as sidebar:
            refs["sidebar"] = sidebar

        with ui.element("main").classes("alu-main"):
            with ui.element("header").classes("alu-header"):
                with ui.row().classes("alu-topbar"):
                    with ui.row().classes("alu-topbar-side"):
                        ui.button("Chats", on_click=toggle_sidebar).props("flat no-caps").classes("alu-glass-button")
                        ui.button("Nouveau", on_click=new_chat).props("flat no-caps").classes("alu-glass-button alu-new-chat")

                    with ui.column().classes("alu-brand"):
                        ui.html('<div class="alu-brand-mark">AI</div>', sanitize=False)
                        ui.html('<h1 class="alu-title">ALUETOO</h1>', sanitize=False)
                        ui.label(f"Compute - Donnees - Algorithmes - Ethique - {now_label()}").classes("alu-subtitle")

                    with ui.row().classes("alu-topbar-side alu-topbar-right"):
                        ui.button("Fichiers", on_click=toggle_context).props("flat no-caps").classes("alu-glass-button")
                        ui.button("Theme", on_click=lambda: apply_theme(not state["settings"].get("dark", True))).props("flat no-caps").classes("alu-glass-button")

            with ui.scroll_area().classes("alu-chat-scroll"):
                with ui.column().classes("alu-chat-inner") as chat:
                    refs["chat"] = chat

            with ui.element("footer").classes("alu-input-zone"):
                with ui.row().classes("alu-command-panel"):
                    ui.button("/sources", on_click=lambda: set_prompt("/sources Reponds avec des sources verifiables et liens cliquables: ")).props("flat no-caps").classes("alu-command")
                    ui.button("/image", on_click=lambda: set_prompt("/image Cree un prompt d'image ultra precis pour: ")).props("flat no-caps").classes("alu-command")
                    ui.button("/code", on_click=lambda: set_prompt("/code Genere un artefact de code propre pour: ")).props("flat no-caps").classes("alu-command")
                    ui.button("/resume", on_click=lambda: set_prompt("/resume Resume clairement le document ou l'image jointe.")).props("flat no-caps").classes("alu-command")

                with ui.row().classes("alu-composer"):
                    ui.button("Mic", on_click=lambda: ui.run_javascript("window.aluStartDictation()")).props("flat no-caps").classes("alu-composer-action")
                    prompt = ui.textarea(placeholder="Message a ALUETOO...").classes("alu-prompt")
                    prompt.props("autogrow borderless")
                    refs["prompt"] = prompt
                    ui.button("Fichier", on_click=open_context_upload).props("flat no-caps").classes("alu-composer-action")
                    ui.button("Envoyer", on_click=send_current_prompt).props("unelevated no-caps color=primary").classes("alu-send-button")
                    if state.get("is_generating"):
                        ui.button("Stop", on_click=stop_generation).props("flat no-caps color=negative").classes("alu-composer-action")

        with ui.element("aside").classes("alu-context") as context:
            refs["context"] = context

    render_sidebar()
    render_chat()
    render_context()


ui.run(title="ALUETOO AI Pro", reload=False, port=8080)
