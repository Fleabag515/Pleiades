"""Email tool: the character's own inbox over IMAP/SMTP.

Credentials come from the profile (hosts/ports + address) and the vault (password).
Designed so `list_unread`/`read` return clean body text — useful for reading account
verification codes the character receives when it signs up for services.
"""

from __future__ import annotations

import email
import imaplib
import smtplib
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Optional

from . import Tool, ToolContext


def _decode(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def _body_text(msg: email.message.Message) -> str:
    """Extract a readable plain-text body, falling back to stripped HTML."""
    if msg.is_multipart():
        plain, html = None, None
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True)
            except Exception:
                continue
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if ctype == "text/plain" and plain is None:
                plain = text
            elif ctype == "text/html" and html is None:
                html = text
        if plain:
            return plain.strip()
        if html:
            return _strip_html(html).strip()
        return ""
    payload = msg.get_payload(decode=True)
    if payload is None:
        return str(msg.get_payload())
    charset = msg.get_content_charset() or "utf-8"
    text = payload.decode(charset, errors="replace")
    if msg.get_content_type() == "text/html":
        return _strip_html(text).strip()
    return text.strip()


def _strip_html(html: str) -> str:
    import re

    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"\s+", " ", html)
    return html


# --------------------------------------------------------------------------- #
# Reusable IMAP/SMTP operations (used by the EmailTool AND the web UI)
# --------------------------------------------------------------------------- #
def imap_connect(imap_host: str, imap_port: int, address: str, password: str,
                 timeout: float = 15.0):
    conn = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=timeout)
    conn.login(address, password)
    return conn


def list_messages(conn, criteria="ALL", limit: int = 20) -> list[dict]:
    """Newest-first [{id, from, subject, date, unread}] from INBOX."""
    conn.select("INBOX")
    typ, unseen_data = conn.search(None, "UNSEEN")
    unseen = set(unseen_data[0].split()) if typ == "OK" and unseen_data and unseen_data[0] else set()
    if isinstance(criteria, tuple):
        typ, data = conn.search(None, *criteria)
    else:
        typ, data = conn.search(None, criteria)
    if typ != "OK":
        return []
    ids = data[0].split()
    out = []
    for mid in ids[-limit:][::-1]:
        typ, msg_data = conn.fetch(mid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
        if typ != "OK" or not msg_data or not msg_data[0]:
            continue
        hdr = email.message_from_bytes(msg_data[0][1])
        out.append({"id": mid.decode(), "from": _decode(hdr.get("From")),
                    "subject": _decode(hdr.get("Subject")) or "(no subject)",
                    "date": _decode(hdr.get("Date")), "unread": mid in unseen})
    return out


def read_message(conn, mid: str) -> dict:
    """Full message {from, to, date, subject, body} (marks it as read)."""
    conn.select("INBOX")
    typ, msg_data = conn.fetch(str(mid).encode(), "(RFC822)")
    if typ != "OK" or not msg_data or not msg_data[0]:
        raise RuntimeError(f"could not read message {mid}")
    msg = email.message_from_bytes(msg_data[0][1])
    return {"id": str(mid), "from": _decode(msg.get("From")),
            "to": _decode(msg.get("To")), "date": _decode(msg.get("Date")),
            "subject": _decode(msg.get("Subject")) or "(no subject)",
            "body": _body_text(msg)}


def send_message(smtp_host: str, smtp_port: int, address: str, password: str,
                 to: str, subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["From"] = address
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    if smtp_port == 465:
        server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20.0)
    else:
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=20.0)
        server.starttls()
    with server:
        server.login(address, password)
        server.send_message(msg)


class EmailTool(Tool):
    name = "email"
    description = (
        "Read and send email from your own inbox. Actions: 'list_unread' (recent unread "
        "messages), 'read' (full body of a message by id), 'send' (compose to/subject/body), "
        "'search' (find messages matching a query). Useful for receiving verification codes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_unread", "read", "send", "search"],
            },
            "id": {"type": "string", "description": "Message id (for 'read')."},
            "to": {"type": "string", "description": "Recipient address (for 'send')."},
            "subject": {"type": "string", "description": "Subject (for 'send')."},
            "body": {"type": "string", "description": "Body text (for 'send')."},
            "query": {"type": "string", "description": "Search text (for 'search')."},
            "limit": {"type": "integer", "description": "Max messages to return.", "default": 10},
        },
        "required": ["action"],
    }

    # -- credentials -------------------------------------------------------- #
    @staticmethod
    def _creds(ctx: ToolContext):
        profile = ctx.profile
        address = ctx.vault.get("email.address") or profile.email_address
        password = ctx.vault.get("email.password")
        if not (address and password and profile.imap_host and profile.smtp_host):
            raise RuntimeError(
                "Email is not fully configured for this character "
                "(need address, password, and IMAP/SMTP hosts)."
            )
        return address, password

    # -- dispatch ----------------------------------------------------------- #
    def run(self, ctx: ToolContext, action: str, **kw) -> str:
        action = action.lower().strip()
        if action == "list_unread":
            return self._list(ctx, criteria="UNSEEN", limit=int(kw.get("limit", 10)))
        if action == "search":
            q = kw.get("query", "")
            if not q:
                return "[email error] 'search' needs a 'query'."
            # IMAP requires quoting for queries containing spaces/specials.
            quoted = '"' + q.replace("\\", "\\\\").replace('"', '\\"') + '"'
            return self._list(ctx, criteria=("TEXT", quoted), limit=int(kw.get("limit", 10)))
        if action == "read":
            if not kw.get("id"):
                return "[email error] 'read' needs an 'id'."
            return self._read(ctx, kw["id"])
        if action == "send":
            return self._send(ctx, kw.get("to", ""), kw.get("subject", ""), kw.get("body", ""))
        return f"[email error] unknown action '{action}'."

    # -- IMAP --------------------------------------------------------------- #
    def _imap(self, ctx: ToolContext):
        address, password = self._creds(ctx)
        conn = imaplib.IMAP4_SSL(ctx.profile.imap_host, ctx.profile.imap_port,
                                 timeout=15.0)
        conn.login(address, password)
        return conn

    def _list(self, ctx: ToolContext, criteria, limit: int) -> str:
        try:
            conn = self._imap(ctx)
        except Exception as e:
            return f"[email error] {e}"
        try:
            conn.select("INBOX")
            if isinstance(criteria, tuple):
                typ, data = conn.search(None, *criteria)
            else:
                typ, data = conn.search(None, criteria)
            if typ != "OK":
                return "[email error] search failed."
            ids = data[0].split()
            if not ids:
                return "No matching messages."
            ids = ids[-limit:][::-1]  # newest first
            lines = []
            for mid in ids:
                typ, msg_data = conn.fetch(mid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue
                hdr = email.message_from_bytes(msg_data[0][1])
                lines.append(
                    f"id={mid.decode()} | from: {_decode(hdr.get('From'))} | "
                    f"subject: {_decode(hdr.get('Subject'))} | {_decode(hdr.get('Date'))}"
                )
            return "\n".join(lines) if lines else "No matching messages."
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    def _read(self, ctx: ToolContext, mid: str) -> str:
        try:
            conn = self._imap(ctx)
        except Exception as e:
            return f"[email error] {e}"
        try:
            conn.select("INBOX")
            typ, msg_data = conn.fetch(str(mid).encode(), "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                return f"[email error] could not read message {mid}."
            msg = email.message_from_bytes(msg_data[0][1])
            header = (
                f"From: {_decode(msg.get('From'))}\n"
                f"To: {_decode(msg.get('To'))}\n"
                f"Date: {_decode(msg.get('Date'))}\n"
                f"Subject: {_decode(msg.get('Subject'))}\n"
            )
            return header + "\n" + _body_text(msg)
        finally:
            try:
                conn.logout()
            except Exception:
                pass

    # -- SMTP --------------------------------------------------------------- #
    def _send(self, ctx: ToolContext, to: str, subject: str, body: str) -> str:
        if not to or not parseaddr(to)[1]:
            return "[email error] 'send' needs a valid 'to' address."
        try:
            address, password = self._creds(ctx)
        except Exception as e:
            return f"[email error] {e}"

        msg = EmailMessage()
        msg["From"] = address
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        host, port = ctx.profile.smtp_host, ctx.profile.smtp_port
        try:
            if port == 465:
                server = smtplib.SMTP_SSL(host, port)
            else:
                server = smtplib.SMTP(host, port)
                server.starttls()
            with server:
                server.login(address, password)
                server.send_message(msg)
        except Exception as e:
            return f"[email error] send failed: {e}"
        return f"Sent email to {to}."
