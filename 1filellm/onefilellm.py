import os
import re
import sys
import json
import tempfile
import pathlib
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr  # For safe XML attribute escaping
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup, Comment
from PyPDF2 import PdfReader

import nbformat
from nbconvert import PythonExporter

import tiktoken
import nltk
from nltk.corpus import stopwords

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

import pyperclip
import wget

from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

# ----------------------------
# Configuration / constants
# ----------------------------

EXCLUDED_DIRS = ["dist", "node_modules", ".git", "__pycache__"]

ALLOWED_EXTS = {
    # Docs
    ".md", ".mdx", ".txt",
    # JVM / Android
    ".java", ".kt", ".kts", ".gradle", ".gradle.kts", ".pro", ".properties", ".toml", ".lock", ".xml",
    # Dart / Flutter
    ".dart",
    # Minecraft
    ".snbt",
    # Data / schema
    ".json", ".jsonl", ".yaml", ".yml",
    # Web
    ".html", ".js", ".mjs", ".ts", ".mts", ".tsx", ".jsx", ".css",
    # Shell
    ".sh", ".bash", ".zsh",
    # Backends / scripting
    ".go", ".proto", ".py", ".ipynb",
    # Native
    ".rs", ".c", ".cpp", ".h", ".hpp",
    # .NET / others
    ".cs", ".csproj", ".fs", ".vb", ".xaml",
    # Git/env
    ".gitignore", ".gitattributes", ".env",
    # Newly added build / platform specific (requested)
    ".cc", ".mm", ".def", ".rc", ".vcxproj", ".filters", ".sln",
}

# Files with no extension to always include
SPECIAL_FILES = {"makefile", ".gitmodules"}

# Image / binary-like assets to mention but not inline contents (including svg per request to not dump body)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".ico", ".icns", ".bmp", ".gif", ".webp", ".svg"}

# Gate Sci-Hub by default for legal/ToS risks
ALLOW_SCIHUB = os.getenv("ALLOW_SCIHUB") == "1"

# Optional token: only required for GitHub API usage
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}


# ----------------------------
# Utilities
# ----------------------------

def require_token(context: str):
    if not TOKEN:
        raise EnvironmentError(
            f"GITHUB_TOKEN is required to process {context}. Provide it via environment variable."
        )


def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "onefilellm/ci"})
    if TOKEN:
        s.headers.update({"Authorization": f"token {TOKEN}"})
    return s


SESSION = build_session()


def safe_file_read(filepath, fallback_encoding="latin1"):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding=fallback_encoding) as file:
            return file.read()


def try_copy_to_clipboard(text, console: Console):
    try:
        pyperclip.copy(text)
        console.print("\n[bright_white]Copied to clipboard.[/bright_white]")
    except Exception as e:
        console.print(f"\n[bright_black]Clipboard unavailable here: {e} — skipping.[/bright_black]")


def escape_xml_text(text: str) -> str:
    """Escape minimal XML characters for text nodes (& and <). Leave > for readability."""
    s = str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;")


def escape_xml_attr(value: str) -> str:
    """Escape and quote an XML attribute value safely (includes surrounding quotes)."""
    return quoteattr(str(value))


def is_excluded_file(filename: str) -> bool:
    excluded_patterns = [
        ".pb.go",
        "_grpc.pb.go",
        "mock_",
        "/generated/",
        "/mocks/",
        ".gen.",
        "_generated.",
    ]
    return any(pattern in filename for pattern in excluded_patterns)


def is_allowed_filetype(path: str) -> bool:
    if is_excluded_file(path):
        return False
    name = os.path.basename(path).lower()
    if name in SPECIAL_FILES:
        return True
    _, ext = os.path.splitext(name)
    return ext in ALLOWED_EXTS


def download_file(url: str, target_path: str):
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)


def download_to_temp(url: str) -> str:
    with tempfile.NamedTemporaryFile(prefix="ofl_", delete=False) as tf:
        path = tf.name
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 15):
            if chunk:
                f.write(chunk)
    return path


def process_ipynb_file(nb_path: str) -> str:
    with open(nb_path, "r", encoding="utf-8", errors="ignore") as f:
        notebook_content = f.read()
    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(
        nbformat.reads(notebook_content, as_version=4)
    )
    return python_code


# ----------------------------
# Tokenization / preprocessing
# ----------------------------

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))


def preprocess_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as input_file_obj:
        input_text = input_file_obj.read()

    def process_text(text):
        text = re.sub(r"[\n\r]+", "\n", text)
        text = re.sub(r"[^a-zA-Z0-9\s_.,!?:;@#$%^&*()+\-=\[\]{}|\\<>`~'\"/]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    try:
        root = ET.fromstring(input_text)
        for elem in root.iter():
            if elem.text:
                elem.text = process_text(elem.text)
            if elem.tail:
                elem.tail = process_text(elem.tail)
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print("Text preprocessing completed with XML structure preserved.")
    except ET.ParseError:
        processed_text = process_text(input_text)
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(processed_text)
        print("XML parsing failed. Text preprocessing completed without XML structure.")


def get_token_count(text, disallowed_special=None, chunk_size=1000):
    if disallowed_special is None:
        disallowed_special = []
    enc = tiktoken.get_encoding("cl100k_base")
    # strip tags for counting
    text_without_tags = re.sub(r"<[^>]+>", "", text)
    chunks = [text_without_tags[i : i + chunk_size] for i in range(0, len(text_without_tags), chunk_size)]
    total_tokens = 0
    for chunk in chunks:
        tokens = enc.encode(chunk, disallowed_special=disallowed_special)
        total_tokens += len(tokens)
    return total_tokens


# ----------------------------
# Local folder processing
# ----------------------------

def process_local_folder(local_path: str) -> str:
    content = [f'<source type="local_directory" path={escape_xml_attr(local_path)}>' ]
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file in files:
            full_path = os.path.join(root, file)
            name_lower = file.lower()
            rel = os.path.relpath(full_path, local_path)
            _, ext = os.path.splitext(name_lower)
            # Image placeholder
            if ext in IMAGE_EXTS:
                try:
                    size = os.path.getsize(full_path)
                except OSError:
                    size = 0
                content.append(f'<file name={escape_xml_attr(rel)} type="image" bytes="{size}" skipped="true" />')
                continue
            if is_allowed_filetype(full_path):
                print(f"Processing {full_path}...")
                content.append(f'<file name={escape_xml_attr(rel)}>' )
                if ext == ".ipynb":
                    content.append(escape_xml_text(process_ipynb_file(full_path)))
                else:
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content.append(escape_xml_text(f.read()))
                    except Exception as e:
                        content.append(f"<error>{escape_xml_text(str(e))}</error>")
                content.append("</file>")
    content.append("</source>")
    print("All files processed.")
    return "\n".join(content)


# ----------------------------
# GitHub repository / PR / Issue processing
# ----------------------------

def process_github_repo(repo_url: str) -> str:
    require_token("GitHub repositories")
    api_base_url = "https://api.github.com/repos/"
    repo_url_parts = repo_url.split("https://github.com/")[-1].split("/")
    repo_name = "/".join(repo_url_parts[:2])

    branch_or_tag = ""
    subdirectory = ""
    if len(repo_url_parts) > 2 and repo_url_parts[2] == "tree":
        if len(repo_url_parts) > 3:
            branch_or_tag = repo_url_parts[3]
        if len(repo_url_parts) > 4:
            subdirectory = "/".join(repo_url_parts[4:])

    contents_url = f"{api_base_url}{repo_name}/contents"
    if subdirectory:
        contents_url = f"{contents_url}/{subdirectory}"
    if branch_or_tag:
        contents_url = f"{contents_url}?ref={branch_or_tag}"

    repo_content = [f'<source type="github_repository" url={escape_xml_attr(repo_url)}>' ]

    def walk(url, out):
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        files = resp.json()

        for f in files:
            if f["type"] == "dir" and f["name"] in EXCLUDED_DIRS:
                continue

            if f["type"] == "file":
                name_lower = f["name"].lower()
                _, ext = os.path.splitext(name_lower)
                if ext in IMAGE_EXTS:
                    out.append(f'<file name={escape_xml_attr(f["path"])} type="image" bytes="{f.get("size",0)}" skipped="true" />')
                    continue
                if is_allowed_filetype(f["name"]):
                    print(f"Processing {f['path']}...")
                    tmp = download_to_temp(f["download_url"])
                    try:
                        out.append(f'<file name={escape_xml_attr(f["path"])}>' )
                        if ext == ".ipynb":
                            out.append(escape_xml_text(process_ipynb_file(tmp)))
                        else:
                            with open(tmp, "r", encoding="utf-8", errors="ignore") as fh:
                                out.append(escape_xml_text(fh.read()))
                        out.append("</file>")
                    finally:
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass

            elif f["type"] == "dir":
                walk(f["url"], out)

    walk(contents_url, repo_content)
    repo_content.append("</source>")
    print("All files processed.")
    return "\n".join(repo_content)


def process_github_pull_request(pull_request_url: str) -> str:
    require_token("GitHub pull requests")
    parts = pull_request_url.split("/")
    repo_owner = parts[3]
    repo_name = parts[4]
    pr_number = parts[-1]

    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}"

    r = SESSION.get(api_url, timeout=30)
    r.raise_for_status()
    pr_data = r.json()

    diff_url = pr_data["diff_url"]
    diff_resp = SESSION.get(diff_url, timeout=60)
    diff_resp.raise_for_status()
    pr_diff = diff_resp.text

    comments_url = pr_data["comments_url"]
    review_comments_url = pr_data["review_comments_url"]
    c1 = SESSION.get(comments_url, timeout=30); c1.raise_for_status()
    c2 = SESSION.get(review_comments_url, timeout=30); c2.raise_for_status()
    comments_data = c1.json()
    review_comments_data = c2.json()

    all_comments = comments_data + review_comments_data
    # sort by 'position' if present
    all_comments.sort(key=lambda c: (c.get("position") is None, c.get("position", 10**9)))

    formatted = [f'<source type="github_pull_request" url={escape_xml_attr(pull_request_url)}>']
    formatted.append("<pull_request_info>")
    formatted.append(f"<title>{escape_xml_text(pr_data.get('title',''))}</title>")
    formatted.append(f"<description>{escape_xml_text(pr_data.get('body',''))}</description>")
    formatted.append("<merge_details>")
    merge_line = (
        f"{escape_xml_text(pr_data['user']['login'])} wants to merge "
        f"{pr_data.get('commits','?')} commit into "
        f"{repo_owner}:{pr_data['base']['ref']} from {pr_data['head']['label']}"
    )
    formatted.append(escape_xml_text(merge_line))
    formatted.append("</merge_details>")
    formatted.append("<diff_and_comments>")

    diff_lines = pr_diff.split("\n")
    comment_index = 0
    for i, line in enumerate(diff_lines):
        formatted.append(escape_xml_text(line))
        while comment_index < len(all_comments) and all_comments[comment_index].get("position") == i:
            c = all_comments[comment_index]
            formatted.append("<review_comment>")
            formatted.append(f"<author>{escape_xml_text(c['user']['login'])}</author>")
            formatted.append(f"<content>{escape_xml_text(c.get('body',''))}</content>")
            if "path" in c:
                formatted.append(f"<path>{escape_xml_text(c['path'])}</path>")
            if "original_line" in c and c["original_line"] is not None:
                formatted.append(f"<line>{c['original_line']}</line>")
            formatted.append("</review_comment>")
            comment_index += 1

    formatted.append("</diff_and_comments>")
    formatted.append("</pull_request_info>")

    # include repository snapshot
    repo_url = f"https://github.com/{repo_owner}/{repo_name}"
    formatted.append("<repository>")
    formatted.append(process_github_repo(repo_url))
    formatted.append("</repository>")
    formatted.append("</source>")

    print(f"Pull request {pr_number} and repository content processed successfully.")
    return "\n".join(formatted)


def process_github_issue(issue_url: str) -> str:
    require_token("GitHub issues")
    parts = issue_url.split("/")
    repo_owner = parts[3]
    repo_name = parts[4]
    issue_number = parts[-1]

    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
    r = SESSION.get(api_url, timeout=30)
    r.raise_for_status()
    issue_data = r.json()

    comments_url = issue_data["comments_url"]
    cr = SESSION.get(comments_url, timeout=30)
    cr.raise_for_status()
    comments_data = cr.json()

    formatted = [f'<source type="github_issue" url={escape_xml_attr(issue_url)}>']
    formatted.append("<issue_info>")
    formatted.append(f"<title>{escape_xml_text(issue_data.get('title',''))}</title>")
    formatted.append(f"<description>{escape_xml_text(issue_data.get('body',''))}</description>")
    formatted.append("<comments>")

    for comment in comments_data:
        formatted.append("<comment>")
        formatted.append(f"<author>{escape_xml_text(comment['user']['login'])}</author>")
        body = comment.get("body", "")
        formatted.append(f"<content>{escape_xml_text(body)}</content>")

        # Extract GitHub code links like ...#L12-L34
        code_snippets = re.findall(r"https://github.com/.*#L\d+-L\d+", body or "")
        for snippet_url in code_snippets:
            url_parts = snippet_url.split("#")
            file_url = url_parts[0].replace("/blob/", "/raw/")
            start_str, end_str = url_parts[1].split("-")
            start_line = int(start_str[1:])
            end_line = int(end_str[1:])

            fr = SESSION.get(file_url, timeout=30)
            fr.raise_for_status()
            file_content = fr.text

            code_lines = file_content.split("\n")[start_line - 1 : end_line]
            code_snippet = "\n".join(code_lines)

            formatted.append("<code_snippet>")
            formatted.append(f"<![CDATA[{code_snippet}]]>")
            formatted.append("</code_snippet>")

        formatted.append("</comment>")

    formatted.append("</comments>")
    formatted.append("</issue_info>")

    # include repository snapshot
    repo_url = f"https://github.com/{repo_owner}/{repo_name}"
    formatted.append("<repository>")
    formatted.append(process_github_repo(repo_url))
    formatted.append("</repository>")
    formatted.append("</source>")

    print(f"Issue {issue_number} and repository content processed successfully.")
    return "\n".join(formatted)


# ----------------------------
# Web crawling / PDFs / YouTube / arXiv
# ----------------------------

def is_same_domain(base_url, new_url):
    return urlparse(base_url).netloc == urlparse(new_url).netloc


def is_within_depth(base_url, current_url, max_depth):
    base_parts = urlparse(base_url).path.rstrip("/").split("/")
    current_parts = urlparse(current_url).path.rstrip("/").split("/")
    if current_parts[: len(base_parts)] != base_parts:
        return False
    return len(current_parts) - len(base_parts) <= max_depth


def process_pdf(url):
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    with open("temp.pdf", "wb") as pdf_file:
        pdf_file.write(resp.content)

    text = []
    with open("temp.pdf", "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            text.append(pdf_reader.pages[page].extract_text())

    try:
        os.remove("temp.pdf")
    except OSError:
        pass
    return " ".join(text)


def crawl_and_extract_text(base_url, max_depth, include_pdfs, ignore_epubs):
    visited_urls = set()
    urls_to_visit = [(base_url, 0)]
    processed_urls = []
    all_text = [f'<source type="web_documentation" url={escape_xml_attr(base_url)}>' ]

    while urls_to_visit:
        current_url, current_depth = urls_to_visit.pop(0)
        clean_url = current_url.split("#")[0]

        if clean_url in visited_urls:
            continue
        if not (is_same_domain(base_url, clean_url) and is_within_depth(base_url, clean_url, max_depth)):
            continue

        if ignore_epubs and clean_url.lower().endswith(".epub"):
            continue

        try:
            resp = SESSION.get(current_url, timeout=60)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            is_pdf = clean_url.lower().endswith(".pdf") or "application/pdf" in content_type

            if is_pdf and include_pdfs:
                text = process_pdf(clean_url)
                soup = None
            else:
                soup = BeautifulSoup(resp.content, "html.parser")
                for e in soup(["script", "style", "noscript", "template", "head", "title", "meta", "link"]):
                    e.decompose()
                for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
                    c.extract()
                text = "\n".join(line for line in soup.get_text("\n").splitlines() if line.strip())

            all_text.append(f"<page url={escape_xml_attr(clean_url)}>")
            all_text.append(escape_xml_text(text))
            all_text.append("</page>")
            processed_urls.append(clean_url)
            visited_urls.add(clean_url)
            print(f"Processed: {clean_url}")

            if soup is not None and current_depth < max_depth:
                for link in soup.find_all("a", href=True):
                    new_url = urljoin(current_url, link["href"]).split("#")[0]
                    if new_url not in visited_urls:
                        if include_pdfs or not new_url.lower().endswith(".pdf"):
                            if not (ignore_epubs and new_url.lower().endswith(".epub")):
                                if is_within_depth(base_url, new_url, max_depth):
                                    urls_to_visit.append((new_url, current_depth + 1))

        except requests.RequestException as e:
            print(f"Failed to retrieve {clean_url}: {e}")

    all_text.append("</source>")
    formatted_content = "\n".join(all_text)
    return {"content": formatted_content, "processed_urls": processed_urls}


def process_arxiv_pdf(arxiv_abs_url):
    pdf_url = arxiv_abs_url.replace("/abs/", "/pdf/") + ".pdf"
    resp = SESSION.get(pdf_url, timeout=60)
    resp.raise_for_status()

    with open("temp.pdf", "wb") as pdf_file:
        pdf_file.write(resp.content)

    text = []
    with open("temp.pdf", "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            text.append(pdf_reader.pages[page].extract_text())

    try:
        os.remove("temp.pdf")
    except OSError:
        pass

    formatted_text = f'<source type="arxiv_paper" url={escape_xml_attr(arxiv_abs_url)}>\n'
    formatted_text += "<paper>\n"
    formatted_text += escape_xml_text(" ".join(text))
    formatted_text += "\n</paper>\n"
    formatted_text += "</source>"

    print("ArXiv paper processed successfully.")
    return formatted_text


def extract_links(input_file, output_file):
    url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
        urls = re.findall(url_pattern, content)
    with open(output_file, "w", encoding="utf-8") as out:
        for url in urls:
            out.write(url + "\n")


def fetch_youtube_transcript(url):
    def extract_video_id(u):
        pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
        m = re.search(pattern, u)
        if m:
            return m.group(1)
        return None

    vid = extract_video_id(url)
    if not vid:
        return f'<source type="youtube_transcript" url={escape_xml_attr(url)}>\n<error>Could not extract video ID from URL.</error>\n</source>'

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(vid)
        formatter = TextFormatter()
        transcript = formatter.format_transcript(transcript_list)

        formatted_text = f'<source type="youtube_transcript" url={escape_xml_attr(url)}>\n'
        formatted_text += "<transcript>\n"
        formatted_text += escape_xml_text(transcript)
        formatted_text += "\n</transcript>\n"
        formatted_text += "</source>"
        return formatted_text
    except Exception as e:
        return f'<source type="youtube_transcript" url={escape_xml_attr(url)}>\n<error>{escape_xml_text(str(e))}</error>\n</source>'


def process_doi_or_pmid(identifier):
    if not ALLOW_SCIHUB:
        return (
            f'<source type="sci_hub_paper" identifier={escape_xml_attr(identifier)}>\n'
            f"<error>Disabled by default. Set ALLOW_SCIHUB=1 to enable (beware legal/ToS).</error>\n"
            f"</source>"
        )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
        "Connection": "keep-alive",
    }

    try:
        payload = {"sci-hub-plugin-check": "", "request": identifier}
        base_url = "https://sci-hub.se/"
        response = SESSION.post(base_url, headers=headers, data=payload, timeout=60)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        pdf_element = soup.find(id="pdf")

        if pdf_element is None:
            raise ValueError(
                f"No PDF found for identifier {identifier}. Sci-hub might be inaccessible or the document is not available."
            )

        # Extract the PDF source attribute (simple robust handling)
        try:
            raw_src = getattr(pdf_element, 'get', lambda *_: None)('src')
            if not raw_src:
                iframe = getattr(pdf_element, 'find', lambda *a, **k: None)('iframe')
                raw_src = getattr(iframe, 'get', lambda *_: None)('src') if iframe else None
            if not raw_src:
                raise ValueError(f"PDF src attribute not found for identifier {identifier}.")
            content = str(raw_src).replace("#navpanes=0&view=FitH", "").replace("//", "/")
        except Exception:
            raise ValueError(f"PDF src attribute not found for identifier {identifier}.")

        if content.startswith("/downloads"):
            pdf_url = "https://sci-hub.se" + content
        elif content.startswith("/tree"):
            pdf_url = "https://sci-hub.se" + content
        elif content.startswith("/uptodate"):
            pdf_url = "https://sci-hub.se" + content
        else:
            pdf_url = "https:/" + content

        pdf_filename = f"{identifier.replace('/', '-')}.pdf"
        wget.download(pdf_url, pdf_filename)

        with open(pdf_filename, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()

        formatted_text = f'<source type="sci_hub_paper" identifier={escape_xml_attr(identifier)}>\n'
        formatted_text += "<paper>\n"
        formatted_text += escape_xml_text(text)
        formatted_text += "\n</paper>\n"
        formatted_text += "</source>"

        os.remove(pdf_filename)
        print(f"Identifier {identifier} processed successfully.")
        return formatted_text
    except (requests.RequestException, ValueError) as e:
        error_text = f'<source type="sci_hub_paper" identifier={escape_xml_attr(identifier)}>\n'
        error_text += f"<error>{escape_xml_text(str(e))}</error>\n"
        error_text += "</source>"
        print(f"Error processing identifier {identifier}: {str(e)}")
        print("Sci-hub appears to be inaccessible or the document was not found. Please try again later.")
        return error_text


# ----------------------------
# Main
# ----------------------------

def main():
    console = Console()

    intro_text = Text("\nInput Paths or URLs Processed:\n", style="dodger_blue1")
    input_types = [
        ("• Local folder path (flattens all files into text)", "bright_white"),
        ("• GitHub repository URL (flattens all files into text)", "bright_white"),
        ("• GitHub pull request URL (PR + Repo)", "bright_white"),
        ("• GitHub issue URL (Issue + Repo)", "bright_white"),
        ("• Documentation URL (base URL)", "bright_white"),
        ("• YouTube video URL (to fetch transcript)", "bright_white"),
        ("• ArXiv Paper URL", "bright_white"),
        ("• DOI or PMID to search on Sci-Hub", "bright_white"),
    ]

    for input_type, color in input_types:
        intro_text.append(f"\n{input_type}", style=color)

    intro_panel = Panel(
        intro_text,
        expand=False,
        border_style="bold",
        title="[bright_white]Copy to File and Clipboard[/bright_white]",
        title_align="center",
        padding=(1, 1),
    )
    console.print(intro_panel)

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = Prompt.ask("\n[bold dodger_blue1]Enter the path or URL[/bold dodger_blue1]", console=console)

    console.print(f"\n[bold bright_green]You entered:[/bold bright_green] [bold bright_yellow]{input_path}[/bold bright_yellow]\n")

    output_file = "uncompressed_output.txt"
    processed_file = "compressed_output.txt"
    urls_list_file = "processed_urls.txt"

    with Progress(
        TextColumn("[bold bright_blue]{task.description}"),
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("[bright_blue]Processing...", total=100)

        try:
            # Route by input type
            if "github.com" in input_path:
                require_token("GitHub repositories/PRs/issues")
                if "/pull/" in input_path:
                    final_output = process_github_pull_request(input_path)
                elif "/issues/" in input_path:
                    final_output = process_github_issue(input_path)
                else:
                    final_output = process_github_repo(input_path)

            elif urlparse(input_path).scheme in ["http", "https"]:
                if "youtube.com" in input_path or "youtu.be" in input_path:
                    final_output = fetch_youtube_transcript(input_path)
                elif "arxiv.org" in input_path:
                    final_output = process_arxiv_pdf(input_path)
                else:
                    crawl_result = crawl_and_extract_text(
                        input_path, max_depth=2, include_pdfs=True, ignore_epubs=True
                    )
                    final_output = crawl_result["content"]
                    with open(urls_list_file, "w", encoding="utf-8") as urls_file:
                        urls_file.write("\n".join(crawl_result["processed_urls"]))

            elif (input_path.startswith("10.") and "/" in input_path) or input_path.isdigit():
                final_output = process_doi_or_pmid(input_path)

            else:
                # Local path
                final_output = process_local_folder(input_path)

            progress.update(task, advance=50)

            # Write uncompressed with context note
            context_note = (
                "<!-- Note: Minimal XML escaping applied. In text: & -> &amp;, < -> &lt; (we keep '>' literal). "
                "Attributes are safely quoted & escaped. -->\n"
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(context_note + final_output)

            # Compressed output
            preprocess_text(output_file, processed_file)
            progress.update(task, advance=50)

            compressed_text = safe_file_read(processed_file)
            compressed_token_count = get_token_count(compressed_text)
            console.print(f"\n[bright_green]Compressed Token Count:[/bright_green] [bold bright_cyan]{compressed_token_count}[/bold bright_cyan]")

            uncompressed_text = safe_file_read(output_file)
            uncompressed_token_count = get_token_count(uncompressed_text)
            console.print(f"[bright_green]Uncompressed Token Count:[/bright_green] [bold bright_cyan]{uncompressed_token_count}[/bold bright_cyan]")

            console.print(
                f"\n[bold bright_yellow]{processed_file}[/bold bright_yellow] and "
                f"[bold bright_blue]{output_file}[/bold bright_blue] have been created in the working directory."
            )

            try_copy_to_clipboard(uncompressed_text, console)

        except Exception as e:
            console.print(f"\n[bold red]An error occurred:[/bold red] {str(e)}")
            console.print("\nPlease check your input and try again.")
            raise  # Re-raise for CI logs


if __name__ == "__main__":
    main()
